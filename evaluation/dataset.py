"""Inspect three pinned dataset shards without loading a teacher or model."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

REPO = "Cukinator/cpu1-ablation-dataset"
REVISION = "bd75d7b811ab730cc912aaa671f9eaaf49ba4bdf"
FILES = ("byte_marginalized/shard_0000.parquet",
         "byte_marginalized/shard_0481.parquet", "bpe_tokenized/shard_0000.parquet")
EXPECTED_HASHES = {
    FILES[0]: "f68a6659ef7b3ddd7c0e7066766f6d1bfbfba639c640a3aa1265b485a2420dae",
    FILES[1]: "a78f45e1510308b7d6be01fbc52c4043ccc7a9ad9a0b47634e16c508496e3dd9",
    FILES[2]: "684797be8d37a18ef3a31af7fc5964cf63d51b2dca1fab29894da267a1229e5b"}
ROOT = Path(__file__).resolve().parents[1]


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_shard(path, filename):
    import numpy as np
    import pyarrow.parquet as pq
    import torch
    from cpu1.components import ByteLevelDataset
    digest = file_hash(path)
    if digest != EXPECTED_HASHES[filename]:
        raise ValueError("Sample hash differs from the reviewed dataset revision")
    parquet = pq.ParquetFile(path)
    first = next(parquet.iter_batches(batch_size=1)).to_pylist()[0]
    result = {"file": filename, "bytes": path.stat().st_size, "sha256": digest,
              "rows": parquet.metadata.num_rows, "schema": str(parquet.schema_arrow),
              "first_row_lengths": {k: len(v) if isinstance(v, list) else v
                                    for k, v in first.items()}}
    if filename.startswith("byte"):
        table = parquet.read()
        rows = table.to_pylist()
        result["rows_with_any_teacher_mask"] = sum(any(r["teacher_mask"]) for r in rows)
        result["all_row_lengths"] = [{k: len(v) if isinstance(v, list) else v
                                      for k,v in r.items()} for r in rows]
        result["mask_lengths"] = sorted(set(len(r["teacher_mask"]) for r in rows))
        source = np.asarray(first["patches"])
        targets = np.asarray(first["targets"])
        result["global_shift_mismatches_first_row"] = int(np.count_nonzero(source[4:] != targets[:-4]))
        result["global_shift_positions_first_row"] = len(source) - 4
        result["shift_note"] = "Comparison across the packed row; document separators/ends can differ. No targets were rewritten."
        seq_len = len(source) // 4
        config = {"seq_len": seq_len, "patch_size": 4, "training_signal": "distillation",
                  "use_embedding_alignment": True, "teacher_dim": first["teacher_dim"]}
        sample = ByteLevelDataset([first], config)[0]
        stored = torch.tensor(first["teacher_probs"], dtype=torch.float32).reshape(-1, 256)
        loaded = sample["teacher_probs"]
        if stored.shape == loaded.shape:
            result["reader_dense_comparison"] = {
                "stored_nonzero_positions": int((stored.sum(-1) != 0).sum()),
                "reader_nonzero_positions": int((loaded.sum(-1) != 0).sum()),
                "max_absolute_difference": float((stored-loaded).abs().max()),
                "mask_true_positions": int(sum(first["teacher_mask"]))}
        result["short_boolean_masks"] = sum(0 < len(r["teacher_mask"]) < seq_len and
                                           all(isinstance(x, bool) for x in r["teacher_mask"])
                                           for r in rows)
    else:
        result["global_shift_mismatches_first_row"] = sum(
            a != b for a, b in zip(first["input_ids"][1:], first["labels"][:-1]))
        result["global_shift_positions_first_row"] = len(first["input_ids"]) - 1
    return result



def inspect_tokenizer(storage, offline=False):
    from transformers import AutoTokenizer
    revision = "3aab1f1954e9cc14eb9509a215f9e5ca08227a9b"
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B", revision=revision,
        cache_dir=storage/"hf/hub", local_files_only=offline, trust_remote_code=False)
    text = "A" + chr(0x1FAE8) + "B"
    encoded = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    return {"model": "Qwen/Qwen2.5-3B", "revision": revision,
            "synthetic_text_codepoints": [ord(c) for c in text],
            "ids": encoded["input_ids"], "character_offsets": encoded["offset_mapping"],
            "individually_decoded_codepoints": [[ord(c) for c in tokenizer.decode([i])]
                                                for i in encoded["input_ids"]],
            "individually_decoded_first_bytes": [tokenizer.decode([i]).encode("utf-8")[0]
                                                 for i in encoded["input_ids"]],
            "original_emoji_first_byte": chr(0x1FAE8).encode("utf-8")[0]}



def parquet_inventory(info):
    return {f.rfilename: (f.size, f.lfs.sha256 if f.lfs else f.blob_id)
            for f in info.siblings if f.rfilename.endswith(".parquet")}


def publish_card(report_path, storage):
    """Publish documentation/evidence only after checking every parquet hash."""
    from huggingface_hub import HfApi, CommitOperationAdd, hf_hub_download
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("Set HF_TOKEN in the environment before publishing")
    api = HfApi(token=token)
    if api.whoami()["name"].lower() != "cukinator":
        raise ValueError("Token does not belong to the authorized repository owner")
    expected = api.dataset_info(REPO, revision=REVISION, files_metadata=True)
    current = api.dataset_info(REPO, files_metadata=True)
    if parquet_inventory(expected) != parquet_inventory(current):
        raise ValueError("Dataset files changed; review changed data before publishing")
    card = ROOT/"docs/model_cards/cpu1-ablation-dataset.md"
    files = [("README.md", card), ("provenance/dataset-review.json", report_path)]
    for name in ("source-publication.json", "history-verification.json"):
        path = ROOT/"outputs/provenance"/name
        if path.exists():
            files.append(("provenance/"+name, path))
    commit = api.create_commit(repo_id=REPO, repo_type="dataset", parent_commit=current.sha,
        commit_message="Document extraction history, reproducible sample checks and reader limitations",
        operations=[CommitOperationAdd(path_in_repo=name, path_or_fileobj=str(path))
                    for name,path in files])
    published = hf_hub_download(REPO, "README.md", repo_type="dataset",
        revision=commit.oid, cache_dir=storage/"hf/hub", token=token)
    if Path(published).read_bytes() != card.read_bytes():
        raise RuntimeError("Published dataset card verification failed")
    result = {"repo": REPO, "oid": commit.oid, "url": commit.commit_url,
              "data_inventory_unchanged": True}
    (ROOT/"outputs/provenance/dataset-publication.json").write_text(
        json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print("Published and verified dataset card:", commit.oid)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--storage", type=Path, default=Path("D:/cpu1-readme-audit"))
    parser.add_argument("--offline", action="store_true", help="Inspect previously downloaded files")
    parser.add_argument("--output", type=Path, default=ROOT/"outputs/provenance/dataset-review.json")
    parser.add_argument("--tokenizer", action="store_true", help="Also inspect a pinned tokenizer on synthetic UTF-8 text")
    parser.add_argument("--publish-card", action="store_true", help="Publish the reviewed card/report with HF_TOKEN; parquet inventory must match")
    args = parser.parse_args()
    storage = args.storage.resolve()
    if sys.platform == "win32" and storage.drive.upper() != "D:":
        parser.error("Use D: for dataset files and caches")
    samples = storage/"provenance-samples"
    os.environ["HF_HOME"] = str(storage/"hf")
    os.environ["HF_HUB_CACHE"] = str(storage/"hf/hub")
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    inventory = {}
    if not args.offline:
        from huggingface_hub import HfApi, hf_hub_download
        info = HfApi().dataset_info(REPO, revision=REVISION, files_metadata=True)
        for subset in ("byte_marginalized", "bpe_tokenized"):
            files = [f for f in info.siblings if f.rfilename.startswith(subset+"/")
                     and f.rfilename.endswith(".parquet")]
            ids = sorted(int(Path(f.rfilename).stem.split("_")[-1]) for f in files)
            inventory[subset] = {"files": len(files), "bytes": sum(f.size for f in files),
                                 "min_shard": min(ids), "max_shard": max(ids),
                                 "missing_shards": sorted(set(range(min(ids), max(ids)+1))-set(ids))}
        for filename in FILES:
            hf_hub_download(REPO, filename, repo_type="dataset", revision=REVISION,
                            local_dir=samples, cache_dir=storage/"hf/hub")
    records = [inspect_shard(samples/filename, filename) for filename in FILES]
    result = {"dataset": REPO, "revision": REVISION, "inventory": inventory,
              "scope": "Three shard footers; all lengths/teacher masks in two byte shards; first BPE row. Not dataset-wide validation.",
              "records": records}
    if args.tokenizer:
        result["tokenizer_conversion"] = inspect_tokenizer(storage, args.offline)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({**result, "records": [{k:v for k,v in r.items() if k not in ("schema", "all_row_lengths")} for r in records]}, indent=2))
    if args.publish_card:
        publish_card(args.output, storage)


if __name__ == "__main__":
    main()
