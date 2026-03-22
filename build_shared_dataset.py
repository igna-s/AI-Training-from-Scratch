"""
Build the shared dataset for the CPU-1 ablation study.

Runs ONCE. Produces both BPE and byte-level datasets from the same corpus,
using Qwen-14B as teacher. All 11 ablation runs share this dataset.

Pipeline:
  1. Download FineWeb corpus
  2. Extract top-k logprobs from Qwen-14B (4-bit NF4) — THE SLOW PART (~6-10h on T4)
  3a. Marginalize BPE→byte for byte-level runs (08-11)
  3b. Build BPE dataset for BPE runs (01-07)
  4. Upload to HuggingFace Hub

Usage:
    # Full pipeline
    python build_shared_dataset.py --n_docs 50000

    # Resume after interruption (extraction supports resume)
    python build_shared_dataset.py --n_docs 50000

    # Skip extraction (if logprobs already exist)
    python build_shared_dataset.py --n_docs 50000 --skip_extraction

    # Only upload existing dataset
    python build_shared_dataset.py --only_upload
"""

import os
import math
import json
import argparse
from pathlib import Path

import numpy as np
import torch
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

# ============================================================
# Configuration
# ============================================================

TEACHER_MODEL = "Qwen/Qwen2.5-14B"
DATASET_REPO = "Cukinator/cpu1-ablation-dataset"
OUTPUT_DIR = Path("./shared_dataset")
TOP_K_LOGPROBS = 64
PATCH_SIZE = 4
SEQ_LEN_PATCHES = 2048
SEQ_LEN_TOKENS = 2048
SEED = 42
BATCH_SIZE_TEACHER = 4


# ============================================================
# STEP 1: Extract teacher logprobs
# ============================================================

def load_teacher(model_name: str = TEACHER_MODEL):
    """Load Qwen-14B in 4-bit NF4 for logprob extraction (~8GB VRAM)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[Teacher] Loading {model_name} in 4-bit NF4...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        device_map="auto",
    )
    model.eval()
    print(f"[Teacher] Ready. Vocab size: {tokenizer.vocab_size}")
    return model, tokenizer


def extract_logprobs_single(
    model,
    tokenizer,
    text: str,
    top_k: int = TOP_K_LOGPROBS,
    seq_len: int = SEQ_LEN_TOKENS,
) -> dict:
    """
    Extract top-k logprobs for every position of a single text.

    Returns dict with:
        text_bytes:  list[int]   — UTF-8 bytes of original text
        positions:   list[dict]  — per-position top-k logprobs
        n_tokens:    int
    """
    enc = tokenizer(
        text, return_tensors='pt',
        truncation=True, max_length=seq_len,
    )
    input_ids = enc['input_ids'].to(model.device)

    with torch.no_grad():
        logits = model(input_ids).logits[0]  # [seq_len, vocab_size]

    log_probs = torch.log_softmax(logits, dim=-1)
    top_logprobs, top_indices = torch.topk(log_probs, k=top_k, dim=-1)

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
    byte_offset = 0
    positions = []

    for pos_idx, (token_id, token_str) in enumerate(
        zip(input_ids[0].tolist(), tokens)
    ):
        try:
            token_byte_seq = tokenizer.convert_tokens_to_string(
                [token_str]
            ).encode('utf-8')
        except Exception:
            token_byte_seq = b'\x00'

        pos_top = []
        for k_idx in range(top_k):
            cand_id = top_indices[pos_idx, k_idx].item()
            cand_lp = top_logprobs[pos_idx, k_idx].item()
            cand_tok = tokenizer.convert_ids_to_tokens([cand_id])[0]
            try:
                cand_bytes = list(
                    tokenizer.convert_tokens_to_string(
                        [cand_tok]
                    ).encode('utf-8')
                )
            except Exception:
                cand_bytes = [0]

            pos_top.append({
                'token': cand_tok,
                'bytes': cand_bytes,
                'logprob': cand_lp,
            })

        positions.append({
            'byte_offset': byte_offset,
            'top_tokens': pos_top,
        })
        byte_offset += len(token_byte_seq)

    return {
        'text_bytes': list(text.encode('utf-8')),
        'positions': positions,
        'n_tokens': len(tokens),
    }


def _append_to_parquet(records: list[dict], path: Path) -> None:
    """Append records to a Parquet file."""
    table = pa.table({
        'doc_id':     pa.array([r['doc_id'] for r in records], type=pa.int32()),
        'text_bytes': pa.array([r['text_bytes'] for r in records]),
        'positions':  pa.array([r['positions'] for r in records], type=pa.string()),
        'n_tokens':   pa.array([r['n_tokens'] for r in records], type=pa.int32()),
    })
    if path.exists():
        existing = pq.read_table(str(path))
        combined = pa.concat_tables([existing, table])
        pq.write_table(combined, str(path))
    else:
        pq.write_table(table, str(path))


def run_logprob_extraction(
    corpus_texts: list[str],
    output_path: str,
    resume: bool = True,
) -> None:
    """Extract logprobs for entire corpus. Supports resume."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    start_idx = 0
    if resume and output_path.exists():
        existing = pq.read_table(str(output_path))
        start_idx = len(existing)
        print(f"[Extraction] Resuming from doc {start_idx}/{len(corpus_texts)}")

    model, tokenizer = load_teacher()
    batch_results = []

    for i in tqdm(range(start_idx, len(corpus_texts)), desc="Extracting logprobs"):
        text = corpus_texts[i]
        if not text.strip():
            continue

        result = extract_logprobs_single(model, tokenizer, text, TOP_K_LOGPROBS)
        batch_results.append({
            'doc_id':     i,
            'text_bytes': result['text_bytes'],
            'positions':  json.dumps(result['positions']),
            'n_tokens':   result['n_tokens'],
        })

        # Save every 100 docs for crash safety
        if len(batch_results) >= 100:
            _append_to_parquet(batch_results, output_path)
            batch_results = []
            print(f"  Saved through doc {i}")

    if batch_results:
        _append_to_parquet(batch_results, output_path)

    print(f"[Extraction] Complete: {output_path}")


# ============================================================
# STEP 2a: Marginalize BPE→Byte
# ============================================================

def build_byte_level_dataset(
    logprobs_parquet: str,
    output_dir: str,
    seq_len: int = SEQ_LEN_PATCHES,
) -> dict:
    """
    Convert BPE logprobs to byte-level distributions.
    Produces dataset for runs 08-11 (byte-level).

    For each BPE token boundary that coincides with a patch boundary:
        first-byte marginalization — sum P(token) for all tokens sharing
        the same first byte to get P(byte_0 | context).

    Output per sequence:
        patches:       [seq_len, 4] uint8
        targets:       [seq_len, 4] uint8
        teacher_probs: [seq_len, 256] float32  (sparse)
        teacher_mask:  [seq_len] bool
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logprobs_table = pq.read_table(logprobs_parquet)

    all_patches = []
    all_targets = []
    all_teacher_probs = []
    all_teacher_mask = []
    stats = {'n_sequences': 0, 'n_patches_total': 0, 'n_with_teacher': 0}

    for row in tqdm(logprobs_table.to_pylist(), desc="Marginalizing BPE→byte"):
        text_bytes = bytes(row['text_bytes'])
        positions = json.loads(row['positions'])

        # Map byte_offset → marginalized byte distribution
        byte_to_teacher = {}
        for pos in positions:
            byte_off = pos['byte_offset']
            top_toks = pos['top_tokens']

            byte_probs = {}
            for item in top_toks:
                if not item['bytes']:
                    continue
                first_byte = item['bytes'][0]
                prob = math.exp(item['logprob'])
                byte_probs[first_byte] = byte_probs.get(first_byte, 0.0) + prob

            total = sum(byte_probs.values())
            if total > 0:
                byte_probs = {k: v / total for k, v in byte_probs.items()}
            byte_to_teacher[byte_off] = byte_probs

        # Pack into patches of 4 bytes
        n_patches = len(text_bytes) // PATCH_SIZE
        for p_idx in range(n_patches - 1):
            patch_start = p_idx * PATCH_SIZE
            target_start = (p_idx + 1) * PATCH_SIZE

            patch = list(text_bytes[patch_start:patch_start + PATCH_SIZE])
            target = list(text_bytes[target_start:target_start + PATCH_SIZE])

            has_teacher = target_start in byte_to_teacher
            if has_teacher:
                bp = byte_to_teacher[target_start]
                teacher_dist = [bp.get(b, 0.0) for b in range(256)]
                stats['n_with_teacher'] += 1
            else:
                teacher_dist = [0.0] * 256

            all_patches.append(patch)
            all_targets.append(target)
            all_teacher_probs.append(teacher_dist)
            all_teacher_mask.append(has_teacher)
            stats['n_patches_total'] += 1

            if len(all_patches) >= seq_len:
                _save_byte_sequence(
                    all_patches[:seq_len],
                    all_targets[:seq_len],
                    all_teacher_probs[:seq_len],
                    all_teacher_mask[:seq_len],
                    output_dir,
                    stats['n_sequences'],
                )
                all_patches = all_patches[seq_len:]
                all_targets = all_targets[seq_len:]
                all_teacher_probs = all_teacher_probs[seq_len:]
                all_teacher_mask = all_teacher_mask[seq_len:]
                stats['n_sequences'] += 1

    coverage = stats['n_with_teacher'] / max(stats['n_patches_total'], 1) * 100
    stats['coverage_pct'] = coverage

    print(f"[Byte dataset] {stats['n_sequences']} sequences")
    print(f"[Byte dataset] Teacher coverage: {coverage:.1f}%")

    _save_split_info(stats['n_sequences'], output_dir)
    return stats


def _save_byte_sequence(patches, targets, teacher_probs, teacher_mask,
                        output_dir: Path, seq_idx: int) -> None:
    """Save one byte-level sequence to a Parquet shard."""
    table = pa.table({
        'patches': pa.array(
            [bytes(p) for p in patches], type=pa.list_(pa.uint8())
        ),
        'targets': pa.array(
            [bytes(t) for t in targets], type=pa.list_(pa.uint8())
        ),
        'teacher_probs': pa.array(teacher_probs),
        'teacher_mask': pa.array(teacher_mask, type=pa.bool_()),
    })
    shard = seq_idx // 1000
    shard_path = output_dir / f"shard_{shard:04d}.parquet"
    if shard_path.exists():
        existing = pq.read_table(str(shard_path))
        pq.write_table(pa.concat_tables([existing, table]), str(shard_path))
    else:
        pq.write_table(table, str(shard_path))


# ============================================================
# STEP 2b: Build BPE dataset
# ============================================================

def build_bpe_dataset(
    logprobs_parquet: str,
    output_dir: str,
    seq_len: int = SEQ_LEN_TOKENS,
) -> dict:
    """
    Build BPE dataset from the same teacher logprobs.
    Produces dataset for runs 01-07 (BPE).

    Teacher logprobs are kept in BPE space (no marginalization).
    """
    from transformers import AutoTokenizer

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(TEACHER_MODEL)
    logprobs_table = pq.read_table(logprobs_parquet)

    stats = {'n_sequences': 0, 'n_tokens_total': 0}
    buffer_ids = []
    buffer_probs = []
    buffer_mask = []

    for row in tqdm(logprobs_table.to_pylist(), desc="Building BPE dataset"):
        text_bytes = bytes(row['text_bytes'])
        positions = json.loads(row['positions'])

        text = text_bytes.decode('utf-8', errors='replace')
        enc = tokenizer(text, truncation=True, max_length=seq_len * 2)
        ids = enc['input_ids']

        for pos_idx, (token_id, pos_data) in enumerate(
            zip(ids, positions[:len(ids)])
        ):
            top_toks = pos_data.get('top_tokens', [])
            probs = [0.0] * TOP_K_LOGPROBS
            has_teacher = len(top_toks) > 0

            if has_teacher:
                for k_idx, item in enumerate(top_toks[:TOP_K_LOGPROBS]):
                    probs[k_idx] = math.exp(item['logprob'])

            buffer_ids.append(token_id)
            buffer_probs.append(probs)
            buffer_mask.append(has_teacher)
            stats['n_tokens_total'] += 1

            if len(buffer_ids) >= seq_len + 1:
                _save_bpe_sequence(
                    buffer_ids[:seq_len],
                    buffer_ids[1:seq_len + 1],
                    buffer_probs[:seq_len],
                    buffer_mask[:seq_len],
                    output_dir,
                    stats['n_sequences'],
                )
                buffer_ids = buffer_ids[seq_len:]
                buffer_probs = buffer_probs[seq_len:]
                buffer_mask = buffer_mask[seq_len:]
                stats['n_sequences'] += 1

    print(f"[BPE dataset] {stats['n_sequences']} sequences")
    _save_split_info(stats['n_sequences'], output_dir)
    return stats


def _save_bpe_sequence(input_ids, labels, teacher_probs, teacher_mask,
                       output_dir: Path, seq_idx: int) -> None:
    """Save one BPE sequence to a Parquet shard."""
    table = pa.table({
        'input_ids': pa.array(input_ids, type=pa.int32()),
        'labels': pa.array(labels, type=pa.int32()),
        'teacher_probs_bpe': pa.array(teacher_probs),
        'teacher_mask_bpe': pa.array(teacher_mask, type=pa.bool_()),
    })
    shard = seq_idx // 1000
    shard_path = output_dir / f"shard_{shard:04d}.parquet"
    if shard_path.exists():
        existing = pq.read_table(str(shard_path))
        pq.write_table(pa.concat_tables([existing, table]), str(shard_path))
    else:
        pq.write_table(table, str(shard_path))


# ============================================================
# STEP 3: Split info & upload
# ============================================================

def _save_split_info(n_sequences: int, output_dir: Path) -> None:
    """Save train/val/test split with fixed seed."""
    rng = np.random.RandomState(SEED)
    indices = rng.permutation(n_sequences).tolist()

    n_train = int(0.95 * n_sequences)
    n_val = int(0.025 * n_sequences)

    split = {
        'train': indices[:n_train],
        'val': indices[n_train:n_train + n_val],
        'test': indices[n_train + n_val:],
        'seed': SEED,
        'n_sequences': n_sequences,
    }
    with open(output_dir / 'split_info.json', 'w') as f:
        json.dump(split, f)


def upload_dataset_to_hub(local_dir: str = str(OUTPUT_DIR)) -> None:
    """Upload complete dataset to HuggingFace Hub."""
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(DATASET_REPO, repo_type="dataset", exist_ok=True)
    api.upload_folder(
        folder_path=local_dir,
        repo_id=DATASET_REPO,
        repo_type="dataset",
        ignore_patterns=["*.tmp", "__pycache__"],
    )
    print(f"\n[Upload] Dataset available at:")
    print(f"  https://huggingface.co/datasets/{DATASET_REPO}")


# ============================================================
# MAIN: Full pipeline
# ============================================================

def build_full_dataset(corpus_name: str = "HuggingFaceFW/fineweb",
                       n_docs: int = 50_000,
                       skip_extraction: bool = False) -> None:
    """
    Complete dataset construction pipeline.

    Step 1: Download FineWeb corpus
    Step 2: Extract logprobs with Qwen-14B (SLOW — ~6-10h on T4)
    Step 3a: Marginalize BPE→byte
    Step 3b: Build BPE dataset
    Step 4: Upload to HuggingFace Hub

    Only needs to run ONCE.
    """
    from datasets import load_dataset

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logprobs_path = OUTPUT_DIR / "teacher_logprobs" / "logprobs_raw.parquet"

    if not skip_extraction:
        # Step 1: Download corpus
        print(f"\n{'=' * 60}")
        print("STEP 1: Downloading FineWeb corpus")
        print(f"{'=' * 60}")

        fw = load_dataset(corpus_name, name="sample-10BT", split="train",
                          streaming=True)
        texts = []
        for i, doc in enumerate(fw):
            if i >= n_docs:
                break
            texts.append(doc['text'])

        print(f"  Corpus: {len(texts):,} documents")

        # Save corpus sample for reference
        corpus_path = OUTPUT_DIR / "corpus_texts.json"
        with open(corpus_path, 'w') as f:
            json.dump(texts[:1000], f)

        # Step 2: Extract logprobs (the slow part)
        print(f"\n{'=' * 60}")
        print("STEP 2: Extracting teacher logprobs (Qwen-14B)")
        print("  (~6-10 hours on T4, supports resume)")
        print(f"{'=' * 60}")

        run_logprob_extraction(texts, str(logprobs_path), resume=True)
    else:
        assert logprobs_path.exists(), (
            f"Logprobs not found at {logprobs_path}. "
            f"Run without --skip_extraction first."
        )

    # Step 3a: Byte-level dataset
    print(f"\n{'=' * 60}")
    print("STEP 3a: Building byte-level dataset (runs 08-11)")
    print(f"{'=' * 60}")

    byte_stats = build_byte_level_dataset(
        str(logprobs_path),
        str(OUTPUT_DIR / "byte_marginalized"),
    )

    # Step 3b: BPE dataset
    print(f"\n{'=' * 60}")
    print("STEP 3b: Building BPE dataset (runs 01-07)")
    print(f"{'=' * 60}")

    bpe_stats = build_bpe_dataset(
        str(logprobs_path),
        str(OUTPUT_DIR / "bpe_tokenized"),
    )

    # Step 4: Upload
    print(f"\n{'=' * 60}")
    print("STEP 4: Uploading to HuggingFace Hub")
    print(f"{'=' * 60}")

    upload_dataset_to_hub()

    # Summary
    print(f"\n{'=' * 60}")
    print("DATASET READY — Summary")
    print(f"{'=' * 60}")
    print(f"  Byte-level: {byte_stats['n_sequences']:,} sequences")
    print(f"  BPE:        {bpe_stats['n_sequences']:,} sequences")
    print(f"  Coverage teacher (byte): {byte_stats['coverage_pct']:.1f}%")
    print(f"  Hub: https://huggingface.co/datasets/{DATASET_REPO}")
    print(f"\n  All 11 ablation runs can start now.")
    print(f"  Teacher does NOT need to run again.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build shared dataset for CPU-1 ablation study"
    )
    parser.add_argument('--n_docs', type=int, default=50_000,
                        help="Number of FineWeb documents")
    parser.add_argument('--skip_extraction', action='store_true',
                        help="Skip logprob extraction (if already exists)")
    parser.add_argument('--only_upload', action='store_true',
                        help="Only upload existing dataset to Hub")
    args = parser.parse_args()

    if args.only_upload:
        upload_dataset_to_hub()
    else:
        build_full_dataset(n_docs=args.n_docs,
                           skip_extraction=args.skip_extraction)
