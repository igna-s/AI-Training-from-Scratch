"""Publication locations and README-only Hub operations."""
from pathlib import Path

GITHUB = "https://github.com/igna-s/New-1.58"
SOURCE_REF = "cpu1-experiment-2026-10-08"
SOURCE_TREE = f"{GITHUB}/tree/{SOURCE_REF}"
SOURCE_BLOB = f"{GITHUB}/blob/{SOURCE_REF}"
AUDIT_PATH = "experiments/evaluation/model-audit-2026-10-06"
AUDIT_TREE = f"{SOURCE_TREE}/{AUDIT_PATH}"
AUDIT_BLOB = f"{SOURCE_BLOB}/{AUDIT_PATH}"
DATASET_REPORT = f"{SOURCE_BLOB}/experiments/evaluation/dataset-review-2026-10-08.json"
HISTORICAL_AUDIT = f"{GITHUB}/blob/bc58efaa7ef00dc24f57ea7479415520fb92705f/eval/audit_readmes.py"


def card_operations(card: Path, siblings, repo_type: str):
    """Keep code and evidence on GitHub; preserve all model/data artifacts."""
    from huggingface_hub import CommitOperationAdd, CommitOperationDelete
    prefixes = ("evaluation/", "provenance/", "cpu1/", "training/", "data_pipeline/",
                "experiments/", "tests/", "runners/", "requirements/", "docs/", "inference/")
    code_suffixes = (".py", ".c", ".cpp", ".h", ".cu", ".ipynb")
    artifact_suffixes = (".pt", ".pth", ".safetensors", ".bin", ".gguf", ".onnx",
                         ".parquet", ".arrow", ".pkl", ".npy", ".npz")
    if repo_type not in ("model", "dataset"):
        raise ValueError("Unsupported Hub repository type")
    removed = sorted(f.rfilename for f in siblings
                     if f.rfilename.startswith(prefixes) or
                     f.rfilename.endswith(code_suffixes) or f.rfilename == ".gitignore")
    if any(name.endswith(artifact_suffixes) for name in removed):
        raise ValueError("Refusing to remove a model or dataset artifact")
    return [CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=str(card))] + [
        CommitOperationDelete(path_in_repo=name) for name in removed]
