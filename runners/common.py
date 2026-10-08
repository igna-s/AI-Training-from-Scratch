"""Explicit plans for historical platform recipes using the current packages."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main(platform: str) -> None:
    parser = argparse.ArgumentParser(description=f"CPU-1 {platform} experiment recipe")
    parser.add_argument("task", choices=("dataset", "ablation"))
    parser.add_argument("--execute", action="store_true", help="Run the plan; otherwise print it")
    parser.add_argument("--n-docs", type=int, default=850_000)
    parser.add_argument("--chunk-size", type=int, default=500 if platform == "kaggle" else 3000)
    parser.add_argument("--shard-offset", type=int, default=0)
    parser.add_argument("--local-only", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("shared_dataset"))
    parser.add_argument("--run")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--backend", choices=("cuda", "amd"), default="cuda")
    parser.add_argument("--resume-ckpt", type=Path)
    parser.add_argument("--phase", type=int, choices=(1, 2))
    parser.add_argument("--phase1-ckpt", type=Path)
    args = parser.parse_args()
    if args.n_docs <= 0 or args.chunk_size <= 0 or args.shard_offset < 0:
        parser.error("Document/chunk counts must be positive; shard offset must be nonnegative")
    if args.task == "dataset":
        plan = {"platform": platform, "task": args.task, "source": "data_pipeline.prepare",
                "n_docs": args.n_docs, "chunk_size": args.chunk_size,
                "shard_offset": args.shard_offset, "stream_to_hub": not args.local_only,
                "output_dir": str(args.output_dir),
                "provenance": "Recovered launcher settings; not an observed document count"}
    else:
        if not args.run:
            parser.error("ablation requires --run")
        module = "training.ablation_amd" if args.backend == "amd" else "training.ablation"
        command = [sys.executable, "-m", module, "--run", args.run, "--skip-install"]
        if args.config:
            command += ["--config", str(args.config.resolve())]
        if args.resume_ckpt:
            command += ["--resume_ckpt", str(args.resume_ckpt.resolve())]
        if args.phase:
            command += ["--phase", str(args.phase)]
        if args.phase1_ckpt:
            command += ["--phase1_ckpt", str(args.phase1_ckpt.resolve())]
        plan = {"platform": platform, "task": args.task, "command": command}
    print(json.dumps(plan, indent=2))
    if not args.execute:
        return
    if platform == "kaggle" and not os.environ.get("HF_TOKEN"):
        try:
            from kaggle_secrets import UserSecretsClient
            os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
        except ImportError:
            if args.task == "dataset" and not args.local_only:
                raise RuntimeError("Configure HF_TOKEN through Kaggle Secrets or the environment") from None
    if args.task == "ablation":
        subprocess.run(plan["command"], check=True, cwd=ROOT)
        return
    if not args.local_only and not os.environ.get("HF_TOKEN"):
        raise RuntimeError("HF_TOKEN is required for dataset uploads")
    from data_pipeline import prepare
    prepare.OUTPUT_DIR = args.output_dir
    prepare.build_full_dataset(n_docs=args.n_docs, chunk_size=args.chunk_size,
                               shard_offset=args.shard_offset,
                               stream_to_hub=not args.local_only)
