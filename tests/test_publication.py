"""Checks that Hub publication cannot upload source or remove model/data files."""
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from huggingface_hub import CommitOperationAdd, CommitOperationDelete
from evaluation.publication import card_operations
from evaluation.audit import restore_evidence


class PublicationTests(unittest.TestCase):
    def test_only_readme_is_uploaded_and_copied_code_is_removed(self):
        card = Path(__file__).resolve().parents[1]/"docs/model_cards/cpu1-ablations-final.md"
        files = [SimpleNamespace(rfilename=p) for p in (
            "README.md", ".gitattributes", "run_04/model.pt",
            "evaluation/source/training/ablation.py", "evaluation/run_04.json",
            "evaluation/source/.gitignore", "evaluation/source/README.md")]
        operations = card_operations(card, files, "model")
        adds = [o.path_in_repo for o in operations if isinstance(o, CommitOperationAdd)]
        deletes = [o.path_in_repo for o in operations if isinstance(o, CommitOperationDelete)]
        self.assertEqual(adds, ["README.md"])
        self.assertEqual(set(deletes), {f.rfilename for f in files if f.rfilename.startswith("evaluation/")})
        self.assertNotIn("run_04/model.pt", deletes)

    def test_dataset_bookkeeping_and_parquet_are_preserved(self):
        card = Path(__file__).resolve().parents[1]/"docs/model_cards/cpu1-ablation-dataset.md"
        files = [SimpleNamespace(rfilename=p) for p in (
            "README.md", ".gitattributes", "byte_marginalized/shard_0000.parquet",
            "stream_checkpoint.json", "stream_carry.pkl", "provenance/dataset-review.json")]
        operations = card_operations(card, files, "dataset")
        deletes = [o.path_in_repo for o in operations if isinstance(o, CommitOperationDelete)]
        self.assertEqual(deletes, ["provenance/dataset-review.json"])

    def test_artifact_in_a_cleanup_directory_stops_the_entire_operation(self):
        for filename in ("evaluation/model.pt", "provenance/shard.parquet", "inference/model.bin"):
            with self.subTest(path=filename), self.assertRaises(ValueError):
                card_operations(Path("README.md"), [SimpleNamespace(rfilename=filename)], "model")

    def test_evidence_restoration_is_offline_and_preserves_bytes(self):
        storage = Path("D:/cpu1-readme-audit/test-checkpoints")
        storage.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=storage) as directory:
            root = Path(directory)
            evidence = root/"experiments/evaluation/model-audit-2026-10-06"
            evidence.mkdir(parents=True)
            raw = b'{"value": 2}\r\n'
            (evidence/"record.json").write_bytes(raw)
            (evidence/"CORPUS_LICENSE.txt").write_bytes(b"attribution\r\n")
            (evidence/"ignored.py").write_bytes(b"raise RuntimeError()")
            with patch("evaluation.audit.ROOT", root), patch("evaluation.audit.OUT", root/"outputs/audit"):
                restore_evidence(storage)
            self.assertEqual((root/"outputs/audit/record.json").read_bytes(), raw)
            self.assertFalse((root/"outputs/audit/ignored.py").exists())


if __name__ == "__main__":
    unittest.main()
