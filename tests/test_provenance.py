"""Offline checks for recovered configuration selection and credential handling."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
from training.configuration import load_config
from training import ablation, ablation_amd
from cpu1 import hub_storage
from runners import common

ROOT = Path(__file__).resolve().parents[1]


class ConfigurationTests(unittest.TestCase):
    def test_all_published_configs_build_recorded_architectures(self):
        manifest = json.loads((ROOT/"experiments/published/manifest.json").read_text())
        self.assertEqual(len(manifest["experiments"]), 34)
        for record in manifest["experiments"]:
            with self.subTest(run=record["run"]):
                path = ROOT/"experiments/published"/record["config"]
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), record["config_sha256"])
                config = load_config(path)
                with torch.device("meta"), contextlib.redirect_stdout(io.StringIO()):
                    model = ablation.build_ablation_model(config)
                self.assertEqual(sum(p.numel() for p in model.parameters()), record["parameters"])
                del model

    def test_v3_cli_uses_record_in_both_backends(self):
        path = ROOT/"experiments/published/configs/run_04_v3.json"
        for module in (ablation, ablation_amd):
            with self.subTest(backend=module.__name__), patch.object(sys, "argv", [
                    "trainer", "--run", "run_04_v3", "--config", str(path), "--skip-install"]):
                args = module.parse_args()
                self.assertEqual(args.run, "run_04_v3")
                self.assertEqual(args.config, path)
                self.assertEqual(args.phase, 1)
                self.assertEqual(load_config(args.config)["chinchilla_tokens_per_param"], 150)

    def test_unknown_preset_does_not_silently_select_defaults(self):
        for module in (ablation, ablation_amd):
            with patch.object(sys, "argv", ["trainer", "--run", "run_04_v3"]), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as exc:
                    module.parse_args()
                self.assertEqual(exc.exception.code, 2)

    def test_config_rejects_missing_and_nonpositive_dimensions(self):
        storage = Path("D:/cpu1-readme-audit/test-checkpoints")
        storage.mkdir(parents=True, exist_ok=True)
        base = load_config(ROOT/"experiments/published/configs/run_04.json")
        for change in ({"d_model": 0}, {"seq_len": True}, {"tokenization": "unknown"}):
            with tempfile.TemporaryDirectory(dir=storage) as directory:
                path = Path(directory)/"config.json"
                path.write_text(json.dumps({**base, **change}))
                with self.assertRaises(ValueError):
                    load_config(path)


class StorageTests(unittest.TestCase):
    def test_mount_credentials_use_child_environment(self):
        storage = Path("D:/cpu1-readme-audit/test-checkpoints")
        storage.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=storage) as directory:
            with patch.object(hub_storage.sys, "platform", "linux"), patch.object(
                    hub_storage.shutil, "which", return_value="/usr/bin/hf-mount"), patch.object(
                    hub_storage.subprocess, "run", return_value=subprocess.CompletedProcess([], 0)) as run, patch.dict(
                    hub_storage.os.environ, {"HF_TOKEN": "parent-placeholder"}), contextlib.redirect_stdout(io.StringIO()):
                hub_storage.mount_repo("datasets/Cukinator/cpu1-ablation-dataset", directory,
                                       token="child-placeholder")
                args, kwargs = run.call_args
                self.assertEqual(args[0][1:4], ["start", "repo", "datasets/Cukinator/cpu1-ablation-dataset"])
                self.assertNotIn("child-placeholder", args[0])
                self.assertEqual(kwargs["env"]["HF_TOKEN"], "child-placeholder")
                self.assertEqual(hub_storage.os.environ["HF_TOKEN"], "parent-placeholder")

    def test_mount_error_does_not_repeat_process_output(self):
        with patch.object(hub_storage.sys, "platform", "linux"), patch.object(
                hub_storage.shutil, "which", return_value="/usr/bin/hf-mount"), patch.object(
                hub_storage.subprocess, "run", return_value=subprocess.CompletedProcess([], 3, stderr="private-placeholder")):
            with self.assertRaisesRegex(RuntimeError, "exit code 3") as exc:
                hub_storage.mount_repo("datasets/example/data", "D:/cpu1-readme-audit/test-checkpoints")
            self.assertNotIn("private-placeholder", str(exc.exception))


class RecipeTests(unittest.TestCase):
    def test_historical_dataset_recipes_are_plans_until_execute(self):
        for platform, chunk in (("kaggle", 500), ("lightning", 3000)):
            with patch.object(sys, "argv", ["recipe", "dataset"]), patch.object(
                    common.subprocess, "run") as run, contextlib.redirect_stdout(io.StringIO()) as output:
                common.main(platform)
                plan = json.loads(output.getvalue())
                self.assertEqual((plan["n_docs"], plan["chunk_size"]), (850000, chunk))
                run.assert_not_called()

    def test_amd_recipe_passes_config_as_separate_argument(self):
        with patch.object(sys, "argv", ["recipe", "ablation", "--run", "run_04_v3",
                "--backend", "amd", "--config", "experiments/published/configs/run_04_v3.json"]), patch.object(
                common.subprocess, "run") as run, contextlib.redirect_stdout(io.StringIO()) as output:
            common.main("lightning")
            command = json.loads(output.getvalue())["command"]
            self.assertIn("training.ablation_amd", command)
            self.assertEqual(command[command.index("--config")+1],
                             str((ROOT/"experiments/published/configs/run_04_v3.json").resolve()))
            run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
