"""Checkpoint compatibility checks for the public package layout."""
from pathlib import Path
import tempfile
import unittest

import torch

from cpu1.components import pack_model_compact
from evaluation.checkpoints import load_run
from training.ablation import build_ablation_model, get_run_config


class CheckpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.storage = Path('D:/cpu1-readme-audit/test-checkpoints')
        cls.storage.mkdir(parents=True, exist_ok=True)

    def config(self):
        cfg = get_run_config('run_04')
        cfg.update(d_model=16, n_layers=1, d_ff=32, n_heads=2,
                   head_dim=8, window_size=8, local_decoder_dim=16,
                   local_decoder_heads=2, use_gradient_checkpointing=False)
        return cfg

    def test_fp32_and_compact_payloads_load_in_new_package(self):
        config = self.config()
        model = build_ablation_model(config).eval()
        x = torch.tensor(list(b'abcdefgh')).reshape(1, 2, 4)
        for format_name in ('fp32', 'compact_2bit'):
            with self.subTest(format=format_name):
                with tempfile.TemporaryDirectory(dir=self.storage) as directory:
                    payload = {'config': config, 'format': format_name}
                    if format_name == 'fp32':
                        payload['state_dict'] = model.state_dict()
                    else:
                        payload['compact'] = pack_model_compact(model)
                    path = Path(directory)/'model.pt'
                    torch.save(payload, path)
                    engine, _, loaded_config, missing, extra = load_run(path, 'test')
                    self.assertEqual(missing, [])
                    self.assertEqual(extra, [])
                    self.assertEqual(loaded_config['d_model'], 16)
                    with torch.inference_mode():
                        heads = engine._get_model()(x, byte_targets=x)[0]
                    self.assertEqual([tuple(h.shape) for h in heads], [(1, 2, 256)]*4)
                    self.assertTrue(all(torch.isfinite(h).all() for h in heads))

    def test_incompatible_payload_is_rejected(self):
        config = self.config()
        state = build_ablation_model(config).state_dict()
        state.pop(next(iter(state)))
        with tempfile.TemporaryDirectory(dir=self.storage) as directory:
            path = Path(directory)/'model.pt'
            torch.save({'config': config, 'format': 'fp32', 'state_dict': state}, path)
            with self.assertRaises(RuntimeError):
                load_run(path, 'incompatible')


if __name__ == '__main__':
    unittest.main()
