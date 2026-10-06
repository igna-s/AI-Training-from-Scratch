"""Offline regression checks for the scoring protocol, not trained-model quality."""
import json
import math
from pathlib import Path
import tempfile
import unittest

import torch

from evaluation.audit import loss_eval, save
from evaluation.helpers import run_check
from cpu1.components import LocalByteDecoder


class AuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_local_teacher_forcing_does_not_look_ahead(self):
        torch.manual_seed(4)
        decoder = LocalByteDecoder(16, 4, 16, 4, use_ternary=False).eval()
        hidden = torch.randn(1, 2, 16)
        targets = torch.tensor([[[5, 6, 7, 8], [9, 10, 11, 12]]])
        future = targets.clone()
        future[:, :, 2:] += 20
        with torch.inference_mode():
            a, b = decoder(hidden, targets), decoder(hidden, future)
        # Head 2 sees only ground-truth bytes 0 and 1; head 3 sees byte 2.
        for k in range(3):
            torch.testing.assert_close(a[k], b[k], rtol=0, atol=0)
        self.assertGreater((a[3]-b[3]).abs().max().item(), 0)

    def test_loss_uses_shifted_targets_and_target_weighting(self):
        class Model:
            def __init__(self):
                self.seen = []

            def __call__(self, x, byte_targets, **kwargs):
                self.seen.append(byte_targets.clone())
                logits = torch.zeros(*byte_targets.shape, 256)
                # Distinct losses per passage and unequal target counts.
                logits[..., 0] = len(self.seen)
                return tuple(logits[:, :, k] for k in range(4)), None, {}
        model = Model()
        result = loss_eval(model, {'tokenization':'byte', 'patch_size':4}, {'texts':['abcdefgh','ijklmnopqrst']}, None)
        self.assertEqual(result['targets'], 12)
        self.assertEqual(model.seen[0].flatten().tolist(), list(b'efgh'))
        self.assertEqual(model.seen[1].flatten().tolist(), list(b'mnopqrst'))
        expected = (4*math.log(255+math.exp(1))+8*math.log(255+math.exp(2)))/12
        self.assertAlmostEqual(result['mean_nll'], expected, places=5)

    def test_skips_are_not_passes(self):
        result = run_check('irrelevant', lambda: {'skipped':True,'reason':'not recurrent'})
        self.assertEqual(result['status'], 'N/A')

    def test_nonfinite_evidence_is_valid_json(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'record.json'
            save(path, {'loss':float('nan'), 'nested':[float('inf'), 2.]})
            self.assertEqual(json.loads(path.read_text()), {'loss':None,'nested':[None,2.]})


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
