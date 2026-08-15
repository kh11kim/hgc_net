"""GPU acceptance seams for the modernized PointNet++ extension.

Run only inside HGC-Net's dedicated uv environment.  The first test is the
public gather operation's forward/backward contract; the second is the public
JustinPointNet2 forward boundary.
"""

from __future__ import annotations

import unittest
import sys
from pathlib import Path

import torch


POINTNET_PARENT = Path(__file__).resolve().parents[1] / "pointnet2"
if str(POINTNET_PARENT) not in sys.path:
    sys.path.insert(0, str(POINTNET_PARENT))


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class PointNet2CudaTest(unittest.TestCase):
    def test_gather_operation_forward_and_backward(self):
        from pointnet2 import pointnet2_utils

        features = torch.tensor(
            [[[1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0]]],
            device="cuda",
            requires_grad=True,
        )
        index = torch.tensor([[3, 1, 1]], dtype=torch.int32, device="cuda")
        output = pointnet2_utils.gather_operation(features.contiguous(), index.contiguous())
        torch.testing.assert_close(
            output,
            torch.tensor([[[4.0, 2.0, 2.0], [40.0, 20.0, 20.0]]], device="cuda"),
        )
        output.sum().backward()
        torch.testing.assert_close(
            features.grad,
            torch.tensor([[[0.0, 2.0, 0.0, 1.0], [0.0, 2.0, 0.0, 1.0]]], device="cuda"),
        )

    def test_justin_pointnet2_full_forward(self):
        from justin_hgc.model import JustinPointNet2

        model = JustinPointNet2().cuda().eval()
        xyz = torch.randn((1, 1024, 3), device="cuda")
        normalized = xyz.transpose(1, 2).contiguous()
        with torch.no_grad():
            gp, pose, contact, squeeze = model(xyz, normalized)
        self.assertEqual(gp.shape, (1, 1024, 2, 3))
        self.assertEqual(pose.shape[0:2], (1, 1024))
        self.assertEqual(contact.shape, (1, 1024, 12, 3))
        self.assertEqual(squeeze.shape, (1, 1024, 12, 3))
        self.assertTrue(torch.isfinite(gp).all())


if __name__ == "__main__":
    unittest.main()
