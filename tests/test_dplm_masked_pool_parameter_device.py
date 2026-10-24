"""Regression: masked_mean_pool must place the mask on hidden_states.device.

``attention_mask.to(hidden_states.dtype)`` keeps the mask on the caller device.
After a parent ``_apply(meta)`` (or any move) leaves ``hidden_states`` off-CPU,
multiplying a CPU mask raises::

    RuntimeError: Tensor on device cpu is not on the expected device meta!

Place the mask with ``.to(device=hidden_states.device, dtype=hidden_states.dtype)``.
This does not rewrite the float16/bf16 accumulation path owned by open #259.
"""

from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path

import torch
from torch import nn


def _load_masked_mean_pool():
    """Load ``masked_mean_pool`` without importing the heavy DPLM package tree."""
    models_dir = (
        Path(__file__).resolve().parents[1]
        / "example"
        / "qdiffusion"
        / "dplm"
        / "models"
    )
    pkg_name = "_dplm_masked_pool_device_test"
    common_name = f"{pkg_name}.common"
    if common_name in sys.modules:
        return sys.modules[common_name].masked_mean_pool

    # Empty __path__ so PathFinder cannot load real sibling modules from disk.
    package = types.ModuleType(pkg_name)
    package.__path__ = []
    sys.modules[pkg_name] = package

    # common.py imports DPLMBackbone at module scope; stub it for this unit.
    backbone = types.ModuleType(f"{pkg_name}.backbone")
    backbone.DPLMBackbone = type("DPLMBackbone", (), {})
    sys.modules[f"{pkg_name}.backbone"] = backbone

    spec = importlib.util.spec_from_file_location(
        common_name,
        models_dir / "common.py",
    )
    module = importlib.util.module_from_spec(spec)
    module.__package__ = pkg_name
    sys.modules[common_name] = module
    spec.loader.exec_module(module)
    return module.masked_mean_pool


masked_mean_pool = _load_masked_mean_pool()


def _dtype_only_mask(attention_mask, hidden_states):
    """Reproduce the pre-fix placement that leaves the mask on the caller device."""
    return attention_mask.to(hidden_states.dtype).unsqueeze(-1)


class _TinyHiddenProducer(nn.Module):
    """Produces token states so a parent ``_apply`` can move them off CPU."""

    def __init__(self, hidden_size: int = 4) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, batch: int = 2, length: int = 3) -> torch.Tensor:
        """Expand the live parameter into a tiny [B, L, H] token tensor."""
        return self.weight.unsqueeze(0).unsqueeze(0).expand(batch, length, -1).clone()


class TestMaskedMeanPoolParameterDevice(unittest.TestCase):
    """CPU/meta coverage matching the #164 device bar."""

    def test_aligned_cpu_mask_keeps_caller_storage(self):
        """Same-device, same-dtype ``.to`` may return the caller's mask storage."""
        hidden = torch.randn(2, 3, 4)
        attention_mask = torch.ones(2, 3, dtype=hidden.dtype)
        pooled = masked_mean_pool(hidden, attention_mask)
        self.assertEqual(pooled.shape, (2, 4))
        self.assertEqual(pooled.device, hidden.device)
        self.assertTrue(torch.isfinite(pooled).all())

        # Direct placement check: already-aligned tensors keep identity.
        placed = attention_mask.to(device=hidden.device, dtype=hidden.dtype)
        self.assertIs(placed, attention_mask)

    def test_same_device_path_keeps_autograd_to_hidden_states(self):
        """Pooling on one device must still backprop into ``hidden_states``."""
        hidden = torch.randn(2, 3, 4, requires_grad=True)
        attention_mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
        pooled = masked_mean_pool(hidden, attention_mask)
        pooled.sum().backward()
        self.assertIsNotNone(hidden.grad)
        self.assertEqual(hidden.grad.shape, hidden.shape)
        self.assertTrue(torch.isfinite(hidden.grad).all())
        # Padded positions contribute zero to the mean gradient.
        self.assertTrue(torch.equal(hidden.grad[0, 2], torch.zeros(4)))
        self.assertTrue(torch.equal(hidden.grad[1, 1], torch.zeros(4)))
        self.assertTrue(torch.equal(hidden.grad[1, 2], torch.zeros(4)))

    def test_parent_apply_meta_with_cpu_mask_pools_on_meta(self):
        """After parent ``_apply(meta)``, a CPU mask must still pool on meta."""
        parent = nn.Module()
        parent.producer = _TinyHiddenProducer(4)
        parent._apply(  # pylint: disable=protected-access
            lambda tensor: tensor.to(torch.device("meta"))
        )
        hidden = parent.producer()
        self.assertEqual(hidden.device, torch.device("meta"))

        attention_mask = torch.ones(2, 3)  # stays on CPU
        self.assertEqual(attention_mask.device, torch.device("cpu"))

        pooled = masked_mean_pool(hidden, attention_mask)
        self.assertEqual(pooled.device, torch.device("meta"))
        self.assertEqual(pooled.shape, (2, 4))

    def test_pre_fix_dtype_only_to_is_red_on_meta(self):
        """Restoring dtype-only placement must raise on meta (documents the bug)."""
        parent = nn.Module()
        parent.producer = _TinyHiddenProducer(4)
        parent._apply(  # pylint: disable=protected-access
            lambda tensor: tensor.to(torch.device("meta"))
        )
        hidden = parent.producer()
        attention_mask = torch.ones(2, 3)

        with self.assertRaises(RuntimeError) as ctx:
            mask = _dtype_only_mask(attention_mask, hidden)
            _ = (hidden * mask).sum(dim=1)

        message = str(ctx.exception).lower()
        self.assertTrue(
            "device" in message or "meta" in message or "cpu" in message,
            f"unexpected error from dtype-only placement: {ctx.exception!r}",
        )


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestMaskedMeanPoolParameterDeviceCuda(unittest.TestCase):
    """CUDA path: parent move leaves a CPU mask; pool must land on CUDA."""

    def test_parent_cuda_move_with_cpu_mask_pools_on_cuda(self):
        """``parent.to(cuda)`` then a CPU mask must still pool on CUDA."""
        parent = nn.Module()
        parent.producer = _TinyHiddenProducer(4)
        parent.to("cuda")
        hidden = parent.producer()
        self.assertEqual(hidden.device.type, "cuda")

        attention_mask = torch.ones(2, 3)  # CPU
        self.assertEqual(attention_mask.device.type, "cpu")

        pooled = masked_mean_pool(hidden, attention_mask)
        self.assertEqual(pooled.device.type, "cuda")
        self.assertEqual(pooled.shape, (2, 4))
        self.assertTrue(torch.isfinite(pooled).all())


if __name__ == "__main__":
    unittest.main()
