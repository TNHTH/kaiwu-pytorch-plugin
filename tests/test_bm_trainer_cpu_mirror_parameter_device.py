"""Regression: Trainer.bm_net_cpu must follow a later bm_net device move.

``Trainer.__init__`` decides once whether ``bm_net_cpu`` is a distinct CPU
mirror or an alias of ``bm_net``, based on the construction-time device. A
later ``.to`` / parent ``_apply`` move of ``bm_net`` leaves that decision
stale: the alias is shared onto the new device, so ``_sync_gpu_to_cpu`` and
worker sampling no longer have a real CPU mirror.

The lazy ensure path must create a distinct CPU BoltzmannMachine whenever the
live ``bm_net`` parameter device is non-CPU, while keeping the CPU-only
early-return in ``_sync_gpu_to_cpu`` (compat with the open #1342 test pin).
"""

from __future__ import annotations

import importlib.util
import os
import unittest
from types import SimpleNamespace

import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _load_trainer_module():
    """Load ``example/bm_generation/trainer.py`` without mutating ``sys.path``."""
    module_path = os.path.join(REPO_ROOT, "example", "bm_generation", "trainer.py")
    spec = importlib.util.spec_from_file_location(
        "bm_generation_trainer_cpu_mirror", module_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


trainer_module = _load_trainer_module()
Trainer = trainer_module.Trainer
BoltzmannMachine = trainer_module.BoltzmannMachine


def _make_trainer(num_visible=4, num_hidden=2, num_output=1):
    """Build a tiny Trainer; ``bm_net`` starts on CPU so the mirror is an alias."""
    return Trainer(
        data=[],
        saver=object(),
        worker=object(),
        num_visible=num_visible,
        num_hidden=num_hidden,
        num_output=num_output,
    )


class ForbiddenCpuCopy:
    """Fails if the CPU parameter iterator is touched (#1342 contract)."""

    def parameters(self):
        raise AssertionError("cpu mirror must not be touched on a CPU-only run")


class TestBmTrainerCpuMirrorParameterDevice(unittest.TestCase):
    """Construction-time CPU alias must not survive a later non-CPU move."""

    def test_cpu_only_sync_still_early_returns(self):
        """Keep the #1342 shortcut: CPU ``bm_net`` must not touch the mirror."""
        trainer = Trainer.__new__(Trainer)
        trainer.bm_net = SimpleNamespace(device=torch.device("cpu"))
        trainer.bm_net_cpu = ForbiddenCpuCopy()
        self.assertIsNone(trainer._sync_gpu_to_cpu())

    def test_sync_after_apply_move_builds_distinct_cpu_mirror_and_copies(self):
        """After ``_apply`` to meta, sync must own a CPU mirror with live weights."""
        trainer = _make_trainer()
        self.assertIs(
            trainer.bm_net_cpu,
            trainer.bm_net,
            "construction on CPU should alias bm_net_cpu to bm_net",
        )

        with torch.no_grad():
            for param in trainer.bm_net.parameters():
                param.fill_(1.25)

        # Preserve values before the meta move (meta tensors have no storage).
        saved = {
            name: tensor.detach().cpu().clone()
            for name, tensor in trainer.bm_net.state_dict().items()
        }

        trainer.bm_net._apply(lambda tensor: tensor.to(torch.device("meta")))
        self.assertEqual(next(trainer.bm_net.parameters()).device, torch.device("meta"))
        self.assertIs(
            trainer.bm_net_cpu,
            trainer.bm_net,
            "without ensure, the construction alias is still shared on meta",
        )

        # Lazy ensure while live params are non-CPU (meta stands in for CUDA).
        trainer._ensure_cpu_mirror()
        self.assertIsNot(trainer.bm_net_cpu, trainer.bm_net)
        for cpu_param in trainer.bm_net_cpu.parameters():
            self.assertEqual(cpu_param.device.type, "cpu")

        # Rematerialize bm_net with the saved weights. Spoof device to meta so
        # the CPU early-return does not fire; live params carry copyable data.
        rematerialized = BoltzmannMachine(
            num_nodes=trainer.num_visible + trainer.num_hidden
        )
        rematerialized.load_state_dict(saved)
        rematerialized.device = torch.device("meta")
        trainer.bm_net = rematerialized

        trainer._sync_gpu_to_cpu()

        for cpu_param, src_param in zip(
            trainer.bm_net_cpu.parameters(), rematerialized.parameters()
        ):
            self.assertEqual(cpu_param.device.type, "cpu")
            self.assertTrue(torch.equal(cpu_param, src_param))
            self.assertTrue(torch.allclose(cpu_param, torch.full_like(cpu_param, 1.25)))

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_sync_after_cuda_move_builds_distinct_cpu_mirror_and_copies(self):
        """End-to-end path when ``bm_net`` actually leaves the CPU."""
        trainer = _make_trainer()
        self.assertIs(trainer.bm_net_cpu, trainer.bm_net)

        with torch.no_grad():
            for param in trainer.bm_net.parameters():
                param.fill_(2.5)

        trainer.bm_net.to("cuda")
        self.assertEqual(next(trainer.bm_net.parameters()).device.type, "cuda")

        trainer._sync_gpu_to_cpu()

        self.assertIsNot(trainer.bm_net_cpu, trainer.bm_net)
        for cpu_param in trainer.bm_net_cpu.parameters():
            self.assertEqual(cpu_param.device.type, "cpu")
            self.assertTrue(torch.allclose(cpu_param, torch.full_like(cpu_param, 2.5)))


if __name__ == "__main__":
    unittest.main()
