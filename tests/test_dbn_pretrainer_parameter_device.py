"""Regression tests for ``DBNPretrainer.device`` after a parent module move.

``UnsupervisedDBN`` stores ``self.device`` when it is constructed. A parent
``nn.Module.to`` / ``_apply`` moves the nested RBM parameters through the
module tree without calling the DBN's ``.to()`` and without refreshing that
attribute. ``DBNPretrainer.device`` used to forward the stale cache, so it kept
reporting the construction device after the wrapped model had moved.

The property must report the live parameter device and only fall back to the
cached attribute while the model has no parameters yet.
"""

import importlib
import importlib.util
from pathlib import Path
import sys
import types
import unittest

import numpy as np
import torch
from torch import nn


def _stub_optional_kaiwu_backends():
    """Provide import-only stand-ins when the Kaiwu SDK is not installed.

    ``dbn_trainer`` imports the CIM and simulated-annealing optimizers at module
    scope, but the regression only builds the no-CIM trainer. The stubs keep the
    test focused on device bookkeeping instead of requiring the full SDK.
    """
    try:
        import kaiwu  # noqa: F401  # pylint: disable=unused-import,import-outside-toplevel
    except ModuleNotFoundError:
        namespace = types.ModuleType("kaiwu")
        namespace.__path__ = []
        sys.modules["kaiwu"] = namespace

    for module_name, attribute_names in (
        ("kaiwu.classical", ("SimulatedAnnealingOptimizer",)),
        ("kaiwu.cim", ("CIMOptimizer", "PrecisionReducer")),
    ):
        try:
            importlib.import_module(module_name)
            continue
        except ModuleNotFoundError:
            pass

        module = types.ModuleType(module_name)
        for attribute_name in attribute_names:
            setattr(
                module,
                attribute_name,
                type(attribute_name, (), {"__init__": lambda self, *a, **k: None}),
            )
        sys.modules[module_name] = module


def _load_dbn_trainer_module():
    """Load ``example/dbn_digits/dbn_trainer.py`` without mutating sys.path."""
    module_path = (
        Path(__file__).resolve().parents[1]
        / "example"
        / "dbn_digits"
        / "dbn_trainer.py"
    )
    _stub_optional_kaiwu_backends()
    spec = importlib.util.spec_from_file_location("dbn_trainer", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


dbn_trainer = _load_dbn_trainer_module()


def _fit_pretrainer():
    """Build a tiny trained model so every RBM owns parameters."""
    pretrainer = dbn_trainer.DBNPretrainer(
        hidden_layers_structure=[4, 3],
        n_epochs_rbm=0,
        batch_size=2,
        verbose=False,
        random_state=0,
    )
    data = np.zeros((2, 6), dtype=np.float32)
    return pretrainer.fit(data)


def _nested_parent(dbn):
    """Wrap the DBN the way an outer estimator wraps its child module."""
    parent = nn.Module()
    parent.dbn = dbn
    return parent


class TestDBNPretrainerParameterDevice(unittest.TestCase):
    """The property follows the parameters, not the construction cache."""

    def test_parent_apply_moves_property_to_live_parameter_device(self):
        """``_apply`` moves RBM parameters but leaves the DBN cache stale."""
        pretrainer = _fit_pretrainer()
        dbn = pretrainer._dbn  # pylint: disable=protected-access
        self.assertEqual(pretrainer.device, torch.device("cpu"))
        self.assertEqual(dbn.device, torch.device("cpu"))

        _nested_parent(dbn)._apply(  # pylint: disable=protected-access
            lambda tensor: tensor.to(torch.device("meta"))
        )

        self.assertEqual(
            dbn.device,
            torch.device("cpu"),
            "the DBN construction cache is expected to stay stale here",
        )
        self.assertEqual(next(dbn.parameters()).device, torch.device("meta"))
        self.assertEqual(pretrainer.device, torch.device("meta"))

    def test_parent_to_moves_property_to_live_parameter_device(self):
        """``parent.to(meta)`` takes the same path as ``_apply``."""
        pretrainer = _fit_pretrainer()
        dbn = pretrainer._dbn  # pylint: disable=protected-access

        _nested_parent(dbn).to(torch.device("meta"))

        self.assertEqual(next(dbn.parameters()).device, torch.device("meta"))
        self.assertEqual(pretrainer.device, torch.device("meta"))

    def test_property_ignores_poisoned_stale_cache(self):
        """A cache poisoned away from the parameters must not win."""
        pretrainer = _fit_pretrainer()
        dbn = pretrainer._dbn  # pylint: disable=protected-access
        dbn.device = torch.device("meta")

        self.assertEqual(next(dbn.parameters()).device, torch.device("cpu"))
        self.assertEqual(pretrainer.device, torch.device("cpu"))

    def test_property_falls_back_to_cache_before_parameters_exist(self):
        """An unbuilt model has no parameters; the cache is the safe fallback."""
        pretrainer = dbn_trainer.DBNPretrainer(
            hidden_layers_structure=[4],
            verbose=False,
        )
        dbn = pretrainer._dbn  # pylint: disable=protected-access

        self.assertEqual(list(dbn.parameters()), [])
        self.assertEqual(pretrainer.device, dbn.device)

    def test_training_config_reports_live_parameter_device(self):
        """``DBNTrainer`` reports the same live device as the pretrainer."""
        pretrainer = _fit_pretrainer()
        trainer = dbn_trainer.DBNTrainer(
            dbn_ref=pretrainer._dbn,  # pylint: disable=protected-access
            verbose=False,
        )
        dbn = pretrainer._dbn  # pylint: disable=protected-access

        _nested_parent(dbn)._apply(  # pylint: disable=protected-access
            lambda tensor: tensor.to(torch.device("meta"))
        )

        self.assertEqual(trainer.get_training_config()["device"], "meta")


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestDBNPretrainerParameterDeviceCUDA(unittest.TestCase):
    """Cover the construct-on-CUDA / move-to-CPU path."""

    def test_parent_move_to_cpu_updates_property(self):
        """A CUDA construction cache must lose to the CPU parameters."""
        pretrainer = _fit_pretrainer()
        dbn = pretrainer._dbn  # pylint: disable=protected-access
        self.assertEqual(pretrainer.device.type, "cuda")

        _nested_parent(dbn).to("cpu")

        self.assertEqual(next(dbn.parameters()).device.type, "cpu")
        self.assertEqual(pretrainer.device.type, "cpu")


if __name__ == "__main__":
    unittest.main()
