"""Reject empty molecular datasets at the shared loader boundary."""

import os
from pathlib import Path
import subprocess
import sys

import pytest

pytest.importorskip("torch")
pytest.importorskip("pandas")
pytest.importorskip("tqdm")


@pytest.mark.parametrize("variant", ["cim_sa", "sqa"])
@pytest.mark.parametrize("shuffle", [False, True])
@pytest.mark.parametrize("case", ["empty", "filtered", "nonempty"])
def test_molecular_loader_boundary(variant, shuffle, case):
    """Exercise real loaders in isolated imports, without SDK or RDKit calls."""
    root = Path(__file__).resolve().parents[1]
    script = r"""
import sys
import numpy as np
import pandas as pd
from qc_mol_design.data import featurizer as f
from qc_mol_design.config import CFG
case, shuffle = sys.argv[1], sys.argv[2] == "True"
columns = ["smiles", "qed", "logp", "sas"]
if case == "filtered":
    f.mol_to_graph = lambda smiles: None
    dataset = f.MoleculeDataset(pd.DataFrame([["invalid", .5, 1., 2.]], columns=columns))
else:
    dataset = f.MoleculeDataset(pd.DataFrame(columns=columns))
if case == "nonempty":
    CFG.batch_size = 2
    dataset.records = [
        (np.ones((1, CFG.atom_feat_dim), dtype=np.float32),
         np.zeros((1, 1), dtype=np.float32), np.zeros(1, dtype=np.int32),
         np.ones(3, dtype=np.float32), np.array([i, 1., 2.], dtype=np.float32))
        for i in range(3)
    ]
    loader = f.make_dataloader(dataset, shuffle=shuffle)
    batches = list(loader)
    assert [batch[0].shape[0] for batch in batches] == [2, 1]
    assert sorted(int(value) for batch in batches for value in batch[-1][:, 0]) == [0, 1, 2]
else:
    assert len(dataset) == 0
    try:
        f.make_dataloader(dataset, shuffle=shuffle)
    except ValueError as exc:
        assert "non-empty molecular dataset" in str(exc), str(exc)
    else:
        raise AssertionError("empty molecular dataset accepted")
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(root / "example/molecular_design" / variant), env.get("PYTHONPATH", "")]
    )
    result = subprocess.run(
        [sys.executable, "-c", script, case, str(shuffle)],
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
