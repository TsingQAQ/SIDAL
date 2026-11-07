#!/usr/bin/env python
"""Precompute evaluation grid for HOnCu13 benchmark.

Generates random eval points in spherical coordinates (r, u=cos(theta), phi),
evaluates the (cluster + probe) energy via MACE-MP-0, applies the same Emax
cap and inverse_temperature scaling that the runtime function uses, and saves
to disk. The runner loads this instead of re-evaluating per trial.

Usage:
  python exps/scripts/create_h_on_cu13_eval_grid.py
"""

import sys
import os
import time
import math

sys.path = [p for p in sys.path if '/codes/botorch' not in p or 'active_learning' in p]
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
os.chdir(os.path.join(os.path.dirname(__file__), "..", ".."))

import torch

torch.set_default_dtype(torch.float64)

from sidal.test_functions import HOnCu13

N_EVAL = 50000
EVAL_SEED = 0
OUTPUT_PATH = "exps/oracles/h_on_cu13_eval_grid.pt"

import json
with open("exps/configs/3d_HOnCu13/matern52/config.json") as _fh:
    _cfg = json.load(_fh)
_kwargs = _cfg.get("problem", {}).get("kwargs", {})
print(f"Using HOnCu13 kwargs from config: {_kwargs}")
fn = HOnCu13(**_kwargs)

# Use bounds defined by the HOnCu13 instance (wedge by default if
# align_5fold_z=True, full sphere otherwise).
bounds = torch.tensor([list(b) for b in fn._bounds], dtype=torch.float64).T  # (2, 3)
print(f"Eval bounds (lo / hi):\n{bounds.numpy()}")

# Generate grid
torch.manual_seed(EVAL_SEED)
eval_X = bounds[0] + (bounds[1] - bounds[0]) * torch.rand(N_EVAL, 3, dtype=torch.float64)

print(f"Evaluating {N_EVAL} points (this is the slow step) ...")
t0 = time.time()
eval_Y = fn(eval_X)
elapsed = time.time() - t0
print(f"Done in {elapsed/60:.1f} min ({elapsed/N_EVAL*1000:.1f} ms/pt)")
print(f"Y range: [{eval_Y.min():.4f}, {eval_Y.max():.4f}]")
print(f"  fraction at Emax cap (in scaled f): "
      f"{(eval_Y >= fn._inv_T * fn._Emax - 1e-6).float().mean().item()*100:.1f}%")

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
torch.save({
    "eval_X": eval_X,
    "eval_Y": eval_Y,
    "n_eval": N_EVAL,
    "eval_seed": EVAL_SEED,
    "bounds": bounds,
}, OUTPUT_PATH)
print(f"Saved to {OUTPUT_PATH}")
