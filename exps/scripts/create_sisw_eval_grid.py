#!/usr/bin/env python
"""
Precompute evaluation grid for SiSWEnergy benchmark.

Generates random eval points, evaluates the SW energy (log-transformed),
and saves to disk. The runner loads this instead of re-evaluating.

Usage:
  python exps/scripts/create_sisw_eval_grid.py
"""

import sys
import os
import time

sys.path = [p for p in sys.path if '/codes/botorch' not in p or 'active_learning' in p]
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
os.chdir(os.path.join(os.path.dirname(__file__), "..", ".."))

import torch

torch.set_default_dtype(torch.float64)

from sidal.test_functions import SiSWEnergy

N_EVAL = 50000
EVAL_SEED = 0
OUTPUT_PATH = "exps/oracles/sisw_eval_grid.pt"

import json
with open("exps/configs/3d_SiSW/matern52/config.json") as _fh:
    _cfg = json.load(_fh)
_kwargs = _cfg.get("problem", {}).get("kwargs", {})
print(f"Using SiSWEnergy kwargs from config: {_kwargs}")
sw = SiSWEnergy(**_kwargs)
bounds = torch.tensor([[0.1, 0.1, 0.1], [0.95, 0.95, 0.95]])

# Generate grid
torch.manual_seed(EVAL_SEED)
eval_X = bounds[0] + (bounds[1] - bounds[0]) * torch.rand(N_EVAL, 3)

print(f"Evaluating {N_EVAL} points...")
t0 = time.time()
eval_Y = sw(eval_X)
elapsed = time.time() - t0
print(f"Done in {elapsed:.1f}s ({elapsed/N_EVAL*1000:.1f}ms/pt)")
print(f"Y range: [{eval_Y.min():.4f}, {eval_Y.max():.4f}]")

# Save
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
torch.save({
    "eval_X": eval_X,
    "eval_Y": eval_Y,
    "n_eval": N_EVAL,
    "eval_seed": EVAL_SEED,
    "bounds": bounds,
}, OUTPUT_PATH)
print(f"Saved to {OUTPUT_PATH}")
