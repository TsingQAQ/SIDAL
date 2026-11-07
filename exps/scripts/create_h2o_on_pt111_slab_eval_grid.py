#!/usr/bin/env python
"""Precompute evaluation grid for H2OonPt111_Slab benchmark.

If a cached diagnostic grid exists at exps/oracles/h2o_on_pt111_slab_diag_10k.pt
(saved with keys 'X', 'Y'), reuse it. Otherwise generate fresh.

Usage:
  python exps/scripts/create_h2o_on_pt111_slab_eval_grid.py
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

from sidal.test_functions import H2OonPt111_Slab

OUTPUT_PATH = "exps/oracles/h2o_on_pt111_slab_eval_grid.pt"
CACHE_DIAG  = "exps/oracles/h2o_on_pt111_slab_diag_10k.pt"
N_EVAL = 10000
EVAL_SEED = 0

import json
with open("exps/configs/6d_H2OonPt111_Slab/matern52/config.json") as fh:
    cfg = json.load(fh)
kwargs = cfg.get("problem", {}).get("kwargs", {})
print(f"Using H2OonPt111_Slab kwargs from config: {kwargs}")
fn = H2OonPt111_Slab(**kwargs)
bounds = torch.tensor([list(b) for b in fn._bounds], dtype=torch.float64).T  # (2, 6)

if os.path.exists(CACHE_DIAG):
    print(f"reusing cached diagnostic grid {CACHE_DIAG}")
    d = torch.load(CACHE_DIAG, weights_only=False)
    eval_X, eval_Y = d["X"], d["Y"]
else:
    torch.manual_seed(EVAL_SEED)
    eval_X = bounds[0] + (bounds[1] - bounds[0]) * torch.rand(N_EVAL, 6, dtype=torch.float64)
    print(f"Evaluating {N_EVAL} points...")
    t0 = time.time()
    eval_Y = fn(eval_X)
    print(f"Done in {(time.time()-t0)/60:.1f} min, Y range=[{eval_Y.min():.2f}, {eval_Y.max():.2f}]")

print(f"Y range: [{eval_Y.min():.4f}, {eval_Y.max():.4f}]  n={len(eval_Y)}")
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
torch.save({
    "eval_X": eval_X,
    "eval_Y": eval_Y,
    "n_eval": len(eval_Y),
    "eval_seed": EVAL_SEED,
    "bounds": bounds,
}, OUTPUT_PATH)
print(f"Saved to {OUTPUT_PATH}")
