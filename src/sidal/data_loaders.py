#!/usr/bin/env python3
r"""
Data loaders for pool-based active learning experiments.

This module provides utilities for loading and preprocessing datasets
for pool-based active learning, including molecular datasets.
"""

from __future__ import annotations

from typing import Dict, Any, Tuple

import torch
from torch import Tensor


def load_molecular_pool_data(
    config: Dict[str, Any],
    verbose: bool = True,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Load and featurize molecular data for pool-based AL.

    Args:
        config: Experiment config with problem and data_split sections.
            Expected structure:
            {
                "problem": {
                    "dataset_path": str,  # Path to CSV with SMILES and objectives
                    "objective": str,     # Column name for target property
                    "fingerprint_type": str,  # "morgan" (default)
                    "fingerprint_bits": int,  # 2048 (default)
                    "fingerprint_radius": int,  # 2 (default)
                },
                "data_split": {
                    "test_size": int,      # Number of held-out test points
                    "test_split_seed": int,  # Seed for reproducible split
                }
            }
        verbose: If True, print progress information.

    Returns:
        pool_X: (N_pool, d) fingerprint tensor for training pool
        pool_Y: (N_pool,) label tensor for training pool
        test_X: (N_test, d) fingerprint tensor for held-out test set
        test_Y: (N_test,) label tensor for held-out test set

    Example:
        >>> config = {
        ...     "problem": {
        ...         "dataset_path": "data/guacamol.csv",
        ...         "objective": "osmb",
        ...     },
        ...     "data_split": {
        ...         "test_size": 2000,
        ...         "test_split_seed": 42,
        ...     }
        ... }
        >>> pool_X, pool_Y, test_X, test_Y = load_molecular_pool_data(config)
    """
    import pandas as pd
    import numpy as np

    problem_config = config["problem"]
    split_config = config.get("data_split", {})

    dataset_path = problem_config["dataset_path"]
    objective = problem_config["objective"]
    fp_type = problem_config.get("fingerprint_type", "morgan")
    fp_bits = problem_config.get("fingerprint_bits", 2048)
    fp_radius = problem_config.get("fingerprint_radius", 2)

    test_size = split_config.get("test_size", 2000)
    test_split_seed = split_config.get("test_split_seed", 42)

    if verbose:
        print(f"Loading molecular data from {dataset_path}")
        print(f"  Objective: {objective}")
        print(f"  Fingerprint: {fp_type}, bits={fp_bits}, radius={fp_radius}")
        print(f"  Test split: size={test_size}, seed={test_split_seed}")

    # Load dataset
    df = pd.read_csv(dataset_path)
    smiles_list = df["smile"].astype(str).tolist()
    y = df[objective].values

    # Featurize molecules
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
    except ImportError:
        raise ImportError("Molecular featurization requires rdkit: pip install rdkit")

    if verbose:
        print(f"Featurizing {len(smiles_list)} molecules...")

    X_list = []
    valid_indices = []
    for i, smiles in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            if fp_type == "morgan":
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=fp_radius, nBits=fp_bits)
            else:
                raise ValueError(f"Unsupported fingerprint type: {fp_type}")
            X_list.append(np.array(fp))
            valid_indices.append(i)

    X = torch.tensor(np.stack(X_list), dtype=torch.float64)
    y = torch.tensor(y[valid_indices], dtype=torch.float64)

    # Min-max normalize y to [0, 1] if requested (pool-based only)
    normalize_y = problem_config.get("normalize_y", False)
    if normalize_y:
        y_min, y_max = y.min(), y.max()
        y = (y - y_min) / (y_max - y_min)
        if verbose:
            print(f"  Normalized y to [0, 1] (original range: [{y_min:.4f}, {y_max:.4f}])")

    if verbose:
        print(f"  Valid molecules: {len(X)} / {len(smiles_list)}")

    # Split into pool and test using fixed seed
    # If test_size=0, evaluate on full pool (no held-out test set)
    if test_size == 0:
        # Full pool for both training and evaluation
        if verbose:
            print(f"  Using full pool for evaluation (no held-out test set)")
            print(f"  Pool size: {len(X)}, Eval size: {len(X)}")
        return X, y, X, y

    torch.manual_seed(test_split_seed)
    n = len(X)
    perm = torch.randperm(n)

    test_idx = perm[:test_size]
    pool_idx = perm[test_size:]

    test_X = X[test_idx]
    test_Y = y[test_idx]
    pool_X = X[pool_idx]
    pool_Y = y[pool_idx]

    if verbose:
        print(f"  Pool size: {len(pool_X)}, Test size: {len(test_X)}")

    return pool_X, pool_Y, test_X, test_Y
