"""H2 on Cu(100) MACE potential energy surface benchmark (2D).

The energy is evaluated with a MACE model (``h2Cu/MACE_model_swa.model``) on a
Cu slab read from ``h2Cu/cu100_0K.in``.

Requires: ``ase`` and ``mace-torch`` (``pip install "sidal[pes]"``).
"""

from __future__ import annotations

import hashlib
import math
import warnings
from pathlib import Path

import torch
from botorch.test_functions.synthetic import SyntheticTestFunction
from torch import Tensor

def _resolve_enable_cueq(enable_cueq: bool, device: str) -> bool:
    """Resolve whether cuEq can be enabled, with safe fallback behavior."""
    if not enable_cueq:
        return False

    if not str(device).startswith("cuda"):
        warnings.warn(
            "enable_cueq=True requested but device is not CUDA; disabling cuEq.",
            RuntimeWarning,
            stacklevel=2,
        )
        return False

    if not torch.cuda.is_available():
        warnings.warn(
            "enable_cueq=True requested but CUDA is not available; disabling cuEq.",
            RuntimeWarning,
            stacklevel=2,
        )
        return False

    try:
        import cuequivariance  # noqa: F401
        import cuequivariance_torch  # noqa: F401
    except (ImportError, ModuleNotFoundError):
        warnings.warn(
            "enable_cueq=True requested but cuequivariance packages are missing; "
            "disabling cuEq.",
            RuntimeWarning,
            stacklevel=2,
        )
        return False

    return True


class H2CuMACEPES(SyntheticTestFunction):
    r"""MACE potential energy surface for H\ :sub:`2` on a Cu surface.

    Two-dimensional function:

        X[..., 0] = Z — H\ :sub:`2` COM height above top Cu layer (Å)
        X[..., 1] = d — H–H bond length (Å)

    Returns E_MACE(Z, d) − E_ref (eV), where E_ref is the MACE energy of H\ :sub:`2`
    far from the surface (Z = z_ref, d = d_ref).

    Default geometry: H\ :sub:`2` COM at the bridge site, molecular axis along [010],
    molecule parallel to the surface. Fig. 3 of Stark et al. (2024) shows elbow
    plots of this kind.

    The MACE model and Cu slab are taken from
    https://github.com/wgst/ml-gas-surface (MIT; see ``h2Cu/LICENSE``):

        W. G. Stark et al., Mach. Learn.: Sci. Technol. 5, 030501 (2024),
        doi:10.1088/2632-2153/ad5f11 — the MACE model.
        W. G. Stark et al., J. Phys. Chem. C 127, 24168 (2023),
        doi:10.1021/acs.jpcc.3c06648 — the DFT training data.
        W. G. Stark et al., Phys. Rev. B 112, 035422 (2025),
        doi:10.1103/h7vd-94pk — the Cu(100) slab.

    Requires: ase, mace-torch
        pip install ase mace-torch

    Optional acceleration: cuequivariance kernels for NVIDIA GPUs
        pip install cuequivariance cuequivariance-torch
    """

    dim = 2
    continuous_inds = [0, 1]
    _bounds = [(0.0, 6.0), (0.5, 2.5)]
    _check_grad_at_opt = False

    def __init__(
        self,
        structure_file: str = "test_functions/h2Cu/cu100_0K.in",
        model_path: str = "test_functions/h2Cu/MACE_model_swa.model",
        x_offset: float = 0.0,
        y_offset: float = 0.0,
        phi_deg: float = 90.0,
        theta_deg: float = 0.0,
        z_ref: float = 7.0,
        d_ref: float = 0.74,
        device: str = "cpu",
        mace_dtype: str = "float32",
        z_min: float = 0.0,
        z_max: float = 6.0,
        d_min: float = 0.5,
        d_max: float = 2.5,
        enable_cueq: bool = False,
        noise_std: float | None = None,
        negate: bool = False,
        dtype: torch.dtype = torch.double,
    ) -> None:
        r"""
        Args:
            structure_file: Path to the Cu slab structure file (FHI-aims format).
            model_path: Path to the MACE model file (.model).
            x_offset: x offset of H2 COM from bridge site (Å). Default 0.
            y_offset: y offset of H2 COM from bridge site (Å). Default 0.
            phi_deg: Azimuthal angle of H-H axis projection onto surface (°).
                     90° = [010] direction (default).
            theta_deg: Tilt of H2 from surface plane (°). 0° = parallel (default).
            z_ref: Z height for reference energy. Default 7.0 Å.
            d_ref: H-H bond length for reference energy. Default 0.74 Å.
            device: Device for MACE evaluation ("cpu" or "cuda"). Default "cpu".
            mace_dtype: Dtype for MACE model ("float32" or "float64"). Default "float32".
            z_min: Lower bound for Z (Å). Default 0.0.
            z_max: Upper bound for Z (Å). Default 6.0.
            d_min: Lower bound for d (Å). Default 0.5.
            d_max: Upper bound for d (Å). Default 2.5.
            enable_cueq: If True, enable MACE cuEquivariance acceleration.
                Requires CUDA plus cuequivariance packages.
            noise_std: Standard deviation of observation noise.
            negate: If True, negate the function.
            dtype: The dtype for the bounds.
        """
        # Keep exact constructor inputs for reproducibility records.
        self._constructor_kwargs = {
            "structure_file": structure_file,
            "model_path": model_path,
            "x_offset": x_offset,
            "y_offset": y_offset,
            "phi_deg": phi_deg,
            "theta_deg": theta_deg,
            "z_ref": z_ref,
            "d_ref": d_ref,
            "device": device,
            "mace_dtype": mace_dtype,
            "z_min": z_min,
            "z_max": z_max,
            "d_min": d_min,
            "d_max": d_max,
            "enable_cueq": enable_cueq,
            "noise_std": noise_std,
            "negate": negate,
            "dtype": str(dtype),
        }

        self.structure_file = self._resolve_asset_path(structure_file)
        self.model_path = self._resolve_asset_path(model_path)
        self.x_offset = x_offset
        self.y_offset = y_offset
        self.phi_deg = phi_deg
        self.theta_deg = theta_deg
        self.z_ref = z_ref
        self.d_ref = d_ref
        self.device = device
        self.mace_dtype = mace_dtype
        self.enable_cueq = enable_cueq

        self._bounds = [(z_min, z_max), (d_min, d_max)]

        self._slab = None
        self._calc = None
        self._e_ref = None

        super().__init__(noise_std=noise_std, negate=negate, dtype=dtype)

    def _resolve_asset_path(self, asset_path: str) -> str:
        """Resolve benchmark asset paths after relocation to package `h2Cu`.

        Supports absolute paths, current-working-directory relative paths,
        and package-relative paths under `sidal/`.
        """
        p = Path(asset_path)
        if p.exists():
            return str(p)

        package_root = Path(__file__).resolve().parents[1]
        candidates = [
            package_root / p,
            package_root / "test_functions" / p,
            package_root / "test_functions" / "h2Cu" / p.name,
        ]
        for cand in candidates:
            if cand.exists():
                return str(cand)

        return asset_path

    def _file_sha256(self, file_path: str) -> str | None:
        p = Path(file_path)
        if not p.exists() or not p.is_file():
            return None
        h = hashlib.sha256()
        with p.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()

    def get_reproducibility_metadata(self) -> dict:
        """Return metadata sufficient to reproduce precomputed evaluations.

        Includes exact constructor parametrization, resolved asset paths,
        asset fingerprints, and lattice/structure information.
        """
        from ase.io import read

        slab = read(self.structure_file)
        structure_path = Path(self.structure_file)
        model_path = Path(self.model_path)

        def _file_info(p: Path) -> dict:
            if not p.exists():
                return {"path": str(p), "exists": False}
            st = p.stat()
            return {
                "path": str(p),
                "exists": True,
                "size_bytes": int(st.st_size),
                "mtime_unix": float(st.st_mtime),
                "sha256": self._file_sha256(str(p)),
            }

        return {
            "class_name": self.__class__.__name__,
            "constructor_kwargs": self._constructor_kwargs,
            "resolved_assets": {
                "structure_file": str(structure_path),
                "model_path": str(model_path),
            },
            "asset_fingerprints": {
                "structure_file": _file_info(structure_path),
                "model_path": _file_info(model_path),
            },
            "lattice": {
                "n_atoms": int(len(slab)),
                "cell": slab.cell.array.tolist(),
                "pbc": list(map(bool, slab.pbc)),
                "symbols": slab.get_chemical_symbols(),
                "atomic_numbers": slab.get_atomic_numbers().tolist(),
                "positions": slab.get_positions().tolist(),
            },
        }

    def _get_slab_and_calc(self):
        """Lazy-initialize the Cu slab and MACE calculator."""
        if self._slab is None or self._calc is None:
            try:
                from ase.constraints import FixAtoms
                from ase.io import read
                from mace.calculators import MACECalculator
            except ImportError:
                raise ImportError(
                    "H2CuMACEPES requires ASE and MACE. "
                    "Install with: pip install ase mace-torch"
                )

            slab = read(self.structure_file)
            slab.set_constraint(FixAtoms(indices=list(range(len(slab)))))
            self._slab = slab

            enable_cueq = _resolve_enable_cueq(self.enable_cueq, self.device)

            self._calc = MACECalculator(
                model_paths=self.model_path,
                device=self.device,
                default_dtype=self.mace_dtype,
                enable_cueq=enable_cueq,
            )
        return self._slab, self._calc

    def _bridge_site(self, slab, tol: float = 0.5):
        """Return (bx, by, z_top) of the bridge site in the top Cu layer."""
        import numpy as np

        z_top = slab.positions[:, 2].max()
        top_idx = np.where(slab.positions[:, 2] > z_top - tol)[0]
        top_pos = slab.positions[top_idx]

        ref = top_pos[np.argmin(top_pos[:, 0] ** 2 + top_pos[:, 1] ** 2)]
        dists = np.linalg.norm(top_pos - ref, axis=1)
        dists[dists < 0.1] = np.inf
        nn = top_pos[np.argmin(dists)]

        mid = (ref + nn) / 2
        return mid[0], mid[1], z_top

    def _make_h2_on_cu(self, slab, Z: float, d: float):
        """Construct a Cu slab + H2 Atoms object for the given Z and d."""
        import numpy as np
        from ase import Atoms

        theta = np.radians(self.theta_deg)
        phi = np.radians(self.phi_deg)
        ux = np.cos(theta) * np.cos(phi)
        uy = np.cos(theta) * np.sin(phi)
        uz = np.sin(theta)

        bx, by, z_top = self._bridge_site(slab)
        bx += self.x_offset
        by += self.y_offset
        z_h = z_top + Z

        atoms = slab.copy()
        h2 = Atoms(
            "H2",
            positions=[
                [bx + (d / 2) * ux, by + (d / 2) * uy, z_h + (d / 2) * uz],
                [bx - (d / 2) * ux, by - (d / 2) * uy, z_h - (d / 2) * uz],
            ],
        )
        atoms.extend(h2)
        atoms.pbc = [True, True, True]
        return atoms

    def _get_e_ref(self, slab, calc) -> float:
        """Lazily compute the reference energy (H2 at large Z, equilibrium d)."""
        if self._e_ref is None:
            ref_atoms = self._make_h2_on_cu(slab, Z=self.z_ref, d=self.d_ref)
            ref_atoms.calc = calc
            # MACE may require autograd even for energy calls; force-enable grads
            # in case the outer benchmark loop is wrapped in no-grad/inference mode.
            with torch.inference_mode(False):
                with torch.enable_grad():
                    self._e_ref = ref_atoms.get_potential_energy()
        return self._e_ref

    def _evaluate_true(self, X: Tensor) -> Tensor:
        """Evaluate MACE PES energy relative to H2 asymptote.

        Args:
            X: Tensor of shape (..., 2) with X[..., 0]=Z and X[..., 1]=d.

        Returns:
            Energy E - E_ref in eV, shape (...).
        """
        slab, calc = self._get_slab_and_calc()
        e_ref = self._get_e_ref(slab, calc)

        X_np = X.detach().cpu().numpy().reshape(-1, 2)
        results = []
        for Z, d in X_np:
            atoms = self._make_h2_on_cu(slab, float(Z), float(d))
            atoms.calc = calc
            with torch.inference_mode(False):
                with torch.enable_grad():
                    results.append(atoms.get_potential_energy() - e_ref)

        return torch.tensor(results, dtype=X.dtype, device=X.device).reshape(
            X.shape[:-1]
        )
