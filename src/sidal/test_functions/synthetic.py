#!/usr/bin/env python3
r"""
Custom synthetic test functions for active learning benchmarks.

These test functions are commonly used in active learning and surrogate
modeling literature but are not included in the standard BoTorch library.

References:

.. [Forrester2008]
    A. Forrester, A. Sobester, A. Keane. Engineering Design via Surrogate
    Modelling: A Practical Guide. Wiley, 2008.

.. [Gramacy2012]
    R. B. Gramacy and H. K. H. Lee. Cases for the nugget in modeling computer
    experiments. Statistics and Computing, 22(3):713-722, 2012.

.. [Higdon2002]
    D. Higdon. Space and space-time modeling using process convolutions.
    In Quantitative Methods for Current Environmental Issues, pages 37-56.
    Springer, 2002.
"""

from __future__ import annotations

import math

import torch
from botorch.test_functions.synthetic import SyntheticTestFunction
from torch import Tensor


class Forrester(SyntheticTestFunction):
    r"""Forrester test function.

    One-dimensional function (evaluated on `[0, 1]`):

        f(x) = (6x - 2)^2 * sin(12x - 4)

    This function is commonly used as a simple 1D benchmark for surrogate
    modeling and Bayesian optimization. It has one global minimum and
    multiple local features.

    f has a global minimum at approximately x* = 0.7572 with f(x*) ≈ -6.0207.
    """

    dim = 1
    continuous_inds = [0]
    _bounds = [(0.0, 1.0)]
    _optimal_value = -6.020740045498103
    _optimizers = [(0.7572485904500308,)]

    def _evaluate_true(self, X: Tensor) -> Tensor:
        x = X[..., 0]
        return (6.0 * x - 2.0) ** 2 * torch.sin(12.0 * x - 4.0)


class Gramacy1D(SyntheticTestFunction):
    r"""Gramacy 1D test function.

    One-dimensional function (evaluated on `[0.5, 2.5]`):

        f(x) = sin(10 * pi * x) / (2 * x) + (x - 1)^4

    This function is used in Gramacy & Lee (2012) for demonstrating
    the importance of the nugget parameter in GP modeling. It has
    a singularity-like behavior near x=0 (avoided by the domain) and
    oscillatory behavior.

    The function has a global minimum at approximately x* = 0.548
    with f(x*) ≈ -0.869.
    """

    dim = 1
    continuous_inds = [0]
    _bounds = [(0.5, 2.5)]
    _optimal_value = -0.8690111349894559
    _optimizers = [(0.5485634346527641,)]

    def _evaluate_true(self, X: Tensor) -> Tensor:
        x = X[..., 0]
        return torch.sin(10.0 * math.pi * x) / (2.0 * x) + (x - 1.0) ** 4


class Higdon(SyntheticTestFunction):
    r"""Higdon test function.

    One-dimensional function (evaluated on `[0, 10]`):

        f(x) = sin(2 * pi * x / 10) + 0.2 * sin(2 * pi * x / 2.5)

    This function combines two sinusoidal components with different
    frequencies, creating a multi-scale pattern. It is used in
    Higdon (2002) for demonstrating space-time modeling with
    process convolutions.

    The function has multiple local minima due to its oscillatory nature.
    The global minimum is approximately at x* = 7.025 with f(x*) ≈ -1.142.
    """

    dim = 1
    continuous_inds = [0]
    _bounds = [(0.0, 10.0)]
    _optimal_value = -1.1417483388086314
    _optimizers = [(7.024814281669934,)]

    def _evaluate_true(self, X: Tensor) -> Tensor:
        x = X[..., 0]
        return torch.sin(2.0 * math.pi * x / 10.0) + 0.2 * torch.sin(2.0 * math.pi * x / 2.5)


class GPSampleWithBias(SyntheticTestFunction):
    r"""GP Sample test function with optional bias term.

    Creates a test function f(x) sampled from a GP prior, with an optional
    independent GP sample b(x) as a bias term. The final function is:
        - y(x) = f(x)           when use_bias=False
        - y(x) = f(x) + b(x)    when use_bias=True

    This is useful for studying how algorithms behave with different target
    distributions P(x) ∝ exp(λ·y(x)) = exp(λ·f(x) + λ·b(x)).

    Both f and b use the same kernel type but can have different lengthscales
    and random seeds.
    """

    _check_grad_at_opt = False

    def __init__(
        self,
        dim: int = 2,
        kernel: str = "matern52",
        lengthscale: float | list[float] = 0.3,
        outputscale: float = 1.0,
        n_features: int = 2048,
        seed: int = 0,
        use_bias: bool = False,
        bias_lengthscale: float | list[float] | None = None,
        bias_outputscale: float = 1.0,
        bias_seed: int = 1,
        noise_std: float | None = None,
        negate: bool = False,
        dtype: torch.dtype = torch.double,
    ) -> None:
        r"""
        Args:
            dim: Dimensionality of the function.
            kernel: Kernel type ("rbf", "matern52", "matern32", "matern12").
            lengthscale: Lengthscale for f. Can be scalar (isotropic) or list (ARD).
            outputscale: Outputscale for f.
            n_features: Number of RFF features for GP sample approximation.
            seed: Random seed for f.
            use_bias: If True, add bias term b(x) to f(x).
            bias_lengthscale: Lengthscale for b. If None, uses same as f.
            bias_outputscale: Outputscale for b.
            bias_seed: Random seed for b (should differ from seed for independence).
            noise_std: Observation noise standard deviation.
            negate: If True, negate the function.
            dtype: Data type for computations.
        """
        self.dim = dim
        self._bounds = [(0.0, 1.0)] * dim
        self.continuous_inds = list(range(dim))

        self.kernel_type = kernel
        self.use_bias = use_bias
        self.seed = seed
        self.bias_seed = bias_seed

        # Convert lengthscales to tensors
        if isinstance(lengthscale, (list, tuple)):
            self.lengthscale = torch.tensor(lengthscale, dtype=dtype)
        else:
            self.lengthscale = torch.full((dim,), lengthscale, dtype=dtype)

        if bias_lengthscale is None:
            self.bias_lengthscale = self.lengthscale.clone()
        elif isinstance(bias_lengthscale, (list, tuple)):
            self.bias_lengthscale = torch.tensor(bias_lengthscale, dtype=dtype)
        else:
            self.bias_lengthscale = torch.full((dim,), bias_lengthscale, dtype=dtype)

        self.outputscale = outputscale
        self.bias_outputscale = bias_outputscale
        self.n_features = n_features

        super().__init__(noise_std=noise_std, negate=negate, dtype=dtype)

        # Create GP sample paths using botorch's RFF
        self._create_sample_paths(dtype)

    def _create_sample_paths(self, dtype: torch.dtype) -> None:
        """Create RFF sample paths for f and optionally b."""
        from botorch.models import SingleTaskGP
        from botorch.sampling.pathwise.prior_samplers import draw_kernel_feature_paths
        from gpytorch.kernels import RBFKernel, MaternKernel, ScaleKernel
        from gpytorch.means import ZeroMean

        def make_kernel(lengthscale: torch.Tensor) -> ScaleKernel:
            if self.kernel_type == "rbf":
                base = RBFKernel(ard_num_dims=self.dim)
            elif self.kernel_type in ["matern52", "matern32", "matern12"]:
                nu_map = {"matern52": 2.5, "matern32": 1.5, "matern12": 0.5}
                base = MaternKernel(nu=nu_map[self.kernel_type], ard_num_dims=self.dim)
            else:
                raise ValueError(f"Unknown kernel: {self.kernel_type}")
            base.lengthscale = lengthscale
            return ScaleKernel(base)

        def draw_path(lengthscale: torch.Tensor, outputscale: float, seed: int):
            torch.manual_seed(seed)
            covar = make_kernel(lengthscale)
            covar.outputscale = outputscale

            dummy_X = torch.zeros(1, self.dim, dtype=dtype)
            dummy_Y = torch.zeros(1, 1, dtype=dtype)
            dummy_gp = SingleTaskGP(dummy_X, dummy_Y)
            dummy_gp.mean_module = ZeroMean()
            dummy_gp.covar_module = covar
            # Feature maps live in the kernel's dtype (torch's default, float32)
            self._path_dtype = covar.outputscale.dtype

            return draw_kernel_feature_paths(
                model=dummy_gp,
                sample_shape=torch.Size([1]),
                num_features=self.n_features,
            )

        # Create f sample path
        self.f_path = draw_path(self.lengthscale, self.outputscale, self.seed)

        # Create b sample path if using bias
        if self.use_bias:
            self.b_path = draw_path(
                self.bias_lengthscale, self.bias_outputscale, self.bias_seed
            )
        else:
            self.b_path = None

    def _evaluate_true(self, X: torch.Tensor) -> torch.Tensor:
        """Evaluate f(x) + b(x) if use_bias else f(x)."""
        # Evaluate in the feature map's dtype; the paper's runs did the same via a
        # patched BoTorch that cast inputs to the kernel parameters' dtype.
        X_flat = X.reshape(-1, self.dim).to(self._path_dtype)

        # Evaluate f
        f_val = self.f_path(X_flat).squeeze(0)

        # Add bias if enabled
        if self.use_bias and self.b_path is not None:
            b_val = self.b_path(X_flat).squeeze(0)
            result = f_val + b_val
        else:
            result = f_val

        return result.reshape(X.shape[:-1]).to(X.dtype)

    def get_rff_params(self) -> dict:
        """Get RFF parameters for reconstruction.

        Returns:
            Dictionary with RFF parameters, compatible with botorch GPSample format.
        """
        params = {
            "kernel_type": self.kernel_type,
            "lengthscale": self.lengthscale.clone(),
            "outputscale": self.outputscale,
            "n_features": self.n_features,
            "dim": self.dim,
            "seed": self.seed,
            "use_bias": self.use_bias,
            "feature_map_weight": self.f_path.feature_map.weight.clone(),
            "path_weight": self.f_path.weight.clone(),
        }

        if self.use_bias and self.b_path is not None:
            params.update({
                "bias_lengthscale": self.bias_lengthscale.clone(),
                "bias_outputscale": self.bias_outputscale,
                "bias_seed": self.bias_seed,
                "bias_feature_map_weight": self.b_path.feature_map.weight.clone(),
                "bias_path_weight": self.b_path.weight.clone(),
            })

        return params

    def set_rff_params(self, rff_params: dict) -> None:
        """Restore RFF weights from saved parameters."""
        if "feature_map_weight" in rff_params:
            self.f_path.feature_map.weight = rff_params["feature_map_weight"]
        if "path_weight" in rff_params:
            self.f_path.weight = rff_params["path_weight"]

        if self.use_bias and self.b_path is not None:
            if "bias_feature_map_weight" in rff_params:
                self.b_path.feature_map.weight = rff_params["bias_feature_map_weight"]
            if "bias_path_weight" in rff_params:
                self.b_path.weight = rff_params["bias_path_weight"]


# =============================================================================
# Stillinger-Weber Si energy test function
# =============================================================================


class SiSWEnergy(SyntheticTestFunction):
    r"""Stillinger-Weber potential energy for two Si atoms in a periodic unit cell.
 
    The system consists of two Si atoms in a cubic unit cell with periodic
    boundary conditions. Atom 1 is fixed at fractional coordinate (0, 0, 0).
    The input is the fractional coordinate of atom 2 in [0, 1]^3.
 
    Returns inverse_temperature * E_SW(x) in eV (scaled by 1/kT so that
    exp(-f(x)) is the Boltzmann distribution at the target temperature).
 
    Requires: ase, quippy-ase
        Install: pip install ase quippy-ase
    """
 
    dim = 3
    continuous_inds = [0, 1, 2]
    _bounds = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]
    _check_grad_at_opt = False
 
    def __init__(
        self,
        a: float = 5.43,
        inverse_temperature: float = 6.9,
        Emax: float = 1.0,
        bounds: list | None = None,
        noise_std: float | None = None,
        negate: bool = False,
        dtype: torch.dtype = torch.double,
    ) -> None:
        r"""
        Args:
            a: Lattice constant in Angstrom. Default is 5.43 (equilibrium Si).
            inverse_temperature: Multiplicative scale on the energy in units
                of 1/(kT). Default 6.9 → kT ≈ 0.145 eV (Si melting point
                T_m ≈ 1687 K). Set 1.0 for raw eV. Used so that exp(-f(x))
                corresponds to the Boltzmann distribution at the target
                temperature without modifying density_fn.
            Emax: Energy cap in eV applied before scaling. Raw SW energies
                are clipped to min(Emax, E_SW) before multiplying by
                inverse_temperature. Default 10.0 eV.
            bounds: Optional list of (lo, hi) tuples per dimension.
                If None, uses the class default [(0.0, 1.0)] * 3.
            noise_std: Standard deviation of observation noise.
            negate: If True, negate the function.
            dtype: The dtype for the bounds.
        """
        self._a = a
        self._inv_T = float(inverse_temperature)
        self._Emax = float(Emax)
        self._cell = None
        self._sw_eval = None  # Lazy init to avoid import at class load time
        if bounds is not None:
            self._bounds = [tuple(b) for b in bounds]
        super().__init__(noise_std=noise_std, negate=negate, dtype=dtype)
 
    def _get_evaluator(self):
        """Lazy-initialize the SW evaluator (requires ase + quippy)."""
        if self._sw_eval is None:
            try:
                import numpy as np
                from ase import Atoms
                from quippy.potential import Potential
            except ImportError:
                raise ImportError(
                    "SiSWEnergy requires ASE and quippy. "
                    "Install with: pip install ase quippy-ase"
                )
 
            SW_PARAM_STR = """<SW_params n_types="1">
<per_type_data type="1" atomic_num="14" />
<per_pair_data atnum_i="14" atnum_j="14" AA="7.049556277" BB="0.6022245584"
      p="4" q="0" a="1.80" sigma="2.0951" eps="2.1683" />
<per_triplet_data atnum_c="14" atnum_j="14" atnum_k="14"
      lambda="21.0" gamma="1.20" eps="2.1683" />
</SW_params>"""
 
            calc = Potential("IP SW", param_str=SW_PARAM_STR)
            self._cell = np.eye(3) * self._a
 
            def evaluate_energy(frac_coords):
                """Evaluate SW energy for a single fractional coordinate."""
                scaled_positions = np.array([
                    [0.0, 0.0, 0.0],
                    frac_coords,
                ])
                atoms = Atoms(
                    symbols="Si2",
                    scaled_positions=scaled_positions,
                    cell=self._cell,
                    pbc=True,
                )
                atoms.calc = calc
                return atoms.get_potential_energy()
 
            self._sw_eval = evaluate_energy
        return self._sw_eval
 
    def _evaluate_true(self, X: Tensor) -> Tensor:
        import numpy as np
 
        evaluate_energy = self._get_evaluator()
        X_np = X.detach().cpu().numpy().reshape(-1, 3)
        raw = np.array([evaluate_energy(x) for x in X_np])
        return torch.tensor(
            self._inv_T * np.minimum(raw, self._Emax), dtype=X.dtype, device=X.device
        ).reshape(X.shape[:-1])
 


class HOnCu13(SyntheticTestFunction):
    r"""H atom adsorption energy on a frozen Cu13 icosahedron, evaluated by MACE-MP-0.

    Setup:
        - 13 Cu atoms held fixed at the icosahedral global-minimum geometry
          (1 center atom + 12 vertices on a sphere of radius `cu_nn_distance`).
        - 1 H probe atom moves in 3D, parametrized in spherical coordinates
          (r, u, phi) where u = cos(theta) and the cluster centroid is at the origin:
              x = r * sqrt(1 - u^2) * cos(phi)
              y = r * sqrt(1 - u^2) * sin(phi)
              z = r * u
        - Energy is computed via MACE-MP-0 (medium) with pbc=False on a non-periodic
          cluster (no surface slab assumption).

    Returns inverse_temperature * min(Emax, E_total - E_cluster - E_H_isolated),
    i.e., the adsorption energy in eV scaled by 1/(kT).

    Requires: mace-torch, ase
        Install: pip install mace-torch ase
    """

    dim = 3
    continuous_inds = [0, 1, 2]
    # Default bounds: bounding box of the I_h fundamental triangle.
    #
    # The cluster is rotated so a 5-fold axis (one Cu vertex) sits at +z.
    # Then the I_h-fundamental triangle on the sphere has corners at:
    #   A = (theta=0)              (5-fold, the apex Cu vertex itself)
    #   B = (theta=theta_3, phi=π/5) (3-fold, nearest face center; theta_3≈37.38°)
    #   C = (theta=theta_2, phi=0)   (2-fold, nearest edge midpoint; theta_2≈31.72°)
    # u_min = cos(theta_3) = sqrt((3*phi_g + 2) / (3*(phi_g + 2))) ≈ 0.7947.
    # The bounding box [u_min, 1] x [0, π/5] covers the triangle with ~20%
    # over-coverage (which is benign — those points are I_h-equivalent to
    # interior points), giving a 97x reduction vs. the full sphere shell.
    _U_MIN = 0.7946544722917661  # cos of the 5-fold→3-fold axis angle
    _bounds = [(0.5, 6.0), (_U_MIN, 1.0), (0.0, math.pi / 5.0)]
    _check_grad_at_opt = False

    def __init__(
        self,
        cu_nn_distance: float = 2.556,
        r_min: float = 0.5,
        r_max: float = 6.0,
        inverse_temperature: float = 1.0,
        Emax: float = 1.0,
        device: str = "auto",
        mace_model: str = "medium",
        align_5fold_z: bool = True,
        bounds: list | None = None,
        noise_std: float | None = None,
        negate: bool = False,
        dtype: torch.dtype = torch.double,
    ) -> None:
        r"""
        Args:
            cu_nn_distance: Center-to-vertex distance of the Cu13 icosahedron in
                Angstrom. Default 2.556 (bulk fcc Cu nearest-neighbor distance).
            r_min: Minimum probe radius in Angstrom (avoids overlap with the
                central Cu atom). Default 0.5.
            r_max: Maximum probe radius in Angstrom. Default 6.0 (well past the
                adsorption layer at ~4 Å).
            inverse_temperature: Multiplicative scale on the energy in units of
                1/(kT). Default 1.0 (raw eV); pick via mass-concentration
                diagnostic for a physically meaningful Boltzmann target.
            Emax: Energy cap in eV applied before scaling. Adsorption energies
                are clipped to min(Emax, E) before multiplying by
                inverse_temperature, to bound repulsive outliers near Cu cores.
                Default 5.0 eV.
            device: "auto" (use CUDA if available), "cuda", or "cpu".
            mace_model: MACE-MP-0 size: "small", "medium", or "large".
                Default "medium".
            align_5fold_z: If True (default), rotate the icosahedron so vertex 0
                sits at (0, 0, cu_nn_distance), aligning a 5-fold axis with +z.
                This makes the I_h fundamental triangle align with a small wedge
                near +z, enabling a 120x reduction of the search space.
            bounds: Optional list of (lo, hi) tuples per dimension. If None,
                uses the bounding box of the I_h fundamental triangle:
                [(r_min, r_max), (u_min, 1.0), (0.0, π/5)] where u_min ≈ 0.7947.
            noise_std: Standard deviation of observation noise.
            negate: If True, negate the function.
            dtype: The dtype for the bounds.
        """
        self._cu_nn = float(cu_nn_distance)
        self._r_min = float(r_min)
        self._r_max = float(r_max)
        self._inv_T = float(inverse_temperature)
        self._Emax = float(Emax)
        self._device_arg = device
        self._mace_model = mace_model
        self._align_5fold_z = bool(align_5fold_z)
        self._calc = None
        self._cu_positions = None
        self._E_cluster = None
        self._E_H = None
        if bounds is not None:
            self._bounds = [tuple(b) for b in bounds]
        elif self._align_5fold_z:
            self._bounds = [(r_min, r_max), (self._U_MIN, 1.0), (0.0, math.pi / 5.0)]
        else:
            self._bounds = [(r_min, r_max), (-1.0, 1.0), (0.0, 2.0 * math.pi)]
        super().__init__(noise_std=noise_std, negate=negate, dtype=dtype)

    def _build_cu13_geometry(self):
        """Cu13 icosahedron: 1 center + 12 vertices on sphere of radius cu_nn.

        If `align_5fold_z` is True, applies a rotation about the x-axis so that
        vertex 0 (originally at (0, 1, phi_g)·scale) lands at (0, 0, cu_nn).
        """
        import numpy as np
        phi_g = (1.0 + np.sqrt(5.0)) / 2.0  # golden ratio
        # 12 vertices of regular icosahedron via golden-ratio coordinates.
        verts = np.array([
            [ 0.0,  1.0,  phi_g], [ 0.0,  1.0, -phi_g],
            [ 0.0, -1.0,  phi_g], [ 0.0, -1.0, -phi_g],
            [ 1.0,  phi_g,  0.0], [ 1.0, -phi_g,  0.0],
            [-1.0,  phi_g,  0.0], [-1.0, -phi_g,  0.0],
            [ phi_g,  0.0,  1.0], [ phi_g,  0.0, -1.0],
            [-phi_g,  0.0,  1.0], [-phi_g,  0.0, -1.0],
        ], dtype=np.float64)
        norm = np.linalg.norm(verts[0])  # = sqrt(1 + phi_g^2)
        if self._align_5fold_z:
            # Rotation about x-axis: (0, 1, phi_g)/norm -> (0, 0, 1).
            cos_a = phi_g / norm
            sin_a = 1.0 / norm
            R = np.array([
                [1.0,    0.0,    0.0],
                [0.0,  cos_a, -sin_a],
                [0.0,  sin_a,  cos_a],
            ])
            verts = verts @ R.T
        # Scale so center-to-vertex distance equals cu_nn.
        verts *= (self._cu_nn / norm)
        return np.vstack([np.zeros((1, 3)), verts])  # (13, 3): center then vertices.

    def _get_calc(self):
        """Lazy-init MACE calculator and reference energies."""
        if self._calc is None:
            try:
                from ase import Atoms
                from mace.calculators import mace_mp
                import torch as _torch
            except ImportError:
                raise ImportError(
                    "HOnCu13 requires mace-torch and ase. "
                    "Install with: pip install mace-torch ase"
                )

            if self._device_arg == "auto":
                device = "cuda" if _torch.cuda.is_available() else "cpu"
            else:
                device = self._device_arg

            self._calc = mace_mp(
                model=self._mace_model,
                dispersion=False,
                default_dtype="float64",
                device=device,
            )
            self._cu_positions = self._build_cu13_geometry()

            # Reference energies (computed once).
            cluster = Atoms(
                symbols=["Cu"] * 13,
                positions=self._cu_positions,
                pbc=False,
            )
            cluster.calc = self._calc
            self._E_cluster = float(cluster.get_potential_energy())

            h_iso = Atoms(symbols=["H"], positions=[[0.0, 0.0, 0.0]], pbc=False)
            h_iso.calc = self._calc
            self._E_H = float(h_iso.get_potential_energy())
        return self._calc

    def _spherical_to_cartesian(self, X_np):
        """X: (N, 3) with columns (r, u=cos(theta), phi). Returns Cartesian (N, 3)."""
        import numpy as np
        r = X_np[:, 0]
        u = np.clip(X_np[:, 1], -1.0, 1.0)
        phi = X_np[:, 2]
        s = np.sqrt(np.clip(1.0 - u * u, 0.0, 1.0))
        x = r * s * np.cos(phi)
        y = r * s * np.sin(phi)
        z = r * u
        return np.column_stack([x, y, z])

    def _evaluate_true(self, X: Tensor) -> Tensor:
        import numpy as np
        from ase import Atoms

        self._get_calc()
        X_np = X.detach().cpu().numpy().reshape(-1, 3)
        cart = self._spherical_to_cartesian(X_np)

        symbols = ["Cu"] * 13 + ["H"]
        positions = np.empty((14, 3), dtype=np.float64)
        positions[:13] = self._cu_positions

        energies = np.empty(len(X_np), dtype=np.float64)
        for i, h_xyz in enumerate(cart):
            positions[13] = h_xyz
            atoms = Atoms(symbols=symbols, positions=positions, pbc=False)
            atoms.calc = self._calc
            E_total = atoms.get_potential_energy()
            energies[i] = E_total - self._E_cluster - self._E_H

        energies = np.minimum(energies, self._Emax)
        return torch.tensor(
            self._inv_T * energies, dtype=X.dtype, device=X.device
        ).reshape(X.shape[:-1])


# ----------------------------------------------------------------------------
# Helpers shared by H2O-on-substrate test functions.
# ----------------------------------------------------------------------------

def _h2o_local_geometry(r_OH: float, theta_HOH_rad: float):
    """H2O atom positions in the molecular frame.

    Convention: O at origin, H-O-H bisector along +z, molecular plane = x-z.
    Returns numpy (3, 3): rows are O, H1, H2.
    """
    import numpy as np
    half = 0.5 * theta_HOH_rad
    sx = math.sin(half)
    sz = math.cos(half)
    return np.array([
        [0.0, 0.0, 0.0],                       # O
        [ r_OH * sx, 0.0, r_OH * sz],          # H1 (+x)
        [-r_OH * sx, 0.0, r_OH * sz],          # H2 (-x)
    ], dtype=np.float64)


def _euler_zyz_R(alpha: float, beta: float, gamma: float):
    """ZYZ-intrinsic Euler-angle rotation matrix: R = R_z(alpha) R_y(beta) R_z(gamma)."""
    import numpy as np
    ca, sa = math.cos(alpha), math.sin(alpha)
    cb, sb = math.cos(beta),  math.sin(beta)
    cg, sg = math.cos(gamma), math.sin(gamma)
    return np.array([
        [ca * cb * cg - sa * sg, -ca * cb * sg - sa * cg, ca * sb],
        [sa * cb * cg + ca * sg, -sa * cb * sg + ca * cg, sa * sb],
        [-sb * cg,                sb * sg,                 cb     ],
    ], dtype=np.float64)


class H2OonPt111_Slab(SyntheticTestFunction):
    r"""Rigid H2O molecule adsorbed on a frozen Pt(111) slab, evaluated by MACE-MP-0.

    Setup:
        - n_layers (default 3) of Pt(111) ABC stacking, in an nx*ny supercell
          (default 3x3) of the primitive in-plane cell. All Pt atoms frozen.
        - Topmost Pt layer at z = 0; deeper layers at z < 0; vacuum above.
        - Cell has 2D periodicity (x, y); pbc=[True, True, False].
        - One rigid H2O molecule with the O atom at (x, y, z) and orientation
          parametrized by ZYZ Euler angles (alpha, u_beta=cos(beta), gamma).

    6D input: (x, y, z, alpha, u_beta, gamma).

    Default bounds use the bounding box of the p3m1 wallpaper IAU for the
    primitive cell (top, bridge, fcc-hollow triangle), z above the slab,
    full Euler (alpha) and cos(beta), and gamma in [0, pi) (H2O C2 fold).
    Combined symmetry reduction vs. unrestricted (full primitive cell + full
    Euler) is ~12x.

    Returns inverse_temperature * min(Emax, E - E_slab - E_H2O_iso) in eV
    scaled by 1/(kT). Cap is applied before scaling.

    Requires: mace-torch, ase
    """

    dim = 6
    continuous_inds = list(range(6))
    _check_grad_at_opt = False

    def __init__(
        self,
        a_pt: float = 3.92,
        n_layers: int = 3,
        nx: int = 3,
        ny: int = 3,
        z_min: float = 0.5,
        z_max: float = 5.0,
        cell_c: float = 30.0,
        OH_bond: float = 0.957,
        HOH_angle_deg: float = 104.5,
        inverse_temperature: float = 38.68,
        Emax: float = 1.0,
        device: str = "auto",
        mace_model: str = "medium-mpa-0",
        bounds: list | None = None,
        noise_std: float | None = None,
        negate: bool = False,
        dtype: torch.dtype = torch.double,
    ) -> None:
        r"""
        Args:
            a_pt: Bulk Pt fcc lattice constant in Å (default 3.92).
            n_layers: Number of Pt layers (default 3).
            nx, ny: In-plane supercell multiplicity (default 3x3 of primitive).
            z_min, z_max: Vertical range of the O atom above the topmost Pt
                layer in Å (default 0.5 to 5.0).
            cell_c: Cell c-vector length in Å (vacuum extent, default 30).
                Only matters for ASE bookkeeping; pbc[2] is False.
            OH_bond: O-H bond length in Å.
            HOH_angle_deg: H-O-H angle in degrees.
            inverse_temperature: 1/(kT) scale on the energy. Default 38.68
                corresponds to room T (300 K). Subject to mass-concentration
                diagnostic.
            Emax: Energy cap in eV applied to the adsorption energy
                (E - E_slab - E_H2O_iso) before scaling.
            device: "auto" (CUDA if available), "cuda", or "cpu".
            mace_model: MACE-MP-0 size: "small", "medium", or "large".
            bounds: Optional explicit bounds. If None, defaults to
                [(0, a/2), (0, a*sqrt(3)/6), (z_min, z_max),
                 (0, 2*pi), (-1, 1), (0, pi)] where a = a_pt/sqrt(2) is the
                in-plane NN distance. The first two cover a bounding box of
                the p3m1 IAU triangle within the primitive cell.
            noise_std, negate, dtype: standard.
        """
        self._a_pt = float(a_pt)
        self._a_2d = self._a_pt / math.sqrt(2.0)        # in-plane NN
        self._d_layer = self._a_2d * math.sqrt(2.0 / 3.0)  # interlayer spacing
        self._n_layers = int(n_layers)
        self._nx = int(nx)
        self._ny = int(ny)
        self._z_min = float(z_min)
        self._z_max = float(z_max)
        self._cell_c = float(cell_c)
        self._r_OH = float(OH_bond)
        self._theta_HOH_rad = math.radians(HOH_angle_deg)
        self._inv_T = float(inverse_temperature)
        self._Emax = float(Emax)
        self._device_arg = device
        self._mace_model = mace_model

        self._calc = None
        self._pt_positions = None
        self._cell = None
        self._E_slab = None
        self._E_H2O = None

        if bounds is not None:
            self._bounds = [tuple(b) for b in bounds]
        else:
            a = self._a_2d
            self._bounds = [
                (0.0, a / 2.0),                   # x: bbox of IAU
                (0.0, a * math.sqrt(3.0) / 6.0),  # y: bbox of IAU
                (z_min, z_max),                   # z above topmost Pt
                (0.0, 2.0 * math.pi),             # alpha
                (-1.0, 1.0),                      # u_beta = cos(beta)
                (0.0, math.pi),                   # gamma (H2O C2)
            ]
        super().__init__(noise_std=noise_std, negate=negate, dtype=dtype)

    def _build_pt111_slab(self):
        """Construct ABC-stacked Pt(111) slab. Topmost layer at z = 0."""
        import numpy as np
        a = self._a_2d
        a1 = a * np.array([1.0, 0.0])
        a2 = a * np.array([0.5, math.sqrt(3.0) / 2.0])
        in_plane_offsets = [
            np.zeros(2),
            (a1 + a2) / 3.0,
            2.0 * (a1 + a2) / 3.0,
        ]
        positions = []
        for k in range(self._n_layers):
            offset_xy = in_plane_offsets[k % 3]
            z = -k * self._d_layer
            for i in range(self._nx):
                for j in range(self._ny):
                    xy = i * a1 + j * a2 + offset_xy
                    positions.append([xy[0], xy[1], z])
        return np.array(positions, dtype=np.float64)

    def _supercell_vectors(self):
        """Return the 3x3 cell matrix (rows = a, b, c)."""
        import numpy as np
        a = self._a_2d
        a1 = self._nx * a * np.array([1.0,             0.0,                 0.0])
        a2 = self._ny * a * np.array([0.5, math.sqrt(3.0) / 2.0,             0.0])
        c  = np.array([0.0, 0.0, self._cell_c])
        return np.stack([a1, a2, c])

    def _get_calc(self):
        """Lazy-init MACE calculator and reference energies."""
        if self._calc is None:
            try:
                from ase import Atoms
                from mace.calculators import mace_mp
                import torch as _torch
            except ImportError:
                raise ImportError(
                    "H2OonPt111_Slab requires mace-torch and ase. "
                    "Install with: pip install mace-torch ase"
                )

            if self._device_arg == "auto":
                device = "cuda" if _torch.cuda.is_available() else "cpu"
            else:
                device = self._device_arg

            self._calc = mace_mp(
                model=self._mace_model,
                dispersion=False,
                default_dtype="float64",
                device=device,
            )
            self._pt_positions = self._build_pt111_slab()
            self._cell = self._supercell_vectors()

            # Reference: bare slab energy, periodic.
            slab = Atoms(
                symbols=["Pt"] * len(self._pt_positions),
                positions=self._pt_positions,
                cell=self._cell,
                pbc=[True, True, False],
            )
            slab.calc = self._calc
            self._E_slab = float(slab.get_potential_energy())

            # Reference: isolated H2O in a large box, no PBC.
            import numpy as np
            local = _h2o_local_geometry(self._r_OH, self._theta_HOH_rad)
            iso_box = max(15.0, 3.0 * self._r_OH)
            iso_cell = np.eye(3) * iso_box
            iso_pos = local + iso_box / 2.0  # center in the box
            iso = Atoms(
                symbols=["O", "H", "H"],
                positions=iso_pos,
                cell=iso_cell,
                pbc=False,
            )
            iso.calc = self._calc
            self._E_H2O = float(iso.get_potential_energy())
        return self._calc

    def _evaluate_true(self, X: Tensor) -> Tensor:
        import numpy as np
        from ase import Atoms

        self._get_calc()
        X_np = X.detach().cpu().numpy().reshape(-1, 6)
        local = _h2o_local_geometry(self._r_OH, self._theta_HOH_rad)

        n_pt = len(self._pt_positions)
        symbols = ["Pt"] * n_pt + ["O", "H", "H"]
        positions = np.empty((n_pt + 3, 3), dtype=np.float64)
        positions[:n_pt] = self._pt_positions

        energies = np.empty(len(X_np), dtype=np.float64)
        for i, row in enumerate(X_np):
            x, y, z, alpha, u_beta, gamma = row
            beta = math.acos(min(max(float(u_beta), -1.0), 1.0))
            R = _euler_zyz_R(float(alpha), beta, float(gamma))
            o_pos = np.array([x, y, z], dtype=np.float64)
            # local rows: O, H1, H2 (O at origin in molecular frame)
            positions[n_pt + 0] = o_pos
            positions[n_pt + 1] = o_pos + R @ local[1]
            positions[n_pt + 2] = o_pos + R @ local[2]
            atoms = Atoms(
                symbols=symbols,
                positions=positions,
                cell=self._cell,
                pbc=[True, True, False],
            )
            atoms.calc = self._calc
            E_total = atoms.get_potential_energy()
            energies[i] = E_total - self._E_slab - self._E_H2O

        energies = np.minimum(energies, self._Emax)
        return torch.tensor(
            self._inv_T * energies, dtype=X.dtype, device=X.device
        ).reshape(X.shape[:-1])


class Gramacy2D(SyntheticTestFunction):
    r"""Gramacy 2D test function from Eq. 21 of Sauer et al. (2023).
    Note that its input space is defined as [-2, 4] in the above paper, but
    we follow another paper https://arxiv.org/pdf/2205.10186 to define the input space as [-2, 6].
    Two-dimensional function (evaluated on `[-2, 6] x [-2, 6]`):

        f(x1, x2) = 10 * x1 * exp(-x1^2 - x2^2)

    This function is used in active learning benchmarks in:
    Sauer, A., Gramacy, R. B., & Higdon, D. (2023). Active Learning for
    Deep Gaussian Process Surrogates. Technometrics, 65(1), 4-18.
    """

    dim = 2
    continuous_inds = list(range(dim))
    _bounds = [(-2.0, 6.0), (-2.0, 6.0)]

    def _evaluate_true(self, X: Tensor) -> Tensor:
        x1 = X[..., 0]
        x2 = X[..., 1]
        return 10 * x1 * torch.exp(-x1.pow(2) - x2.pow(2))


class Ishigami(SyntheticTestFunction):
    r"""Ishigami test function.

    Three-dimensional function (evaluated on `[-π, π]³`):

        f(x1, x2, x3) = sin(x1) + 7 * sin²(x2) + 0.1 * x3^4 * sin(x1)

    This is a well-known test function for sensitivity analysis, featuring
    strong nonlinearity and interactions between variables. It is commonly
    used in uncertainty quantification and global sensitivity analysis.

    Reference: Ishigami, T., & Homma, T. (1990). An importance quantification
    technique in uncertainty analysis for computer models.
    """

    dim = 3
    continuous_inds = list(range(dim))
    _bounds = [(-math.pi, math.pi), (-math.pi, math.pi), (-math.pi, math.pi)]

    def _evaluate_true(self, X: Tensor) -> Tensor:
        x1 = X[..., 0]
        x2 = X[..., 1]
        x3 = X[..., 2]
        return torch.sin(x1) + 7.0 * torch.sin(x2).pow(2) + 0.1 * x3.pow(4) * torch.sin(x1)
