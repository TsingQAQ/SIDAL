"""Tests for runner's lambda alignment check.

Covers two pieces:
  - extract_lambda_from_density_fn (regex parsing of density_fn strings)
  - check_lambda_alignment (cross-checks acq lambda vs density_fn lambda for
    SID and HAL acquisitions)

These guards exist to prevent silent mismatches between the evaluation target
(weighted_mse weighting) and the acq function's internal Boltzmann sharpness.
"""
import unittest

from sidal.benchmarking.runner import (
    extract_lambda_from_density_fn,
    check_lambda_alignment,
    apply_lambda_val_to_acq,
    density_fn_str_for_lambda,
)


class TestExtractLambdaFromDensityFn(unittest.TestCase):
    """Regex must support common Boltzmann forms used in configs."""

    def test_k_times_f(self):
        # exp(K*f) format (eval-density emitted by --lambda_val)
        self.assertEqual(extract_lambda_from_density_fn("lambda f: torch.exp(2*f)"), 2.0)
        self.assertEqual(extract_lambda_from_density_fn("lambda f: torch.exp(-2*f)"), -2.0)
        self.assertEqual(extract_lambda_from_density_fn("lambda f: torch.exp(0.5*f)"), 0.5)
        self.assertEqual(extract_lambda_from_density_fn("lambda f: torch.exp(-0.5*f)"), -0.5)
        self.assertEqual(extract_lambda_from_density_fn("lambda f: torch.exp(-2.0*f)"), -2.0)

    def test_f_over_t(self):
        # exp(-f/T) format (used by run_lambda_bias_ablation.sh)
        self.assertEqual(extract_lambda_from_density_fn("lambda f: torch.exp(-f/2)"), -0.5)
        self.assertEqual(extract_lambda_from_density_fn("lambda f: torch.exp(-f/0.5)"), -2.0)
        self.assertEqual(extract_lambda_from_density_fn("lambda f: torch.exp(-f / 2.0)"), -0.5)
        self.assertEqual(extract_lambda_from_density_fn("lambda f: torch.exp(f/2)"), 0.5)

    def test_f_times_k(self):
        # exp(-f*K) format (f-first variant)
        self.assertEqual(extract_lambda_from_density_fn("lambda f: torch.exp(-f*2)"), -2.0)
        self.assertEqual(extract_lambda_from_density_fn("lambda f: torch.exp(f*2)"), 2.0)

    def test_bare_f(self):
        self.assertEqual(extract_lambda_from_density_fn("lambda f: torch.exp(-f)"), -1.0)
        self.assertEqual(extract_lambda_from_density_fn("lambda f: torch.exp(f)"), 1.0)

    def test_unparseable_boltzmann_raises(self):
        # Looks like exp(...f...) but not a supported pattern → must raise to
        # prevent silent bypass of alignment check.
        with self.assertRaises(ValueError):
            extract_lambda_from_density_fn("lambda f: torch.exp(f**2)")
        with self.assertRaises(ValueError):
            extract_lambda_from_density_fn("lambda f: torch.exp(2*f + 1)")
        with self.assertRaises(ValueError):
            extract_lambda_from_density_fn("lambda f: torch.exp(-(f-1)/2)")

    def test_no_exp_returns_none(self):
        # Non-Boltzmann density_fn (no torch.exp(...f...)) → None, no error.
        self.assertIsNone(extract_lambda_from_density_fn("lambda f: torch.ones_like(f)"))
        self.assertIsNone(extract_lambda_from_density_fn("lambda f: 1.0"))


class TestCheckLambdaAlignment(unittest.TestCase):
    """Alignment check covers SID-iVAR, SID-EPIG, and HAL acqs."""

    def test_target_agnostic_acqs_skipped(self):
        # RS / US / IMSE / EPIG don't use lambda → check returns None silently.
        for name in ["RS", "US", "IMSE", "EPIG"]:
            self.assertIsNone(
                check_lambda_alignment(name, {}, "lambda f: torch.exp(-f)")
            )

    def test_aligned_default_sid(self):
        # SID with default temperature/sign (acq λ = -1) and density exp(-f) (eval λ = -1) → ok.
        for name in ["AB-SID-iVAR", "TS-SID-iVAR", "mu-SID-iVAR", "mu-SID-EPIG"]:
            result = check_lambda_alignment(
                name, {"boltzmann_sign": -1.0}, "lambda f: torch.exp(-f)"
            )
            self.assertEqual(result, -1.0)

    def test_aligned_hal(self):
        # HAL: temperature=1.0, sign=-1.0 (config default) ↔ density exp(-f).
        result = check_lambda_alignment(
            "HAL",
            {"temperature": 1.0, "tau": [0.0, 3.0]},  # no sign → default -1
            "lambda f: torch.exp(-f)",
        )
        self.assertEqual(result, -1.0)

    def test_mismatch_magnitude_raises(self):
        # acq λ = -1 (default), eval λ = -2 → magnitude mismatch
        with self.assertRaises(ValueError) as cm:
            check_lambda_alignment(
                "AB-SID-iVAR", {"boltzmann_sign": -1.0}, "lambda f: torch.exp(-2*f)"
            )
        self.assertIn("Mismatch", str(cm.exception))

    def test_mismatch_sign_raises(self):
        # acq λ = +50 (positive), eval λ = -1 (negative) → sign mismatch
        with self.assertRaises(ValueError) as cm:
            check_lambda_alignment(
                "AB-SID-iVAR", {"boltzmann_lambda": 50}, "lambda f: torch.exp(-f)"
            )
        # Either "Sign mismatch" or "Mismatch" since magnitudes also differ
        self.assertTrue(
            "ismatch" in str(cm.exception),  # noqa: covers both
        )

    def test_hal_mismatch_raises(self):
        # HAL with temperature=1.0 (acq λ=-1), eval λ=-2 → mismatch
        with self.assertRaises(ValueError) as cm:
            check_lambda_alignment(
                "HAL", {"temperature": 1.0}, "lambda f: torch.exp(-2*f)"
            )
        self.assertIn("HAL", str(cm.exception))

    def test_mu_sid_epig_aligned_with_boltzmann_lambda(self):
        # mu-SID-EPIG with boltzmann_lambda=50 ↔ density exp(50*f) → aligned
        result = check_lambda_alignment(
            "mu-SID-EPIG",
            {"boltzmann_lambda": 50},
            "lambda f: torch.exp(50*f)",
        )
        self.assertEqual(result, 50)

    def test_mu_sid_epig_mismatch_raises(self):
        with self.assertRaises(ValueError):
            check_lambda_alignment(
                "mu-SID-EPIG", {"boltzmann_lambda": 50}, "lambda f: torch.exp(-f)"
            )

    def test_temperature_sign_combo(self):
        # Effective acq λ = sign / temperature
        # sign=+1, temperature=0.02 → λ=+50 ↔ eval exp(50*f) → ok
        result = check_lambda_alignment(
            "AB-SID-iVAR",
            {"temperature": 0.02, "boltzmann_sign": 1.0},
            "lambda f: torch.exp(50*f)",
        )
        self.assertAlmostEqual(result, 50.0)

    def test_density_fn_division_format(self):
        # Common script format exp(-f/T) — must work after regex fix.
        # T=2 → eval λ=-0.5; SID with sign=-1, temp=2 → acq λ=-0.5 → aligned.
        result = check_lambda_alignment(
            "AB-SID-iVAR",
            {"boltzmann_sign": -1.0, "temperature": 2.0},
            "lambda f: torch.exp(-f/2)",
        )
        self.assertEqual(result, -0.5)


class TestLambdaValOverride(unittest.TestCase):
    """--lambda_val CLI must align acq's lambda AND eval density together.

    Both must agree so that check_lambda_alignment passes after override.
    """

    def test_density_fn_str_format(self):
        # Output uses K*f format (regex-parseable).
        s = density_fn_str_for_lambda(2.5)
        self.assertEqual(extract_lambda_from_density_fn(s), 2.5)
        s = density_fn_str_for_lambda(-1.0)
        self.assertEqual(extract_lambda_from_density_fn(s), -1.0)
        s = density_fn_str_for_lambda(50)
        self.assertEqual(extract_lambda_from_density_fn(s), 50.0)

    def test_apply_to_sid_acq_sets_boltzmann_lambda(self):
        for name in ["AB-SID-iVAR", "TS-SID-iVAR", "mu-SID-iVAR", "mu-SID-EPIG"]:
            h = {"boltzmann_sign": -1.0}  # config defaults
            apply_lambda_val_to_acq(name, h, -2.0)
            self.assertEqual(h["boltzmann_lambda"], -2.0)

    def test_apply_to_hal_sets_temperature_and_sign(self):
        h = {"tau": [0.0, 3.0], "temperature": 1.0}
        apply_lambda_val_to_acq("HAL", h, -2.0)
        self.assertAlmostEqual(h["temperature"], 0.5)
        self.assertEqual(h["boltzmann_sign"], -1.0)

        h2 = {"tau": [0.0, 3.0]}
        apply_lambda_val_to_acq("HAL", h2, 50.0)
        self.assertAlmostEqual(h2["temperature"], 0.02)
        self.assertEqual(h2["boltzmann_sign"], 1.0)

    def test_apply_no_op_for_target_agnostic(self):
        for name in ["RS", "US", "IMSE", "EPIG"]:
            h = {"some_param": 1}
            apply_lambda_val_to_acq(name, h, -5.0)
            self.assertEqual(h, {"some_param": 1})  # unchanged

    def test_round_trip_aligned_after_override(self):
        # Simulate the full --lambda_val flow:
        # 1. Set density_fn_type from lambda_val
        # 2. Apply to acq hyperparams
        # 3. Run alignment check — must pass
        for lam in [-0.5, -1.0, -2.0, -5.0, 1.0, 50.0, 0.5]:
            density_str = density_fn_str_for_lambda(lam)
            for name in ["AB-SID-iVAR", "TS-SID-iVAR", "mu-SID-iVAR", "mu-SID-EPIG", "HAL"]:
                h = {"boltzmann_sign": -1.0, "tau": [0.0, 3.0]}
                apply_lambda_val_to_acq(name, h, lam)
                # After override, alignment check must succeed (no raise) and return lam.
                result = check_lambda_alignment(name, h, density_str)
                self.assertAlmostEqual(
                    result, lam,
                    msg=f"After lambda_val={lam} override on {name}, "
                        f"alignment returned {result}",
                )

    def test_misaligned_then_override_fixes_it(self):
        # Start with a misaligned config (acq λ=-1, eval λ=-2):
        h_acq = {"boltzmann_sign": -1.0}  # λ_acq = -1
        density_str = "lambda f: torch.exp(-2*f)"
        with self.assertRaises(ValueError):
            check_lambda_alignment("AB-SID-iVAR", h_acq, density_str)

        # Apply --lambda_val -2.0 to fix: should now pass.
        apply_lambda_val_to_acq("AB-SID-iVAR", h_acq, -2.0)
        result = check_lambda_alignment("AB-SID-iVAR", h_acq, density_str)
        self.assertEqual(result, -2.0)

    def test_hal_misalignment_fixed_by_override(self):
        h_acq = {"temperature": 1.0}  # λ_HAL = -1 (default sign=-1)
        density_str = "lambda f: torch.exp(-5*f)"
        with self.assertRaises(ValueError):
            check_lambda_alignment("HAL", h_acq, density_str)

        apply_lambda_val_to_acq("HAL", h_acq, -5.0)
        result = check_lambda_alignment("HAL", h_acq, density_str)
        self.assertAlmostEqual(result, -5.0)


if __name__ == "__main__":
    unittest.main()
