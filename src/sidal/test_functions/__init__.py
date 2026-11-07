#!/usr/bin/env python3
r"""Custom test functions for active learning benchmarks.

Synthetic: Forrester, Gramacy1D, Gramacy2D, Higdon, Ishigami, GPSample,
GPSampleWithBias.

Physics-based potential energy surfaces (require ASE and the respective
calculators): H2CuMACEPES, HOnCu13, SiSWEnergy, H2OonPt111_Slab.
"""

from sidal.test_functions.synthetic import (
    Forrester,
    GPSampleWithBias,
    Gramacy1D,
    Higdon,
    H2OonPt111_Slab,
    HOnCu13,
    SiSWEnergy,
)
from sidal.test_functions.gp_sample import GPSample
from sidal.test_functions.h2cu import H2CuMACEPES
from sidal.test_functions.synthetic import Gramacy2D, Ishigami

__all__ = [
    "Forrester",
    "GPSampleWithBias",
    "GPSample",
    "Gramacy2D",
    "H2CuMACEPES",
    "Ishigami",
    "Gramacy1D",
    "H2OonPt111_Slab",
    "HOnCu13",
    "Higdon",
    "SiSWEnergy",
]


def _build_benchmark_namespace():
    """BoTorch's test functions plus the ones SIDAL adds, looked up by class name.

    Configs name problems by class (e.g. ``"Branin"``, ``"Ishigami"``); this lets
    the runner resolve both kinds with one ``getattr``.
    """
    import types

    import botorch.test_functions as botorch_test_functions

    names = {n: getattr(botorch_test_functions, n) for n in dir(botorch_test_functions)
             if not n.startswith("_")}
    names.update(GPSample=GPSample, Gramacy2D=Gramacy2D, Ishigami=Ishigami)
    return types.SimpleNamespace(**names)


benchmark_test_functions = _build_benchmark_namespace()
