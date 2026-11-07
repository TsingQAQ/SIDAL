"""Active Learning for Unknown Target Regions.

This module implements acquisition functions for active learning scenarios where
the target distribution is unknown and depends on the function values themselves.
"""

from sidal.active_learning import (
    PosteriorVariance,
    qExpectedPredictiveInformationGain,
    qNegIntegratedPosteriorVariance,
    VarianceConstraintMixin,
    BoltzmannEnergyDistribution,
    # SID-iVAR classes with proper discrete/continuous handling
    AB_SID_iVAR_Discrete,
    AB_SID_iVAR_Continuous,
    TS_SID_iVAR_Discrete,
    TS_SID_iVAR_Continuous,
    Mu_SID_iVAR_Continuous,
    Mu_SID_EPIG_Continuous,
    Mu_SID_EPIG_Discrete,
    # HAL-style acquisition (single-chain MCMC with threshold stopping)
    HAL_Acquisition,
)
from sidal.design_space import (
    ContinuousDesignSpace,
    DesignSpace,
    DiscreteDesignSpace,
    GridDiscreteDesignSpace,
)
from sidal.sampler import (
    sample_smc,
    print_smc_diagnostics,
    smc_reference_point_generator,
    uniform_reference_point_generator,
)
from sidal.test_functions import (
    Forrester,
    Gramacy1D,
    Higdon,
)

__all__ = [
    # Acquisition functions
    "PosteriorVariance",
    "qExpectedPredictiveInformationGain",
    "qNegIntegratedPosteriorVariance",
    "VarianceConstraintMixin",
    "BoltzmannEnergyDistribution",
    # SID-iVAR classes with proper discrete/continuous handling
    "AB_SID_iVAR_Discrete",
    "AB_SID_iVAR_Continuous",
    "TS_SID_iVAR_Discrete",
    "TS_SID_iVAR_Continuous",
    "Mu_SID_iVAR_Continuous",
    "Mu_SID_EPIG_Continuous",
    "Mu_SID_EPIG_Discrete",
    # HAL-style acquisition (single-chain MCMC with threshold stopping)
    "HAL_Acquisition",
    # Design spaces
    "DesignSpace",
    "DiscreteDesignSpace",
    "GridDiscreteDesignSpace",
    "ContinuousDesignSpace",
    # Samplers
    "sample_smc",
    "print_smc_diagnostics",
    "smc_reference_point_generator",
    "uniform_reference_point_generator",
    # Test functions
    "Forrester",
    "Gramacy1D",
    "Higdon",
]
