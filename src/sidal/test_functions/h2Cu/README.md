# H₂/Cu benchmark assets

Both files are taken unchanged from [wgst/ml-gas-surface](https://github.com/wgst/ml-gas-surface) (MIT License, Copyright (c) 2023 Wojciech Stark; see [`LICENSE`](LICENSE)):

- `MACE_model_swa.model`: `models/h2cu/pes/MACE_model_swa.model`, a MACE potential for H₂ on Cu.
- `cu100_0K.in`: `scripts/dynamics/state-to-state-scattering/initial_conditions/input_structures/cu100_0K.in`, the Cu(100) slab (FHI-aims format), an input structure for the scattering simulations of Stark et al. (2025).

If you use them, please cite:

- W. G. Stark, C. van der Oord, I. Batatia, Y. Zhang, B. Jiang, G. Csányi, R. J. Maurer, Benchmarking of machine learning interatomic potentials for reactive hydrogen dynamics at metal surfaces, *Mach. Learn.: Sci. Technol.* **5**, 030501 (2024). [doi:10.1088/2632-2153/ad5f11](https://doi.org/10.1088/2632-2153/ad5f11)
- W. G. Stark, J. Westermayr, O. A. Douglas-Gallardo, J. Gardner, S. Habershon, R. J. Maurer, Machine learning interatomic potentials for reactive hydrogen dynamics at metal surfaces based on iterative refinement of reaction probabilities, *J. Phys. Chem. C* **127**, 24168–24182 (2023). [doi:10.1021/acs.jpcc.3c06648](https://doi.org/10.1021/acs.jpcc.3c06648)
- W. G. Stark, C. L. Box, M. Sachs, N. Hertl, R. J. Maurer, Nonadiabatic reactive scattering of hydrogen on different surface facets of copper, *Phys. Rev. B* **112**, 035422 (2025). [doi:10.1103/h7vd-94pk](https://doi.org/10.1103/h7vd-94pk)
- J. Gardner, O. A. Douglas-Gallardo, W. G. Stark, J. Westermayr, S. M. Janke, S. Habershon, R. J. Maurer, NQCDynamics.jl: A Julia package for nonadiabatic quantum classical molecular dynamics in the condensed phase, *J. Chem. Phys.* **156**, 174801 (2022). [doi:10.1063/5.0089436](https://doi.org/10.1063/5.0089436)

The first paper trains and benchmarks the MACE model, the second provides the DFT (SRP48) training data, and the slab ships with the scattering scripts of the third. The upstream repository asks users of its code and data to cite the second, third and fourth.
