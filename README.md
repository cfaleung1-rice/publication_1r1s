# Numerical Simulation and Analytical Calculation Codes of "A Minimal Stochastic Model of Microbial Ecological Dynamics in a Single-Species-Single-Resource Setting"

Reference implementation of the analytical and deterministic solutions for a minimal two-step chemical-kinetic model of microbial ecological dynamics.

## 1. run_all.py:
Sequentially runs the code in each subdirectory and reproduces high-resolution figures included in the publication.

## 2. theory_deterministic_1r1s.py:
Includes functions that compute analytical approximations to the system of deterministic, mean-field ordinary differential equations detailed in Section S8 of the Supplementary Information.

## 3. theory_stochastic_1r1s.py:
Includes functions that compute analytical solutions to the coarse-grained effective master equations in the limiting regimes and numerically simulate the reaction system by Gillespie algorithm detailed in Sections S1-S7 of the Supplementary Information.

## References
Experimental data used for fitting are retrieved from the following publications:
- Monod, [1949](https://doi.org/10.1146/annurev.mi.03.100149.002103)
- Varma and Palsson, [1994](https://doi.org/10.1128/aem.60.10.3724-3731.1994)
- Enjalbert et al., [2015](https://doi.org/10.1128/jb.00128-15)
