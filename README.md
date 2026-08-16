# Edge-cusped weak compactons in degenerate nonlinear dispersive equations

Reproducibility code for the manuscript

**Zenodo DOI (all versions):** https://doi.org/10.5281/zenodo.21970590  
**Archived release v1.0.0:** https://doi.org/10.5281/zenodo.21970591

> F. R. Villatoro, *Edge-cusped weak compactons in degenerate nonlinear dispersive equations*, submitted to **Studies in Applied Mathematics** (2026).

The paper classifies one-hump nonnegative weak compactons of

$$\tfrac{1}{m}(u^m)_t + \alpha (u^n)_x + \beta\left[u^a (u^b)_{xx}\right]_x = 0,
\qquad m,n,b>0,\quad m\neq n,\quad a\ge 0,$$

in a measure-admissible weak class, and identifies the two mechanisms that
produce an infinite physical slope at the free boundary. This repository
contains the code that generates every figure, table, and numerical constant
reported in the paper.

## Contents

| File | Description |
| --- | --- |
| `generate_figures.py` | Regenerates all figures (PDF and PNG) and the data tables. Runtime is a few minutes on a laptop. |
| `verify_claims.py` | Recomputes the closed-form constants of the paper to 30 digits with `mpmath`, by quadrature routes independent of the ODE integrator. Exits with a nonzero status if any check fails. |
| `figures/` | Figures as included in the manuscript. |
| `data/` | CSV output of the numerical verification. |
| `requirements.txt` | Python dependencies. |

## Reproducing the results

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python verify_claims.py      # 13 high-precision checks, seconds
python generate_figures.py   # all figures and CSV data, minutes
```

Tested with Python 3.12, NumPy 2.x, SciPy 1.14, Matplotlib 3.9 and mpmath 1.3.
The scripts write only inside `figures/` and `data/`.

## Map from output to manuscript

| Output | Manuscript item |
| --- | --- |
| `figures/fig01_universal_regime_diagram.*` | Figure 1, universal edge-regime diagram in the variables $(b,s)$, $s=a+b-p$ |
| `figures/fig02_edge_asymptotic_regimes.*` | Figure 2, representative local edge laws |
| `figures/fig09_critical_example_profile.*` | Figure 3, explicit critical compacton ($m=2$, $n=1$, $a=3$, $b=2$, $E=-1$) |
| `figures/fig03_kinetic_cusp_family.*` | Figure 4, square-root cusp family, $0<A<2$ |
| `figures/fig05_energy_geometry_transition.*` | Figure 5, turning-point geometry |
| `figures/fig06_critical_broadening_profiles.*` | Figure 6, critical broadening as $A\to2^-$ |
| `figures/fig04_exact_balanced_cusp_compacton.*` | Figure 7, exact $r^{2/3}$ cusp compacton |
| `figures/weak_residual_convergence_A1.*`, `data/weak_residual_A1.csv` | Figure 8 and Table 1, weak-residual convergence on aligned and shifted grids |
| `figures/fig10_extended_residual_convergence.*`, `data/extended_residual_critical.csv` | Figure 9, residual of the once-integrated identity for the critical compacton |
| `figures/width_critical_asymptotics.*`, `data/critical_width.csv` | Figure 10, logarithmic width divergence |

## Numerical constants verified by `verify_claims.py`

Semiwidth $L=1.915346448610\ldots$ and edge coefficient $(2E)^{1/4}=1.074569931823\ldots$
of the $A=1$ square-root cusp compacton; the width constant
$C_*=1.677672331176593\ldots$ and the $O(\varepsilon)$ remainder $\sqrt2/3$ of the
critical-width asymptotics; the small-amplitude coefficient $\pi/8-1/6$; the
semiwidth $4\pi$ of the exact $r^{2/3}$ compacton, obtained both by quadrature
and from the incomplete-beta representation; the edge amplitude $A_0=(3/2)^{2/3}$;
and, for the critical example, the simple root $W_*=1$, the semiwidth
$L=1.695756643916\ldots$, the double root $W=4$ and the corresponding
$E_{\rm crit}=2\log2-2$.

## License

Code and data are released under the MIT License (see `LICENSE`). The figures
are also covered by it; if you reuse them in a publication, please cite the
paper.

## Citation

For the repository as a whole, cite the concept DOI:

> F. R. Villatoro, *Reproducibility code for "Edge-cusped weak compactons in
> degenerate nonlinear dispersive equations"*, Zenodo.  
> https://doi.org/10.5281/zenodo.21970590

For the exact software version used for the manuscript, cite:

> F. R. Villatoro, *Reproducibility code for "Edge-cusped weak compactons in
> degenerate nonlinear dispersive equations"*, version 1.0.0, Zenodo, 2026.  
> https://doi.org/10.5281/zenodo.21970591

See also `CITATION.cff`.
