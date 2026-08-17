#!/usr/bin/env python3
"""Discrete residual test for the exact zero-energy r^{2/3} cusp compacton.

Parameters m=2, n=1, a=2, b=2, alpha=-1, beta=1, c=-1, k=1 (delta=1).
The profile is known in closed parametric form,

    U = 4 sin^2(theta),   L - |xi| = 8 theta - 4 sin(2 theta),   L = 4 pi,

so no ODE integration is needed. The integrated traveling-wave equation is

    Q = W D^2 W - W^{1/2} + W/2 = 0,   W = U^2,

which contains no boundary atom because W' -> 0 at the free boundary. The test
compares an edge-aligned grid with a shifted one; unlike the nonzero-energy
family, both give the same rate.

Outputs: figures/fig11_balanced_residual_convergence.{pdf,png}
         data/balanced_residual_zero_energy.csv

Dependencies: numpy, scipy, matplotlib.
"""
from pathlib import Path
import csv

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parent
FIG = ROOT / "figures"
DATA = ROOT / "data"
FIG.mkdir(exist_ok=True)
DATA.mkdir(exist_ok=True)

L = 4.0 * np.pi


def U_of_distance(r):
    """Invert L-|xi| = 8 theta - 4 sin(2 theta) and return U = 4 sin^2(theta)."""
    r = np.atleast_1d(np.asarray(r, dtype=float))
    out = np.zeros_like(r)
    for i, ri in enumerate(r):
        if ri <= 0.0:
            out[i] = 0.0
        elif ri >= L:
            out[i] = 4.0
        else:
            theta = brentq(lambda th: 8*th - 4*np.sin(2*th) - ri,
                           0.0, 0.5*np.pi, xtol=1e-15, rtol=8.9e-16, maxiter=200)
            out[i] = 4.0*np.sin(theta)**2
    return out


def profile_W(x):
    x = np.abs(np.asarray(x, dtype=float))
    U = np.zeros_like(x)
    inside = x < L
    U[inside] = U_of_distance(L - x[inside])
    return U**2


def weak_test_derivative(x, R):
    """phi'(x) for the odd bump phi(x) = z exp(1-1/(1-z^2)), z = x/R."""
    z = x/R
    phip = np.zeros_like(x)
    good = np.abs(z) < 1
    zz = z[good]
    bump = np.exp(1.0 - 1.0/(1.0 - zz**2))
    dbdx = bump*(-2.0*zz/(R*(1.0 - zz**2)**2))
    phip[good] = bump/R + zz*dbdx
    return phip


def residual(N, shift=0.0):
    h = L/N
    pad = max(8, N//4)
    j = np.arange(-N-pad, N+pad+1)
    x = (j + shift)*h
    W = profile_W(x)
    W[np.abs(x) >= L] = 0.0
    D2W = np.zeros_like(W)
    D2W[1:-1] = (W[:-2] - 2*W[1:-1] + W[2:])/h**2
    Q = W*D2W - np.sqrt(W) + 0.5*W
    l1 = h*np.sum(np.abs(Q))
    weak = abs(-h*np.sum(Q*weak_test_derivative(x, 1.2*L)))
    return h, l1, weak


Ns = [50, 100, 200, 400, 800, 1600, 3200]
rows = []
for N in Ns:
    h, l1, weak = residual(N, 0.0)
    _, l1s, weaks = residual(N, 1.0/3.0)
    rows.append((N, h, l1, weak, l1s, weaks))

with (DATA / "balanced_residual_zero_energy.csv").open("w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["N", "h", "L1_Q_aligned", "weak_test_aligned",
                "L1_Q_shifted", "weak_test_shifted"])
    w.writerows(rows)

hs = np.array([r[1] for r in rows])
cols = [np.array([r[k] for r in rows]) for k in (2, 3, 4, 5)]
rate = lambda v: np.polyfit(np.log(hs[-4:]), np.log(v[-4:]), 1)[0]
labels = [r"$\|Q_h\|_{L^1}$, aligned", "weak test, aligned",
          r"$\|Q_h\|_{L^1}$, shifted", "weak test, shifted"]

fig = plt.figure(figsize=(7.2, 5.0))
ax = fig.add_subplot(111)
for v, lab, mk, ls in zip(cols, labels, "osos", ["-", "-", "--", "--"]):
    ax.loglog(hs, v, marker=mk, linestyle=ls,
              label=rf"{lab}, slope $\approx {rate(v):.2f}$")
ax.loglog(hs, cols[0][0]*(hs/hs[0])**(5.0/3.0), color="0.5", linewidth=0.9,
          linestyle=":", label=r"$O(h^{5/3})$ reference")
ax.set_xlabel(r"$h$")
ax.set_ylabel("residual")
ax.set_title("Mesh convergence of the residual, zero-energy cusp compacton")
ax.legend(fontsize=8.5)
ax.grid(True, which="both", alpha=0.25)
fig.tight_layout()
fig.savefig(FIG / "fig11_balanced_residual_convergence.pdf")
fig.savefig(FIG / "fig11_balanced_residual_convergence.png", dpi=220)
plt.close(fig)

print("slopes:", [round(rate(v), 3) for v in cols])
print("figure and CSV written")
