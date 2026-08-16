#!/usr/bin/env python3
"""Reproduce the figures and numerical data used in the manuscript.

Dependencies: numpy, scipy, matplotlib.
Run from the manuscript directory:
    python generate_figures.py
"""
from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

ROOT = Path(__file__).resolve().parent
FIG = ROOT / "figures"
DATA = ROOT / "data"
FIG.mkdir(exist_ok=True)
DATA.mkdir(exist_ok=True)


def save(fig, stem, dpi=260):
    fig.tight_layout()
    fig.savefig(FIG / f"{stem}.pdf")
    fig.savefig(FIG / f"{stem}.png", dpi=dpi)
    plt.close(fig)


# ----------------------------------------------------------------------
# Figure 1: universal edge-regime diagram for the E=0 branch.
# s = a+b-p, p=min(m,n).
# ----------------------------------------------------------------------
b = np.linspace(0.2, 3.0, 600)
fig = plt.figure(figsize=(8.2, 5.7))
ax = fig.add_subplot(111)
ax.fill_between(b, -2.0, 0.0, alpha=0.18, label="Algebraic tail ($E=0$)")
ax.fill_between(
    b, 0.0, np.minimum(2.0, 2*b), alpha=0.18,
    label="Noncusped compact edge ($E=0$)"
)
mask = 2*b > 2.0
ax.fill_between(b[mask], 2.0, 2*b[mask], alpha=0.18,
                label="Cusp compacton ($E=0$)")
ax.fill_between(b, 2*b, 6.4, alpha=0.18, label="Critical/supersingular")
ax.plot(b, np.zeros_like(b), label=r"$s=0$ (exponential-tail threshold)")
ax.plot(b, np.full_like(b, 2.0), label=r"$s=2$ (linear-edge threshold)")
ax.plot(b, 2*b, label=r"$s=2b$ (critical $a=p+b$)")
ax.set_xlim(0.2, 3.0)
ax.set_ylim(-1.6, 6.2)
ax.set_xlabel(r"$b$")
ax.set_ylabel(r"$s=a+b-p$")
ax.set_title(r"Universal edge-regime diagram for the $E=0$ branch")
ax.legend(loc="upper left", fontsize=9)
ax.grid(True, alpha=0.25)
save(fig, "fig01_universal_regime_diagram")


# ----------------------------------------------------------------------
# Figure 2: representative edge laws.
# ----------------------------------------------------------------------
r = np.linspace(1e-4, 1.0, 1600)
profiles = [
    (r**1.5, r"$r^{3/2}$: smooth compact edge"),
    (r, r"$r$: linear edge"),
    (r**(2/3), r"$r^{2/3}$: balanced cusp"),
    (r**0.5, r"$r^{1/2}$: nonzero-energy cusp"),
    ((r**0.5) * np.log(np.e/r)**0.25,
     r"$r^{1/2}\log(e/r)^{1/4}$: critical"),
    (r**0.4, r"$r^{2/5}$: supersingular"),
]
fig = plt.figure(figsize=(8.0, 5.5))
ax = fig.add_subplot(111)
for y, label in profiles:
    ax.plot(r, y / y[-1], label=label)
ax.set_xlabel(r"distance to the edge $r=L-|\xi|$")
ax.set_ylabel(r"normalized local profile $U(r)/U(1)$")
ax.set_title("Representative edge regularities")
ax.legend(fontsize=8.7)
ax.grid(True, alpha=0.25)
save(fig, "fig02_edge_asymptotic_regimes")


# ----------------------------------------------------------------------
# Square-root cusp family: m=2,n=1,a=1,b=2, alpha=beta=c=k=1.
# W=U^2, W''=sqrt(W)/2-1.
# ----------------------------------------------------------------------
def energy(A):
    return A*A - A**3/3.0


def solve_kinetic(A, rtol=2e-12, atol=2e-14, max_step=0.01):
    def rhs(x, y):
        W = max(y[0], 0.0)
        return [y[1], 0.5*np.sqrt(W) - 1.0]

    def edge(x, y):
        return y[0]
    edge.terminal = True
    edge.direction = -1

    sol = solve_ivp(rhs, (0.0, 100.0), (A*A, 0.0),
                    events=edge, dense_output=True,
                    rtol=rtol, atol=atol, max_step=max_step)
    if len(sol.t_events[0]) == 0:
        raise RuntimeError(f"A={A}: trajectory did not reach W=0")
    return sol, float(sol.t_events[0][0])


def kinetic_profile(A, x):
    sol, L = solve_kinetic(A)
    x = np.asarray(x)
    U = np.zeros_like(x, dtype=float)
    inside = np.abs(x) < L
    U[inside] = np.sqrt(np.maximum(sol.sol(np.abs(x[inside]))[0], 0.0))
    return U, L


# Figure 3: kinetic cusp family.
As = [0.4, 0.8, 1.2, 1.6, 1.9]
Ls = [solve_kinetic(A)[1] for A in As]
x = np.linspace(-1.03*max(Ls), 1.03*max(Ls), 3200)
fig = plt.figure(figsize=(8.2, 5.4))
ax = fig.add_subplot(111)
for A in As:
    U, L = kinetic_profile(A, x)
    ax.plot(x, U, label=rf"$A={A:.1f}$")
ax.set_xlabel(r"$\xi$")
ax.set_ylabel(r"$U(\xi)$")
ax.set_title(r"Nonzero-energy cusp compactons: $m=2,n=1,a=1,b=2$, $0<A<2$")
ax.legend()
ax.grid(True, alpha=0.25)
save(fig, "fig03_kinetic_cusp_family")


# Figure 4: energy geometry.
W = np.linspace(0.0, 7.0, 2200)
fig = plt.figure(figsize=(8.2, 5.2))
ax = fig.add_subplot(111)
for E in [0.45, 4/3, 1.55]:
    F = 2*E - 2*W + (2/3)*W**1.5
    ax.plot(W, F, label=rf"$E={E:.3f}$")
ax.axhline(0.0)
ax.set_xlabel(r"$W=U^2$")
ax.set_ylabel(r"$F_E(W)=(W')^2$")
ax.set_title("Turning-point geometry: compacton / critical front / no compacton")
ax.set_ylim(-1.2, 5.2)
ax.legend()
ax.grid(True, alpha=0.25)
save(fig, "fig05_energy_geometry_transition")


# Figure 5: critical broadening, aligned at left edge.
fig = plt.figure(figsize=(8.2, 5.2))
ax = fig.add_subplot(111)
for A in [1.6, 1.8, 1.9, 1.95, 1.98]:
    sol, L = solve_kinetic(A, max_step=0.006)
    s = np.linspace(0.0, L, 1400)
    xi = -L + s
    U = np.sqrt(np.maximum(sol.sol(np.abs(xi))[0], 0.0))
    ax.plot(s, U, label=rf"$A={A:.2f}$")
ax.axhline(2.0)
ax.set_xlabel(r"distance from the compact edge $s$")
ax.set_ylabel(r"$U$")
ax.set_title(r"Critical broadening as $A\to2^-$")
ax.legend()
ax.grid(True, alpha=0.25)
save(fig, "fig06_critical_broadening_profiles")


# ----------------------------------------------------------------------
# Figure 6: exact zero-energy balanced cusp compacton.
# U=4 sin^2(theta), r=8 theta-4 sin(2 theta), L=4 pi.
# ----------------------------------------------------------------------
theta = np.linspace(0.0, np.pi/2, 1800)
Uhalf = 4*np.sin(theta)**2
rhalf = 8*theta - 4*np.sin(2*theta)
L_exact = 4*np.pi
xi_right = L_exact - rhalf
xi_full = np.concatenate((-xi_right[::-1], xi_right[1:]))
U_full = np.concatenate((Uhalf[::-1], Uhalf[1:]))
xext = np.linspace(-1.12*L_exact, 1.12*L_exact, 3000)
order = np.argsort(xi_full)
Uext = np.interp(xext, xi_full[order], U_full[order], left=0.0, right=0.0)
fig = plt.figure(figsize=(8.2, 5.2))
ax = fig.add_subplot(111)
ax.plot(xext, Uext)
ax.set_xlabel(r"$\xi$")
ax.set_ylabel(r"$U(\xi)$")
ax.set_title(r"Exact balanced cusp compacton: $U\sim(3r/2)^{2/3}$, $L=4\pi$")
ax.grid(True, alpha=0.25)
save(fig, "fig04_exact_balanced_cusp_compacton")


# ----------------------------------------------------------------------
# Weak residual test for A=1.
# Q = U + U W'' - U^2/2 is the integrated travelling-wave residual.
# ----------------------------------------------------------------------
def weak_test_derivative(x, L):
    """phi'(x) for the odd bump test function phi(x)=z exp(1-1/(1-z^2)), z=x/(1.2L)."""
    R = 1.2*L
    z = x/R
    phip = np.zeros_like(x)
    good = np.abs(z) < 1
    zz = z[good]
    bump = np.exp(1.0 - 1.0/(1.0-zz**2))
    dbdx = bump * (-2.0*zz/(R*(1.0-zz**2)**2))
    phip[good] = bump/R + zz*dbdx
    return phip


def discrete_residual(A, N, shift=0.0):
    """Integrated residual Q_h = U_h + U_h D_h^2 W_h - U_h^2/2 on the grid
    x_j=(j+shift)h, h=L/N. shift=0 places nodes exactly at +-L."""
    sol, L = solve_kinetic(A)
    h = L / N
    pad = max(8, N // 4)
    j = np.arange(-N-pad, N+pad+1)
    x = (j+shift)*h

    W = np.zeros_like(x, dtype=float)
    inside = np.abs(x) < L*(1.0-5e-15)
    W[inside] = np.maximum(sol.sol(np.abs(x[inside]))[0], 0.0)
    U = np.sqrt(W)

    D2W = np.zeros_like(W)
    D2W[1:-1] = (W[:-2] - 2*W[1:-1] + W[2:]) / h**2
    Q = U + U*D2W - 0.5*U**2
    l1 = h*np.sum(np.abs(Q))
    weak_action = abs(-h*np.sum(Q*weak_test_derivative(x, L)))
    return h, l1, weak_action


Ns = np.array([50, 100, 200, 400, 800, 1600, 3200])
rows = []
for N in Ns:
    h, l1, weak = discrete_residual(1.0, int(N), 0.0)
    _, l1s, weaks = discrete_residual(1.0, int(N), 1.0/3.0)
    rows.append((int(N), h, l1, weak, l1s, weaks))
with (DATA / "weak_residual_A1.csv").open("w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["N", "h", "L1_Qh_aligned", "weak_test_aligned",
                "L1_Qh_shifted", "weak_test_shifted"])
    w.writerows(rows)

hs = np.array([r[1] for r in rows])
l1s = np.array([r[2] for r in rows])
weak = np.array([r[3] for r in rows])
l1sh = np.array([r[4] for r in rows])
weaksh = np.array([r[5] for r in rows])
rate = lambda v: np.polyfit(np.log(hs[-4:]), np.log(v[-4:]), 1)[0]
fig = plt.figure(figsize=(7.2, 5.0))
ax = fig.add_subplot(111)
ax.loglog(hs, l1s, marker="o",
          label=rf"$\|Q_h\|_{{L^1}}$, aligned grid, slope $\approx {rate(l1s):.2f}$")
ax.loglog(hs, weak, marker="s",
          label=rf"weak test, aligned grid, slope $\approx {rate(weak):.2f}$")
ax.loglog(hs, l1sh, marker="o", linestyle="--",
          label=rf"$\|Q_h\|_{{L^1}}$, shifted grid, slope $\approx {rate(l1sh):.2f}$")
ax.loglog(hs, weaksh, marker="s", linestyle="--",
          label=rf"weak test, shifted grid, slope $\approx {rate(weaksh):.2f}$")
ax.set_xlabel(r"$h$")
ax.set_ylabel("residual")
ax.set_title(r"Mesh convergence of the weak residual, $A=1$")
ax.legend(fontsize=8.5)
ax.grid(True, which="both", alpha=0.25)
save(fig, "weak_residual_convergence_A1", dpi=220)

# Semiwidth of the A=1 compacton by adaptive quadrature of Eq. (85).
from scipy.integrate import quad as _quad
_E1 = energy(1.0)
_L1 = np.sqrt(2)*_quad(lambda U: U/np.sqrt(_E1 - U*U + U**3/3.0), 0.0, 1.0,
                       limit=400, epsabs=1e-13, epsrel=1e-13)[0]
print("L(A=1) by quadrature =", repr(_L1))


# ----------------------------------------------------------------------
# Critical example: m=2, n=1, a=3, b=2, alpha=beta=c=k=1, E=-1.
# (W')^2 = F(W) = 2 sqrt(W) - 2 - 2 log W, W_* = 1, W'' = W^{-1/2}/2 - W^{-1}.
# ----------------------------------------------------------------------
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import PchipInterpolator


def Fcrit(W):
    return 2*np.sqrt(W) - 2.0 - 2*np.log(W)


def build_critical_profile():
    # Crest layer: W = 1 - t^2 removes the simple-root singularity.
    t = np.linspace(0.0, np.sqrt(0.5), 200001)
    W1 = 1.0 - t**2
    g1 = np.empty_like(t)
    g1[0] = 2.0          # limit 2t/sqrt(F) = 2/sqrt(-F'(1)), F'(1) = -1
    g1[1:] = 2*t[1:]/np.sqrt(Fcrit(W1[1:]))
    xi1 = cumulative_trapezoid(g1, t, initial=0.0)
    # Edge layer: W = exp(-u) resolves the logarithmic cusp.
    u = np.linspace(np.log(2.0), 40.0, 800001)
    W2 = np.exp(-u)
    g2 = W2/np.sqrt(Fcrit(W2))
    xi2 = xi1[-1] + cumulative_trapezoid(g2, u, initial=0.0)
    tail = _quad(lambda uu: np.exp(-uu)/np.sqrt(Fcrit(np.exp(-uu))), 40.0, np.inf)[0]
    L = xi2[-1] + tail
    xi = np.concatenate((xi1, xi2[1:]))
    Wt = np.concatenate((W1, W2[1:]))
    keep = np.concatenate(([True], np.diff(xi) > 0))
    ip = PchipInterpolator(xi[keep], np.log(Wt[keep]))
    ximax = xi[keep][-1]

    def Wprof(x):
        x = np.abs(np.asarray(x, dtype=float))
        out = np.zeros_like(x)
        m = x < ximax
        out[m] = np.exp(ip(x[m]))
        return out
    return L, Wprof


Lcrit, Wcrit = build_critical_profile()
print("Critical example semiwidth L =", repr(Lcrit))

# Figure: critical profile and its logarithmically corrected edge law.
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.6))
xx = np.linspace(-1.15*Lcrit, 1.15*Lcrit, 4001)
ax1.plot(xx, np.sqrt(Wcrit(xx)))
ax1.set_xlabel(r"$\xi$")
ax1.set_ylabel(r"$U(\xi)$")
ax1.set_title(r"Critical compacton: $m=2,n=1,a=3,b=2$, $E=-1$")
ax1.grid(True, alpha=0.25)
rr = np.logspace(-8, -0.5, 400)
Wr = Wcrit(Lcrit - rr)
ax2.semilogx(rr, Wr/(np.sqrt(2)*rr*np.sqrt(np.log(1/Wr))),
             label=r"$W/[\sqrt{2}\,r\sqrt{\log(1/W)}]$")
ax2.semilogx(rr, Wr/(np.sqrt(2)*rr*np.sqrt(np.log(1/rr))),
             label=r"$W/[\sqrt{2}\,r\sqrt{\log(1/r)}]$")
ax2.axhline(1.0, color="k", linewidth=0.8)
ax2.set_xlabel(r"distance to the edge $r$")
ax2.set_ylabel("ratio")
ax2.set_title("Logarithmically corrected edge law")
ax2.legend(fontsize=9)
ax2.grid(True, which="both", alpha=0.25)
save(fig, "fig09_critical_example_profile", dpi=220)


def extended_residual(N, shift=0.0, nu=1.5):
    """Discrete residual of the divergence-form identity (65) for the critical
    example on the grid x_j=(j+shift)h, h=L/N."""
    h = Lcrit/N
    pad = max(8, N//4)
    j = np.arange(-N-pad, N+pad+1)
    x = (j+shift)*h
    W = Wcrit(x)
    W[np.abs(x) >= Lcrit] = 0.0
    Wh = 0.5*(W[:-1] + W[1:])
    dWh = (W[1:] - W[:-1])/h
    flux = Wh**nu*dWh
    quadr = Wh**(nu-1.0)*dWh**2
    P = np.zeros_like(W)
    P[1:-1] = (flux[1:] - flux[:-1])/h - nu*0.5*(quadr[1:] + quadr[:-1])
    Q = P + np.sqrt(W) - 0.5*W
    l1 = h*np.sum(np.abs(Q))
    weak_action = abs(-h*np.sum(Q*weak_test_derivative(x, Lcrit)))
    return h, l1, weak_action


rows = []
for N in Ns:
    h, l1, weak = extended_residual(int(N), 0.0)
    _, l1s, weaks = extended_residual(int(N), 1.0/3.0)
    rows.append((int(N), h, l1, weak, l1s, weaks))
with (DATA / "extended_residual_critical.csv").open("w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["N", "h", "L1_Q_aligned", "weak_test_aligned",
                "L1_Q_shifted", "weak_test_shifted"])
    w.writerows(rows)
hs = np.array([r[1] for r in rows])
cols = [np.array([r[k] for r in rows]) for k in (2, 3, 4, 5)]
labels = [r"$\|Q_h\|_{L^1}$, aligned grid", "weak test, aligned grid",
          r"$\|Q_h\|_{L^1}$, shifted grid", "weak test, shifted grid"]
fig = plt.figure(figsize=(7.2, 5.0))
ax = fig.add_subplot(111)
for v, lab, mk, ls in zip(cols, labels, "osos", ["-", "-", "--", "--"]):
    ax.loglog(hs, v, marker=mk, linestyle=ls,
              label=rf"{lab}, slope $\approx {rate(v):.2f}$")
ax.set_xlabel(r"$h$")
ax.set_ylabel("residual")
ax.set_title("Mesh convergence of the extended-operator residual, critical example")
ax.legend(fontsize=8.5)
ax.grid(True, which="both", alpha=0.25)
save(fig, "fig10_extended_residual_convergence", dpi=220)


# ----------------------------------------------------------------------
# Critical width asymptotics.
# ----------------------------------------------------------------------
Ccrit = (-6*np.sqrt(2) + 2*np.sqrt(6)
         + 2*np.sqrt(2)*np.log(24*(2-np.sqrt(3))))
eps = np.array([1e-1, 5e-2, 2e-2, 1e-2, 5e-3, 2e-3, 1e-3])
Lnum = np.array([solve_kinetic(2.0-e, max_step=0.004)[1] for e in eps])
Lasym = 2*np.sqrt(2)*np.log(1/eps) + Ccrit
with (DATA / "critical_width.csv").open("w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["epsilon", "A", "L_numerical", "L_asymptotic", "difference"])
    for e, ln, la in zip(eps, Lnum, Lasym):
        w.writerow([e, 2-e, ln, la, ln-la])

fig = plt.figure(figsize=(7.2, 5.0))
ax = fig.add_subplot(111)
ax.semilogx(eps, Lnum, marker="o", label=r"numerical $L(A)$")
ax.semilogx(eps, Lasym, marker="s", label="critical asymptotic formula")
ax.invert_xaxis()
ax.set_xlabel(r"$\varepsilon=2-A$")
ax.set_ylabel(r"$L(A)$")
ax.set_title(r"Logarithmic width divergence as $A\to2^-$")
ax.legend()
ax.grid(True, alpha=0.25)
save(fig, "width_critical_asymptotics", dpi=220)

print("Figures written to", FIG)
print("Data written to", DATA)
