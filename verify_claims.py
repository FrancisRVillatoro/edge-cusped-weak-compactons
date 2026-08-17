#!/usr/bin/env python3
"""Independent high-precision verification of the closed-form numerical
claims of the manuscript.

Every quantity is recomputed with mpmath at 30 decimal digits, by a route
that does not use the ODE integrator of ``generate_figures.py``, and is
compared with the value printed in the paper. Run with

    python verify_claims.py

Dependencies: mpmath.
"""
from mpmath import mp, mpf, sqrt, log, pi, quad, beta, findroot

mp.dps = 30

TOL = mpf(10) ** (-12)
checks = []


def check(label, computed, reference, equation, tol=TOL):
    """Record a comparison between a recomputed and a published value."""
    err = abs(mpf(computed) - mpf(reference))
    checks.append((label, equation, mp.nstr(computed, 15), err, err < tol))


# ----------------------------------------------------------------------
# Section 6: square-root cusp family, m=2, n=1, a=1, b=2.
# (W')^2 = 2E - 2W + (2/3) W^{3/2},  W = U^2,  E(A) = A^2 - A^3/3.
# ----------------------------------------------------------------------
def energy(A):
    return A**2 - A**3 / 3


def semiwidth(A):
    """L(A) from Eq. (91) of the manuscript, by adaptive quadrature."""
    E = energy(A)
    f = lambda U: sqrt(2) * U / sqrt(E - U * U + U**3 / 3)
    return quad(f, [0, A / 2, A])


check("semiwidth L(A=1)", semiwidth(mpf(1)),
      mpf("1.915346448610"), "(116)")
check("edge coefficient (2E)^{1/4} at A=1", (2 * energy(mpf(1))) ** (mpf(1) / 4),
      mpf("1.074569931823"), "(117)")

# Critical amplitude and energy of the family: V(W) = W - W^{3/2}/3 peaks at W=4.
check("critical energy E_crit", energy(mpf(2)), mpf(4) / 3, "(96)")

# Exact constant of the logarithmic width divergence, Eq. (92).
C_star = -6 * sqrt(2) + 2 * sqrt(6) + 2 * sqrt(2) * log(24 * (2 - sqrt(3)))
check("width constant C_*", C_star, mpf("1.677672331176593"), "(100)")

# Remainder of Eq. (99): [L(A) - 2 sqrt(2) log(1/eps) - C_*]/eps -> sqrt(2)/3.
eps = mpf("1e-5")
rem = (semiwidth(2 - eps) - (2 * sqrt(2) * log(1 / eps) + C_star)) / eps
check("O(eps) remainder of Eq. (99)", rem.real, sqrt(2) / 3, "(B17)", tol=mpf("1e-4"))

# Small-amplitude expansion, Eq. (94): L(A) = sqrt(2) A [1 + (pi/8 - 1/6) A + ...].
A = mpf("1e-4")
coef = (semiwidth(A) / (sqrt(2) * A) - 1) / A
check("small-amplitude coefficient", coef.real, pi / 8 - mpf(1) / 6, "(94)",
      tol=mpf("1e-4"))

# ----------------------------------------------------------------------
# Section 7: exact balanced cusp compacton, m=2, n=1, a=2, b=2.
# (W')^2 = 4 sqrt(W) - W,  W_* = 16,  U_max = 4,  L = 4 pi.
# ----------------------------------------------------------------------
L_balanced = quad(lambda W: 1 / sqrt(4 * sqrt(W) - W), [0, 4, 16])
check("semiwidth of the r^{2/3} compacton", L_balanced, 4 * pi, "(113)")

# The same width from the incomplete-beta representation, Eqs. (45)-(47).
# Here p = n = 1, q = m = 2, r_j = j/2, so r_p = 1/2 and Delta = 1/2; with
# sigma_m = c/(m delta) = -1/2 and sigma_n = -alpha/delta = 1, K_p = 2 sigma_n/r_n = 4
# and W_* = 16, whence mu = (1 - r_p/2)/Delta = 3/2.
r_p, Delta, K_p, W_star = mpf(1) / 2, mpf(1) / 2, mpf(4), mpf(16)
mu = (1 - r_p / 2) / Delta
L_beta = W_star ** (1 - r_p / 2) / (Delta * sqrt(K_p)) * beta(mu, mpf(1) / 2)
check("same width via incomplete beta", L_beta, 4 * pi, "(45)")

# Edge law U ~ (3r/2)^{2/3}: check the amplitude A_0 of Eq. (38).
# s = a + b - p = 3, gamma = 2/s = 2/3, A_0 = [(1 - r_p/2) sqrt(K_p)]^{2/s}.
A0 = ((1 - r_p / 2) * sqrt(K_p)) ** (mpf(2) / 3)
check("edge amplitude A_0", A0, (mpf(3) / 2) ** (mpf(2) / 3), "(38)")

# ----------------------------------------------------------------------
# Section 5.3: explicit critical compacton, m=2, n=1, a=3, b=2, E=-1.
# (W')^2 = 2 sqrt(W) - 2 - 2 log W,  simple root W_* = 1.
# ----------------------------------------------------------------------
Fcrit = lambda W: 2 * sqrt(W) - 2 - 2 * log(W)
check("root W_* of the critical example", findroot(Fcrit, mpf("0.9")), 1, "(77)")
L_crit = quad(lambda W: 1 / sqrt(Fcrit(W)), [0, mpf("1e-8"), mpf("1e-3"), mpf("0.5"), 1])
check("semiwidth of the critical example", L_crit, mpf("1.695756643916"), "(79)")

# Double root of the same family: F'(W) = W^{-1/2} - 2/W vanishes at W = 4,
# where the edge energy is E_crit = 2 log 2 - 2.
W0 = findroot(lambda W: 1 / sqrt(W) - 2 / W, mpf(3))
check("double root of the critical family", W0, 4, "Sec. 5.3")
check("critical energy of the critical family",
      -(2 * sqrt(W0) - 2 * log(W0)) / 2, 2 * log(2) - 2, "Sec. 5.3")

# ----------------------------------------------------------------------
if __name__ == "__main__":
    width = max(len(c[0]) for c in checks)
    print(f"{'quantity'.ljust(width)}  {'eq.':>7}  {'recomputed':>22}  "
          f"{'|error|':>10}  status")
    for label, equation, value, err, ok in checks:
        print(f"{label.ljust(width)}  {equation:>7}  {value:>22}  "
              f"{mp.nstr(err, 3):>10}  {'ok' if ok else 'FAILED'}")
    failed = [c[0] for c in checks if not c[4]]
    if failed:
        raise SystemExit(f"\n{len(failed)} check(s) failed: {', '.join(failed)}")
    print(f"\nAll {len(checks)} checks passed at {mp.dps} digits.")
