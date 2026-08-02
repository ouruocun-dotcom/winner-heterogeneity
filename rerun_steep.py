"""
==============================================================================
RE-MEASUREMENT OF THE STEEP-WELL ROWS  (c >= 2)
==============================================================================
Levy flights in U(x) = |x|^c/c under symmetric alpha-stable noise.
Prediction:  P(X > x) ~ x^{-beta},  beta = c + alpha - 2  (Chechkin 2004).

HOW TO RUN
----------
Paste into one Colab cell and press play.  Nothing to configure, no input
files.  About 9 minutes.  Send me everything it prints.

WHY THESE ROWS NEED REDOING
---------------------------
The c >= 2 entries in the old table were taken with the same broken sampling
that produced the spurious c=1.5 result: the state was recorded straight
after the noise increment, so the increment's own x^{-alpha} tail was baked
into the measurement.  For c > 2 the prediction beta = c+alpha-2 exceeds
alpha, which means the spurious tail is HEAVIER than the real one and takes
over the far tail, biasing the fit downward.  The c=4, alpha=1.5 row came
out at 4.267 against 3.500 predicted, which is not a small discrepancy and
cannot be quoted.  Everything here uses the corrected order: noise, then
drift, and record after the drift.

TWO THINGS THAT MAKE STEEP WELLS DIFFERENT FROM SHALLOW ONES
------------------------------------------------------------
(1) THE DISCRETE-TIME CAP.  With superlinear drift the exact flow brings a
    walker in from infinity in finite time: one step of length dt lands it
    at |x| <= [(c-2)dt]^{-1/(c-2)} at the very most.  Sampling at spacing dt
    therefore truncates the recorded tail at that cap.  It is real, it is
    computable, and it is why dt has to be chosen per c rather than fixed.
    Here dt is set so the cap sits well above the fitting window, and both
    are printed so you can see the margin.

(2) THE WINDOW BIAS, which is the important one.  For large beta the tail is
    reachable only at moderate x, where subleading corrections have not yet
    died away, so ANY finite-window fit reads high -- including a fit to the
    exact answer.  Fitting the exact c=4, alpha=1 density over [1.4, 7.7]
    gives 3.12, not 3.00.  So a measured 3.09 is not 3% error; it is correct.
    The script therefore prints, for that row, the same fit applied to the
    exact law, as a self-calibration: measured and exact-window numbers
    should agree to about 1%, and their common offset from c+alpha-2 is the
    window effect, not a discrepancy.  Quote the comparison, not the raw gap.

The local-slope method used for c < 2 is the wrong tool here -- it needs
large x, where a steep tail has no samples left -- so this script uses a
regression of log P(X>x) on log x over an automatically chosen window,
which is what the original table used.

Each row runs from a narrow and a broad initial condition; their agreement
certifies equilibration.
==============================================================================
"""

import time
import numpy as np
from scipy import integrate

SEED = 20260730
M = 50_000          # walkers
SNAPS = 20          # late-time snapshots pooled -> 1e6 samples per run
CAP_TARGET = 100.0  # keep the discrete-time cap this far out
NMIN = 300          # smallest survivor count admitted into the fit
NPTS = 14           # points in the log-log regression

# (c, alpha, physical time).  Steep wells relax fast, so t is short; the cost
# is driven by dt, which the cap forces down as c grows.
ROWS = [(2.0, 1.0, 50.0),
        (2.0, 1.5, 50.0),
        (3.0, 0.5, 10.0),
        (3.0, 1.0, 10.0),
        (4.0, 1.0, 2.0),
        (4.0, 1.5, 2.0)]


def stable(alpha, n, rng):
    """Symmetric alpha-stable, unit scale, phi(k)=exp(-|k|^alpha).  CMS method."""
    if alpha >= 2.0:
        return rng.standard_normal(n) * np.sqrt(2.0)
    U = rng.uniform(-np.pi / 2, np.pi / 2, n)
    W = rng.exponential(1.0, n)
    if abs(alpha - 1.0) < 1e-9:
        return np.tan(U)
    return (np.sin(alpha * U) / np.cos(U) ** (1.0 / alpha)
            * (np.cos(U - alpha * U) / W) ** ((1.0 - alpha) / alpha))


def drift_exact(x, c, dt):
    """Exact flow of dx/dt = -sign(x)|x|^{c-1} over one step."""
    a, s = np.abs(x), np.sign(x)
    if abs(c - 2.0) < 1e-12:
        return x * np.exp(-dt)
    p = 2.0 - c
    v = a ** p + (c - 2.0) * dt
    if c < 2.0:
        v = np.maximum(v, 0.0)
    return s * v ** (1.0 / p)


def pick_dt(c):
    """Small enough that the discrete-time cap clears CAP_TARGET."""
    if abs(c - 2.0) < 1e-12:
        return 0.002
    return min(0.02, 1.0 / ((c - 2.0) * CAP_TARGET ** (c - 2.0)))


def cap_of(c, dt):
    return np.inf if c <= 2.0 else ((c - 2.0) * dt) ** (-1.0 / (c - 2.0))


def run(c, alpha, t_phys, dt, seed, broad):
    rng = np.random.default_rng(seed)
    if broad:
        x = np.minimum(rng.pareto(0.3, M) + 1.0, 50.0) * rng.choice([-1.0, 1.0], M)
    else:
        x = rng.standard_normal(M) * 0.5
    T = int(round(t_phys / dt))
    amp = dt ** (1.0 / alpha)
    gap = max(T // (4 * SNAPS), 1)
    pool = []
    for step in range(T):
        # noise first, then drift, and record AFTER the drift
        x = drift_exact(x + amp * stable(alpha, M, rng), c, dt)
        x = np.clip(x, -1e14, 1e14)
        if step >= T - SNAPS * gap and (T - step) % gap == 0:
            pool.append(np.abs(x).copy())
    return np.sort(np.concatenate(pool))


def fit_tail(a, c, dt):
    """Regress log P(X>x) on log x over an automatic window; return also the
    window so the same fit can be applied to an exact law for calibration."""
    n = a.size
    lo = a[int(0.90 * n)]
    hi = min(cap_of(c, dt) / 10.0, a[n - NMIN])
    if not (hi > lo * 1.5):
        return np.nan, lo, hi
    xs = np.logspace(np.log10(lo), np.log10(hi), NPTS)
    S = np.array([(n - np.searchsorted(a, x, 'right')) / n for x in xs])
    m = S > 0
    return -np.polyfit(np.log(xs[m]), np.log(S[m]), 1)[0], lo, hi


def exact_c4_a1(lo, hi):
    """Same fit applied to the exact stationary law for c=4, alpha=1:
    f(x) = 1/(pi(1-x^2+x^4)), tail exponent exactly 3 (Chechkin 2004 Eq.35)."""
    f = lambda x: 1.0 / (np.pi * (1.0 - x ** 2 + x ** 4))
    xs = np.logspace(np.log10(lo), np.log10(hi), NPTS)
    S = np.array([2 * integrate.quad(f, x, np.inf)[0] for x in xs])
    return -np.polyfit(np.log(xs), np.log(S), 1)[0]


t0 = time.time()
print("=" * 88)
print("  Re-measurement of the steep-well rows, corrected sampling")
print(f"  walkers={M}  snapshots pooled={SNAPS}  cap target={CAP_TARGET}")
print("  'exact-window' = the SAME fit applied to the known exact law, which")
print("  is the number the measurement should reproduce (not c+alpha-2).")
print("=" * 88)
print(f"\n{'c':>5} {'alpha':>5} {'pred':>7} {'dt':>9} {'cap':>7}  {'init':>7} "
      f"{'beta':>7}  {'window':>16}")

SUMMARY = []
for c, al, t in ROWS:
    dt = pick_dt(c)
    got = []
    for broad in (False, True):
        a = run(c, al, t, dt, SEED + int(100 * c) + int(10 * al) + int(broad), broad)
        b, lo, hi = fit_tail(a, c, dt)
        got.append((b, lo, hi))
        print(f"{c:5.2f} {al:5.2f} {c + al - 2:7.3f} {dt:9.2g} {cap_of(c, dt):7.3g}  "
              f"{'broad' if broad else 'narrow':>7} {b:7.3f}  "
              f"[{lo:6.2f},{hi:8.2f}]", flush=True)
    b = np.nanmean([g[0] for g in got])
    gap = abs(got[0][0] - got[1][0])
    lo, hi = got[0][1], got[0][2]
    ex = np.nan
    if abs(c - 4.0) < 1e-9 and abs(al - 1.0) < 1e-9:
        ex = exact_c4_a1(lo, hi)
        print(f"{'':>36}  exact law, same window -> {ex:7.3f}"
              f"   (asymptote is 3.000; the offset is the window)")
    elif abs(c - 2.0) < 1e-9:
        ex = al          # stationary law is alpha-stable of index alpha
        print(f"{'':>36}  exact asymptote     -> {ex:7.3f}"
              f"   (OU under stable noise: beta = alpha exactly)")
    print(f"{'':>36}  narrow-vs-broad gap {gap:.3f}"
          f" ({'equilibrated' if gap < 0.08 else 'TOO SHORT'})\n")
    SUMMARY.append((c, al, c + al - 2, b, ex, gap))

print("=" * 88)
print("SUMMARY")
print(f"{'c':>5} {'alpha':>5} {'c+alpha-2':>10} {'measured':>9} "
      f"{'exact/window':>13} {'eq.gap':>7}")
for c, al, pred, b, ex, gap in SUMMARY:
    e = f"{ex:13.3f}" if np.isfinite(ex) else f"{'--':>13}"
    print(f"{c:5.2f} {al:5.2f} {pred:10.3f} {b:9.3f} {e} {gap:7.3f}")

print("""
HOW TO READ THIS
  * c=4, alpha=1 is the calibration row.  'measured' should match
    'exact/window' to about 1%.  If it does, the integrator and the estimator
    are both sound, and the gap between either of them and c+alpha-2 is the
    finite-window effect -- expect roughly +3-4% at these window sizes.
  * c=2 rows: beta = alpha is exact with no window caveat at all, since the
    stationary law is itself alpha-stable.  These should land within ~1-2%.
  * The remaining rows have no closed form.  Read them as: measured value,
    minus the window offset calibrated above, compared with c+alpha-2.
  * 'TOO SHORT' on any row means raise its physical time in ROWS.
  * alpha < 1 with c > 2 (the 3.0/0.5 row) is covered by neither Chechkin
    2004 (which assumes alpha >= 1) nor Dybiec 2010 (which assumes c < 2),
    so it is a genuine prediction of the formula rather than a check against
    a known result.
""")
print(f"elapsed {time.time() - t0:.1f} s")
