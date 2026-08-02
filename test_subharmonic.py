"""
==============================================================================
DOES  beta = c + alpha - 2  HOLD FOR SUBHARMONIC POTENTIALS (c < 2)?
==============================================================================
Levy flights in U(x) = |x|^c/c driven by symmetric alpha-stable noise.
Prediction:  P(X > x) ~ x^{-beta},  beta = c + alpha - 2.

HOW TO RUN
----------
Paste into one Colab cell and press play.  Nothing to configure, no files
needed.  Takes about 8 minutes.  Send me everything it prints -- the SUMMARY
block at the end is the part that matters.

WHY
---
Chechkin et al., J. Stat. Phys. 115, 1505 (2004) derive this tail ONLY for
c >= 2.  Dybiec, Sokolov & Chechkin, JSTAT P07008 (2010) treat c < 2 and
report the same exponent, valid iff c > 2-alpha (equivalently beta > 0).
An earlier probe of ours at (c=1.5, alpha=1) gave 0.872 instead of 0.5.
That probe had two bugs, both fixed here:

(1) It sampled straight after adding the noise, so the noise's own x^{-alpha}
    tail was baked into every measurement -- which is why beta came out stuck
    near alpha whatever c was.  Here the state is recorded after the drift
    step, which reabsorbs the last kick.
(2) It ran for physical time 8, not 4000: T=4000 steps at dt=2e-3 is t=8.
    In a subharmonic well the return time from x grows like x^{2-c}/(2-c),
    so at time t only the region below x_eq ~ [(2-c)t]^{1/(2-c)} has settled.

WHAT IT MEASURES
----------------
The local slope of P(X>x) over half-decade windows, printed against x.  A
power-law tail appears as a PLATEAU.  This is more honest than one fitted
number, which would average a crossover and hide whether a plateau exists.

Each configuration runs twice, from a NARROW start (tail fills in from
below) and a BROAD start (tail starts over-populated).  They approach the
stationary state from opposite sides, so their agreement is the proof of
equilibration.  x_eq is printed on every row: slopes far beyond it are not
measurements, that region is still filling and reads too steep.
==============================================================================
"""

import time
import numpy as np

SEED = 4242
M = 20_000                # walkers
SNAPS = 20                # late-time snapshots pooled
XS = np.array([1.0, 3.16, 10.0, 31.6, 100.0, 316.0, 1000.0])
MIN_COUNT = 50            # fewer survivors than this and a slope is noise

# The drift is exact, so dt is limited only by the Lie splitting.  Steep wells
# need a fine dt (one exact drift step from x caps at [(c-2)dt]^{-1/(c-2)},
# which for c=4, dt=0.02 is only x=5 -- it would clip the tail being measured)
# but they relax fast, so they get a short run.  Subharmonic wells relax
# slowly and need a long run, but tolerate a coarse dt.
DT_SUB, T_SUB = 0.02, 1000.0
DT_STEEP, T_STEEP = 0.002, 50.0


def stable(alpha, size, rng):
    """Symmetric alpha-stable, unit scale, phi(k)=exp(-|k|^alpha).  CMS method."""
    if alpha >= 2.0:
        return rng.standard_normal(size) * np.sqrt(2.0)
    U = rng.uniform(-np.pi / 2, np.pi / 2, size)
    W = rng.exponential(1.0, size)
    if abs(alpha - 1.0) < 1e-9:
        return np.tan(U)
    t1 = np.sin(alpha * U) / np.cos(U) ** (1.0 / alpha)
    t2 = (np.cos(U - alpha * U) / W) ** ((1.0 - alpha) / alpha)
    return t1 * t2


def drift_exact(x, c, dt):
    """
    Exact flow of dx/dt = -sign(x)|x|^{c-1} over one step:
        |x(t)|^{2-c} = |x0|^{2-c} + (c-2)t   (c != 2)
        x(t) = x0 exp(-t)                    (c = 2)
    Unconditionally stable for every c, unlike an Euler step near the origin.
    """
    a, s = np.abs(x), np.sign(x)
    if abs(c - 2.0) < 1e-12:
        return x * np.exp(-dt)
    p = 2.0 - c
    v = a ** p + (c - 2.0) * dt
    if c < 2.0:
        v = np.maximum(v, 0.0)          # flow reaches the origin in finite time
    return s * v ** (1.0 / p)


def x_eq(c, t):
    return np.inf if c >= 2.0 else ((2.0 - c) * t) ** (1.0 / (2.0 - c))


def run(c, alpha, t_phys, dt, seed, broad):
    rng = np.random.default_rng(seed)
    if broad:
        x = np.minimum(rng.pareto(0.3, M) + 1.0, 100.0) * rng.choice([-1.0, 1.0], M)
    else:
        x = rng.standard_normal(M) * 0.5
    T = int(round(t_phys / dt))
    amp = dt ** (1.0 / alpha)
    gap = max(T // (4 * SNAPS), 1)
    pool = []
    for step in range(T):
        # noise FIRST, then drift, and record after the drift: this is what
        # keeps the noise's own tail out of the measurement.
        x = x + amp * stable(alpha, M, rng)
        x = drift_exact(x, c, dt)
        x = np.clip(x, -1e14, 1e14)
        if step >= T - SNAPS * gap and (T - step) % gap == 0:
            pool.append(np.abs(x).copy())
    return np.sort(np.concatenate(pool))


def local_slope(a):
    """-dlogS/dlogx over half-decade windows; nan where survivors run out."""
    n, out = a.size, []
    for x in XS:
        x2 = x * 10 ** 0.5
        n1 = n - np.searchsorted(a, x, 'right')
        n2 = n - np.searchsorted(a, x2, 'right')
        out.append(np.nan if (n2 < MIN_COUNT or n1 <= 0)
                   else -np.log(n2 / n1) / np.log(x2 / x))
    return np.array(out)


def plateau(s, c, t):
    """Flattest run of 3 consecutive windows lying below the front x_eq."""
    lim = x_eq(c, t)
    ok = [i for i in range(len(XS)) if np.isfinite(s[i]) and XS[i] <= lim]
    best, span = np.nan, np.inf
    for j in range(len(ok) - 2):
        i0, i1, i2 = ok[j], ok[j + 1], ok[j + 2]
        if i2 - i0 != 2:
            continue
        w = s[[i0, i1, i2]]
        if w.max() - w.min() < span:
            best, span = w.mean(), w.max() - w.min()
    if np.isnan(best) and ok:
        best, span = s[ok].mean(), np.ptp(s[ok])
    return best, span


HDR = ("    c alpha  b_pred    init  " + " ".join(f"{x:>7.4g}" for x in XS)
       + "     x_eq")
SUMMARY = []


def row(c, alpha, t_phys, dt, label):
    pred = c + alpha - 2.0
    print(HDR)
    s = {}
    for broad in (False, True):
        a = run(c, alpha, t_phys, dt,
                seed=SEED + int(100 * c) + int(10 * alpha) + int(broad) + len(label),
                broad=broad)
        s[broad] = local_slope(a)
        print(f"{c:5.2f} {alpha:5.2f} {pred:7.3f} "
              f"{'broad' if broad else 'narrow':>7}  "
              + " ".join(f"{v:7.3f}" if np.isfinite(v) else "      -" for v in s[broad])
              + f"   {x_eq(c, t_phys):8.3g}")
    # judge equilibration only where the measurement is meaningful: inside
    # the front, and where both starts gave a finite slope
    lim = x_eq(c, t_phys)
    valid = np.isfinite(s[False]) & np.isfinite(s[True]) & (XS <= lim)
    gap = np.nanmax(np.abs(s[False] - s[True])[valid]) if valid.any() else np.nan
    eq = gap < 0.08
    pl = np.nanmean([plateau(s[False], c, t_phys)[0], plateau(s[True], c, t_phys)[0]])
    tol = max(0.08, 0.15 * abs(pred))
    verdict = ("not equilibrated" if not eq else
               "MATCHES" if abs(pl - pred) <= tol else "DIFFERS")
    print(f"{'':>21}narrow-vs-broad gap {gap:.3f} ({'equilibrated' if eq else 'TOO SHORT'})"
          f" | plateau {pl:.3f} vs predicted {pred:.3f} -> {verdict}\n")
    SUMMARY.append((label, c, alpha, t_phys, pred, pl, gap, verdict))


t0 = time.time()
print("=" * 94)
print("  Levy flights in U(x)=|x|^c/c :  P(X>x) ~ x^{-beta},  beta = c+alpha-2")
print(f"  walkers={M}  snapshots pooled={SNAPS}")
print("  columns = local slope of P(X>x) over half-decade windows starting at x")
print("  a power law shows as a PLATEAU;  ignore any column with x >> x_eq")
print("=" * 94)

print("\n(C) CONTROLS -- exact answers, so these validate the integrator.")
print("    c=4,alpha=1: f_st = 1/(pi(1-x^2+x^4)) exactly (Chechkin 2004 Eq.35),")
print("                 beta=3.  Steep tail, so the x~1-3 columns are the ones.")
print("    c=2:         OU driven by stable noise, stationary law is alpha-stable,")
print("                 so beta=alpha EXACTLY, for any alpha.\n")
for c, al in [(4.0, 1.0), (2.0, 1.0), (2.0, 1.5)]:
    row(c, al, T_STEEP, DT_STEEP, "control")

print("\n(A) c-SCAN at alpha=1.0.  Existence boundary at c = 2-alpha = 1.00.")
print("    THE KEY TEST: the plateau must TRACK c, falling roughly linearly")
print("    from 0.8 to 0.2 as c goes 1.8 -> 1.2.  If it instead sits pinned")
print("    near alpha=1 regardless of c, the formula fails below 2.\n")
for c in [1.2, 1.4, 1.6, 1.8]:
    row(c, 1.0, T_SUB, DT_SUB, "c-scan a=1")

print("\n(A') same at alpha=1.5, where the boundary moves to c = 0.50.  c=0.8")
print("     sits below the alpha=1 boundary, so a stationary state here shows")
print("     the condition really is beta>0 and not c>1.\n")
for c in [0.8, 1.2, 1.6]:
    row(c, 1.5, T_SUB, DT_SUB, "c-scan a=1.5")

print("\n(B) TIME SCAN at c=1.5, alpha=1.0 (beta_pred = 0.500).")
print("    The plateau must STAY at 0.500 while its right edge moves out with")
print("    x_eq.  A plateau that drifts with run length is not a real tail.\n")
for t in (300.0, 1000.0, 3000.0):
    row(1.5, 1.0, t, DT_SUB, f"t={t:.0f}")

print("=" * 94)
print("SUMMARY")
print(f"{'block':>14} {'c':>5} {'alpha':>5} {'t':>6} {'pred':>7} "
      f"{'measured':>9} {'eq.gap':>7}  verdict")
for lab, c, al, t, pred, pl, gap, v in SUMMARY:
    print(f"{lab:>14} {c:5.2f} {al:5.2f} {t:6.0f} {pred:7.3f} "
          f"{pl:9.3f} {gap:7.3f}  {v}")

bad = [s for s in SUMMARY if s[0] == 'control' and s[7] == 'DIFFERS']
scan = [s for s in SUMMARY if s[0].startswith('c-scan')]
ok = sum(1 for s in scan if s[7] == 'MATCHES')
print("\n" + "-" * 94)
if bad:
    print("CONTROLS FAILED -- integrator problem, ignore everything else.")
else:
    print(f"Controls pass.  Subharmonic rows matching: {ok}/{len(scan)}.")
    print("Rows marked 'not equilibrated' are not evidence either way; they need")
    print("a longer run (raise T_SUB).  Rows near the boundary (small beta) are")
    print("intrinsically hard: x_eq only grows as t^{1/(2-c)}, so the honest")
    print("window is narrow there, and that is a limit of the numerics, not a")
    print("result about the formula.")
print(f"elapsed {time.time() - t0:.1f} s")
