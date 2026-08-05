"""
============================================================================
GLOBAL SCAN OVER THE AXES THAT WERE NEVER VARIED
============================================================================
mu_k := k(1 - Y_{k+1}/Y_k).  Constant in k <=> Poisson-Dirichlet.
Same display coordinate as before, same matched-Y_2 protocol, same common
random numbers.  What changes is WHAT IS SWEPT.

Every measurement so far varied only: the noise index alpha, the value
distribution, Y_2, N, k.  Three things were held fixed the whole time and are
opened here.  A fourth block settles a specific question left by the last run.

BLOCK S -- SKEWNESS OF THE NOISE
  The stable family has a skewness parameter kappa in [-1, 1] that has been
  set to zero in every run so far.  It is a standard parameter of the family
  and simply was never switched on.  kappa is swept at several alpha.  Note
  the skewed CMS formula collapses to the symmetric one at alpha = 2, where
  skewness has no effect, so the alpha dependence of the kappa effect is
  itself part of the measurement.

BLOCK C -- COUPLING BETWEEN A COMPETITOR'S VALUE AND ITS OWN NOISE
  Every run so far assumed the increment Z_i is independent of the value s_i.
  Here the scale is tied to the value, sigma_i = sigma (s_i / median s)^rho.
  rho = 0 is the old case.  rho = 1 is exactly multiplicative noise, i.e. the
  Kesten / Bouchaud-Mezard class, which the theory currently excludes by
  assumption.  Negative rho means leaders are the quieter ones.

BLOCK H -- HETEROGENEOUS NOISE AMPLITUDE
  sigma_i = sigma * exp(h * xi_i), xi_i standard normal and independent of
  the values.  h = 0 is the old case.  Competitors differ in how noisy they
  are, without that being tied to how good they are.

BLOCK N -- DOES alpha* MOVE WITH N?
  The previous run found DRIFT crossing zero at alpha* = 1.97 +- 0.006, the
  same for Frechet and Gumbel values and the same at two values of Y_2, with
  the profile flat below the noise floor there.  Whether that is a constant or
  a finite-size crossover is decided by one measurement: alpha* at N = 50,
  200, 800.  Both readings are cheap to distinguish and only this block can
  do it.

All blocks report the same thing: the mu_k profile at matched Y_2, its DRIFT
and FLAT, and a bootstrap CI.  Everything is saved to global_scan.npz.

RUNTIME ~70-90 min.  numpy only.
============================================================================
"""

import time
import numpy as np

KMAX  = 8
CONF  = 1200
REPL  = 350
NSIG  = 4
Y2T   = 0.65
BOOT  = 250
SEED  = 20260804
VSEED = 555_000_000


def stable_skew(U, W, al, kappa):
    """Chambers-Mallows-Stuck with skewness kappa in [-1,1].
    kappa = 0 reduces to the symmetric formula used in all previous runs.
    At alpha = 2, tan(pi*alpha/2) = 0 so B = 0 and S = 1: skewness drops out,
    as it must for a Gaussian."""
    if abs(al - 1.0) < 1e-12:
        # alpha = 1 needs the separate CMS branch when skewed
        if abs(kappa) < 1e-12:
            return np.tan(U)
        t = 2.0/np.pi
        return t*((np.pi/2 + kappa*U)*np.tan(U)
                  - kappa*np.log((np.pi/2)*W*np.cos(U)/(np.pi/2 + kappa*U)))
    tp = np.tan(np.pi*al/2.0)
    B = np.arctan(kappa*tp)/al
    S = (1.0 + (kappa*tp)**2)**(1.0/(2.0*al))
    return S*(np.sin(al*(U + B))/np.cos(U)**(1.0/al)
              * (np.cos(U - al*(U + B))/W)**((1.0 - al)/al))


def falling(c, k):
    out = np.ones_like(c)
    for m in range(k):
        out = out * (c - m)
    return out


def run_cell(al, sigma, n, kappa=0.0, rho=0.0, het=0.0,
             conf=CONF, repl=REPL, noise_seed=0, kmax=KMAX):
    Y = np.empty((conf, kmax - 1))
    dens = [float(np.prod([float(repl - m) for m in range(k)]))
            for k in range(2, kmax + 1)]
    nrng = np.random.default_rng(noise_seed)
    for c in range(conf):
        vrng = np.random.default_rng(VSEED + c)
        s = vrng.pareto(1.5, n) + 1.0                 # values fixed across cells
        sc = np.ones(n)
        if rho != 0.0:
            sc = sc * (s/np.median(s))**rho
        if het != 0.0:
            sc = sc * np.exp(het*vrng.standard_normal(n))
        U = nrng.uniform(-np.pi/2, np.pi/2, (repl, n))
        W = nrng.exponential(1.0, (repl, n))
        z = sigma * sc[None, :] * stable_skew(U, W, al, kappa)
        w = np.argmax(s[None, :] + z, axis=1)
        cnt = np.bincount(w, minlength=n).astype(np.float64)
        for j, k in enumerate(range(2, kmax + 1)):
            Y[c, j] = falling(cnt, k).sum() / dens[j]
    return Y


def profile(Y):
    m = Y.mean(axis=0)
    return m[0], np.arange(2, KMAX)*(1.0 - m[1:]/m[:-1])


def coarse(al, target, n, kappa, rho, het, ns, lo=-6.0, hi=8.0, iters=11):
    for _ in range(iters):
        mid = 0.5*(lo + hi)
        Y = run_cell(al, 10**mid, n, kappa, rho, het, 40, 250, ns, kmax=2)
        if Y[:, 0].mean() > target: lo = mid
        else:                       hi = mid
    return 0.5*(lo + hi)


def measure(tag, al, n, kappa, rho, het, rng, store):
    ns = int(rng.integers(1, 2**62))
    a = coarse(al, 0.80, n, kappa, rho, het, ns)
    b = coarse(al, 0.50, n, kappa, rho, het, ns)
    lo, hi = min(a, b), max(a, b)
    grid = np.logspace(lo - 0.10, hi + 0.10, NSIG)
    y2s, raws = [], []
    for s in grid:
        Y = run_cell(al, s, n, kappa, rho, het, CONF, REPL, ns)
        y2s.append(Y[:, 0].mean()); raws.append(Y)
        store[f"Y|{tag}|{s:.6g}"] = Y
    y2s = np.array(y2s); o = np.argsort(y2s)
    if Y2T < y2s[o][0] or Y2T > y2s[o][-1]:
        return None
    j = int(np.searchsorted(y2s[o], Y2T)); j = min(max(j, 1), len(o) - 1)
    A, B = raws[o[j-1]], raws[o[j]]
    w = float(np.clip((Y2T - y2s[o[j-1]])/(y2s[o[j]] - y2s[o[j-1]] + 1e-15), 0, 1))
    _, mu = profile((1-w)*A + w*B)
    db = np.empty(BOOT)
    for bi in range(BOOT):
        idx = rng.integers(0, A.shape[0], A.shape[0])
        _, m2 = profile((1-w)*A[idx] + w*B[idx])
        db[bi] = m2[-1] - m2[0]
    return mu, mu[-1] - mu[0], np.nanmax(mu) - np.nanmin(mu), np.percentile(db, [2.5, 97.5])


def row(label, res):
    if res is None:
        print(f"{label:>26}   (outside scanned range)"); return
    mu, d, f, ci = res
    print(f"{label:>26}   " + "".join(f"{x:<8.3f}" for x in mu)
          + f"{d:+9.3f}{f:8.3f}   [{ci[0]:+.3f},{ci[1]:+.3f}]", flush=True)


HDR = (f"{'cell':>26}   " + "".join(f"mu_{k:<6d}" for k in range(2, KMAX))
       + f"{'DRIFT':>9}{'FLAT':>8}{'95% CI':>20}")

rng = np.random.default_rng(SEED)
store = {}
t0 = time.time()
print("=" * 112)
print(f"  GLOBAL SCAN.  values = Pareto(1.5) throughout, matched at Y_2 = {Y2T}")
print(f"  conf={CONF} repl={REPL} sigma points={NSIG}")
print("=" * 112)

print("\nBLOCK S -- skewness of the noise (never varied before)")
print(HDR)
for al in (1.0, 1.5, 1.9, 1.97):
    for kap in (0.0, 0.5, 1.0):
        r = measure(f"S|{al}|{kap}", al, 200, kap, 0.0, 0.0, rng, store)
        row(f"alpha={al} kappa={kap}", r)

print("\nBLOCK C -- competitor's own value coupled to its own noise scale")
print("           sigma_i = sigma (s_i/median s)^rho ;  rho=1 is multiplicative noise")
print(HDR)
for al in (1.0, 1.5, 1.97):
    for rho in (-0.6, -0.3, 0.0, 0.3, 0.6, 1.0):
        r = measure(f"C|{al}|{rho}", al, 200, 0.0, rho, 0.0, rng, store)
        row(f"alpha={al} rho={rho}", r)

print("\nBLOCK H -- heterogeneous noise amplitude, independent of the values")
print("           sigma_i = sigma exp(h xi_i)")
print(HDR)
for al in (1.0, 1.5, 1.97):
    for het in (0.0, 0.5, 1.0, 1.5):
        r = measure(f"H|{al}|{het}", al, 200, 0.0, 0.0, het, rng, store)
        row(f"alpha={al} h={het}", r)

print("\nBLOCK N -- does the zero crossing of DRIFT move with N?")
print(HDR)
for n in (50, 200, 800):
    for al in (1.93, 1.95, 1.96, 1.97, 1.98, 1.99, 2.00):
        r = measure(f"N|{n}|{al}", al, n, 0.0, 0.0, 0.0, rng, store)
        row(f"N={n} alpha={al}", r)
    print()

np.savez_compressed("global_scan.npz", **store)
print(f"saved global_scan.npz ({len(store)} arrays)")
print(f"elapsed {time.time()-t0:.0f} s")

print("""
COLUMNS
  mu_k    k(1 - Y_{k+1}/Y_k).  Constant in k <=> Poisson-Dirichlet.
  DRIFT   mu_7 - mu_2.        FLAT  max_k mu_k - min_k mu_k.
  95% CI  bootstrap over configurations, on DRIFT.
  Per-configuration Y_k for every cell is in global_scan.npz.

Send the entire output.
""")
