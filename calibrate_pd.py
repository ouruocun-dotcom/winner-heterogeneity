"""
============================================================================
CALIBRATING THE mu_k PIPELINE ON CASES THAT ARE POISSON-DIRICHLET BY
CONSTRUCTION
============================================================================
mu_k := k(1 - Y_{k+1}/Y_k).

Poisson-Dirichlet PD(c) satisfies BOTH
      mu_k = c  for every k        AND        Y_2 = 1 - c .
Only the first was ever checked.  Define

      w  :=  Y_2 / (1 - mean_k mu_k)

so that w = 1 exactly for PD, and w != 1 means the ratios are PD-like while
the overall level is not.  In the global scan the flat cells gave w = 0.82,
i.e. flat but NOT Poisson-Dirichlet.  Before anything is built on that, the
pipeline has to be shown to return w = 1 on cases that really are PD.  It
never has been: the original control was checked through a different
statistic, and every Exponential row in the mu_k tables used stable noise.

THREE CASES, SHARING ONE CODE PATH
  A.  PD BY CONSTRUCTION.  Weights drawn from the stick-breaking
      representation of PD(c), V_i ~ Beta(1-c, i c), winners sampled from
      them.  No competition at all.  This tests the ESTIMATOR alone: the
      falling-factorial U-statistics, the finite number of replicas, the
      configuration average.  The unresolved stick mass is assigned to
      distinct fresh atoms, which is what the infinitely many infinitesimal
      atoms of a true PD do, and contributes exactly zero to Y_k for k >= 2.

  B.  GUMBEL NOISE, EXPONENTIAL VALUES.  Here the competition is exactly
      the logit / random-energy structure, P_i = e^{s_i/sigma} / sum_j
      e^{s_j/sigma}, and with s ~ Exp(1) the weights are Pareto of index
      sigma, so the law is PD(sigma) with Y_2 = 1 - sigma.  This tests the
      MODEL path -- values, noise, argmax, the sigma scan and the
      interpolation -- on a case whose answer is known.

  C.  STABLE NOISE, PARETO VALUES, alpha = 1.97.  The cell from the global
      scan, run through the identical code, for side-by-side comparison.

READING IT
  A and B returning w = 1 within error validates everything and makes the
  w = 0.82 of case C a property of stable noise.
  A returning w = 1 but B not means the Gumbel competition is not PD at
  finite N, which would be about the model rather than the estimator.
  A not returning w = 1 means the estimator is biased and every mu_k table
  produced so far has to be re-read.

RUNTIME ~5-8 min.  numpy + scipy.
============================================================================
"""

import time
import numpy as np

KMAX = 8
N    = 200
CONF = 1500
REPL = 400
NSIG = 4
BOOT = 300
SEED = 20260805
VSEED = 424_000_000
NSTICK = 3000


def falling(c, k):
    out = np.ones_like(c)
    for m in range(k):
        out = out * (c - m)
    return out


def Yk_from_counts(cnt, repl, kmax=KMAX):
    out = np.empty(kmax - 1)
    for j, k in enumerate(range(2, kmax + 1)):
        den = float(np.prod([float(repl - m) for m in range(k)]))
        out[j] = falling(cnt, k).sum() / den
    return out


def profile(Y):
    m = Y.mean(axis=0)
    return m[0], np.arange(2, KMAX) * (1.0 - m[1:] / m[:-1])


def report(tag, Y, rng, extra=""):
    y2, mu = profile(Y)
    c = mu.mean()
    w = y2 / (1.0 - c)
    wb = np.empty(BOOT)
    for b in range(BOOT):
        idx = rng.integers(0, Y.shape[0], Y.shape[0])
        y2b, mub = profile(Y[idx])
        wb[b] = y2b / (1.0 - mub.mean())
    ci = np.percentile(wb, [2.5, 97.5])
    print(f"{tag:>34}  Y2={y2:.4f}  1-Y2={1-y2:.4f}  "
          + "mu_k=" + " ".join(f"{m:.3f}" for m in mu)
          + f"  flat={mu.max()-mu.min():.4f}"
          + f"  w={w:.4f} [{ci[0]:.4f},{ci[1]:.4f}]" + extra, flush=True)
    return w, ci


# ----------------------------------------------------------------- CASE A
def pd_sticks(c, rng, nst=NSTICK):
    V = rng.beta(1.0 - c, np.arange(1, nst + 1) * c)
    logrest = np.concatenate([[0.0], np.cumsum(np.log1p(-V))[:-1]])
    P = V * np.exp(logrest)
    return P, max(0.0, 1.0 - P.sum())


def run_A(c, rng, conf=CONF, repl=REPL):
    Y = np.empty((conf, KMAX - 1))
    for i in range(conf):
        P, rem = pd_sticks(c, rng)
        probs = np.concatenate([P, [rem]])
        probs = probs / probs.sum()
        cnt = rng.multinomial(repl, probs)[:-1].astype(np.float64)
        # the draws that fell in 'rem' each land on a distinct fresh atom and
        # therefore contribute nothing to Y_k for k >= 2; they are simply
        # excluded from the count vector, which is exactly right.
        Y[i] = Yk_from_counts(cnt, repl)
    return Y


# --------------------------------------------------------- CASES B and C
def stable_sym(U, W, al):
    if abs(al - 1.0) < 1e-12:
        return np.tan(U)
    return (np.sin(al*U)/np.cos(U)**(1.0/al)
            * (np.cos(U - al*U)/W)**((1.0 - al)/al))


def run_comp(kind, al, sigma, rng, conf=CONF, repl=REPL, n=N):
    Y = np.empty((conf, KMAX - 1))
    for i in range(conf):
        vrng = np.random.default_rng(VSEED + i)
        if kind == "gumbel":
            s = vrng.exponential(1.0, n)
            z = sigma * rng.gumbel(0.0, 1.0, (repl, n))
        else:
            s = vrng.pareto(1.5, n) + 1.0
            U = rng.uniform(-np.pi/2, np.pi/2, (repl, n))
            W = rng.exponential(1.0, (repl, n))
            z = sigma * stable_sym(U, W, al)
        w = np.argmax(s[None, :] + z, axis=1)
        Y[i] = Yk_from_counts(np.bincount(w, minlength=n).astype(np.float64), repl)
    return Y


def coarse(kind, al, target, rng, n=N, lo=-6.0, hi=8.0, iters=11):
    for _ in range(iters):
        mid = 0.5*(lo + hi)
        Y = run_comp(kind, al, 10**mid, rng, 40, 250, n)
        if Y[:, 0].mean() > target: lo = mid
        else:                       hi = mid
    return 0.5*(lo + hi)


rng = np.random.default_rng(SEED)
t0 = time.time()
print("=" * 118)
print("  w := Y_2 / (1 - mean_k mu_k).   PD  <=>  mu_k flat AND w = 1.")
print(f"  N={N} conf={CONF} repl={REPL}")
print("=" * 118)

print("\nCASE A -- PD by construction (stick-breaking).  Tests the estimator only.")
for c in (0.15, 0.25, 0.35, 0.50):
    Y = run_A(c, rng)
    report(f"PD(c={c}) direct", Y, rng, extra=f"   [true c = {c}]")

print("\nCASE B -- Gumbel noise, Exponential values.  Exactly PD(sigma), Y_2 = 1-sigma.")
for sig in (0.15, 0.25, 0.35, 0.50):
    Y = run_comp("gumbel", 0.0, sig, rng)
    report(f"Gumbel sigma={sig}", Y, rng, extra=f"   [PD predicts c = {sig}]")

print("\nCASE C -- stable noise, Pareto(1.5) values, alpha=1.97.  The global-scan cell.")
lo = coarse("stable", 1.97, 0.85, rng); hi = coarse("stable", 1.97, 0.50, rng)
for s in np.logspace(min(lo, hi) - 0.05, max(lo, hi) + 0.05, NSIG):
    Y = run_comp("stable", 1.97, s, rng)
    report(f"stable a=1.97 sigma={s:.3g}", Y, rng)

print(f"\nelapsed {time.time()-t0:.0f} s")
print("""
COLUMNS
  mu_k   k(1 - Y_{k+1}/Y_k)
  flat   max_k mu_k - min_k mu_k
  w      Y_2 / (1 - mean_k mu_k), with a 95% bootstrap CI over configurations.
         w = 1 is what Poisson-Dirichlet requires.  Flatness alone is not.

Send the entire output.
""")
