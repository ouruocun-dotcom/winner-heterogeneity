"""
============================================================================
THE DISTRIBUTION OF P_W  --  THE WINNER'S OWN WINNING PROBABILITY
============================================================================
Two identities collapse the whole problem.

(1)  Y_k = E[ sum_i P_i^k ] = E[ P_W^{k-1} ],  where W is the winner of one
     realization.  Sampling proportional to P IS running the contest once.
     So the entire Y_k sequence measured over six rounds is nothing but the
     MOMENT SEQUENCE of one scalar random variable, P_W.  k was never a
     separate dimension.

(2)  The size-biased pick from PD(c) is Beta(1-c, c), and
     E[Beta(1-c,c)^{k-1}] = Gamma(k-c)/(Gamma(k)Gamma(1-c)) = Y_k^PD.
     So   PD(c)  <=>  P_W ~ Beta(1-c, c).

Together with the measured structure Y_k = w * Y_k^{PD(c)} for all k >= 2,
i.e. E[P_W^m] = w E[B^m] for every m >= 1 while the m = 0 moments differ,
the only solution is

     P_W = 0            with probability 1 - w
     P_W ~ Beta(1-c,c)  with probability w.

So 1 - w is not a fitted constant: it is the probability that the winner is
an ACCIDENTAL winner, one whose own chance of winning was negligible.

MEASURING IT DIRECTLY INSTEAD OF THROUGH MOMENTS
P_W is already computed inside every run ever done -- it is just
count[winner]/replicas -- and was thrown away every time.  Moments compress
the distribution into six noisy numbers; the histogram shows it whole.  The
prediction to look at is a Beta(1-c,c) body plus a spike at P_W ~ 1/N
carrying mass 1 - w, which the H^2 law says should be H^2.

Estimator: for each replica r with winner i, the leave-one-out value
(count_i - 1)/(replicas - 1) is used, so the replica that selected the winner
does not inflate its own probability.

CONTROLS FIRST
  PD by construction, and the Gumbel/exponential contest, are both exactly
  PD.  Their P_W histograms must be Beta with NO spike at zero.  If a spike
  appears there it is an artifact and the stable-noise spike means nothing.

RUNTIME ~10-15 min.  numpy + scipy.
============================================================================
"""

import time
import numpy as np
from scipy.stats import beta as Beta

N      = 200
REPL   = 1000          # sets the resolution in P_W: 1/REPL
CONF   = 2500
NSTICK = 50000
SEED   = 20260806
VSEED  = 313_000_000
KMAX   = 8

# The spike sits at P_W ~ 1/N, so thresholds are set in units of 1/N, not of
# the replica resolution.  Rather than choosing a boundary for it, the readout
# is the CUMULATIVE EXCESS over Beta: measured CDF(t) - Beta CDF(t).  That
# rises across the spike and then flattens once past it, and the height of the
# plateau is 1 - w no matter where the spike happens to end.
THRESH = np.array([0.5, 1, 2, 3, 5, 8, 12, 20, 40])/N
_raw   = np.concatenate([[0.0], THRESH, [0.35, 0.50, 0.70, 0.85, 1.01]])
EDGES  = np.unique(np.round(_raw, 10))          # strictly increasing
THRESH = THRESH[THRESH < 0.30]                  # excess is read below the body


def falling(c, k):
    out = np.ones_like(c)
    for m in range(k):
        out = out * (c - m)
    return out


def stats_from_counts(cnt, repl):
    """exact U-statistic Y_k, and the leave-one-out sample of P_W"""
    Y = np.empty(KMAX - 1)
    for j, k in enumerate(range(2, KMAX + 1)):
        den = float(np.prod([float(repl - m) for m in range(k)]))
        Y[j] = falling(cnt, k).sum() / den
    hit = cnt > 0
    pw = (cnt[hit] - 1.0) / (repl - 1.0)      # value of P_W for those winners
    wt = cnt[hit]                              # how many replicas saw it
    return Y, pw, wt


def stable_sym(U, W, al):
    return (np.sin(al*U)/np.cos(U)**(1.0/al)
            * (np.cos(U - al*U)/W)**((1.0 - al)/al))


def run(kind, param, sigma, rng, conf=CONF, repl=REPL, n=N):
    Ys = np.empty((conf, KMAX - 1))
    PW, WT = [], []
    for i in range(conf):
        if kind == "pd":
            V = rng.beta(1.0 - param, np.arange(1, NSTICK + 1)*param)
            lr = np.concatenate([[0.0], np.cumsum(np.log1p(-V))[:-1]])
            P = V*np.exp(lr)
            rem = max(0.0, 1.0 - P.sum())
            probs = np.concatenate([P, [rem]]); probs /= probs.sum()
            cnt = rng.multinomial(repl, probs)[:-1].astype(np.float64)
        else:
            vrng = np.random.default_rng(VSEED + i)
            if kind == "gumbel":
                s = vrng.exponential(1.0, n)
                z = sigma*rng.gumbel(0.0, 1.0, (repl, n))
            else:
                s = vrng.pareto(1.5, n) + 1.0
                U = rng.uniform(-np.pi/2, np.pi/2, (repl, n))
                Wx = rng.exponential(1.0, (repl, n))
                z = sigma*stable_sym(U, Wx, param)
            wn = np.argmax(s[None, :] + z, axis=1)
            cnt = np.bincount(wn, minlength=n).astype(np.float64)
        Y, pw, wt = stats_from_counts(cnt, repl)
        Ys[i] = Y; PW.append(pw); WT.append(wt)
    return Ys, np.concatenate(PW), np.concatenate(WT)


def describe(tag, Ys, pw, wt):
    m = Ys.mean(axis=0)
    mu = np.arange(2, KMAX)*(1.0 - m[1:]/m[:-1])
    c = mu.mean(); w = m[0]/(1.0 - c); H = 1.0 - m[0]
    hist, _ = np.histogram(pw, bins=EDGES, weights=wt)
    hist = hist/wt.sum()
    bb = Beta(1.0 - c, c)
    print(f"\n{tag}")
    print(f"   H={H:.4f}   c={c:.4f}   w={w:.4f}   flat={mu.max()-mu.min():.4f}"
          f"   1-w={1-w:.4f}   H^2={H*H:.4f}")
    print(f"   {'P_W bin':>18}{'measured':>10}{'Beta':>9}      "
          f"{'t':>8}{'CDF meas':>10}{'CDF Beta':>10}{'EXCESS':>9}")
    cm = 0.0
    for j in range(len(EDGES) - 1):
        lo, hi = EDGES[j], min(EDGES[j+1], 1.0)
        cm += hist[j]
        line = (f"   [{lo:7.5f},{hi:7.5f}){hist[j]:10.4f}"
                f"{bb.cdf(hi)-bb.cdf(lo):9.4f}")
        if hi <= 0.45:
            line += (f"      {hi:8.4f}{cm:10.4f}{bb.cdf(hi):10.4f}"
                     f"{cm-bb.cdf(hi):+9.4f}")
        print(line)
    exc = []
    for t_ in THRESH:
        j = int(np.searchsorted(EDGES, t_ - 1e-12))
        exc.append(hist[:j].sum() - bb.cdf(t_))
    exc = np.array(exc)
    plateau = float(np.max(exc))
    print(f"   excess plateau (max over thresholds) = {plateau:.4f}"
          f"    1-w = {1-w:.4f}    H^2 = {H*H:.4f}")
    return H, 1 - w, plateau


rng = np.random.default_rng(SEED)
t0 = time.time()
print("=" * 78)
print("  P_W = the winning probability of whoever actually won.")
print("  PD(c) <=> P_W ~ Beta(1-c, c) with NO mass at zero.")
print(f"  N={N} conf={CONF} repl={REPL}  (resolution in P_W = {1/REPL:.4f})")
print("=" * 78)

print("\n" + "-"*78 + "\nCONTROLS -- these are Poisson-Dirichlet by construction")
for c in (0.15, 0.35):
    describe(f"CONTROL  PD(c={c}) by stick-breaking", *run("pd", c, 0.0, rng))
for sig in (0.15, 0.35):
    describe(f"CONTROL  Gumbel noise, sigma={sig}", *run("gumbel", 0.0, sig, rng))

print("\n" + "-"*78 + "\nSTABLE NOISE, Pareto(1.5) values, alpha=1.97")
res = []
for sig in (2.4, 3.9, 6.2, 9.9):
    res.append(describe(f"stable alpha=1.97 sigma={sig}", *run("stable", 1.97, sig, rng)))

print("\n" + "="*78)
print(f"{'H':>10}{'1-w (moments)':>16}{'excess (histogram)':>20}{'H^2':>10}")
for H, omw, sp in res:
    print(f"{H:10.4f}{omw:16.4f}{sp:20.4f}{H*H:10.4f}")
print(f"\nelapsed {time.time()-t0:.0f} s")
print("""
The two middle columns are independent measurements of the same thing: one
from the moment sequence, one by counting how often the winner had almost no
chance.  If they agree, the decomposition is right.  Whether either equals
H^2 is the separate question.

Send the entire output.
""")
