"""
============================================================================
TESTING THE DERIVED FORMULA FOR THE ACCIDENTAL-WINNER MASS
============================================================================
P_W is the winning probability of whoever actually won.  Measurement showed
P_W has a spike at P_W ~ 1/N -- the winner had almost no chance -- carrying
mass m, and that m tracked H^2 with no fitted constant, where H is the
probability that two independent realizations end with different winners.

The derivation says the exponent is not 2 but alpha, and that it only looked
like 2 because everything was measured at alpha = 1.97.  H and m are carried
by different configurations:

  H comes from NEAR-DEGENERATE configurations.  With probability ~ rho sigma
  the runner-up sits within sigma of the leader and has an O(1) chance.
  These produce the Beta body, not the spike.      H  ~  2 a rho sigma

  m comes from SINGLE LONG JUMPS by distant competitors, each with a small
  probability q_i = 2 C_alpha sigma^alpha Delta_i^{-alpha}.  Every such
  winner is accidental.
                                m  =  2 C_alpha sigma^alpha T_alpha
  with T_alpha = sum_{i>=2} Delta_i^{-alpha}, which is exactly the sum whose
  scaling exponent theta = 1 - alpha/beta the whole classification is built
  on.  Eliminating sigma gives  m = K N H^alpha.

THREE TESTS, IN ORDER OF SHARPNESS

  1. THE FORMULA ITSELF, WITH NOTHING FITTED.  T_alpha is computed directly
     from the order statistics of the same values used in the simulation, so
     m_pred = 2 C_alpha sigma^alpha <T_alpha> is a number, not a fit.  It is
     printed next to the measured m.

  2. THE EXPONENT IS alpha, NOT 2.  Measured at alpha = 0.5, 1.0, 1.5, 1.97.
     At 1.97 the two predictions are indistinguishable; at 0.5 they differ by
     a factor of four.  If the fitted exponent comes out near 2 at alpha=0.5,
     the derivation is wrong.

  3. m / H^alpha IS PROPORTIONAL TO N.  Measured at N = 50, 200, 800 at fixed
     alpha.  There is already one piece of evidence against this: the global
     scan gave A = 1.119 at (N=200, alpha=1.97) and A = 0.947 at (N=800,
     alpha=1.99), where the formula wants a factor 0.75 and the data show
     1.18.  Those two cells differ in alpha as well as N, so they settle
     nothing, but the discrepancy is on the record.

m is read off as the excess of the P_W distribution over Beta(1-c,c) --
the boundary-free plateau readout, already validated on two exactly-PD
controls where it returned 0.001 and 0.006.  The raw mass below 5/N is
printed too, so the reading does not depend on the Beta subtraction.

RUNTIME ~25-35 min.  numpy + scipy.
============================================================================
"""

import time
import numpy as np
from scipy.stats import beta as Beta
from scipy.special import gamma as Gam

REPL  = 800
CONF  = 2000
KMAX  = 8
SEED  = 20260807
VSEED = 191_000_000


def C_alpha(al):
    """P(Z > x) ~ C_alpha x^{-alpha} for the symmetric stable with
    characteristic function exp(-|k|^alpha)"""
    return Gam(al)*np.sin(np.pi*al/2.0)/np.pi


def stable_sym(U, W, al):
    if abs(al - 1.0) < 1e-12:
        return np.tan(U)
    return (np.sin(al*U)/np.cos(U)**(1.0/al)
            * (np.cos(U - al*U)/W)**((1.0 - al)/al))


def falling(c, k):
    out = np.ones_like(c)
    for m in range(k):
        out = out * (c - m)
    return out


def run(al, sigma, n, rng, conf=CONF, repl=REPL):
    """returns per-config Y_k, the pooled P_W sample with weights, and <T_alpha>"""
    Ys = np.empty((conf, KMAX - 1))
    PW, WT, Ts = [], [], np.empty(conf)
    for i in range(conf):
        vrng = np.random.default_rng(VSEED + i)
        s = vrng.pareto(1.5, n) + 1.0
        ss = np.sort(s)[::-1]
        d = ss[0] - ss[1:]
        Ts[i] = np.sum(d[d > 0]**(-al))                 # T_alpha, no simulation
        U = rng.uniform(-np.pi/2, np.pi/2, (repl, n))
        W = rng.exponential(1.0, (repl, n))
        wn = np.argmax(s[None, :] + sigma*stable_sym(U, W, al), axis=1)
        cnt = np.bincount(wn, minlength=n).astype(np.float64)
        for j, k in enumerate(range(2, KMAX + 1)):
            den = float(np.prod([float(repl - mm) for mm in range(k)]))
            Ys[i, j] = falling(cnt, k).sum()/den
        hit = cnt > 0
        PW.append((cnt[hit] - 1.0)/(repl - 1.0)); WT.append(cnt[hit])
    return Ys, np.concatenate(PW), np.concatenate(WT), np.median(Ts)


def analyse(Ys, pw, wt, n):
    mm = Ys.mean(axis=0)
    mu = np.arange(2, KMAX)*(1.0 - mm[1:]/mm[:-1])
    c = float(np.clip(mu.mean(), 1e-4, 0.999))
    H = 1.0 - mm[0]
    bb = Beta(1.0 - c, c)
    tot = wt.sum()
    thr = np.array([1, 2, 3, 5, 8, 12, 20, 40])/n
    exc = [np.sum(wt[pw < t])/tot - bb.cdf(t) for t in thr]
    raw5 = np.sum(wt[pw < 5.0/n])/tot
    return H, c, float(np.max(exc)), raw5


def block(title, cells, rng):
    print(f"\n{'-'*104}\n{title}")
    print(f"{'alpha':>6}{'N':>6}{'sigma':>9}{'H':>8}{'c':>7}"
          f"{'m measured':>12}{'m raw<5/N':>11}{'m predicted':>13}"
          f"{'meas/pred':>11}{'m/H^alpha':>11}")
    out = []
    for al, n, sig in cells:
        Ys, pw, wt, T = run(al, sig, n, rng)
        H, c, m, raw5 = analyse(Ys, pw, wt, n)
        mpred = 2.0*C_alpha(al)*sig**al*T
        out.append((al, n, H, m))
        print(f"{al:6.2f}{n:6d}{sig:9.3g}{H:8.4f}{c:7.4f}{m:12.4f}{raw5:11.4f}"
              f"{mpred:13.4f}{(m/mpred if mpred > 0 else np.nan):11.3f}"
              f"{m/H**al:11.3f}", flush=True)
    return out


def fit_exp(out):
    print(f"\n{'alpha':>7}{'N':>6}{'fitted exponent':>18}{'predicted':>11}")
    import collections
    g = collections.defaultdict(list)
    for al, n, H, m in out:
        if m > 1e-4: g[(al, n)].append((H, m))
    for (al, n), v in sorted(g.items()):
        if len(v) < 3: continue
        H = np.array([x[0] for x in v]); M = np.array([x[1] for x in v])
        p = np.polyfit(np.log(H), np.log(M), 1)[0]
        print(f"{al:7.2f}{n:6d}{p:18.3f}{al:11.2f}")


rng = np.random.default_rng(SEED)
t0 = time.time()
print("=" * 104)
print("  m = mass of the spike at P_W ~ 1/N   ('the winner had almost no chance')")
print("  derivation:  m = 2 C_alpha sigma^alpha T_alpha  =  K N H^alpha")
print(f"  conf={CONF} repl={REPL}   T_alpha computed from the order statistics, not fitted")
print("=" * 104)

# sigma ranges chosen so H lands roughly in [0.10, 0.45] for each alpha
SIG = {0.50: [0.05, 0.12, 0.3, 0.7, 1.6],
       1.00: [0.25, 0.5, 1.0, 1.9, 3.6],
       1.50: [0.8, 1.4, 2.4, 4.0, 6.5],
       1.97: [1.3, 2.2, 3.6, 5.8, 9.0]}

out1 = block("TEST 1+2 -- the formula, and the exponent, at N = 200",
             [(al, 200, s) for al in (0.5, 1.0, 1.5, 1.97) for s in SIG[al]], rng)
fit_exp(out1)

out2 = block("TEST 3 -- is m/H^alpha proportional to N?  (alpha = 1.97 throughout)",
             [(1.97, n, s) for n in (50, 800) for s in SIG[1.97]], rng)
print("\nm/H^alpha averaged over the sigma scan, per N:")
import collections
g = collections.defaultdict(list)
for al, n, H, m in out1 + out2:
    if abs(al - 1.97) < 1e-9 and m > 1e-4: g[n].append(m/H**al)
base = None
for n in sorted(g):
    v = float(np.mean(g[n]))
    if base is None: base = v/n
    print(f"   N={n:4d}   m/H^alpha = {v:.3f}    "
          f"(N=50 scaled by N would give {base*n:.3f})")

print(f"\nelapsed {time.time()-t0:.0f} s")
print("""
COLUMNS
  m measured    excess of the P_W distribution over Beta(1-c,c), plateau readout
  m raw<5/N     the same mass without any Beta subtraction, as a cross-check
  m predicted   2 C_alpha sigma^alpha <T_alpha>, computed from the order
                statistics of the same values.  Nothing is fitted.
  m/H^alpha     should be a constant times N if the derivation holds

Send the entire output.
""")
