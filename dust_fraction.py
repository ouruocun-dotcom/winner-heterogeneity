"""
============================================================================
IS HALF OF ALL RESHUFFLING ACCIDENTAL?
============================================================================
In the scaling limit the winner measure is one atom of random size
q = exp(-eta u^r), u ~ Exp(1), r = alpha/beta, plus dust of mass 1-q whose
individual grains carry O(1/N).  The atom is the competitor the values
single out; the dust is everyone else, each reachable only by one long jump.

Two consequences follow, with no free constant:

    m  := Pr[the winner is dust]          = 1 - int_0^inf e^{-u - eta u^r} du
    H  := Pr[two replicas differ]         = 1 - int_0^inf e^{-u - 2 eta u^r} du

and for small eta both are linear in eta, so

    m / H  ->  1/2 .

Half of all reshuffling events produce a winner who had almost no chance.

WHY THIS NEEDS CHECKING SEPARATELY
The manuscript currently states m ~ H^alpha, measured at alpha = 1.97 where
m/H^2 came out near unity.  That measurement sits in the corner where the
stable tail constant C_alpha vanishes: the Levy tail barely acts, reshuffling
is carried almost entirely by a near-degenerate runner-up drawn from the
noise CORE rather than its tail, and the winner is a serious contender rather
than dust.  So it validated a statement about Levy tails in the one place
where Levy tails do nothing.  Away from alpha = 2 the two predictions differ
by an order of magnitude: at H = 0.1, H^2 = 0.01 while H/2 = 0.05.

WHAT IS MEASURED
Values with Pr[S>x] = A x^{-beta}, symmetric alpha-stable increments of
scale sigma.  For each configuration the winner counts over R replicas give

    H  from  1 - sum_i P_i^2   (unbiased U-statistic)
    m  from  the fraction of replicas whose winner took less than a
             threshold share of the replicas.

The threshold is the delicate part, so m is reported at several of them,
in units of 1/N.  Dust grains carry O(1/N); the atom carries O(1).  If the
picture holds, m should be flat in the threshold across a wide plateau,
and that plateau is the number to compare with H/2.  If instead m keeps
growing with the threshold there is no separation of scales and the
decomposition fails -- which is itself the answer.

alpha = 0.5, 1.0, 1.5 are scanned, and 1.97 is included so the old result
can be seen failing in its own regime rather than only asserted to.

RUNTIME ~15-20 min.  numpy + scipy.
============================================================================
"""

import time
import numpy as np
from scipy.integrate import quad
from scipy.special import gamma as Gam

N     = 400
CONF  = 1500
REPL  = 800
BETA  = 1.5
A     = 1.0
THRESH = (2, 5, 10, 20, 40)      # in units of 1/N
SEED  = 20260802


def C_alpha(al):
    return Gam(al)*np.sin(np.pi*al/2.0)/np.pi


def stable(al, size, rng):
    U = rng.uniform(-np.pi/2, np.pi/2, size)
    W = rng.exponential(1.0, size)
    if abs(al - 1.0) < 1e-12:
        return np.tan(U)
    return (np.sin(al*U)/np.cos(U)**(1.0/al)
            * (np.cos(U - al*U)/W)**((1.0 - al)/al))


def predict(eta, r, k):
    return quad(lambda u: np.exp(-u - k*eta*u**r), 0, np.inf, limit=200)[0]


def run(al, sig, rng, n=N, conf=CONF, repl=REPL, be=BETA):
    """returns H, and m at each threshold"""
    H = 0.0
    dust = np.zeros(len(THRESH))
    den2 = float(repl*(repl - 1))
    for _ in range(conf):
        S = (A/rng.uniform(0, 1, n))**(1.0/be)
        w = np.argmax(S[None, :] + sig*stable(al, (repl, n), rng), axis=1)
        c = np.bincount(w, minlength=n).astype(np.float64)
        H += 1.0 - (c*(c - 1.0)).sum()/den2
        for j, t in enumerate(THRESH):
            small = c < t*repl/n          # winners holding less than t/N share
            dust[j] += c[small].sum()/repl
    return H/conf, dust/conf


rng = np.random.default_rng(SEED)
t0 = time.time()
print("=" * 96)
print("  m = Pr[winner is dust],  H = Pr[two replicas differ].")
print("  Prediction, with no free constant:   m/H -> 1/2   as H -> 0.")
print(f"  N={N} conf={CONF} repl={REPL} beta={BETA}")
print("=" * 96)

SIG = {0.50: [0.02, 0.05, 0.12, 0.30],
       1.00: [0.10, 0.25, 0.60, 1.40],
       1.50: [0.30, 0.70, 1.50, 3.00],
       1.97: [0.60, 1.40, 3.00, 6.00]}

for al in (0.5, 1.0, 1.5, 1.97):
    r = al/BETA
    Ca = C_alpha(al)
    print(f"\n{'-'*96}\nalpha = {al}   (r = alpha/beta = {r:.3f},  C_alpha = {Ca:.4f})")
    print(f"{'sigma':>7}{'H':>9}{'H pred':>9}{'H/2':>8}"
          + "".join(f"{'m@'+str(t)+'/N':>10}" for t in THRESH)
          + f"{'m/H':>8}{'H^2':>8}")
    for sig in SIG[al]:
        H, m = run(al, sig, rng)
        eta = Ca*sig**al*A**(-r)*N**(1 - r)
        Hp = 1.0 - predict(eta, r, 2)
        best = m[len(THRESH)//2]
        print(f"{sig:7.2f}{H:9.4f}{Hp:9.4f}{H/2:8.4f}"
              + "".join(f"{x:10.4f}" for x in m)
              + f"{best/H:8.3f}{H*H:8.4f}", flush=True)

print("\n" + "=" * 96)
print(f"elapsed {time.time()-t0:.0f} s")
print("""
COLUMNS
  H, H pred   measured reshuffling probability and the crossover value
  m@t/N       dust mass, i.e. the share of replicas whose winner held less
              than t/N of them.  Flat across thresholds = clean separation
              between the atom and the dust; growing with t = no separation.
  m/H         quoted at the middle threshold.  The prediction is 0.5.
  H^2         what the manuscript currently implies for m.

Send the entire output.
""")
