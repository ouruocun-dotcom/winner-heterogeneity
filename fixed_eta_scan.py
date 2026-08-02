"""
============================================================================
DOES m/H APPROACH 1/2 WHEN THE NOISE CORE IS SCALED AWAY?
============================================================================
In the scaling limit the winner measure is one atom of random size
q = exp(-eta u^r), u ~ Exp(1), r = alpha/beta, plus dust of individual
grains O(1/N).  Writing m for the dust mass and H for the probability that
two replicas disagree, both are linear in eta at small eta, giving the
parameter-free ratio  m/H -> 1/2.

WHY THE PREVIOUS SCAN COULD NOT SETTLE THIS
That scan lowered sigma at fixed N.  But H has two sources and only one of
them makes dust:

  tail   a distant competitor crosses in one long jump.  Weight ~ eta.
         Its winner holds O(1/N): this is dust, and it is what the
         crossover describes.
  core   the runner-up sits within sigma of the leader and wins by an
         ordinary fluctuation.  Weight ~ kappa sigma N^{-1/beta}.
         Its winner is a serious contender, not dust.

Lowering sigma shrinks the tail term as sigma^alpha and the core term only
as sigma, so for alpha > 1 the core takes over exactly where the ratio was
supposed to converge.  The measured ratio duly fell through 1/2 and kept
going (0.768, 0.624, 0.504, 0.428 at alpha=1.5), which is a statement about
the core, not a refutation of the prediction.

THE RIGHT KNOB IS N, NOT sigma
At fixed eta the tail term is fixed while the core term carries an explicit
N^{-1/beta}.  So

    H = H_tail (1 + c N^{-1/beta}),      m = m_tail,
    m/H = (1/2) / (1 + c N^{-1/beta}),

i.e. the ratio approaches its limit from the core-contaminated side, with
the excess falling as a clean power N^{-1/beta}.  That is a two-part
prediction: the limit and the exponent of the approach, and both are checked.

At finite eta the limit is not 1/2 but the exact ratio of the two crossover
integrals,
    m/H  =  [1 - I(eta)] / [1 - I(2 eta)],     I(x) = int e^{-u - x u^r} du,
which tends to 1/2 only as eta -> 0.  The comparison below is against that,
not against 1/2, so no small-eta approximation is folded in.

IDENTIFYING THE DUST WITHOUT A THRESHOLD
The atom of the crossover picture is the competitor the values single out:
the top score.  So "the winner is dust" is simply "the winner is not
argmax_i s_i", which is exact, needs no cut, and costs nothing.  A
threshold on winner counts, the obvious alternative, is unusable here: dust
grains hold O(1/N) of the replicas, so resolving them at all would require
many more replicas than competitors, which defeats the point of raising N.

At alpha = beta the variable eta does not depend on N at all, so fixing it
means holding sigma fixed and simply enlarging N -- the cleanest case, with
nothing else moving.  That row is the decisive one.

RUNTIME ~20-30 min.  numpy + scipy.
============================================================================
"""

import time
import numpy as np
from scipy.integrate import quad
from scipy.special import gamma as Gam

REPL   = 400
BUDGET = 900_000      # conf is set so that conf*N stays near this
CONF_MIN, CONF_MAX = 60, 900
A      = 1.0
SEED   = 20260803

CASES = [
    # alpha, beta, eta values
    (1.5, 1.5, (0.05, 0.15)),   # r = 1: sigma is constant across N
    (1.0, 1.5, (0.05, 0.15)),   # r < 1: sigma falls with N
]
NS = (200, 800, 3200, 12800)


def C_alpha(al):
    return Gam(al)*np.sin(np.pi*al/2.0)/np.pi


def stable(al, size, rng):
    U = rng.uniform(-np.pi/2, np.pi/2, size)
    W = rng.exponential(1.0, size)
    if abs(al - 1.0) < 1e-12:
        return np.tan(U)
    return (np.sin(al*U)/np.cos(U)**(1.0/al)
            * (np.cos(U - al*U)/W)**((1.0 - al)/al))


def sigma_for(eta, al, be, n):
    r = al/be
    return (eta/(C_alpha(al)*A**(-r)*n**(1 - r)))**(1.0/al)


def run(al, be, sig, n, conf, rng, repl=REPL):
    """H, and m = Pr[winner is not the top-score competitor]"""
    H = m = 0.0
    den = float(repl*(repl - 1))
    for _ in range(conf):
        S = (A/rng.uniform(0, 1, n))**(1.0/be)
        top = int(np.argmax(S))
        w = np.argmax(S[None, :] + sig*stable(al, (repl, n), rng), axis=1)
        m += np.mean(w != top)
        c = np.bincount(w, minlength=n).astype(np.float64)
        H += 1.0 - (c*(c - 1.0)).sum()/den
    return H/conf, m/conf


def predict(eta, r, k):
    return quad(lambda u: np.exp(-u - k*eta*u**r), 0, np.inf, limit=200)[0]


rng = np.random.default_rng(SEED)
t0 = time.time()
print("=" * 100)
print("  m/H at FIXED eta, scanning N, against the exact crossover ratio.")
print("  The excess over it is the noise core, predicted to fall as N^{-1/beta}.")
print("  dust = winner is not the top-score competitor (no threshold)")
print(f"  repl={REPL}")
print("=" * 100)

for al, be, etas in CASES:
    r = al/be
    for eta in etas:
        print(f"\n{'-'*100}")
        print(f"alpha={al}  beta={be}  r={r:.3f}  eta={eta}"
              + ("   (sigma constant in N)" if abs(r - 1) < 1e-9 else ""))
        mp = 1.0 - predict(eta, r, 1)
        Hp = 1.0 - predict(eta, r, 2)
        target = mp/Hp
        print(f"{'N':>7}{'sigma':>9}{'conf':>6}{'m':>9}{'m pred':>9}"
              f"{'H':>9}{'H pred':>9}{'m/H':>8}{'target':>8}{'excess':>9}")
        rows = []
        for n in NS:
            conf = int(np.clip(BUDGET//n, CONF_MIN, CONF_MAX))
            sig = sigma_for(eta, al, be, n)
            H, m = run(al, be, sig, n, conf, rng)
            exc = m/H - target
            rows.append((n, exc))
            print(f"{n:7d}{sig:9.4g}{conf:6d}{m:9.4f}{mp:9.4f}"
                  f"{H:9.4f}{Hp:9.4f}{m/H:8.3f}{target:8.3f}{exc:+9.4f}",
                  flush=True)
        good = [(n, e) for n, e in rows if e > 0]
        if len(good) >= 3:
            p = np.polyfit(np.log([g[0] for g in good]),
                           np.log([g[1] for g in good]), 1)[0]
            print(f"{'':>30}excess decays with N as exponent {p:+.3f}"
                  f"   (predicted {-1/be:+.3f})")

print("\n" + "=" * 100)
print(f"elapsed {time.time()-t0:.0f} s")
print("""
COLUMNS
  m, H        dust mass and reshuffling probability, with the crossover
              values at this eta beside them
  m/H         measured ratio
  target      the exact crossover ratio at this eta; it tends to 1/2 only
              as eta -> 0, so this, not 1/2, is what the data should meet
  excess      m/H minus target: the core contamination, predicted to fall
              as N^{-1/beta}

Send the entire output.
""")
