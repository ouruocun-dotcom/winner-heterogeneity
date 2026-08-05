"""
==============================================================================
  alpha = 2 CONSISTENCY CHECK
  Does our framework contain the Brownian universality of
  Burda & Kieburg, Phys. Rev. E 112, 014114 (2025)
  Burda, Kieburg & Maciocha, Phys. Rev. E (2026), arXiv:2603.24151 ?
==============================================================================

WHAT IS BEING TESTED
--------------------
Those works consider N particles diffusing in a confining potential
V(x) ~ x^gamma / 2, so the stationary law has tail exp(-x^gamma) (Gumbel
class for every gamma > 0).  They find that the reshuffling statistics of
the leaders are UNIVERSAL -- independent of gamma once time is measured in
units of

        c_N  ~  (ln N)^{2(1-gamma)/gamma}.

Our framework predicts the same thing, for a simple reason.  Writing
Delta_i for the gaps below the leader, we have

        H  ~  2 sum_i Q(Delta_i),      Q(x) = Pr[Z - Z' > x].

For Gumbel-class values Delta_i ~ b_N ln i, with the spacing scale
b_N = (1/gamma)(ln N)^{(1-gamma)/gamma}.  Then:

  alpha = 2 :  Q(x) ~ exp(-x^2/4 sigma^2), the sum converges and is
               dominated by i = 2, so H depends only on sigma / Delta_2
               and carries NO explicit N dependence.  Since Brownian
               motion has sigma^2 ~ t, the reshuffling time is
               t ~ b_N^2 ~ (ln N)^{2(1-gamma)/gamma}, which is exactly
               their c_N.  Their universality is recovered.

  alpha < 2 :  Q(x) ~ C (sigma/x)^alpha, the sum picks up every
               competitor and grows like N (ln N)^{-alpha}.  An explicit
               factor of N survives and the universality breaks.

SHARP NUMERICAL CRITERION
-------------------------
Measure sigma*(N), the noise scale at which H reaches a fixed target, and
divide by the typical top gap Delta_2(N).  Then

        sigma* / Delta_2  ~  N^{-1/alpha} (ln N)   for alpha < 2
                          ~  const                  for alpha = 2

so after dividing out the ln N factor the slope must be exactly -1/alpha
below two, and exactly zero at two.  The raw slope carries an extra
+1/ln N, which is sizeable at accessible N and is reported separately.

Flat means the Brownian universality holds; a slope of -1/alpha means it
has broken.

Also checked: our winner heterogeneity and their overlap ratio satisfy
H = 2(1 - Omega_1) + O(H^2), where Omega_1 is the probability that the
initial leader is still the leader.

OUTPUT   alpha2_results.csv, fig_alpha2.png, and a printed summary
RUNTIME  ~5-10 min.   QUICK = True gives ~1 min.
==============================================================================
"""
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

QUICK = False
SEED = 902

if QUICK:
    N_VALUES = [30, 300, 3000]
    GAMMAS = [1.0, 2.0]
    ALPHAS = [1.0, 2.0]
    LAND, K, BISECT = 8, 400, 10
else:
    N_VALUES = [30, 100, 300, 1000, 3000, 10000]
    GAMMAS = [0.5, 1.0, 2.0, 3.0]
    ALPHAS = [0.5, 1.0, 1.5, 2.0]
    LAND, K, BISECT = 12, 700, 13

H_TARGET = 0.20


# ------------------------------------------------------------------ variates
def draw_stable(alpha, size, rng):
    """Symmetric alpha-stable, unit scale (Chambers-Mallows-Stuck)."""
    if alpha >= 2.0:
        return rng.standard_normal(size)
    U = rng.uniform(-np.pi / 2, np.pi / 2, size)
    W = rng.exponential(1.0, size)
    if abs(alpha - 1.0) < 1e-9:
        return np.tan(U)
    t1 = np.sin(alpha * U) / np.cos(U) ** (1.0 / alpha)
    t2 = (np.cos(U - alpha * U) / W) ** ((1.0 - alpha) / alpha)
    return t1 * t2


def draw_values(gamma, N, rng):
    """
    Stationary law of Burda et al.: tail  P(X > x) = exp(-x^gamma).
    Sampling: X = (-ln U)^{1/gamma}.  Gumbel class for every gamma > 0.
    """
    U = np.clip(rng.random(N), 1e-300, 1 - 1e-16)
    return (-np.log(U)) ** (1.0 / gamma)


# --------------------------------------------------------------- measurement
def measure(s_desc, alpha, sigma, K, rng):
    """Returns (H, Omega_1) from one landscape and one set of increments."""
    N = s_desc.size
    Z = sigma * draw_stable(alpha, (K, N), rng)
    win = np.argmax(s_desc[None, :] + Z, axis=1)
    p = np.bincount(win, minlength=N) / K
    H = min((1.0 - np.sum(p ** 2)) * K / (K - 1), 1.0)
    Omega1 = float(np.mean(win == 0))     # leader keeps the lead
    return H, Omega1


def mean_H(lands, alpha, sigma, K, rng):
    return float(np.median([measure(s, alpha, sigma, K, rng)[0]
                            for s in lands]))


def sigma_star(lands, alpha, K, rng, target=H_TARGET, steps=BISECT):
    scale = float(np.median([s[0] - s[1] for s in lands]))
    lo, hi = scale * 1e-3, scale * 1e2
    for _ in range(20):
        if mean_H(lands, alpha, lo, K, rng) < target:
            break
        lo /= 10.0
    for _ in range(20):
        if mean_H(lands, alpha, hi, K, rng) > target:
            break
        hi *= 10.0
    for _ in range(steps):
        mid = np.sqrt(lo * hi)
        if mean_H(lands, alpha, mid, K, rng) < target:
            lo = mid
        else:
            hi = mid
    return float(np.sqrt(lo * hi))


# ===========================================================================
if __name__ == '__main__':
    t0 = time.time()
    print('=' * 78)
    print('  DOES OUR FRAMEWORK CONTAIN THE BROWNIAN UNIVERSALITY?')
    print(f'  gammas={GAMMAS}  alphas={ALPHAS}  N={N_VALUES}')
    print('=' * 78)

    rows = []
    for gamma in GAMMAS:
        for alpha in ALPHAS:
            for N in N_VALUES:
                rng = np.random.default_rng(
                    SEED + hash((gamma, alpha, N)) % 90000)
                lands = [np.sort(draw_values(gamma, N, rng))[::-1]
                         for _ in range(LAND)]
                lands = [s for s in lands if s[0] - s[1] > 0]
                if len(lands) < 4:
                    continue
                d2 = float(np.median([s[0] - s[1] for s in lands]))
                ss = sigma_star(lands, alpha, K, rng)

                # observable correspondence, evaluated at sigma*
                Hs, Om = [], []
                for s in lands:
                    h, o = measure(s, alpha, ss, K, rng)
                    Hs.append(h); Om.append(o)
                H_med = float(np.median(Hs))
                Om_med = float(np.median(Om))

                rows.append(dict(gamma=gamma, alpha=alpha, N=N,
                                 d2=d2, sigma_star=ss, ratio=ss / d2,
                                 H=H_med, Omega1=Om_med,
                                 two_one_minus_Omega=2 * (1 - Om_med)))
            print(f'    gamma={gamma}  alpha={alpha}  done '
                  f'({time.time()-t0:.0f}s)')

    R = pd.DataFrame(rows)
    R.to_csv('alpha2_results.csv', index=False)

    # ------------------------------------------------------- main criterion
    print('\n' + '=' * 78)
    print('  d log(sigma*/Delta_2) / d log N')
    print('  prediction:  0 for alpha=2 (universality holds)')
    print('               -1/alpha for alpha<2 (universality breaks)')
    print('=' * 78)
    print(f"\n  {'gamma':>6} {'alpha':>6} {'raw':>9} {'-lnN':>9} "
          f"{'predicted':>10} {'verdict':>10}")
    print('  ' + '-' * 58)
    summary = []
    for (g, a), grp in R.groupby(['gamma', 'alpha']):
        grp = grp.sort_values('N')
        if len(grp) < 3:
            continue
        x = np.log(grp['N'].values.astype(float))
        y = np.log(grp['ratio'].values.astype(float))
        sl_raw = float(np.polyfit(x, y, 1)[0])
        # the Gumbel class carries an extra factor ln N below alpha = 2;
        # divide it out so the slope is the bare -1/alpha
        sl_cor = (float(np.polyfit(x, y - np.log(x), 1)[0]) if a < 2
                  else sl_raw)
        pred = 0.0 if a >= 2 else -1.0 / a
        if a >= 2:
            ok = 'FLAT' if abs(sl_raw) < 0.10 else 'not flat'
        else:
            ok = 'matches' if abs(sl_cor - pred) < 0.20 else 'off'
        print(f"  {g:>6.1f} {a:>6.1f} {sl_raw:>9.3f} {sl_cor:>9.3f} "
              f"{pred:>10.3f} {ok:>10}")
        summary.append(dict(gamma=g, alpha=a, slope_raw=sl_raw,
                            slope_logcorr=sl_cor, pred=pred))
    S = pd.DataFrame(summary)
    S.to_csv('alpha2_slopes.csv', index=False)

    # --------------------------------------------- their time-scale formula
    print('\n' + '=' * 78)
    print('  Burda et al. time scale:  t ~ (ln N)^{2(1-gamma)/gamma}')
    print('  Brownian sigma^2 ~ t, so  sigma*^2  must follow the same law')
    print('=' * 78)
    print(f"\n  {'gamma':>6} {'measured':>10} {'predicted':>10}")
    print('  ' + '-' * 28)
    for g in GAMMAS:
        grp = R[(R['gamma'] == g) & (R['alpha'] == 2.0)].sort_values('N')
        if len(grp) < 3:
            continue
        x = np.log(np.log(grp['N']))
        y = np.log(grp['sigma_star'] ** 2)
        sl = float(np.polyfit(x, y, 1)[0])
        print(f"  {g:>6.1f} {sl:>10.3f} {2*(1-g)/g:>10.3f}")

    # ------------------------------------------- observable correspondence
    print('\n' + '=' * 78)
    print('  H  vs  2(1 - Omega_1)     [our observable vs theirs]')
    print('=' * 78)
    rr = R[R['H'] > 0]
    if len(rr):
        rel = (rr['H'] - rr['two_one_minus_Omega']).abs() / rr['H']
        print(f'\n  median relative difference: {rel.median():.3f}')
        print(f'  90th percentile:            {rel.quantile(0.9):.3f}')
        print(f'  (they agree to O(H); H ~ {H_TARGET} here)')

    # ------------------------------------------------------------- figure
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    cm = plt.get_cmap('viridis')

    ax = axes[0]
    for i, g in enumerate(GAMMAS):
        grp = R[(R['gamma'] == g) & (R['alpha'] == 2.0)].sort_values('N')
        if len(grp):
            ax.plot(grp['N'], grp['ratio'], marker='o', ms=5, lw=1.5,
                    color=cm(i / max(len(GAMMAS) - 1, 1)),
                    label=rf'$\gamma={g}$')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('$N$', fontsize=11)
    ax.set_ylabel(r'$\sigma^*/\Delta_2$', fontsize=11)
    ax.set_title(r'(a) $\alpha=2$: flat $\Rightarrow$ Brownian universality',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=8); ax.grid(alpha=0.25, which='both')

    ax = axes[1]
    for i, a in enumerate(ALPHAS):
        grp = R[(R['gamma'] == 1.0) & (R['alpha'] == a)].sort_values('N')
        if len(grp):
            ax.plot(grp['N'], grp['ratio'], marker='s', ms=5, lw=1.5,
                    color=cm(i / max(len(ALPHAS) - 1, 1)),
                    label=rf'$\alpha={a}$')
            if a < 2:
                n = np.array(grp['N'], float)
                ax.plot(n, grp['ratio'].iloc[0] * (n / n[0]) ** (-1.0 / a),
                        ':', lw=1.0, color=cm(i / max(len(ALPHAS) - 1, 1)))
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('$N$', fontsize=11)
    ax.set_ylabel(r'$\sigma^*/\Delta_2$', fontsize=11)
    ax.set_title(r'(b) $\gamma=1$: breakdown for $\alpha<2$',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=8); ax.grid(alpha=0.25, which='both')

    plt.tight_layout()
    plt.savefig('fig_alpha2.png', dpi=300, bbox_inches='tight')
    print(f'\n  saved fig_alpha2.png   TOTAL {time.time()-t0:.0f}s')
    print("""
==============================================================================
  READING THE RESULT
==============================================================================
  The first table is decisive.  If the alpha=2 rows are flat for every
  gamma, our framework reproduces the Brownian universality; if the
  alpha<2 rows follow -1/alpha, the universality breaks exactly as the
  sum over competitors predicts.

  The second table checks their explicit time law
  t ~ (ln N)^{2(1-gamma)/gamma} against sigma*^2.

  The third confirms that our observable and theirs measure the same
  thing to leading order.
==============================================================================
""")
