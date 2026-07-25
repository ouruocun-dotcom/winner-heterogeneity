"""
==============================================================================
  FIGURE: VALIDATION OF THE PREDICTABILITY THEORY
==============================================================================
 (a) EQ. (1) ACROSS ALL EXTREME-VALUE CLASSES
     H_measured vs H_predicted = 2 * sum_i Pr[Z_i - Z_1 > Delta_i].
     Both come from the SAME increment draws, so no analytic tail constant
     or scale convention can enter.  Eq.(1) is a union bound, so the ratio
     approaches 1 from below as H -> 0; the binned table prints this.

 (b) THE TAIL DICHOTOMY  (N = 2, exact)
     H vs sigma/Delta on log-log axes.
       alpha < 2 : straight line of slope alpha   (power law)
       alpha = 2 : bends sharply upward           (essential singularity)

 (c) PHASE-DIAGRAM EXPONENTS  (alpha < 2 only)
     theta = d log H / d log N, predicted vs measured, from the median of
     T_alpha = sum_i Delta_i^(-alpha) (pure order statistics, no Monte
     Carlo).

     RESTRICTED TO alpha < 2 ON PURPOSE.  For alpha = 2 the increment tail
     is Gaussian, Q(x) ~ exp(-x^2/4 sigma^2), so H is dominated by the
     single smallest gap and depends on sigma exponentially rather than as
     a power.  There is then no sigma-independent exponent theta to test:
     T_2 = sum_i Delta_i^-2 still has a well-defined scaling, but it is not
     the scaling of H.  The alpha = 2 case is instead shown in panel (b),
     where its essential singularity is manifest.

     Predictions:
         Gumbel(b)                theta = 1        [x (ln N)^-alpha]
         Frechet(beta)            theta = 1 - alpha/beta
         Weibull(p)               theta = max(1, alpha/p)
     The Frechet row is the only one whose exponent depends on alpha, and
     the only one where theta changes sign -- at alpha = beta.

OUTPUT   collapse_data.csv, collapse_n2.csv, exponents.csv, fig_collapse.png
RUNTIME  ~4-6 min.   QUICK = True gives ~10 s.
DEPS     numpy, pandas, matplotlib
==============================================================================
"""
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

QUICK = False
SEED = 20260725

if QUICK:
    N_MC = [30, 300]
    LAND_MC, K_MC = 8, 400
    SIG_RATIOS = np.geomspace(0.03, 1.5, 6)
    N_EXP = [100, 300, 1000, 3000, 10000]
    LAND_EXP = 120
else:
    N_MC = [30, 100, 300, 1000]
    LAND_MC, K_MC = 12, 1200
    SIG_RATIOS = np.geomspace(0.02, 1.5, 8)
    N_EXP = [100, 300, 1000, 3000, 10000, 30000]
    LAND_EXP = 150

ALPHAS = [0.5, 1.0, 1.5, 2.0]      # panels (a) and (b)
ALPHAS_EXP = [0.5, 1.0, 1.5]       # panel (c): power-law regime only
CLASSES_MC = [('gumbel', 1.0), ('frechet', 2.0), ('weibull', 1.0)]
# Frechet beta values include 0.5, 1.0, 1.5 so that the diagonal
# alpha = beta is sampled three times -- that is where theta must vanish.
CELLS_EXP = [('gumbel', 1.0),
             ('frechet', 0.5), ('frechet', 1.0), ('frechet', 1.5),
             ('frechet', 2.0), ('frechet', 3.0),
             ('weibull', 1.0), ('weibull', 2.0)]

H_LO, H_HI = 3e-3, 0.20


# --------------------------------------------------------------- variates
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


def draw_fitness(kind, prm, shape, rng):
    U = np.clip(rng.random(shape), 1e-15, 1 - 1e-15)
    if kind == 'gumbel':
        return -prm * np.log(U)
    if kind == 'frechet':
        return (-np.log(U)) ** (-1.0 / prm)
    if kind == 'weibull':
        return 1.0 - U ** (1.0 / prm)
    raise ValueError(kind)


def theta_pred(kind, prm, alpha):
    if kind == 'gumbel':
        return 1.0
    if kind == 'frechet':
        return 1.0 - alpha / prm
    if kind == 'weibull':
        return max(1.0, alpha / prm)


def has_log_corr(kind, prm, alpha):
    if kind == 'gumbel':
        return True
    if kind == 'weibull' and abs(alpha - prm) < 1e-9:
        return True
    return False


# ------------------------------------------------------------ panel (a)
def one_landscape(s_desc, alpha, sigma, K, rng):
    N = s_desc.size
    delta = s_desc[0] - s_desc
    Z = sigma * draw_stable(alpha, (K, N), rng)
    win = np.argmax(s_desc[None, :] + Z, axis=1)
    p = np.bincount(win, minlength=N) / K
    H_meas = min((1.0 - np.sum(p ** 2)) * K / (K - 1), 1.0)
    ups = (Z[:, 1:] - Z[:, 0:1]) > delta[None, 1:]
    return H_meas, 2.0 * float(ups.sum()) / K


def sweep_mc():
    rows = []
    for kind, prm in CLASSES_MC:
        for alpha in ALPHAS:
            for N in N_MC:
                rng = np.random.default_rng(
                    SEED + hash((kind, prm, alpha, N)) % 90000)
                lands = []
                for _ in range(LAND_MC):
                    s = np.sort(draw_fitness(kind, prm, N, rng))[::-1]
                    if s[0] - s[1] > 0:
                        lands.append(s)
                if not lands:
                    continue
                d2 = float(np.median([s[0] - s[1] for s in lands]))
                if not np.isfinite(d2) or d2 <= 0:
                    continue
                for r in SIG_RATIOS:
                    for s in lands:
                        Hm, Hp = one_landscape(s, alpha, r * d2, K_MC, rng)
                        rows.append(dict(cls=kind, prm=prm, alpha=alpha,
                                         N=N, ratio=r,
                                         H_meas=Hm, H_pred=Hp))
                print(f'    (a) {kind:>8} a={alpha} N={N:>5} done')
    return pd.DataFrame(rows)


# ------------------------------------------------------------ panel (b)
def sweep_n2():
    rows = []
    K2 = 60000
    s = np.array([1.0, 0.0])
    for alpha in ALPHAS:
        ratios = (np.geomspace(0.15, 3.0, 24) if alpha >= 2.0
                  else np.geomspace(1e-4, 3.0, 34))
        rng = np.random.default_rng(SEED + 777 + int(10 * alpha))
        for r in ratios:
            Hm, _ = one_landscape(s, alpha, r, K2, rng)
            rows.append(dict(alpha=alpha, ratio=r, H_meas=Hm))
    return pd.DataFrame(rows)


# ------------------------------------------------------------ panel (c)
def median_T(kind, prm, alpha, N, land, rng):
    """Batched: draw all landscapes at once, sort, form T = sum Delta^-alpha."""
    A = np.sort(draw_fitness(kind, prm, (land, N), rng), axis=1)
    d = A[:, -1][:, None] - A[:, :-1]        # gaps to the leader
    with np.errstate(divide='ignore', invalid='ignore'):
        contrib = np.where(d > 0, d ** (-alpha), 0.0)
    T = contrib.sum(axis=1)
    T = T[np.isfinite(T) & (T > 0)]
    return float(np.median(T)) if T.size else np.nan


def sweep_exponents():
    rows = []
    for kind, prm in CELLS_EXP:
        for alpha in ALPHAS_EXP:
            Ts = []
            for N in N_EXP:
                rng = np.random.default_rng(
                    SEED + hash((kind, prm, alpha, N, 'e')) % 90000)
                Ts.append(median_T(kind, prm, alpha, N, LAND_EXP, rng))
            Ts = np.array(Ts, float)
            ok = np.isfinite(Ts) & (Ts > 0)
            if ok.sum() < 3:
                continue
            x, y = np.log(np.array(N_EXP)[ok]), np.log(Ts[ok])
            # fit the upper half of the range when there is enough of it,
            # otherwise use everything
            k = len(x) // 2 if len(x) >= 6 else 0
            th_raw = float(np.polyfit(x[k:], y[k:], 1)[0])
            # remove the known logarithmic factor where it applies
            if kind == 'gumbel':
                th_cor = float(np.polyfit(x[k:], y[k:] + alpha * np.log(x[k:]),
                                          1)[0])
            elif has_log_corr(kind, prm, alpha):
                th_cor = float(np.polyfit(x[k:], y[k:] - np.log(x[k:]), 1)[0])
            else:
                th_cor = th_raw
            rows.append(dict(cls=kind, prm=prm, alpha=alpha,
                             pred=theta_pred(kind, prm, alpha),
                             raw=th_raw, corrected=th_cor,
                             logcorr=has_log_corr(kind, prm, alpha)))
            print(f'    (c) {kind:>8} prm={prm} a={alpha} done')
    return pd.DataFrame(rows)


# ==================================================================== main
if __name__ == '__main__':
    t0 = time.time()
    print('=' * 74)
    print('  VALIDATION FIGURE FOR THE PREDICTABILITY THEORY')
    print('=' * 74)

    print('\n  panel (a): Monte Carlo sweep ...')
    D = sweep_mc()
    D.to_csv('collapse_data.csv', index=False)

    print('\n  panel (b): N=2 sweep ...')
    D2 = sweep_n2()
    D2.to_csv('collapse_n2.csv', index=False)

    print('\n  panel (c): exponent sweep ...')
    E = sweep_exponents()
    E.to_csv('exponents.csv', index=False)

    # ------------------------------------------------------ diagnostics
    W = D[(D['H_meas'] > H_LO) & (D['H_meas'] < H_HI) &
          (D['H_pred'] > H_LO)].copy()
    print('\n' + '=' * 74)
    print(f'  (a) points in the validity window: {len(W)}')
    if len(W) > 20:
        print('\n  union bound tightens as H falls:')
        print(f"    {'H_pred bin':>20} {'n':>5} {'median ratio':>13}")
        ed = np.geomspace(W['H_pred'].min() * 0.99,
                          W['H_pred'].max() * 1.01, 6)
        for lo, hi in zip(ed[:-1], ed[1:]):
            b = W[(W['H_pred'] >= lo) & (W['H_pred'] < hi)]
            if len(b) >= 5:
                print(f'    {lo:>8.3g} - {hi:<9.3g} {len(b):>5} '
                      f'{np.median(b["H_meas"] / b["H_pred"]):>13.3f}')

    print('\n  (b) N=2 slopes over the small-H tail:')
    n2fit = {}
    for a in ALPHAS:
        g = D2[(D2['alpha'] == a) & (D2['H_meas'] > 3e-4) &
               (D2['H_meas'] < 0.12)]
        if len(g) >= 4:
            sl = float(np.polyfit(np.log(g['ratio']),
                                  np.log(g['H_meas']), 1)[0])
            n2fit[a] = sl
            tgt = f'{a:+.2f}' if a < 2 else 'none (essential sing.)'
            print(f'    alpha={a}: slope {sl:+.3f}   power-law target {tgt}')

    if len(E):
        print('\n  (c) exponents  (alpha < 2; alpha = 2 has no power-law regime\n      and is shown in panel (b) instead):')
        print(f"    {'class':>8} {'prm':>5} {'alpha':>6} {'pred':>7} "
              f"{'raw':>7} {'corrected':>10} {'dev':>7}")
        for _, r in E.iterrows():
            print(f"    {r['cls']:>8} {r['prm']:>5.1f} {r['alpha']:>6.1f} "
                  f"{r['pred']:>+7.3f} {r['raw']:>+7.3f} "
                  f"{r['corrected']:>+10.3f} {r['corrected']-r['pred']:>+7.3f}")
        dev = (E['corrected'] - E['pred']).abs()
        print(f'\n    mean |deviation| = {dev.mean():.4f}   '
              f'max = {dev.max():.4f}')
        diag = E[np.isclose(E['pred'], 0.0)]
        if len(diag):
            print('    alpha = beta cells (theta must vanish):')
            for _, r in diag.iterrows():
                print(f"      beta=alpha={r['alpha']:.1f}: "
                      f"theta = {r['corrected']:+.4f}")
        fr = E[E['cls'] == 'frechet']
        if len(fr) > 2:
            rr = np.corrcoef(fr['pred'], fr['corrected'])[0, 1]
            print(f'    Frechet row: Pearson(pred, measured) = {rr:.4f}')

    # =========================================================== FIGURE
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6))
    amap = {0.5: '#4575b4', 1.0: '#74add1', 1.5: '#f46d43', 2.0: '#a50026'}
    mmap = {'gumbel': 'o', 'frechet': '^', 'weibull': 's'}

    # (a)
    ax = axes[0]
    if len(W):
        for (cls, a), g in W.groupby(['cls', 'alpha']):
            ax.scatter(g['H_pred'], g['H_meas'], s=10, alpha=0.4,
                       color=amap.get(a, 'k'), marker=mmap.get(cls, 'o'),
                       edgecolors='none')
        xx = np.geomspace(W['H_pred'].min(), W['H_pred'].max(), 40)
        ax.plot(xx, xx, 'k--', lw=1.4, alpha=0.85)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel(r'$\mathcal{H}_{\rm pred}$ from Eq. (1)', fontsize=11)
    ax.set_ylabel(r'$\mathcal{H}$ measured', fontsize=11)
    ax.set_title('(a) Eq. (1), all classes and $N$', fontsize=11,
                 fontweight='bold')
    hs = [plt.Line2D([], [], marker=mmap[c], ls='', color='gray', ms=5,
                     label=c) for c in mmap]
    hs += [plt.Line2D([], [], marker='o', ls='', color=amap[a], ms=5,
                      label=rf'$\alpha$={a}') for a in ALPHAS]
    hs += [plt.Line2D([], [], ls='--', color='k', label='$y=x$')]
    ax.legend(handles=hs, fontsize=6.5, ncol=2, loc='upper left')
    ax.grid(alpha=0.25, which='both')

    # (b)
    ax = axes[1]
    for a in ALPHAS:
        g = D2[(D2['alpha'] == a) & (D2['H_meas'] > 1e-5)].sort_values('ratio')
        if len(g):
            ax.plot(g['ratio'], g['H_meas'], marker='o', ms=3, lw=1.5,
                    color=amap[a], label=rf'$\alpha$={a}')
    for a, sl in n2fit.items():
        if a < 2:
            g = D2[(D2['alpha'] == a) & (D2['H_meas'] > 3e-4) &
                   (D2['H_meas'] < 0.12)].sort_values('ratio')
            if len(g) >= 2:
                x = np.array([g['ratio'].iloc[0], g['ratio'].iloc[-1]])
                ax.plot(x, g['H_meas'].iloc[0] *
                        (x / g['ratio'].iloc[0]) ** a, ':', lw=1.1,
                        color=amap[a], alpha=0.9)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel(r'$\sigma/\Delta$', fontsize=11)
    ax.set_ylabel(r'$\mathcal{H}$', fontsize=11)
    ax.set_title(r'(b) $N=2$: power law vs essential singularity',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=8, loc='lower right')
    ax.grid(alpha=0.25, which='both')

    # (c)
    ax = axes[2]
    if len(E):
        for cls, g in E.groupby('cls'):
            ax.scatter(g['pred'], g['corrected'], s=60,
                       marker=mmap.get(cls, 'o'),
                       facecolors=['none' if lc else 'C0'
                                   for lc in g['logcorr']],
                       edgecolors='C0' if cls == 'gumbel' else
                                  ('C1' if cls == 'frechet' else 'C2'),
                       linewidth=1.3, label=cls, zorder=3)
        lim = [min(E['pred'].min(), E['corrected'].min()) - 0.25,
               max(E['pred'].max(), E['corrected'].max()) + 0.25]
        ax.plot(lim, lim, 'k--', lw=1.3, alpha=0.7, zorder=1)
        ax.axhline(0, color='gray', lw=0.6)
        ax.axvline(0, color='gray', lw=0.6)
        ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel(r'predicted $\theta$', fontsize=11)
    ax.set_ylabel(r'measured $\theta$', fontsize=11)
    ax.set_title(r'(c) $\theta=d\log\mathcal{H}/d\log N$   ($\alpha<2$)',
                 fontsize=11, fontweight='bold')
    if len(E):
        ax.legend(fontsize=8, loc='upper left')
    ax.grid(alpha=0.25)

    plt.tight_layout()
    plt.savefig('fig_collapse.png', dpi=300, bbox_inches='tight')
    print('\n  saved fig_collapse.png')
    print(f'  TOTAL RUNTIME: {time.time() - t0:.0f} s')
