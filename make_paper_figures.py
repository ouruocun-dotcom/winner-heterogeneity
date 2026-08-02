"""
==============================================================================
  BUILD fig1.png AND figS1.png
==============================================================================
  fig1.png  THE RESULT
            (a) phase diagram in the (alpha, beta) plane: theta = 1 - alpha/beta
                for Frechet initial values, with the alpha = beta line where
                theta changes sign.  Measured exponents overlaid.
            (b) measured vs predicted theta, all three extreme-value classes.

  figS1.png  THE VALIDATION
            (a) N = 2 tail dichotomy: power law for alpha < 2, essential
                singularity for alpha = 2.
            (b) Eq. (2) across all classes.  NOTE: the window is applied to
                BOTH H_meas and H_pred -- a bound on H_meas alone lets in
                points whose H_pred is far outside the single-upset regime,
                which makes the union bound look like a failure of theory.
            The window 0.10 is set by the slowest class: Weibull gaps grow
            only as (i/N)^(1/p), so many competitors sit at comparable
            distance from the leader and simultaneous upsets appear at
            lower H than for the other two classes.

Reads collapse_n2.csv, exponents.csv, collapse_data.csv.
==============================================================================
"""
# ---------------------------------------------------------------- data access
# Looks for the file locally first, then downloads it from the repository, so
# the script runs as-is in a fresh Colab cell with nothing uploaded.
REPO = ('https://raw.githubusercontent.com/'
        'ouruocun-dotcom/winner-heterogeneity/{branch}/data/')


def fetch(name):
    import os
    import urllib.request
    for p in (name, f'data/{name}', f'../data/{name}', f'/content/{name}'):
        if os.path.exists(p):
            return p
    err = None
    for branch in ('main', 'master'):
        try:
            urllib.request.urlretrieve(REPO.format(branch=branch) + name, name)
            return name
        except Exception as e:
            err = e
    raise FileNotFoundError(
        f"{name} is not here and could not be downloaded ({err}).\n"
        f"Fetch it from {REPO.format(branch='main') + name} and put it in "
        f"this directory, or upload it with the folder icon in the Colab "
        f"sidebar.")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

# the single-upset regime, applied to BOTH quantities
H_LO, H_HI = 3e-3, 0.10

ALPHAS = [0.5, 1.0, 1.5, 2.0]
AC = {0.5: '#3b6fb0', 1.0: '#7fb3d5', 1.5: '#e8834a', 2.0: '#9e1b32'}
MK = {'gumbel': 'o', 'frechet': '^', 'weibull': 's'}
CC = {'gumbel': '#3b6fb0', 'frechet': '#e8834a', 'weibull': '#2a9d4a'}

D2 = pd.read_csv(fetch('collapse_n2.csv'))
E = pd.read_csv(fetch('exponents.csv'))
D = pd.read_csv(fetch('collapse_data.csv'))

# ======================================================== FIGURE 1
fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))

# ---- (a) phase diagram -------------------------------------------------
ax = axes[0]
a_grid = np.linspace(0.05, 2.0, 400)
b_grid = np.linspace(0.15, 3.0, 400)
A, B = np.meshgrid(a_grid, b_grid)
TH = 1.0 - A / B
TH = np.clip(TH, -2.0, 1.0)

im = ax.pcolormesh(A, B, TH, cmap='RdBu_r',
                   norm=TwoSlopeNorm(vmin=-2.0, vcenter=0.0, vmax=1.0),
                   shading='auto', rasterized=True)
ax.plot([0.05, 2.0], [0.05, 2.0], 'k-', lw=2.0, zorder=4)

# measured Frechet cells
fr = E[E['cls'] == 'frechet']
if len(fr):
    ax.scatter(fr['alpha'], fr['prm'], s=34, facecolors='none',
               edgecolors='k', linewidth=1.1, zorder=5)

# labels sit in gaps between the measured points
ax.text(0.75, 2.62, r'$\alpha<\beta$: less predictable', fontsize=7.6,
        ha='center', zorder=6,
        bbox=dict(boxstyle='round,pad=0.18', fc='white', ec='none',
                  alpha=0.72))
ax.text(1.30, 0.26, r'$\alpha>\beta$: more predictable', fontsize=7.6,
        ha='center', zorder=6, color='white',
        bbox=dict(boxstyle='round,pad=0.18', fc='#1b3a6b', ec='none',
                  alpha=0.55))
ax.annotate(r'$\alpha=\beta$', xy=(1.85, 1.85), xytext=(1.42, 2.18),
            fontsize=8.5, zorder=6,
            bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none',
                      alpha=0.72),
            arrowprops=dict(arrowstyle='-', lw=0.8))

ax.set_xlim(0.05, 2.0); ax.set_ylim(0.12, 3.0)
ax.set_xlabel(r'stable index $\alpha$ (noise)', fontsize=9.5)
ax.set_ylabel(r'Fr\'echet index $\beta$ (initial values)'
              if False else r'Fréchet index $\beta$ (initial values)',
              fontsize=9.5)
ax.set_title(r'(a) $\theta=1-\alpha/\beta$', fontsize=10, fontweight='bold')
ax.tick_params(labelsize=8)
cb = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.046)
cb.set_label(r'$\theta$', fontsize=9)
cb.ax.tick_params(labelsize=7)

# ---- (b) measured vs predicted theta -----------------------------------
ax = axes[1]
STY = {'weibull': dict(s=105, dx=+0.045, z=3),
       'gumbel':  dict(s=62,  dx=-0.045, z=4),
       'frechet': dict(s=32,  dx=0.0,    z=5)}
for cls in ['weibull', 'gumbel', 'frechet']:
    g = E[E['cls'] == cls]
    if not len(g):
        continue
    st = STY[cls]
    if 'err' in g.columns:
        ax.errorbar(g['pred'] + st['dx'], g['corrected'], yerr=g['err'],
                    fmt='none', ecolor=CC[cls], elinewidth=0.9,
                    capsize=1.8, zorder=st['z'] - 1)
    ax.scatter(g['pred'] + st['dx'], g['corrected'], s=st['s'],
               marker=MK[cls], facecolors='none', edgecolors=CC[cls],
               linewidth=1.4, label=cls, zorder=st['z'])
lo = min(E['pred'].min(), E['corrected'].min()) - 0.22
hi = max(E['pred'].max(), E['corrected'].max()) + 0.22
ax.plot([lo, hi], [lo, hi], 'k--', lw=1.0, alpha=0.7, zorder=1)
ax.axhline(0, color='gray', lw=0.5); ax.axvline(0, color='gray', lw=0.5)
ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
ax.set_xlabel(r'predicted $\theta$', fontsize=9.5)
ax.set_ylabel(r'measured $\theta$', fontsize=9.5)
ax.set_title(r'(b) all classes, $\alpha<2$', fontsize=10, fontweight='bold')
ax.legend(fontsize=7, loc='upper left', framealpha=0.95)
ax.tick_params(labelsize=8)
ax.grid(alpha=0.2, lw=0.4)

plt.tight_layout()
plt.savefig('fig1.png', dpi=300, bbox_inches='tight')
plt.close()
print('saved fig1.png')

# ======================================================== FIGURE 2
fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))

# ---- (a) N = 2 ----------------------------------------------------------
ax = axes[0]
slopes = {}
for a in ALPHAS:
    g = D2[(D2['alpha'] == a) & (D2['H_meas'] > 1e-5)].sort_values('ratio')
    if not len(g):
        continue
    ax.plot(g['ratio'], g['H_meas'], marker='o', ms=2.4, lw=1.2,
            color=AC[a], label=rf'$\alpha={a}$')
    f = g[(g['H_meas'] > 3e-4) & (g['H_meas'] < 0.12)]
    if len(f) >= 4:
        slopes[a] = float(np.polyfit(np.log(f['ratio']),
                                     np.log(f['H_meas']), 1)[0])
        if a < 2:
            x = np.array([f['ratio'].iloc[0], f['ratio'].iloc[-1]])
            ax.plot(x, f['H_meas'].iloc[0] * (x / f['ratio'].iloc[0]) ** a,
                    ':', lw=0.9, color=AC[a], alpha=0.85)
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel(r'$\sigma/\Delta$', fontsize=9.5)
ax.set_ylabel(r'$\mathcal{H}$', fontsize=9.5)
ax.set_title(r'(a) $N=2$', fontsize=10, fontweight='bold')
ax.legend(fontsize=7, loc='lower right', framealpha=0.9)
ax.tick_params(labelsize=8)
ax.grid(alpha=0.2, which='both', lw=0.35)

# ---- (b) Eq. (2) --------------------------------------------------------
ax = axes[1]
W = D[(D['H_meas'] > H_LO) & (D['H_meas'] < H_HI) &
      (D['H_pred'] > H_LO) & (D['H_pred'] < H_HI)].copy()
for (cls, a), g in W.groupby(['cls', 'alpha']):
    ax.scatter(g['H_pred'], g['H_meas'], s=7, alpha=0.4,
               color=AC.get(a, 'k'), marker=MK.get(cls, 'o'),
               edgecolors='none')
xx = np.geomspace(W['H_pred'].min(), W['H_pred'].max(), 40)
ax.plot(xx, xx, 'k--', lw=1.1, alpha=0.85)
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel(r'$2\sum_i\Pr[Z_i-Z_1>\Delta_i]$', fontsize=9.5)
ax.set_ylabel(r'$\mathcal{H}$ measured', fontsize=9.5)
ax.set_title(r'(b) Eq. (2), all classes', fontsize=10, fontweight='bold')
hs = [plt.Line2D([], [], marker=MK[c], ls='', color='gray', ms=4,
                 label=c) for c in MK]
hs += [plt.Line2D([], [], marker='o', ls='', color=AC[a], ms=4,
                  label=rf'$\alpha={a}$') for a in ALPHAS]
ax.legend(handles=hs, fontsize=5.6, ncol=2, loc='upper left',
          framealpha=0.92)
ax.tick_params(labelsize=8)
ax.grid(alpha=0.2, which='both', lw=0.35)

plt.tight_layout()
plt.savefig('figS1.png', dpi=300, bbox_inches='tight')
plt.close()
print('saved figS1.png')

# ======================================================== numbers
print('\n--- numbers for the text ---')
print('N=2 slopes:', {k: round(v, 3) for k, v in slopes.items()})
dev = (E['corrected'] - E['pred']).abs()
print(f'exponents: n={len(E)}  mean|dev|={dev.mean():.3f}  max={dev.max():.3f}')
if len(fr) > 2:
    print(f'Frechet Pearson = {np.corrcoef(fr["pred"], fr["corrected"])[0,1]:.3f}')
diag = E[np.isclose(E['pred'], 0.0)]
if len(diag):
    print('alpha=beta cells:', [round(v, 3) for v in diag['corrected']])
print(f'\nEq.(2) window [{H_LO}, {H_HI}] on BOTH axes: {len(W)} points')
r = W['H_meas'] / W['H_pred']
print(f'  median ratio overall = {np.median(r):.3f}')
ed = np.geomspace(W['H_pred'].min() * 0.99, W['H_pred'].max() * 1.01, 5)
for lo_, hi_ in zip(ed[:-1], ed[1:]):
    b = W[(W['H_pred'] >= lo_) & (W['H_pred'] < hi_)]
    if len(b) >= 5:
        print(f'  H_pred {lo_:.3g}-{hi_:.3g}: n={len(b):>4}  '
              f'median ratio {np.median(b["H_meas"]/b["H_pred"]):.3f}')
print('\n  by class, over a range of windows (for the Supplement):')
print(f"    {'window':>10}" + ''.join(f'{c:>10}' for c in MK))
for hi_ in [0.30, 0.20, 0.15, 0.10, 0.06, 0.03]:
    Wc = D[(D['H_meas'] > H_LO) & (D['H_meas'] < hi_) &
           (D['H_pred'] > H_LO) & (D['H_pred'] < hi_)]
    print(f'    H < {hi_:<6.2f}', end='')
    for c in MK:
        gg = Wc[Wc['cls'] == c]
        v = np.median(gg['H_meas'] / gg['H_pred']) if len(gg) else np.nan
        print(f'{v:>10.3f}', end='')
    print()
if 'err' in E.columns:
    z = (E['corrected'] - E['pred']).abs() / E['err'].replace(0, np.nan)
    print(f"\n  theta error bars: median {E['err'].median():.3f}; "
          f"{(z < 2).sum()}/{z.notna().sum()} cells within 2 sigma")
    print('  (residual offsets are finite-N, see the convergence study)')
