"""
==============================================================================
  BUILD fig1.png AND fig2.png FOR THE MANUSCRIPT
==============================================================================
Reads the three CSV files written by collapse_figure.py and re-plots them in
the two-figure layout the paper refers to.  No simulation is re-run.

  fig1.png   (a) N = 2 tail dichotomy        [from collapse_n2.csv]
             (b) measured vs predicted theta [from exponents.csv]

  fig2.png       Eq. (1) across all classes  [from collapse_data.csv]

USAGE
  run collapse_figure.py first (it writes the CSVs), then run this.

RUNTIME  a few seconds.
==============================================================================
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

H_LO, H_HI = 3e-3, 0.20
ALPHAS = [0.5, 1.0, 1.5, 2.0]
AC = {0.5: '#4575b4', 1.0: '#74add1', 1.5: '#f46d43', 2.0: '#a50026'}
MK = {'gumbel': 'o', 'frechet': '^', 'weibull': 's'}
CC = {'gumbel': '#4575b4', 'frechet': '#f46d43', 'weibull': '#1a9850'}

D2 = pd.read_csv('collapse_n2.csv')
E = pd.read_csv('exponents.csv')
D = pd.read_csv('collapse_data.csv')

# ============================================================ FIGURE 1
fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.1))

# ---- (a) N = 2 tail dichotomy
ax = axes[0]
slopes = {}
for a in ALPHAS:
    g = D2[(D2['alpha'] == a) & (D2['H_meas'] > 1e-5)].sort_values('ratio')
    if not len(g):
        continue
    ax.plot(g['ratio'], g['H_meas'], marker='o', ms=2.6, lw=1.3,
            color=AC[a], label=rf'$\alpha={a}$')
    f = g[(g['H_meas'] > 3e-4) & (g['H_meas'] < 0.12)]
    if len(f) >= 4:
        slopes[a] = float(np.polyfit(np.log(f['ratio']),
                                     np.log(f['H_meas']), 1)[0])
        if a < 2:
            x = np.array([f['ratio'].iloc[0], f['ratio'].iloc[-1]])
            ax.plot(x, f['H_meas'].iloc[0] * (x / f['ratio'].iloc[0]) ** a,
                    ':', lw=1.0, color=AC[a], alpha=0.85)
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel(r'$\sigma/\Delta$', fontsize=10)
ax.set_ylabel(r'$\mathcal{H}$', fontsize=10)
ax.set_title(r'(a) $N=2$', fontsize=10, fontweight='bold')
ax.legend(fontsize=7, loc='lower right', framealpha=0.9)
ax.tick_params(labelsize=8)
ax.grid(alpha=0.22, which='both', lw=0.4)

# ---- (b) measured vs predicted theta
ax = axes[1]
for cls, g in E.groupby('cls'):
    face = ['none' if lc else CC[cls] for lc in g['logcorr']]
    ax.scatter(g['pred'], g['corrected'], s=46, marker=MK[cls],
               facecolors=face, edgecolors=CC[cls], linewidth=1.2,
               label=cls, zorder=3)
lo = min(E['pred'].min(), E['corrected'].min()) - 0.22
hi = max(E['pred'].max(), E['corrected'].max()) + 0.22
ax.plot([lo, hi], [lo, hi], 'k--', lw=1.1, alpha=0.7, zorder=1)
ax.axhline(0, color='gray', lw=0.5); ax.axvline(0, color='gray', lw=0.5)
ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
ax.set_xlabel(r'predicted $\theta$', fontsize=10)
ax.set_ylabel(r'measured $\theta$', fontsize=10)
ax.set_title(r'(b) exponents, $\alpha<2$', fontsize=10, fontweight='bold')
ax.legend(fontsize=7, loc='upper left', framealpha=0.9)
ax.tick_params(labelsize=8)
ax.grid(alpha=0.22, lw=0.4)

plt.tight_layout()
plt.savefig('fig1.png', dpi=300, bbox_inches='tight')
plt.close()
print('saved fig1.png')

# ============================================================ FIGURE 2
W = D[(D['H_meas'] > H_LO) & (D['H_meas'] < H_HI) &
      (D['H_pred'] > H_LO)].copy()

fig, ax = plt.subplots(figsize=(3.5, 3.2))
for (cls, a), g in W.groupby(['cls', 'alpha']):
    ax.scatter(g['H_pred'], g['H_meas'], s=7, alpha=0.38,
               color=AC.get(a, 'k'), marker=MK.get(cls, 'o'),
               edgecolors='none')
xx = np.geomspace(W['H_pred'].min(), W['H_pred'].max(), 40)
ax.plot(xx, xx, 'k--', lw=1.2, alpha=0.85)
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel(r'$2\sum_i\Pr[Z_i-Z_1>\Delta_i]$', fontsize=10)
ax.set_ylabel(r'$\mathcal{H}$ measured', fontsize=10)
hs = [plt.Line2D([], [], marker=MK[c], ls='', color='gray', ms=4.5,
                 label=c) for c in MK]
hs += [plt.Line2D([], [], marker='o', ls='', color=AC[a], ms=4.5,
                  label=rf'$\alpha={a}$') for a in ALPHAS]
ax.legend(handles=hs, fontsize=6, ncol=2, loc='upper left', framealpha=0.9)
ax.tick_params(labelsize=8)
ax.grid(alpha=0.22, which='both', lw=0.4)
plt.tight_layout()
plt.savefig('fig2.png', dpi=300, bbox_inches='tight')
plt.close()
print('saved fig2.png')

# ============================================================ numbers
print('\n--- numbers quoted in the text ---')
print('N=2 slopes:', {k: round(v, 3) for k, v in slopes.items()})
dev = (E['corrected'] - E['pred']).abs()
print(f'exponents: n={len(E)}  mean|dev|={dev.mean():.3f}  max={dev.max():.3f}')
fr = E[E['cls'] == 'frechet']
if len(fr) > 2:
    print(f'Frechet Pearson = {np.corrcoef(fr["pred"], fr["corrected"])[0,1]:.3f}')
diag = E[np.isclose(E['pred'], 0.0)]
if len(diag):
    print(f'alpha=beta cells: theta = '
          f'{[round(v,3) for v in diag["corrected"]]}')
print(f'Eq.(1) window: {len(W)} points')
ed = np.geomspace(W['H_pred'].min() * 0.99, W['H_pred'].max() * 1.01, 6)
for lo_, hi_ in zip(ed[:-1], ed[1:]):
    b = W[(W['H_pred'] >= lo_) & (W['H_pred'] < hi_)]
    if len(b) >= 5:
        print(f'  H_pred {lo_:.3g}-{hi_:.3g}: n={len(b):>4}  '
              f'median ratio {np.median(b["H_meas"]/b["H_pred"]):.3f}')
