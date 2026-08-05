"""
Re-plot the alpha=2 consistency figure from alpha2_results.csv.
No simulation is re-run.

Paste into one Colab cell and run.  It finds alpha2_results.csv wherever it
is, so the working directory does not matter.

The point of panel (a) is that sigma*/Delta_2 is FLAT at alpha=2.  Plotting
it on a narrow axis turns a 1% wobble into visual chaos, so each curve is
normalised to its value at the smallest N and drawn on a wide log axis,
with the alpha=1 curves shown for contrast.  Flat versus plunging is then
the message of the panel.

Output: figS2.png
"""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import _figstyle
_figstyle.use()

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


path = fetch('alpha2_results.csv')

print(f'reading {path}')
R = pd.read_csv(path)
GAMMAS = sorted(R['gamma'].unique())
ALPHAS = sorted(R['alpha'].unique())
print(f'  {len(R)} rows | gamma={GAMMAS} | alpha={ALPHAS} '
      f'| N={sorted(R["N"].unique())}')
if len(GAMMAS) < 4 or R['N'].max() < 10000:
    print('  WARNING: this looks like a QUICK run, not the full sweep.')

cm = plt.get_cmap('viridis')

fig, axes = plt.subplots(1, 2, figsize=(_figstyle.COL, 1.55))

# ---------------- (a) universality at alpha=2, breakdown at alpha=1 -------
ax = axes[0]
for i, g in enumerate(GAMMAS):
    c = cm(i / max(len(GAMMAS) - 1, 1))
    for a, ls, mk, lw in [(2.0, '-', 'o', 1.8), (1.0, '--', 'v', 1.2)]:
        s = R[(R['gamma'] == g) & (np.isclose(R['alpha'], a))].sort_values('N')
        if len(s) < 2:
            continue
        y = s['ratio'].values / s['ratio'].values[0]
        ax.plot(s['N'], y, ls=ls, marker=mk, ms=2.2, lw=lw, color=c,
                label=(rf'$\gamma={g}$' if a == 2.0 else None))
ax.axhline(1.0, color='k', lw=0.8, alpha=0.5)
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_ylim(4e-3, 3)
ax.set_xlabel('$N$', fontsize=7.9)
ax.set_ylabel(r'$\sigma^*/\Delta_2$', fontsize=7.9)
ax.text(0.03, 0.22, '(a)', transform=ax.transAxes, fontsize=7.5,
        fontweight='bold', va='bottom', bbox=dict(fc='white', ec='none', alpha=0.75, pad=1.0), zorder=9)
h, l = ax.get_legend_handles_labels()
h += [plt.Line2D([], [], ls='-', marker='o', ms=2.2, color='gray',
                 label=r'$\alpha=2$'),
      plt.Line2D([], [], ls='--', marker='v', ms=2.2, color='gray',
                 label=r'$\alpha=1$')]
ax.legend(handles=h, fontsize=5.2, ncol=3, loc='lower left',
          bbox_to_anchor=(-0.02, 1.02), framealpha=0.0, borderpad=0.1,
          labelspacing=0.12, columnspacing=0.6, handlelength=0.9)
ax.grid(alpha=0.25, which='both', lw=0.4)

# ---------------- (b) all alpha at one gamma, with N^{-1/alpha} guides ----
ax = axes[1]
g0 = 1.0 if 1.0 in GAMMAS else GAMMAS[0]
for i, a in enumerate(ALPHAS):
    s = R[(R['gamma'] == g0) & (np.isclose(R['alpha'], a))].sort_values('N')
    if len(s) < 2:
        continue
    c = cm(i / max(len(ALPHAS) - 1, 1))
    y = s['ratio'].values / s['ratio'].values[0]
    ax.plot(s['N'], y, marker='s', ms=2.5, lw=1.6, color=c,
            label=rf'$\alpha={a}$')
    if a < 2:
        n = s['N'].values.astype(float)
        ax.plot(n, (n / n[0]) ** (-1.0 / a) * np.log(n) / np.log(n[0]),
                ':', lw=1.0, color=c)
ax.axhline(1.0, color='k', lw=0.8, alpha=0.5)
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel('$N$', fontsize=7.9)
ax.set_ylabel(r'$\sigma^*/\Delta_2$', fontsize=7.9)
ax.text(0.03, 0.22, '(b)', transform=ax.transAxes, fontsize=7.5,
        fontweight='bold', va='bottom', bbox=dict(fc='white', ec='none', alpha=0.75, pad=1.0), zorder=9)
ax.legend(fontsize=5.6, ncol=4, loc='lower left',
          bbox_to_anchor=(-0.02, 1.02), framealpha=0.0,
          borderpad=0.1, labelspacing=0.12, columnspacing=0.6,
          handlelength=0.9)
ax.grid(alpha=0.25, which='both', lw=0.4)

plt.subplots_adjust(left=0.155, right=0.985, bottom=0.235, top=0.845, wspace=1.05)
plt.savefig('figS2.png', dpi=300, bbox_inches='tight')
print('saved figS2.png')

print('\nflatness at alpha=2 (slope of log ratio vs log N):')
for g in GAMMAS:
    s = R[(R['gamma'] == g) & (np.isclose(R['alpha'], 2.0))].sort_values('N')
    if len(s) >= 3:
        sl = np.polyfit(np.log(s['N']), np.log(s['ratio']), 1)[0]
        rng = s['ratio'].max() / s['ratio'].min()
        print(f'  gamma={g}: slope {sl:+.4f}, total spread {rng:.2f}x')
