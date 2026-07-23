"""
Supplemental Material Figures:

  Fig S1: Bradley-Terry pairwise-to-trio prediction (failure analysis)
  Fig S2: N=3 basin structure comparison with N=2

Requires data files in ./data/:
    meroz2021_counts.xlsx

Output: figS1_bradley_terry.png, figS2_n3_basins.png
Runtime: ~2 minutes
Dependencies: numpy, matplotlib, pandas, scipy, openpyxl
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from itertools import combinations
from collections import Counter
from scipy.stats import spearmanr

# ================================================================
# HELPER: deterministic replicator to find winner
# ================================================================
def run_to_win(x0, w, gamma, dt=0.05, max_steps=1000):
    x = x0.copy()
    for _ in range(max_steps):
        fit = w * x**gamma
        fbar = np.dot(x, fit)
        x = x + dt * x * (fit - fbar)
        x = np.maximum(x, 0)
        s = x.sum()
        if s > 0:
            x = x / s
        if np.max(x) > 0.99:
            return np.argmax(x)
    return np.argmax(x)

# ================================================================
# LOAD BACTERIAL DATA
# ================================================================
print("Loading data...")
df_bac = pd.read_excel('data/meroz2021_counts.xlsx')
sp_all = ['Ea', 'Pa', 'Pch', 'Pci', 'Pv', 'Pf', 'Sm', 'Ab',
          'H77', 'H79', 'H82', 'H97', 'Fj', 'IN63', 'IN65', 'IN72']

# Extract pairwise outcomes
pair_outcomes = {}
pairs_df = df_bac[df_bac['sample_kind'] == 'Pair']
for sample in sorted(pairs_df['sample'].unique()):
    sub = pairs_df[(pairs_df['sample'] == sample) &
                   (pairs_df['transfer'] == 38)]
    sp = [s for s in sp_all
          if pairs_df[pairs_df['sample'] == sample][s].sum() > 0]
    if len(sp) != 2:
        continue
    fracs_sp0 = []
    for _, row in sub.iterrows():
        c = np.array([row[sp[0]], row[sp[1]]], dtype=float)
        t = c.sum()
        if t > 0:
            fracs_sp0.append(c[0] / t)
    if len(fracs_sp0) < 2:
        continue
    mf = np.mean(fracs_sp0)
    pair_outcomes[(sp[0], sp[1])] = mf
    pair_outcomes[(sp[1], sp[0])] = 1 - mf

# Bradley-Terry fit
all_sp_bt = sorted(set(s for p in pair_outcomes for s in p))
sp_idx = {s: i for i, s in enumerate(all_sp_bt)}
n_sp_bt = len(all_sp_bt)
obs = [(s1, s2, p) for (s1, s2), p in pair_outcomes.items()
       if s1 < s2 and 0.01 < p < 0.99]

GAMMA_BT = 2.0
A_bt, b_bt = [], []
for s1, s2, p in obs:
    row = np.zeros(n_sp_bt)
    row[sp_idx[s1]] = 1
    row[sp_idx[s2]] = -1
    A_bt.append(row)
    b_bt.append(GAMMA_BT * np.log(p / (1 - p)))
A_bt = np.vstack([np.array(A_bt), np.ones(n_sp_bt)])
b_bt = np.append(np.array(b_bt), 0)
log_w, _, _, _ = np.linalg.lstsq(A_bt, b_bt, rcond=None)
w_dict = {all_sp_bt[i]: float(np.exp(log_w[i]))
          for i in range(n_sp_bt)}

# Predict trio outcomes
trios_df = df_bac[df_bac['sample_kind'] == 'Trio']
bt_het_obs, bt_het_pred = [], []
bt_dist_obs, bt_dist_pred = [], []

for sample in sorted(trios_df['sample'].unique()):
    sub = trios_df[trios_df['sample'] == sample]
    sp = [s for s in sp_all if sub[s].sum() > 0]
    if not all(s in w_dict for s in sp):
        continue
    final = sub[sub['transfer'] == 38]
    if len(final) < 3:
        continue
    fracs, winners = [], []
    for _, row in final.iterrows():
        c = np.array([row[s] for s in sp], dtype=float)
        t = c.sum()
        if t > 0:
            fracs.append(c / t)
            winners.append(sp[np.argmax(c / t)])
    if len(fracs) < 3:
        continue
    fa = np.array(fracs)
    K = len(fa)
    dists = [np.linalg.norm(fa[i] - fa[j])
             for i, j in combinations(range(K), 2)]
    P_obs = np.array([winners.count(s) / K for s in set(winners)])
    het_obs = 1 - np.sum(P_obs**2)

    w_vals = np.array([w_dict[s] for s in sp])
    rng = np.random.default_rng(hash(sample) % 2**31)
    wins = np.zeros(len(sp))
    for _ in range(500):
        x0 = rng.dirichlet(np.ones(len(sp)))
        wins[run_to_win(x0, w_vals, GAMMA_BT)] += 1
    P_pred = wins / 500
    het_pred = 1 - np.sum(P_pred**2)

    bt_het_obs.append(het_obs)
    bt_het_pred.append(het_pred)
    bt_dist_obs.append(np.mean(dists))
    bt_dist_pred.append(np.sqrt(2) * het_pred)

r_bt_h, p_bt_h = spearmanr(bt_het_obs, bt_het_pred)
r_bt_d, p_bt_d = spearmanr(bt_dist_obs, bt_dist_pred)

# ================================================================
# FIG S1: Bradley-Terry failure
# ================================================================
print("Generating Fig S1...")
fig, axes = plt.subplots(1, 2, figsize=(7, 3.2))

ax = axes[0]
ax.scatter(bt_het_pred, bt_het_obs, s=25, c='#d6604d',
           alpha=0.7, edgecolors='k', lw=0.3)
ax.plot([0, 0.7], [0, 0.7], 'k--', lw=1, alpha=0.5,
        label='Perfect prediction')
ax.set_xlabel(r'Predicted $\mathcal{H}$ (Bradley-Terry)', fontsize=10)
ax.set_ylabel(r'Observed $\mathcal{H}$', fontsize=10)
ax.set_title(f'(a) $r_s$={r_bt_h:.2f}, $p$={p_bt_h:.3f}',
             fontsize=10)
ax.legend(fontsize=8)
ax.set_xlim(-0.03, 0.72)
ax.set_ylim(-0.03, 0.72)
ax.set_aspect('equal')
ax.tick_params(labelsize=8)

ax = axes[1]
ax.scatter(bt_dist_pred, bt_dist_obs, s=25, c='#2166ac',
           alpha=0.7, edgecolors='k', lw=0.3)
ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5,
        label='Perfect prediction')
ax.set_xlabel(r'Predicted $d$ (from BT $\mathcal{H}$)', fontsize=10)
ax.set_ylabel(r'Observed $d$', fontsize=10)
ax.set_title(f'(b) $r_s$={r_bt_d:.2f}, $p$={p_bt_d:.3f}',
             fontsize=10)
ax.legend(fontsize=8)
ax.set_xlim(-0.03, 1.05)
ax.set_ylim(-0.03, 1.05)
ax.set_aspect('equal')
ax.tick_params(labelsize=8)

plt.tight_layout()
plt.savefig('figS1_bradley_terry.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: figS1_bradley_terry.png")

# ================================================================
# FIG S2: N=3 basin structure
# ================================================================
print("Generating Fig S2...")
fig, axes = plt.subplots(1, 2, figsize=(7, 3.2))

dws = np.arange(0, 0.96, 0.05)
P1_n3, het_n3 = [], []
P1_n2, het_n2 = [], []

for dw in dws:
    # N=3
    w3 = np.array([1 + dw, 1.0, max(1 - dw, 0.05)])
    rng = np.random.default_rng(42)
    wins3 = np.zeros(3)
    for _ in range(1000):
        x0 = rng.dirichlet(np.ones(3))
        wins3[run_to_win(x0, w3, 2.0)] += 1
    P3 = wins3 / 1000
    P1_n3.append(P3[0])
    het_n3.append(1 - np.sum(P3**2))
    # N=2 exact
    w1, w2 = 1 + dw, max(1 - dw, 0.01)
    xs = 1 / (1 + (w1 / w2)**(1 / 2.0))
    p1 = 1 - xs
    P1_n2.append(p1)
    het_n2.append(2 * p1 * (1 - p1))

ax = axes[0]
ax.plot(dws, P1_n2, 'b-', lw=2, label=r'$N=2$ (exact)')
ax.plot(dws, P1_n3, 'r--', lw=2, label=r'$N=3$ (simulation)')
ax.set_xlabel(r'Asymmetry $\delta_w$', fontsize=10)
ax.set_ylabel(r'$P_1$ (strongest species)', fontsize=10)
ax.set_title('(a) Winning probability', fontsize=10, fontweight='bold')
ax.legend(fontsize=8)
ax.axhline(1 / 3, color='gray', ls=':', lw=0.5)
ax.set_xlim(0, 0.95)
ax.set_ylim(0.3, 0.75)
ax.tick_params(labelsize=8)

ax = axes[1]
ax.plot(dws, het_n2, 'b-', lw=2, label=r'$N=2$ (exact)')
ax.plot(dws, het_n3, 'r--', lw=2, label=r'$N=3$ (simulation)')
ax.set_xlabel(r'Asymmetry $\delta_w$', fontsize=10)
ax.set_ylabel(r'Winner heterogeneity $\mathcal{H}$', fontsize=10)
ax.set_title('(b) Heterogeneity vs asymmetry', fontsize=10,
             fontweight='bold')
ax.legend(fontsize=8)
ax.axhline(2 / 3, color='gray', ls=':', lw=0.5)
ax.set_xlim(0, 0.95)
ax.set_ylim(0, 0.72)
ax.tick_params(labelsize=8)

plt.tight_layout()
plt.savefig('figS2_n3_basins.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: figS2_n3_basins.png")

# Print summary
print(f"\nBradley-Terry: r_s(het)={r_bt_h:.3f}, r_s(dist)={r_bt_d:.3f}")
acc = np.mean(
    (np.array(bt_het_pred) < 0.15) == (np.array(bt_het_obs) < 0.15))
print(f"Classification accuracy: {acc:.0%}")
