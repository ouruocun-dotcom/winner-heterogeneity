"""
Figure 1: Theoretical predictions for winner heterogeneity theory.

(a) Winning probability P_1 = logistic(ln(w1/w2)/gamma) for different gamma.
(b) Between-replicate distance d = sqrt(2*P*(1-P)) vs competitive asymmetry.

Output: fig1_theory.png
Runtime: < 1 second
Dependencies: numpy, matplotlib (standard in Colab)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ================================================================
# Panel (a): Logistic winning probability
# ================================================================
fig, axes = plt.subplots(1, 2, figsize=(7, 3.2))

ax = axes[0]
log_ratio = np.linspace(-3, 3, 200)
for gamma, color, ls in [(0.5, '#2166ac', '-'), (1.0, '#4393c3', '--'),
                          (2.0, '#d6604d', '-.'), (5.0, '#b2182b', ':')]:
    P1 = 1 / (1 + np.exp(-log_ratio / gamma))
    ax.plot(log_ratio, P1, color=color, ls=ls, lw=1.8,
            label=rf'$\gamma={gamma}$')

ax.axhline(0.5, color='gray', lw=0.5, ls=':')
ax.axvline(0, color='gray', lw=0.5, ls=':')
ax.set_xlabel(r'$\ln(w_1/w_2)$', fontsize=10)
ax.set_ylabel(r'$P_1$', fontsize=10)
ax.set_title(r'(a) Winning probability', fontsize=10, fontweight='bold')
ax.legend(fontsize=7.5, loc='lower right', framealpha=0.9)
ax.set_xlim(-3, 3)
ax.set_ylim(-0.02, 1.02)
ax.tick_params(labelsize=8)

# ================================================================
# Panel (b): d vs asymmetry for different gamma
# ================================================================
ax = axes[1]
asym = np.linspace(0, 0.5, 200)
P = 0.5 + asym
d_theory = np.sqrt(2 * P * (1 - P))

for gamma, color, ls in [(0.5, '#2166ac', '-'), (1.0, '#4393c3', '--'),
                          (2.0, '#d6604d', '-.'), (5.0, '#b2182b', ':')]:
    dw_vals = np.linspace(0, 2, 200)
    P1_vals = 1 / (1 + np.exp(-dw_vals / gamma))
    asym_vals = np.abs(P1_vals - 0.5)
    d_vals = np.sqrt(2 * P1_vals * (1 - P1_vals))
    ax.plot(asym_vals, d_vals, color=color, ls=ls, lw=1.8)

ax.fill_between(asym, d_theory, alpha=0.08, color='gray')
ax.plot(asym, d_theory, 'k-', lw=2, alpha=0.4, label=r'$d=\sqrt{2P(1-P)}$')

ax.set_xlabel(r'Competitive asymmetry $|P_1 - 1/2|$', fontsize=10)
ax.set_ylabel(r'Between-replicate distance $d$', fontsize=10)
ax.set_title(r'(b) Replicate divergence', fontsize=10, fontweight='bold')
ax.set_xlim(0, 0.5)
ax.set_ylim(0, 0.75)
ax.tick_params(labelsize=8)
ax.legend(fontsize=8, loc='upper right')

plt.tight_layout()
plt.savefig('fig1_theory.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: fig1_theory.png")
