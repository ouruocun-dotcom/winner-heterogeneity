"""
Verification of the prefactor in Eq.(1): the leader's own downward jump is a
single shared event, not one per challenger.

With gaps PRESCRIBED rather than drawn from a value distribution, H can be
measured exactly by simulation and compared against both forms:

  independent pairs   H = 4 C_a sigma^a T_a
                      (treats the N-1 differences Z_i - Z_1 as independent,
                       which counts the leader's jump N times)

  Eq.(1)              H = 2 C_a sigma^a [ T_a + Delta_2^{-a} ]

Produces Table S1 of the Supplemental Material.  Runtime ~5 min.
"""
import numpy as np
from scipy.special import gamma as G


def stable(al, size, rng):
    U = rng.uniform(-np.pi/2, np.pi/2, size)
    W = rng.exponential(1.0, size)
    if abs(al - 1.0) < 1e-9:
        return np.tan(U)
    return (np.sin(al*U)/np.cos(U)**(1./al)
            * (np.cos(U - al*U)/W)**((1.-al)/al))


def H_exact(gaps, al, rng, M=3_000_000, chunk=150_000):
    """H = 1 - sum_i P_i^2 for prescribed gaps, leader at 0"""
    n = len(gaps) + 1
    s = np.concatenate([[0.0], -gaps])
    cnt = np.zeros(n)
    for _ in range(M // chunk):
        Z = stable(al, (chunk, n), rng)
        cnt += np.bincount(np.argmax(s[None, :] + Z, axis=1), minlength=n)
    P = cnt/cnt.sum()
    return 1.0 - np.sum(P**2)


rng = np.random.default_rng(7)
N = 60
i = np.arange(2, N + 1)
SHAPES = {"Gumbel":  np.log(i),                 # b ln i
          "Frechet": 1 - i**(-1/1.5),           # N^{1/b}(1 - i^{-1/b})
          "Weibull": i**(1/0.7) - 1}            # N^{-1/p}(i^{1/p} - 1)

print(f"{'class':>9}{'alpha':>6}{'D_2':>6}{'H exact':>10}"
      f"{'pairs':>9}{'ratio':>7}{'Eq.(1)':>10}{'ratio':>7}")
for al in (1.0, 1.5):
    Ca = G(al)*np.sin(np.pi*al/2)/np.pi
    for name, sh in SHAPES.items():
        for sc in (40.0, 150.0):
            gaps = sc*sh/sh[0]                  # normalise so Delta_2 = sc
            H = H_exact(gaps, al, rng)
            T = (gaps**(-al)).sum()
            pairs = 4*Ca*T
            eq1 = 2*Ca*(T + gaps.min()**(-al))
            print(f"{name:>9}{al:6.1f}{sc:6.0f}{H:10.6f}"
                  f"{pairs:9.6f}{H/pairs:7.3f}{eq1:10.6f}{H/eq1:7.3f}", flush=True)
