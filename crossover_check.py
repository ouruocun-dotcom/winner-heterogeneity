"""
Verification of the exact crossover in the Frechet domain (SM Sec. S6).

  E[H] -> 1 - int_0^inf exp(-u - 2 eta_N u^{alpha/beta}) du
  eta_N = B A^{-alpha/beta} N^{1-alpha/beta},  B = C_alpha sigma^alpha

and on the critical line alpha = beta,  E[H] -> 2B/(A+2B).

Values are drawn with Pr(S>x) = A x^{-beta} exactly; two independent
realizations share the same values, and H is the frequency with which they
end with different winners.  Produces Tables S3 and S4.  Runtime ~10 min.
"""
import numpy as np
from scipy.special import gamma as G
from scipy.integrate import quad


def stable(al, size, rng):
    U = rng.uniform(-np.pi/2, np.pi/2, size)
    W = rng.exponential(1.0, size)
    if abs(al - 1.0) < 1e-9:
        return np.tan(U)
    return (np.sin(al*U)/np.cos(U)**(1./al)
            * (np.cos(U - al*U)/W)**((1.-al)/al))


def H_mean(N, al, be, A, sig, rng, reps=24000):
    """ensemble-averaged H: two realizations share the values"""
    diff = tot = 0
    B = max(1, min(200, 4_000_000//N))
    for _ in range(max(1, reps//B)):
        S = (A/rng.uniform(0, 1, (B, N)))**(1./be)
        w1 = np.argmax(S + sig*stable(al, (B, N), rng), axis=1)
        w2 = np.argmax(S + sig*stable(al, (B, N), rng), axis=1)
        diff += np.count_nonzero(w1 != w2)
        tot += B
    return diff/tot


def H_cross(eta, r):
    return 1 - quad(lambda u: np.exp(-u - 2*eta*u**r), 0, np.inf, limit=200)[0]


rng = np.random.default_rng(17)

print("Table S3 -- full crossover, alpha != beta  (A=1, sigma=1)")
print(f"{'alpha':>6}{'beta':>6}{'N':>8}{'eta':>10}{'measured':>10}{'Eq.(10)':>10}{'ratio':>7}")
for al, be in [(1.0, 1.5), (1.0, 2.0), (0.5, 1.5), (1.5, 1.0)]:
    Ca = G(al)*np.sin(np.pi*al/2)/np.pi
    r = al/be
    for N in (100, 1000, 10000):
        eta = Ca*N**(1 - r)
        h = H_mean(N, al, be, 1.0, 1.0, rng)
        hc = H_cross(eta, r)
        print(f"{al:6.1f}{be:6.1f}{N:8d}{eta:10.4f}{h:10.4f}{hc:10.4f}{h/hc:7.3f}", flush=True)

print("\nTable S4 -- critical line alpha = beta")
print(f"{'alpha':>6}{'A':>5}{'sigma':>7}{'N':>8}{'measured':>10}{'Eq.(11)':>10}{'ratio':>7}")
for al, A, sig, N in [(1.0, 1, 1.0, 1000), (1.0, 1, 0.3, 1000), (1.0, 4, 1.0, 1000),
                      (1.5, 1, 1.0, 20000), (1.5, 4, 1.0, 20000)]:
    Ca = G(al)*np.sin(np.pi*al/2)/np.pi
    B = Ca*sig**al
    pred = 2*B/(A + 2*B)
    h = H_mean(N, al, al, A, sig, rng)
    print(f"{al:6.1f}{A:5d}{sig:7.2f}{N:8d}{h:10.4f}{pred:10.4f}{h/pred:7.3f}", flush=True)
