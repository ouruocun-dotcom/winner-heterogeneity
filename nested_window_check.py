import numpy as np, sys
from scipy import integrate
def stable(al,n,rng):
    U=rng.uniform(-np.pi/2,np.pi/2,n); W=rng.exponential(1.,n)
    if abs(al-1.)<1e-9: return np.tan(U)
    return np.sin(al*U)/np.cos(U)**(1./al)*(np.cos(U-al*U)/W)**((1.-al)/al)
def drift(x,c,dt):
    a,s=np.abs(x),np.sign(x); p=2.-c
    return s*(a**p+(c-2.)*dt)**(1./p)
def run(c,al,t,dt,M,snaps,seed):
    rng=np.random.default_rng(seed); x=rng.standard_normal(M)*.5
    T=int(t/dt); amp=dt**(1./al); gap=max(T//(4*snaps),1); pool=[]
    for s in range(T):
        x=drift(x+amp*stable(al,M,rng),c,dt); x=np.clip(x,-1e14,1e14)
        if s>=T-snaps*gap and (T-s)%gap==0: pool.append(np.abs(x).copy())
    return np.sort(np.concatenate(pool))
def bwin(a,lo,hi,npts=14):
    n=a.size; xs=np.logspace(np.log10(lo),np.log10(hi),npts)
    S=np.array([(n-np.searchsorted(a,x,'right'))/n for x in xs]); m=S>0
    if m.sum()<5: return np.nan
    return -np.polyfit(np.log(xs[m]),np.log(S[m]),1)[0]
WINS=[(1.3,3.),(2.,5.),(3.,9.)]
f4=lambda x:1./(np.pi*(1.-x**2+x**4))
def exact4(lo,hi,npts=14):
    xs=np.logspace(np.log10(lo),np.log10(hi),npts)
    S=np.array([2*integrate.quad(f4,x,np.inf)[0] for x in xs])
    return -np.polyfit(np.log(xs),np.log(S),1)[0]
print(f"{'c':>4}{'al':>5}{'pred':>7}  " + "  ".join(f"[{l},{h}]".rjust(11) for l,h in WINS))
for c,al in [(4.,1.,),(4.,1.5)]:
    a=run(c,al,1.5,5e-5,20000,20,seed=11+int(10*al))
    print(f"{c:4.1f}{al:5.1f}{c+al-2:7.3f}  " + "  ".join(f"{bwin(a,l,h):11.3f}" for l,h in WINS)+"   sim")
    if abs(al-1.)<1e-9:
        print(f"{'':>16}  " + "  ".join(f"{exact4(l,h):11.3f}" for l,h in WINS)+"   EXACT law")
