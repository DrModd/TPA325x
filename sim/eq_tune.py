import numpy as np, itertools
from double_blend import *
Zi=Zint(1.8e-9,33e3,22e-9,1.0e6)
best=[]
for Cd_,Cm_ in itertools.product((3.3e-9,3.9e-9,4.7e-9,5.6e-9,6.8e-9,8.2e-9),(4.7e-9,6.8e-9,8.2e-9,10e-9,12e-9,15e-9)):
    yi=Yin(Cd_,Cm_); g=abs(yi[i_(1e3)]/Ypost[i_(1e3)])
    o=analyze(Zi,yi); band=(f>=20)&(f<=2e4)
    cls=[x['cl']+20*np.log10(G/g) for x in o]
    dev=max(np.abs(c[band]).max() for c in cls); at20=[c[i_(2e4)] for c in cls]
    bw=[f[np.where((f>1e3)&(c<-3))[0][0]] for c in cls]
    best.append((dev,Cd_,Cm_,at20,bw))
best.sort(key=lambda b:b[0])
for b in best[:5]: print(f"Cd {b[1]*1e9:.1f}n Cm {b[2]*1e9:.1f}n dev {b[0]:.2f} @20k {['%+.2f'%v for v in b[3]]} -3dB {['%.0fk'%(v/1e3) for v in b[4]]}")
