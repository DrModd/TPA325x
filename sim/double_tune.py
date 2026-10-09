import numpy as np, itertools
from double_blend import *
base=summary(analyze(Zint(1.0e-9,None,None,2.2e6)))
res=[]
for Cd,Rt,Cx,Rdc in itertools.product((1.8e-9,2.0e-9,2.2e-9,2.4e-9),(6.8e3,10e3,15e3,22e3,33e3,47e3),(10e-9,22e-9,47e-9),(1e6,2.2e6)):
    o=analyze(Zint(Cd,Rt,Cx,Rdc)); sm=summary(o)
    if sm['pm']>=48 and sm['gm']>=10 and sm['lfpm']>=35 and sm['T20k']>=base['T20k']-0.7 and sm['T10k']>=base['T10k']-0.3:
        res.append((sm['T1k'],Cd,Rt,Cx,Rdc,sm))
res.sort(key=lambda r:-r[0])
for r in res[:10]:
    sc,Cd,Rt,Cx,Rdc,sm=r
    fz=1/(4*np.pi*Rt*Cd)
    print(f"Cd {Cd*1e9:.1f}n Rt {Rt/1e3:.0f}k (fz {fz/1e3:.1f} кГц) Cx {Cx*1e9:.0f}n Rdc {Rdc/1e6:.1f}M |",{k:round(float(v),1) for k,v in sm.items()})
print(len(res))
