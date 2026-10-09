import numpy as np, itertools, lf_check as L
f,s,i_=L.f,L.s,L.i_
def metrics(Cfb,Cc,Rdc,Cin):
    Ypost=L.lad([1/(s*Cfb)+20e3,20e3],[1.5e-9]); Ypre=L.lad([1/(s*390e-12)+13.3e3,13.3e3,13.3e3],[56e-12,56e-12])
    Yin=L.lad([470,1/(s*Cin)+768,768],[2*6.8e-9,3.3e-9])
    Zt=2/(s*1.8e-9)+(1/(33e3+1/(s*22e-9)))/(s*1.8e-9)**2; Zi=1/(1/Rdc+1/Zt)
    HPc=s*Cc*L.Rchip/(1+s*Cc*L.Rchip)
    h=L.H(4); T=L.K*np.exp(-s*L.tau)*Zi*HPc*(Ypost*h+Ypre)
    cl=abs(h*L.K*np.exp(-s*L.tau)*Zi*HPc*Yin/(1+T)); cl=20*np.log10(cl/cl[i_(1e3)])
    m=abs(T); ph=np.angle(T,deg=True)
    idx=np.where((f[:-1]<20)&(m[:-1]<1)&(m[1:]>=1))[0]
    pmu=180-abs(ph[idx[0]]) if len(idx) else 180
    lf=f<20; pk=cl[lf].max()
    return pmu,pk,cl[i_(20)],cl[i_(10)],20*np.log10(m[i_(20)]),20*np.log10(m[i_(100)])
res=[]
for Cfb,Cc,Rdc,Cin in itertools.product((0.1e-6,0.15e-6,0.22e-6,0.33e-6,0.47e-6,1e-6),(10e-6,22e-6,47e-6,100e-6),(1e6,470e3,220e3),(2.2e-6,3.3e-6,4.7e-6,6.8e-6,10e-6,22e-6)):
    pm,pk,c20,c10,t20,t100=metrics(Cfb,Cc,Rdc,Cin)
    if pm>=50 and pk<=0.5 and c20>-0.6:
        res.append((-(t20)+0.1*Cc*1e6,Cfb,Cc,Rdc,Cin,pm,pk,c20,c10,t20,t100))
res.sort(key=lambda r:r[0])
for r in res[:12]:
    _,Cfb,Cc,Rdc,Cin,pm,pk,c20,c10,t20,t100=r
    print(f"Cfb {Cfb*1e6:.2f}µ Cc {Cc*1e6:.0f}µ Rdc {Rdc/1e3:.0f}k Cin {Cin*1e6:.1f}µ | ЗФ на НЧ-срезе {pm:.0f}°, пик {pk:+.2f} дБ, АЧХ 20 Гц {c20:+.2f}, 10 Гц {c10:+.2f} | |T| 20 Гц {t20:.0f} дБ, 100 Гц {t100:.0f} дБ")
print(len(res))
