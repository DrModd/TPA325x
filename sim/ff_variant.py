"""Вариант с прямой подачей: U1 — интегратор (только ошибка), U3 — суммирующий FDA (×1 от U1, ×kff от входа),
Rout 470 Ом → диодный ограничитель → Cc → TPA3255 (Rвх 20 кОм)."""
import numpy as np, itertools
import double_blend as db
from double_blend import f, s, i_, H, Ypost, Ypre, Zint, loads, tau
Rout=470; K=12*20e3/(20e3+Rout)
Rsrc, Cin, Ra, Rb = 470, 22e-6, 768, 768
R3=4.99e3; Rfb_dc=40e3
Zi=Zint(1.8e-9,33e3,22e-9,1.0e6)
def paths(Cdi,Cm,Rff,Cffc=10e-6,Cfx=3.3e-9):
    # основной путь — как в double_blend (Rsrc, Cdi, Cin, Ra–Cm–Rb); прямая подача — прямо со входа:
    # Cffc → Rff/2 → (Cfx на землю) → Rff/2 → узел U3 (виртуальная земля)
    I1=db.Yin(Cdi,Cm)
    Iff=db.lad([1/(s*Cffc)+Rff/2, Rff/2],[Cfx])
    return I1, Iff
def run(Cdi,Cm,Rff,ff=True,Cfx=3.3e-9):
    I1,Iff=paths(Cdi,Cm,Rff,Cfx=Cfx); Iff=Iff if ff else 0*Iff
    Gn=abs(db.Yin(Cdi,Cm)[i_(1e3)]/Ypost[i_(1e3)])
    out=[]
    for R,lab in loads:
        h=H(R); e=np.exp(-s*tau); Yf=Ypost*h+Ypre; T=K*e*Zi*Yf
        Vpre=K*e*(Zi*I1+R3*Iff)/(1+T); Vout=h*Vpre
        Vint=Zi*(I1-Yf*Vpre); Vchip=Vint+R3*Iff
        cl=20*np.log10(abs(Vout)/Gn)
        m=abs(T); ph=np.unwrap(np.angle(T))*180/np.pi
        idx=np.where((m[:-1]>=1)&(m[1:]<1))[0]; ic=idx[-1]
        j=np.where((f[:-1]>f[ic])&(ph[:-1]>-180)&(ph[1:]<=-180))[0]
        out.append(dict(lab=lab,cl=cl,T=T,pm=180+ph[ic],gm=-20*np.log10(m[j[0]]),fc=f[ic],
                        r=abs(Vint)/abs(Vchip),lfpm=180+ph[(f<f[ic])&(m>1)].min()))
    return out
if __name__=="__main__":
    Rff=R3*K/20.0; print("K=%.2f, Rff расч. = %.0f Ом"%(K,Rff))
    band=(f>=20)&(f<=2e4)
    best=[]
    for Cdi,Cm,Rf in itertools.product((1e-9,2.2e-9,3.3e-9,4.7e-9,6.8e-9,8.2e-9),(2.2e-9,3.3e-9,4.7e-9,6.8e-9,10e-9),(2.94e3,)):
        o=run(Cdi,Cm,Rf); dev=max(np.abs(x['cl'][band]).max() for x in o)
        best.append((dev,Cdi,Cm,Rf,o))
    best.sort(key=lambda b:b[0])
    dev,Cdi,Cm,Rf,o=best[0]
    print(f"лучшее: Cdi {Cdi*1e9:.1f}n Cm {Cm*1e9:.1f}n Rff {Rf:.0f} → АЧХ ±{dev:.2f} дБ")
    for x in o:
        print("  %s: fc %.0fk PM %.0f GM %.1f НЧ-фаза-запас %.0f | АЧХ %+.2f…%+.2f @20k %+.2f | доля интегратора 1к/5к/10к/20к = %.3f/%.2f/%.2f/%.2f"%(
            x['lab'],x['fc']/1e3,x['pm'],x['gm'],x['lfpm'],x['cl'][band].min(),x['cl'][band].max(),x['cl'][i_(2e4)],
            *(x['r'][i_(v)] for v in (1e3,5e3,1e4,2e4))))
    rmax=max(x['r'][(f>=20)&(f<=2e4)].max() for x in o)
    Vfull=34*np.sqrt(2)/K/2
    print("вход чипа на полной мощности: %.2f В пик на плечо; макс. выход интегратора в полосе: %.2f В пик"%(Vfull,rmax*Vfull))
    print("без прямой подачи для сравнения:")
    for x in run(Cdi,Cm,Rf,ff=False): print("  %s АЧХ %+.2f…%+.2f"%(x['lab'],x['cl'][band].min(),x['cl'][band].max()))
