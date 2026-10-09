"""Смешанная ООС с двойным интегратором и прямой подачей: влияние частоты стыка путей ООС на требуемый Зобель."""
import numpy as np, itertools
import double_blend as db
from double_blend import f, s, i_, lad, loads
K=12*20e3/(20e3+100); tau=1e-6; R3=4.99e3
Zi=db.Zint(1.8e-9,33e3,22e-9,1.0e6)
def H(R,Rz,Cz,L=6.8e-6,C=470e-9):
    Yp=s*C+(0 if Rz is None else 1/(Rz+1/(s*Cz)))+(0 if R is None else 2/R); Zp=1/Yp; return Zp/(s*L+0.02+Zp)
def evaluate(Cp,Cpre,Rz,Cz,Cdi,Cm,Rff=3.0e3,Cl=56e-12):
    Ypost=lad([1/(s*1e-6)+20e3,20e3],[Cp]); Ypre=lad([1/(s*Cpre)+13.3e3,13.3e3,13.3e3],[Cl,Cl])
    I1=db.Yin(Cdi,Cm); Iff=lad([1/(s*10e-6)+Rff/2,Rff/2],[3.3e-9])
    Gn=abs(I1[i_(1e3)]/Ypost[i_(1e3)]); out=[]
    for R,lab in loads:
        h=H(R,Rz,Cz); e=np.exp(-s*tau); T=K*e*Zi*(Ypost*h+Ypre)
        Vout=h*K*e*(Zi*I1+R3*Iff)/(1+T); cl=20*np.log10(abs(Vout)/Gn)
        m=abs(T); ph=np.unwrap(np.angle(T))*180/np.pi
        idx=np.where((m[:-1]>=1)&(m[1:]<1))[0]
        if not len(idx): return None
        ic=idx[-1]; pms=[180+ph[k] for k in idx if f[k]>1e3]
        j=np.where((f[:-1]>f[ic])&(ph[:-1]>-180)&(ph[1:]<=-180))[0]
        gm=-20*np.log10(m[j[0]]) if len(j) else 99
        out.append(dict(lab=lab,pm=min(pms),gm=gm,cl=cl,T=T))
    return out
def summarize(o):
    band=(f>=20)&(f<=2e4)
    return dict(pm=min(x['pm'] for x in o),gm=min(x['gm'] for x in o),
                dev=max(np.abs(x['cl'][band]).max() for x in o),
                pk=max(x['cl'][(f>2e4)&(f<3e5)].max() for x in o),
                T1k=min(20*np.log10(abs(1+x['T'][i_(1e3)])) for x in o),
                T10k=min(20*np.log10(abs(1+x['T'][i_(1e4)])) for x in o),
                T20k=min(20*np.log10(abs(1+x['T'][i_(2e4)])) for x in o))
if __name__=="__main__":
    zobels=[(3.3,2.2e-6),(3.3,1e-6),(3.9,680e-9),(3.9,470e-9),(4.7,330e-9),(1.0,220e-9),(None,None)]
    blends=[("10,6 кГц (как сейчас)",1.5e-9,390e-12),("7 кГц",2.2e-9,560e-12),("5 кГц",3.3e-9,820e-12),("3,5 кГц",4.7e-9,1.2e-9)]
    for bname,Cp,Cpre in blends:
        print("Стык путей ООС", bname)
        for Rz,Cz in zobels:
            best=None
            for Cdi,Cm in itertools.product((2.2e-9,3.3e-9,4.7e-9,6.8e-9,10e-9),(1.5e-9,2.2e-9,3.3e-9,4.7e-9,6.8e-9,10e-9)):
                o=evaluate(Cp,Cpre,Rz,Cz,Cdi,Cm)
                if o is None: continue
                sm=summarize(o)
                if best is None or sm['dev']<best[0]['dev']: best=(sm,Cdi,Cm)
            name="без Зобеля" if Rz is None else f"{Rz} Ом/{Cz*1e9:.0f} нФ"
            if best is None: print(f"   {name:13s}: неустойчиво"); continue
            sm,Cdi,Cm=best
            print(f"   {name:13s}: запас {sm['pm']:.0f}°/{sm['gm']:.1f} дБ | АЧХ ±{sm['dev']:.2f} (Cdi {Cdi*1e9:.1f}н, Cm {Cm*1e9:.1f}н), пик >20к {sm['pk']:+.1f} | подавление 1к/10к/20к {sm['T1k']:.0f}/{sm['T10k']:.0f}/{sm['T20k']:.1f} дБ")
