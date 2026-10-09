"""Смешанная ООС + двойной интегратор (T-цепочка Cd–Rt–Cd, ветвь Rt+Cx на землю, Rdc поперёк).
Ограничитель по образцу NC400 в покое не влияет на петлю (единицы пФ на суммирующем узле) и в модель не входит."""
import numpy as np, itertools, sys
K=12*20e3/(20e3+100); tau=1e-6; L=6.8e-6; C=470e-9; DCR=0.02; Rz,Cz=3.3,2.2e-6
f=np.logspace(0.5,6,30000); s=2j*np.pi*f; i_=lambda x:np.argmin(abs(f-x))
def H(R): Yp=s*C+1/(Rz+1/(s*Cz))+(0 if R is None else 2/R); Zp=1/Yp; return Zp/(s*L+DCR+Zp)
def lad(Zs,Cs,sv=s):
    M=(1,0,0,1)
    mul=lambda M,N:(M[0]*N[0]+M[1]*N[2],M[0]*N[1]+M[1]*N[3],M[2]*N[0]+M[3]*N[2],M[2]*N[1]+M[3]*N[3])
    for k,Z in enumerate(Zs):
        M=mul(M,(1,Z,0,1))
        if k<len(Cs): M=mul(M,(1,0,sv*Cs[k],1))
    return 1/M[1]
Ypost=lad([1/(s*1e-6)+20e3,20e3],[1.5e-9]); Ypre=lad([1/(s*390e-12)+13.3e3,13.3e3,13.3e3],[56e-12,56e-12])
def Yin(Cd_=5.6e-9,Cm_=10e-9): return lad([470,1/(s*22e-6)+768,768],[2*Cd_,Cm_])
G=abs(Yin()[i_(1e3)]/Ypost[i_(1e3)])
loads=[(4,"4 Ом"),(8,"8 Ом"),(None,"х.х.")]
def Zint(Cd,Rt,Cx,Rdc):
    if Rt is None: Zt=1/(s*Cd)
    else:
        Ysh=1/(Rt+(0 if Cx is None else 1/(s*Cx)))
        Zt=2/(s*Cd)+Ysh/(s*Cd)**2
    return 1/(1/Rdc+1/Zt)
def analyze(Zi,yin=None):
    yin=Yin() if yin is None else yin
    A=K*np.exp(-s*tau)*Zi; out=[]
    for R,lab in loads:
        h=H(R); T=A*(Ypost*h+Ypre); m=abs(T); ph=np.unwrap(np.angle(T))*180/np.pi
        # unwrap reference: make phase at highest |T| region near -90..-180
        idx=np.where((m[:-1]>=1)&(m[1:]<1))[0]; ic=idx[-1]
        pm=180+ph[ic]
        j=np.where((f[:-1]>f[ic])&(ph[:-1]>-180)&(ph[1:]<=-180))[0]; gm=-20*np.log10(m[j[0]]) if len(j) else 99
        lf=(f<f[ic])&(m>1); lfpm=180+ph[lf].min()       # "запас" на НЧ (условная устойчивость)
        cl=20*np.log10(abs(h*A*yin/(1+T))/G)
        out.append(dict(lab=lab,T=T,cl=cl,fc=f[ic],pm=pm,gm=gm,lfpm=lfpm,
            Tk=[20*np.log10(m[i_(v)]) for v in (100,1e3,5e3,1e4,2e4)]))
    return out
def summary(o):
    band=(f>=20)&(f<=2e4)
    return dict(pm=min(x['pm'] for x in o),gm=min(x['gm'] for x in o),lfpm=min(x['lfpm'] for x in o),
        dev=max(np.abs(x['cl'][band]).max() for x in o),T1k=min(x['Tk'][1] for x in o),T10k=min(x['Tk'][3] for x in o),T20k=min(x['Tk'][4] for x in o))
if __name__=="__main__":
    base=analyze(Zint(1.0e-9,None,None,2.2e6)); b=summary(base)
    print("одинарный 1 нФ:",{k:round(v,1) for k,v in b.items()})
    res=[]
    for Cd,Rt,Cx,Rdc in itertools.product((1.8e-9,2.2e-9,2.7e-9,3.3e-9),(2.2e3,3.3e3,4.7e3,6.8e3,10e3,15e3),(None,10e-9,22e-9,47e-9,100e-9),(1e6,2.2e6,4.7e6)):
        o=analyze(Zint(Cd,Rt,Cx,Rdc)); sm=summary(o)
        if sm['pm']>=50 and sm['gm']>=10 and sm['lfpm']>=25:
            res.append((sm['T1k']+0.5*sm['T10k']+sm['T20k'],Cd,Rt,Cx,Rdc,sm))
    res.sort(key=lambda r:-r[0])
    for r in res[:12]:
        sc,Cd,Rt,Cx,Rdc,sm=r
        print(f"Cd {Cd*1e9:.1f}n Rt {Rt/1e3:.1f}k Cx {'—' if Cx is None else f'{Cx*1e9:.0f}n'} Rdc {Rdc/1e6:.1f}M |",{k:round(v,1) for k,v in sm.items()})
    print(len(res))
