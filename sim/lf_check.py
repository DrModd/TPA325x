"""Инфранизкие частоты: петля с учётом ФВЧ Cc·Rвх(TPA3255) внутри петли и Cfb в цепи ООС."""
import numpy as np
import double_blend as db
K=12*20e3/(20e3+100); Rchip=20e3; tau=1e-6
f=np.logspace(-3,5,40000); s=2j*np.pi*f; i_=lambda x:np.argmin(abs(f-x))
def lad(Zs,Cs):
    M=(1,0,0,1)
    mul=lambda M,N:(M[0]*N[0]+M[1]*N[2],M[0]*N[1]+M[1]*N[3],M[2]*N[0]+M[3]*N[2],M[2]*N[1]+M[3]*N[3])
    for k_,Z in enumerate(Zs):
        M=mul(M,(1,Z,0,1))
        if k_<len(Cs): M=mul(M,(1,0,s*Cs[k_],1))
    return 1/M[1]
def H(R,L=6.8e-6,C=470e-9,Rz=3.3,Cz=2.2e-6):
    Yp=s*C+1/(Rz+1/(s*Cz))+(0 if R is None else 2/R); Zp=1/Yp; return Zp/(s*L+0.02+Zp)
def run(Cfb=1e-6,Cc=10e-6,Rdc=1e6,Cin=22e-6,Cd=1.8e-9,Rt=33e3,Cx=22e-9,label=""):
    Ypost=lad([1/(s*Cfb)+20e3,20e3],[1.5e-9]); Ypre=lad([1/(s*390e-12)+13.3e3,13.3e3,13.3e3],[56e-12,56e-12])
    Yin=lad([470,1/(s*Cin)+768,768],[2*6.8e-9,3.3e-9])
    Zt=2/(s*Cd)+(1/(Rt+1/(s*Cx)))/(s*Cd)**2; Zi=1/(1/Rdc+1/Zt)
    HPc=s*Cc*Rchip/(1+s*Cc*Rchip)
    res=[]
    for R in (4,8):
        h=H(R); T=K*np.exp(-s*tau)*Zi*HPc*(Ypost*h+Ypre)
        cl=abs(h*K*np.exp(-s*tau)*Zi*HPc*Yin/(1+T))
        cl=20*np.log10(cl/cl[i_(1e3)])
        lf=f<20
        m=abs(T); ph=np.angle(T,deg=True)
        # LF unity crossing
        idx=np.where((f[:-1]<20)&(m[:-1]<1)&(m[1:]>=1))[0]
        fu=f[idx[0]] if len(idx) else np.nan; pmu=180-abs(ph[idx[0]]) if len(idx) else np.nan
        pk=cl[lf].max(); fpk=f[lf][np.argmax(cl[lf])]
        res.append((R,fu,pmu,pk,fpk,cl[i_(20)],20*np.log10(m[i_(20)])))
    print(label)
    for R,fu,pmu,pk,fpk,c20,t20 in res:
        print(f"   {R} Ом: НЧ-срез петли {fu:.3f} Гц, запас по фазе там {pmu:.0f}°; пик АЧХ ниже 20 Гц {pk:+.1f} дБ на {fpk:.3f} Гц; АЧХ на 20 Гц {c20:+.2f} дБ; |T|(20 Гц) {t20:.0f} дБ")
if __name__=="__main__":
    run(label="Как в схеме: Cfb 1 мкФ, Cc 10 мкФ, Rdc 1 МОм")
