"""Итоговая проверка рекомендуемого варианта со всеми исправлениями:
 - стык путей ООС 5 кГц (Cp 3,3 нФ, Cpre 820 пФ), Зобель 3,9 Ом / 470 нФ;
 - инфранизкие: Cfb 0,1 мкФ, Cc 100 мкФ внутри петли (с Rвх TPA3255 20 кОм), Cin подбирается;
 - двойной интегратор Cd 1,8 нФ – (Rt 33 к + Cx 22 нФ) – Cd, Rdc 1 МОм; прямая подача через U3."""
import numpy as np, itertools
f=np.logspace(-3,6,60000); s=2j*np.pi*f; i_=lambda x:np.argmin(abs(f-x))
K=12*20e3/(20e3+100); tau=1e-6; R3=4.99e3; Rchip=20e3
def lad(Zs,Cs):
    M=(1,0,0,1)
    mul=lambda M,N:(M[0]*N[0]+M[1]*N[2],M[0]*N[1]+M[1]*N[3],M[2]*N[0]+M[3]*N[2],M[2]*N[1]+M[3]*N[3])
    for k,Z in enumerate(Zs):
        M=mul(M,(1,Z,0,1))
        if k<len(Cs): M=mul(M,(1,0,s*Cs[k],1))
    return 1/M[1]
P=dict(Cp=3.3e-9,Cpre=820e-12,Rz=3.9,Cz=470e-9,Cfb=0.1e-6,Cc=100e-6,Cin=2.2e-6,Cdi=2.2e-9,Cm=6.8e-9,Rff=3.0e3,Cffc=1e-6)
def H(R,p):
    Yp=s*470e-9+1/(p['Rz']+1/(s*p['Cz']))+(0 if R is None else 2/R); Zp=1/Yp; return Zp/(s*6.8e-6+0.02+Zp)
def model(p):
    Ypost=lad([1/(s*p['Cfb'])+20e3,20e3],[p['Cp']]); Ypre=lad([1/(s*p['Cpre'])+13.3e3,13.3e3,13.3e3],[56e-12,56e-12])
    I1=lad([470,1/(s*p['Cin'])+768,768],[2*p['Cdi'],p['Cm']]); Iff=lad([1/(s*p['Cffc'])+p['Rff']/2,p['Rff']/2],[3.3e-9])
    Zt=2/(s*1.8e-9)+(1/(33e3+1/(s*22e-9)))/(s*1.8e-9)**2; Zi=1/(1/1e6+1/Zt)
    HPc=s*p['Cc']*Rchip/(1+s*p['Cc']*Rchip)
    Gn=abs(I1[i_(1e3)]/Ypost[i_(1e3)]); out=[]
    for R,lab in ((4,"4 Ом"),(8,"8 Ом"),(None,"х.х.")):
        h=H(R,p); A=K*np.exp(-s*tau)*HPc
        T=A*Zi*(Ypost*h+Ypre)
        Vout=h*A*(Zi*I1+R3*Iff)/(1+T); cl=20*np.log10(abs(Vout)/Gn)
        Vint=Zi*(I1-(Ypost*h+Ypre)*Vout/h); r=abs(Vint)/abs(Vout/h/K)
        m=abs(T); ph=np.unwrap(np.angle(T))*180/np.pi
        hi=np.where((f[:-1]>1e3)&(m[:-1]>=1)&(m[1:]<1))[0]; ic=hi[-1]
        pmh=min(180-abs(((ph[k]+180)%360)-180) for k in hi)
        j=np.where((f[:-1]>f[ic])&(np.cos(np.radians(ph[:-1]))>-1)&(np.sin(np.radians(ph[:-1]))*np.sin(np.radians(ph[1:]))<0)&(np.cos(np.radians(ph[1:]))<0))[0]
        gm=-20*np.log10(m[j[0]]) if len(j) else 99
        lo=np.where((f[:-1]<20)&(m[:-1]<1)&(m[1:]>=1))[0]
        pml=180-abs(((ph[lo[0]]+180)%360)-180) if len(lo) else 180
        out.append(dict(lab=lab,cl=cl,T=T,pmh=pmh,gm=gm,pml=pml,fu=f[lo[0]] if len(lo) else 0,fc=f[ic],r=r))
    return out
def report(p,title):
    o=model(p); band=(f>=20)&(f<=2e4); sub=f<20
    print(title)
    for x in o:
        print("  %-5s срез %.0f кГц, запас %.0f°/%.1f дБ | НЧ-срез %.3f Гц, запас %.0f° | АЧХ 20–20к %+.2f…%+.2f, ниже 20 Гц макс %+.2f, 10 Гц %+.2f | подавление 1к/10к/20к %.0f/%.0f/%.1f дБ"%(
            x['lab'],x['fc']/1e3,x['pmh'],x['gm'],x['fu'],x['pml'],x['cl'][band].min(),x['cl'][band].max(),x['cl'][sub&(f>0.01)].max(),x['cl'][i_(10)],
            *(20*np.log10(abs(1+x['T'][i_(v)])) for v in (1e3,1e4,2e4))))
    return o
if __name__=="__main__":
    best=None
    for Cin in (1.8e-6,2.0e-6,2.2e-6,2.7e-6):
        p=dict(P,Cin=Cin); o=model(p); sub=(f>0.01)&(f<20); band=(f>=20)&(f<=2e4)
        e=max(max(abs(x['cl'][sub]).max(),abs(x['cl'][band]).max()) for x in o)
        if best is None or e<best[0]: best=(e,Cin)
    P['Cin']=best[1]; print("Cin = %.1f мкФ"%(best[1]*1e6))
    report(P,"ИТОГ")
    V=17.3; Rz,Cz=P['Rz'],P['Cz']
    print("Зобель %.1f Ом/%.0f нФ, синус полной мощности (300 Вт/4 Ом): "%(Rz,Cz*1e9)+", ".join("%g кГц %.2f Вт"%(fr/1e3,V**2*Rz/(Rz**2+(1/(2*np.pi*fr*Cz))**2)) for fr in (2e3,5e3,1e4,2e4)))
    # розовый шум со средней мощностью 1/8 от 300 Вт (≈ музыка на пределе)
    ff=np.logspace(np.log10(20),np.log10(2e4),2000); Vtot=np.sqrt(300/8*4)/2
    dens=Vtot**2/np.log(1000)
    Pz=np.trapezoid(dens*Rz/(Rz**2+(1/(2*np.pi*ff*Cz))**2), np.log(ff))
    print("   розовый шум, средняя мощность 37,5 Вт/4 Ом: %.2f Вт на резисторе Зобеля"%Pz)
