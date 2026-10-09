import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10})
K=12*20e3/(20e3+1e3); tau=1e-6; L=6.8e-6; C=470e-9; DCR=0.02; Rz,Cz=3.3,2.2e-6
Rin=2e3; Rfb=40.2e3; Rdc=2.2e6; G=Rfb/Rin
Cff,Rs=75e-12,2.0e3; Cffi,Rsi=1.5e-9,100.0
f=np.logspace(1,6,40000); s=2j*np.pi*f
def H(R): Yp=s*C+1/(Rz+1/(s*Cz))+(0 if R is None else 2/R); Zp=1/Yp; return Zp/(s*L+DCR+Zp)
Yfb=1/Rfb+1/(Rs+1/(s*Cff)); Yin=1/Rin+1/(Rsi+1/(s*Cffi))
Zs=1/(1/Rdc+s*2.2e-9)
Cd,Rt=4.7e-9,6.8e3; Zt=2/(s*Cd)+1/(s**2*Cd**2*Rt); Zd=1/(1/Rdc+1/Zt)
loads=[(4,"4 Ом","#c0392b"),(8,"8 Ом","#2471a3"),(None,"Х.х.","#7d7d7d")]
def stats(Z):
    rows=[]
    for R,lab,_ in loads:
        T=K*H(R)*np.exp(-s*tau)*Z*Yfb; m=np.abs(T); ph=np.unwrap(np.angle(T))*180/np.pi
        idx=np.where((m[:-1]>=1)&(m[1:]<1))[0]; pm=min(180+ph[i] for i in idx); fc=f[idx[-1]]
        j=np.where((f[:-1]>2e3)&(ph[:-1]>-180)&(ph[1:]<=-180))[0]; gm=-20*np.log10(m[j[0]])
        cl=Yin/Yfb*T/(1+T)/G; cl=20*np.log10(abs(cl))
        i=lambda x:np.argmin(abs(f-x))
        rows.append((lab,fc,pm,gm,*[20*np.log10(m[i(x)]) for x in (1e3,5e3,1e4,2e4)],cl[i(2e4)],cl[f<3e5].max()))
    return rows
for name,Z in (("Одинарный",Zs),("Двойной",Zd)):
    print(name)
    for r in stats(Z): print("  %s fc=%.0fk PM=%.0f GM=%.1f T1k=%.0f T5k=%.0f T10k=%.0f T20k=%.1f CL20k=%+.2f pk=%+.2f"%(r[0],r[1]/1e3,*r[2:]))
print("Без PFFB:", " ".join("%s %+.2f"%(lab,20*np.log10(abs(H(R)[np.argmin(abs(f-2e4))]))) for R,lab,_ in loads))
fig,ax=plt.subplots(3,1,figsize=(9,11),sharex=True)
for Z,ls,nm in ((Zs,"-","PFFB"),):
    for R,lab,c in loads:
        T=K*H(R)*np.exp(-s*tau)*Z*Yfb
        ax[0].semilogx(f,20*np.log10(abs(T)),ls,color=c,lw=1.4,label=f"{lab}")
        ax[1].semilogx(f,np.unwrap(np.angle(T))*180/np.pi,ls,color=c,lw=1.4)
for R,lab,c in loads:
    T=K*H(R)*np.exp(-s*tau)*Zs*Yfb
    ax[2].semilogx(f,20*np.log10(abs(Yin/Yfb*T/(1+T)/G)),"-",color=c,lw=1.6,label=f"{lab}, с PFFB")
    ax[2].semilogx(f,20*np.log10(abs(H(R))),":",color=c,lw=1.4,label=f"{lab}, без PFFB")
ax[0].axhline(0,color="k",lw=.6); ax[1].axhline(-180,color="k",lw=.6)
ax[0].set_ylim(-40,70); ax[0].set_ylabel("Петлевое усиление |T|, дБ"); ax[0].legend(fontsize=8,ncol=2); ax[0].set_title("Петля PFFB: TPA3255 + LC 6,8 мкГн/470 нФ + Zobel 3,3 Ом/2,2 мкФ\nинтегратор Ci = 2,2 нФ, опережение 75 пФ + 2 кОм")
ax[1].set_ylim(-270,0); ax[1].set_ylabel("Фаза T, °")
ax[2].set_ylim(-8,5); ax[2].set_ylabel("АЧХ на клеммах АС, дБ (норм.)"); ax[2].legend(fontsize=8,ncol=2); ax[2].set_xlabel("Частота, Гц")
for a in ax: a.grid(True,which="both",alpha=.3); a.axvline(2e4,color="g",lw=.6,alpha=.6)
ax[2].set_xlim(10,1e6); ax[0].annotate("запас по фазе 66–80°\nзапас по усилению ≥ 8,8 дБ",xy=(2.5e4,0),xytext=(3e4,30),fontsize=9,arrowprops=dict(arrowstyle="->",lw=.6)); plt.tight_layout(); plt.savefig("pffb_bode.png",dpi=130)
