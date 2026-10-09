import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10})
from double_blend import *
yi=Yin(6.8e-9,10e-9); g=abs(yi[i_(1e3)]/Ypost[i_(1e3)])
var=[("двойной интегратор","-",Zint(1.8e-9,33e3,22e-9,1.0e6)),("одинарный 1 нФ","--",Zint(1.0e-9,None,None,2.2e6))]
cols={"4 Ом":"#c0392b","8 Ом":"#2471a3","х.х.":"#7d7d7d"}
fig,ax=plt.subplots(3,1,figsize=(9,11),sharex=True)
for name,ls,Zi in var:
    o=analyze(Zi, yi if ls=="-" else Yin())
    sm=summary(o); print(name,{k:round(float(v),1) for k,v in sm.items()})
    for x in o:
        print("   %s fc %.0fk PM %.0f GM %.1f |T| 100/1k/5k/10k/20k = %s"%(x['lab'],x['fc']/1e3,x['pm'],x['gm'],"/".join("%.0f"%t for t in x['Tk'])))
        c=cols[x['lab']]; lw=1.6 if ls=="-" else 1.0
        ax[0].semilogx(f,20*np.log10(abs(x['T'])),ls,color=c,lw=lw,label=f"{x['lab']}, {name}")
        ax[1].semilogx(f,np.unwrap(np.angle(x['T']))*180/np.pi,ls,color=c,lw=lw)
        if ls=="-": ax[2].semilogx(f,x['cl']+20*np.log10(G/g),"-",color=c,lw=1.8,label=x['lab'])
ax[0].axhline(0,color="k",lw=.6); ax[0].set_ylim(-30,90); ax[0].set_ylabel("Петлевое усиление |T|, дБ"); ax[0].legend(fontsize=7.5,ncol=2)
ax[0].set_title("Смешанная ООС: двойной интегратор (Cd 1,8 нФ, Rt 33 к, Cx 22 нФ, Rdc 1 М) против одинарного")
ax[1].axhline(-180,color="k",lw=.6); ax[1].set_ylim(-270,0); ax[1].set_ylabel("Фаза T, °")
ax[2].set_ylim(-6,3); ax[2].set_ylabel("АЧХ на клеммах АС, дБ"); ax[2].set_xlabel("Частота, Гц"); ax[2].legend(fontsize=8)
for a in ax: a.grid(True,which="both",alpha=.3); a.axvline(2e4,color="g",lw=.6,alpha=.6)
ax[2].set_xlim(10,3e5); plt.tight_layout(); plt.savefig("double_blend_bode.png",dpi=130)
