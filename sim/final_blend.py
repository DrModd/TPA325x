import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10})
K=12; tau=1e-6; L=6.8e-6; C=470e-9; DCR=0.02; Rz,Cz=3.3,2.2e-6; Rdc=2.2e6
f=np.logspace(1,6,40000); s=2j*np.pi*f; i_=lambda x:np.argmin(abs(f-x))
def H(R,sv=s): Yp=sv*C+1/(Rz+1/(sv*Cz))+(0 if R is None else 2/R); Zp=1/Yp; return Zp/(sv*L+DCR+Zp)
def ladder_Y(Zs_list, Csh, sv=s):
    # series elements Zs_list[0..n], shunt caps between them, output into virtual ground: transadmittance I_out/V_in
    # ABCD cascade
    A=np.ones_like(sv); B=np.zeros_like(sv); Cc=np.zeros_like(sv); D=np.ones_like(sv)
    def mul(M,N): a,b,c,d=M; e,f_,g,h=N; return (a*e+b*g, a*f_+b*h, c*e+d*g, c*f_+d*h)
    M=(A,B,Cc,D)
    for k,Z in enumerate(Zs_list):
        M=mul(M,(1,Z,0,1))
        if k<len(Csh): M=mul(M,(1,0,sv*Csh[k],1))
    return 1/M[1]   # short-circuit output: I = V/B
# ---- values
Cfb=1e-6; Rp1=Rp2=20.0e3; Cmid=1.5e-9             # post T-network
Cpre=390e-12; Rl=[13.3e3,13.3e3,13.3e3]; Cl=[56e-12,56e-12]  # pre ladder
Ci=1.0e-9
Rsrc=470; Cdiff=5.6e-9; Cin=22e-6; Ra=Rb=768; Cin_mid=10e-9
def Ypost(sv=s): return ladder_Y([1/(sv*Cfb)+Rp1, Rp2],[Cmid],sv)
def Ypre(sv=s):  return ladder_Y([1/(sv*Cpre)+Rl[0],Rl[1],Rl[2]],Cl,sv)
# input: per half: Vin -> Rsrc -> node(Cdiff to other half => 2*Cdiff to virtual gnd for diff) -> Cin -> Ra -> node(Cin_mid) -> Rb -> sum
def Yin(sv=s): return ladder_Y([Rsrc, 1/(sv*Cin)+Ra, Rb],[2*Cdiff, Cin_mid],sv)
Zi=1/(1/Rdc+s*Ci); A=K*np.exp(-s*tau)*Zi
Rfb_eff=40.0e3; Gnom=abs(Yin()[i_(1e3)])/abs(Ypost()[i_(1e3)])
loads=[(4,"4 Ом","#c0392b"),(8,"8 Ом","#2471a3"),(None,"х.х.","#7d7d7d")]
print("G(1k)=%.2f (%.1f dB)"%(Gnom,20*np.log10(Gnom)))
rows=[]
for R,lab,c in loads:
    h=H(R); T=A*(Ypost()*h+Ypre()); m=abs(T); ph=np.unwrap(np.angle(T))*180/np.pi
    idx=np.where((m[:-1]>=1)&(m[1:]<1))[0]; pm=min(180+ph[k] for k in idx)
    j=np.where((f[:-1]>2e3)&(ph[:-1]>-180)&(ph[1:]<=-180))[0]; gm=-20*np.log10(m[j[0]])
    cl=20*np.log10(abs(h*A*Yin()/(1+T))/Gnom); band=(f>=20)&(f<=2e4)
    rows.append((lab,c,T,cl))
    print("%s: fc=%.0fk PM=%.0f GM=%.1f |T| 1k/5k/10k/20k=%.0f/%.0f/%.0f/%.1f  АЧХ 20–20k %+.2f…%+.2f  @20k %+.2f  -3дБ %.0fk"%(lab,f[idx[-1]]/1e3,pm,gm,*(20*np.log10(m[i_(v)]) for v in (1e3,5e3,1e4,2e4)),cl[band].min(),cl[band].max(),cl[i_(2e4)],f[np.where(cl<-3)[0][0]]/1e3))
ss=np.array([2j*np.pi*600e3]); Ci_=Ci
print("ripple at FDA out ≈ %.0f mVpk"%(4/np.pi*25*abs(Ypre(ss))[0]/abs(ss[0]*Ci_)*1e3))
np.save("blend_rows.npy",np.array([0]))
# ---- figure: loop gain + closed loop vs other variants
exec(open("bode.py").read().split("loads=")[0].split("K=12")[0])
fig,ax=plt.subplots(2,1,figsize=(9,8.5),sharex=True)
for lab,c,T,cl in rows:
    ax[0].semilogx(f,20*np.log10(abs(T)),color=c,lw=1.6,label=f"{lab}, смешанная ООС")
    ax[1].semilogx(f,cl,color=c,lw=1.8,label=f"{lab}, смешанная ООС")
# post-only reference (single integrator variant)
Rin0=2e3; Rfb0=40.2e3; Yfb0=1/Rfb0+1/(2e3+1/(s*75e-12)); Zs0=1/(1/Rdc+s*2.2e-9); G0=Rfb0/Rin0
for R,lab,c in loads:
    T0=K*H(R)*np.exp(-s*tau)*Zs0*Yfb0
    ax[0].semilogx(f,20*np.log10(abs(T0)),"--",color=c,lw=1,alpha=.7,label=f"{lab}, только после фильтра")
    ax[1].semilogx(f,20*np.log10(abs(T0/(1+T0))),"--",color=c,lw=1,alpha=.7)
ax[0].axhline(0,color="k",lw=.6); ax[0].set_ylim(-30,70); ax[0].set_ylabel("Петлевое усиление |T|, дБ")
ax[0].set_title("Смешанная ООС (до + после LC) против ООС только после фильтра"); ax[0].legend(fontsize=7.5,ncol=2)
ax[1].set_ylim(-6,3); ax[1].set_ylabel("АЧХ на клеммах АС, дБ (норм.)"); ax[1].set_xlabel("Частота, Гц")
for a in ax: a.grid(True,which="both",alpha=.3); a.axvline(2e4,color="g",lw=.6,alpha=.6)
ax[1].set_xlim(10,3e5); plt.tight_layout(); plt.savefig("pffb_blend_bode.png",dpi=130)
