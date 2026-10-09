import numpy as np, itertools
src=open("final_blend.py").read().split("Zi=1/(1/Rdc")[0]
exec(src)
best=[]
for Clv,Cdv,Cmv,Civ,Cmidv in itertools.product((47e-12,56e-12,68e-12),(3.3e-9,4.7e-9,5.6e-9),(6.8e-9,10e-9,12e-9),(0.82e-9,1.0e-9,1.2e-9),(1.2e-9,1.5e-9,1.8e-9)):
    Cl=[Clv,Clv]; Cdiff=Cdv; Cin_mid=Cmv; Ci=Civ; Cmid=Cmidv
    A=K*np.exp(-s*tau)/(1/Rdc+s*Ci); G=abs(Yin()[i_(1e3)])/abs(Ypost()[i_(1e3)])
    pms=[];gms=[];dev=[];at20=[];T20=[]
    yp=Ypost(); ypr=Ypre(); yi=Yin()
    ok=True
    for R in (4,8,None):
        h=H(R); T=A*(yp*h+ypr); m=abs(T); ph=np.unwrap(np.angle(T))*180/np.pi
        idx=np.where((m[:-1]>=1)&(m[1:]<1))[0]; pms.append(min(180+ph[k] for k in idx))
        j=np.where((f[:-1]>2e3)&(ph[:-1]>-180)&(ph[1:]<=-180))[0]; gms.append(-20*np.log10(m[j[0]]) if len(j) else 99)
        cl=20*np.log10(abs(h*A*yi/(1+T))/G); band=(f>=20)&(f<=2e4)
        dev.append(np.max(np.abs(cl[band]))); at20.append(cl[i_(2e4)]); T20.append(20*np.log10(m[i_(2e4)]))
    ss=np.array([2j*np.pi*600e3]); rip=4/np.pi*25*abs(Ypre(ss))[0]/abs(ss[0]*Ci)
    if min(pms)>=50 and min(gms)>=9:
        score=max(dev)+0.003*rip*1e3-0.05*min(T20)
        best.append((score,Clv,Cdv,Cmv,Civ,Cmidv,min(pms),min(gms),dev,at20,T20,rip))
best.sort(key=lambda b:b[0])
for b in best[:6]:
    print("Cl %.0fp Cdiff %.1fn Cmid_in %.1fn Ci %.2fn Cmid_post %.1fn | PM%.0f GM%.1f | maxdev %s | @20k %s | T20 %s | ripple %.0f mV"%(b[1]*1e12,b[2]*1e9,b[3]*1e9,b[4]*1e9,b[5]*1e9,b[6],b[7],["%.2f"%x for x in b[8]],["%+.2f"%x for x in b[9]],["%.1f"%x for x in b[10]],b[11]*1e3))
