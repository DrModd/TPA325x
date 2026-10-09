"""Статика ограничителя по образцу NC400: порог по выходу FDA с делителем 10к/3,0к на базы T26/T28."""
import numpy as np
Vt=0.02585; Is_q=2.8e-14; Is_m=2.8e-14      # BC846 / половинка BCV62 (оценка)
R99=100; R96=10e3
def inj(Vout, r1=10e3, r2=3.0e3, dT=0.0):
    Vb=Vout*r2/(r1+r2); vt=Vt*(273+25+dT)/298; is_=Is_q*np.exp(0.08*dT)   # ≈ -2 мВ/°C
    lo,hi=0.0,Vb/R99
    for _ in range(200):
        Ic=(lo+hi)/2
        if is_*np.exp((Vb-Ic*R99)/vt)>Ic: lo=Ic
        else: hi=Ic
    # вход отражателя: Ic делится между R96 и диодным транзистором
    lo,hi=0,Ic
    for _ in range(100):
        im=(lo+hi)/2; vbe=vt*np.log(im/Is_m+1)
        if im+vbe/R96>Ic: hi=im
        else: lo=im
    return Ic, im
if __name__ == "__main__":
    for dT in (0,40):
        print(f"ΔT={dT} °C")
        for V in (2.1,2.4,2.6,2.8,3.0,3.2):
            Ic,im=inj(V,dT=dT); print(f"  Vвых {V:.1f} В: ток T26 {Ic*1e6:8.2f} мкА, впрыск в узел A {im*1e6:8.3f} мкА")
