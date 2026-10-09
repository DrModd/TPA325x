"""Прямая подача входа на вход TPA3255 (feed-forward, как R63/R72 в NC400) — что она меняет."""
import numpy as np
from double_blend import *
Zi=Zint(1.8e-9,33e3,22e-9,1.0e6)
yi=Yin(6.8e-9,10e-9)
Gn=abs(yi[i_(1e3)]/Ypost[i_(1e3)])          # 20
EQ=yi/yi[i_(1e3)]                            # та же входная коррекция и для прямой подачи
kff=Gn/K                                     # чип сам даёт G без участия интегратора
for ff in (0.0,1.0):
    print("прямая подача" if ff else "без прямой подачи")
    for R,lab in loads:
        h=H(R); e=np.exp(-s*tau); Yf=Ypost*h+Ypre; T=K*e*Zi*Yf
        U=Zi*yi + ff*kff*EQ                  # «вход чипа» на 1 В входа, до замыкания петли
        Vpre=K*e*U/(1+T); Vout=h*Vpre
        Vint=Zi*(yi - Yf*Vpre)               # выход интегратора
        Vchip=Vint+ff*kff*EQ
        cl=20*np.log10(abs(Vout)/Gn); band=(f>=20)&(f<=2e4)
        r=lambda x: abs(Vint[i_(x)])/abs(Vchip[i_(x)])
        print("  %s: АЧХ 20–20к %+.2f…%+.2f дБ, @20к %+.2f; доля выхода интегратора в сигнале чипа 1к/10к/20к = %.3f / %.2f / %.2f"
              %(lab,cl[band].min(),cl[band].max(),cl[i_(2e4)],r(1e3),r(1e4),r(2e4)))
