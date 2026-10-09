import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon, Arc
plt.rcParams.update({"font.family": "DejaVu Sans"})

fig, ax = plt.subplots(figsize=(19, 14))
ax.set_aspect("equal"); ax.axis("off")
LW = 1.3; K = "#1a1a1a"; BLUE = "#1f4e8c"; RED = "#a3302d"; GR = "#555"

def W(*pts, c=K):
    xs, ys = zip(*pts); ax.plot(xs, ys, color=c, lw=LW, solid_capstyle="round")

def dot(x, y, c=K): ax.plot(x, y, "o", ms=5, color=c)

def T(x, y, s, size=9, ha="center", va="center", c=K, w="normal"):
    ax.text(x, y, s, fontsize=size, ha=ha, va=va, color=c, weight=w)

def _frame(p1, p2):
    p1, p2 = np.array(p1, float), np.array(p2, float)
    d = p2 - p1; L = np.hypot(*d); u = d / L; n = np.array([-u[1], u[0]])
    m = (p1 + p2) / 2
    return p1, p2, u, n, m, L

def _local(m, u, n, pts):
    return [tuple(m + a * u + b * n) for a, b in pts]

def element(p1, p2, kind, label="", lab_side=1, body=1.4, c=K, lab_off=0.55, lab_size=8.5):
    p1, p2, u, n, m, L = _frame(p1, p2)
    h = body / 2
    W(tuple(p1), tuple(m - h * u), c=c); W(tuple(m + h * u), tuple(p2), c=c)
    P = lambda pts: _local(m, u, n, pts)
    if kind == "R":
        ax.add_patch(Polygon(P([(-h, -.22), (h, -.22), (h, .22), (-h, .22)]), closed=True, fill=False, ec=c, lw=LW))
    elif kind == "C":
        W(*P([(-h, 0), (-.12, 0)]), c=c); W(*P([(.12, 0), (h, 0)]), c=c)
        W(*P([(-.12, -.42), (-.12, .42)]), c=c); W(*P([(.12, -.42), (.12, .42)]), c=c)
    elif kind == "L":
        k = 4; r = body / (2 * k)
        for i in range(k):
            cx = -h + r + 2 * r * i
            t = np.linspace(0, np.pi, 30)
            pts = [(cx - r * np.cos(tt), r * np.sin(tt)) for tt in t]
            xs, ys = zip(*P(pts)); ax.plot(xs, ys, color=c, lw=LW)
    elif kind in ("D", "Dr", "Z", "Zr"):
        s = 1 if kind in ("D", "Z") else -1
        W(*P([(-h, 0), (h, 0)]), c=c)
        a, k_ = -0.3 * s, 0.3 * s
        ax.add_patch(Polygon(P([(a, -.3), (a, .3), (k_, 0)]), closed=True, fc="white", ec=c, lw=LW, zorder=3))
        W(*P([(k_, -.3), (k_, .3)]), c=c)
        if kind[0] == "Z":
            W(*P([(k_, .3), (k_ - .12 * s, .3)]), c=c); W(*P([(k_, -.3), (k_ + .12 * s, -.3)]), c=c)
    if label:
        if abs(u[0]) > .5:
            T(m[0], m[1] + lab_off * lab_side, label, size=lab_size, va="bottom" if lab_side > 0 else "top")
        else:
            T(m[0] - lab_off * lab_side, m[1], label, size=lab_size, ha="right" if lab_side > 0 else "left")

def gnd(x, y, up=False, c=K):
    s = 1 if up else -1
    W((x, y), (x, y + .25 * s), c=c)
    for i, w in enumerate((.45, .3, .15)):
        yy = y + (.25 + .14 * i) * s; W((x - w, yy), (x + w, yy), c=c)

def rail(x, y, txt, up=True):
    s = 1 if up else -1
    W((x, y), (x, y + .35 * s)); W((x - .3, y + .35 * s), (x + .3, y + .35 * s))
    T(x, y + .7 * s, txt, size=8.5)

# ======================= one half (mirror = -1 for bottom) =======================
def half(mir):
    Y = lambda y: 12 + mir * (y - 12)
    top = mir == 1
    vin = "Vin+" if top else "Vin−"
    spk = "SPK−" if top else "SPK+"
    fin = "VIN+" if top else "VIN−"
    fout = "VOUT−" if top else "VOUT+"
    cin_ = "IN+" if top else "IN−"
    cout_ = "OUT−" if top else "OUT+"
    side = 1 if top else -1

    # --- input network
    T(-0.4, Y(14), vin, ha="right", size=10, w="bold")
    dot(-0.2, Y(14))
    element((-0.2, Y(14)), (2.8, Y(14)), "C", "Cin 22 мкФ\nплёнка", side)
    W((2.8, Y(14)), (3.6, Y(14))); dot(3.6, Y(14))
    element((3.6, Y(14)), (8.6, Y(14)), "R", "Rin 2,00 к", -side, body=1.6)
    W((3.6, Y(14)), (3.6, Y(15.6))); W((8.6, Y(15.6)), (8.6, Y(14)))
    element((3.6, Y(15.6)), (6.1, Y(15.6)), "R", "Rsi 100", side, body=1.2)
    element((6.1, Y(15.6)), (8.6, Y(15.6)), "C", "Cffi 1,5 н", side)
    dot(8.6, Y(14)); W((8.6, Y(14)), (12, Y(14)))
    dot(12, Y(14))
    # bus
    W((12, Y(14)), (12, Y(26)))
    # to FDA input
    W((12, Y(14)), (14, Y(14)))
    T(14.15, Y(14), fin, ha="left", size=8)

    # --- integrator network (between bus and VOUT node at x=20)
    xo = 20
    W((18, Y(14)), (xo, Y(14))); dot(xo, Y(14))
    W((xo, Y(14)), (xo, Y(18.5)))
    T(17.85, Y(14), fout, ha="right", size=8)
    rows = [(17.0, "C", "Ci 2,2 нФ C0G"), (18.5, "R", "Rdc 2,2 М")]
    for yy, k, lab in rows:
        dot(12, Y(yy)); dot(xo, Y(yy))
        element((12, Y(yy)), (xo, Y(yy)), k, lab, side, body=1.5, lab_off=.45)

    # --- protection diodes to rails
    W((12, Y(24.4)), (10, Y(24.4))); dot(12, Y(24.4))
    element((10, Y(24.4)), (10, Y(25.7)), "D" if top else "Dr", "", body=0.9)
    element((10, Y(24.4)), (10, Y(23.1)), "Dr" if top else "D", "", body=0.9)
    T(10, Y(26.1), "+15 В" if top else "−15 В", size=8)
    T(10, Y(22.7), "−15 В" if top else "+15 В", size=8)
    T(9.4, Y(24.4), "BAV99", ha="right", size=8)

    # --- output to chip: Rout -> input limiter (to gnd) -> Cc
    element((xo, Y(14)), (22.3, Y(14)), "R", "Rout 1,0 к", side, body=1.1)
    W((22.3, Y(14)), (23.3, Y(14))); dot(22.8, Y(14))
    W((22.8, Y(14)), (22.8, Y(14.6)))
    W((22.35, Y(14.6)), (23.25, Y(14.6)))
    element((22.35, Y(14.6)), (22.35, Y(16.4)), "D" if top else "Dr", "", body=0.9)
    element((23.25, Y(14.6)), (23.25, Y(16.4)), "Dr" if top else "D", "", body=0.9)
    W((22.35, Y(16.4)), (23.25, Y(16.4))); W((22.8, Y(16.4)), (22.8, Y(16.8)))
    gnd(22.8, Y(16.8), up=top)
    T(21.8, Y(15.5), "VD: 5×BAS416\nв каждой ветви", ha="right", size=7.5)
    element((23.3, Y(14)), (25.6, Y(14)), "C", "Cc 10 мкФ", side)
    W((25.6, Y(14)), (26, Y(14)))
    T(26.15, Y(14), cin_, ha="left", size=8)

    # --- chip output -> LC -> speaker
    T(29.85, Y(14), cout_, ha="right", size=8)
    W((30, Y(14)), (30.4, Y(14)))
    element((30.4, Y(14)), (33.4, Y(14)), "L", "L 6,8 мкГн", side, body=1.8)
    W((33.4, Y(14)), (38, Y(14)))
    dot(34.2, Y(14)); dot(36.2, Y(14)); dot(38, Y(14))
    element((34.2, Y(14)), (34.2, Y(17.4)), "C", "Cf\n470 нФ", 1, lab_off=.6)
    gnd(34.2, Y(17.4), up=top)
    element((36.2, Y(14)), (36.2, Y(15.9)), "R", "Rz 3,3 Ом\n5 Вт", -1, body=1.0, lab_off=.45)
    element((36.2, Y(15.9)), (36.2, Y(17.4)), "C", "", body=.6)
    T(36.75, Y(16.9), "Cz 2,2 мкФ", ha="left", size=8.5)
    gnd(36.2, Y(17.4), up=top)
    T(38.3, Y(14) + .35 * side, spk, ha="left", size=10, w="bold")

    # --- feedback path along y=26
    W((38, Y(14)), (38, Y(26)))
    W((38, Y(26)), (35.5, Y(26)))
    element((35.5, Y(26)), (31.5, Y(26)), "C", "Cfb 1 мкФ / 100 В", side)
    W((31.5, Y(26)), (28, Y(26))); dot(28, Y(26))
    element((28, Y(26)), (21, Y(26)), "R", "Rfb 40,2 к / 0,5 Вт", -side, body=1.8)
    W((28, Y(26)), (28, Y(27.6))); W((21, Y(27.6)), (21, Y(26)))
    element((28, Y(27.6)), (24.5, Y(27.6)), "C", "Cff 75 пФ C0G", side)
    element((24.5, Y(27.6)), (21, Y(27.6)), "R", "Rs 2,0 к", side, body=1.3)
    dot(21, Y(26)); W((21, Y(26)), (12, Y(26)))

half(1); half(-1)

# --- FDA box
ax.add_patch(Rectangle((14, 8.6), 4, 6.8, fill=False, ec=BLUE, lw=1.8))
T(15.6, 12.9, "U1", size=10, c=BLUE, w="bold"); T(15.6, 12.3, "OPA1632", size=10, c=BLUE, w="bold")
T(15.6, 11.6, "FDA", size=8.5, c=BLUE)
W((16, 15.4), (16, 15.9), c=BLUE); T(16.15, 15.75, "+15 В", size=8, c=BLUE, ha="left")
W((16, 8.6), (16, 8.1), c=BLUE); T(16.15, 8.25, "−15 В", size=8, c=BLUE, ha="left")
W((18, 12), (19, 12), c=BLUE); T(17.85, 12, "VOCM", ha="right", size=7.5, c=BLUE)
element((19, 12), (19, 11.3), "C", "", body=.35, c=BLUE)
gnd(19, 11.3, c=BLUE)
T(19.55, 11.65, "100 н", ha="left", size=7.5, c=BLUE)

# --- TPA3255 box
ax.add_patch(Rectangle((26, 8.2), 4, 7.6, fill=False, ec=RED, lw=1.8))
T(28, 12.7, "U2", size=10, c=RED, w="bold"); T(28, 12.1, "TPA3255", size=10, c=RED, w="bold")
T(28, 11.4, "1 канал BTL", size=8, c=RED)
T(28, 10.95, "K ≈ −12 (инверт.)", size=8, c=RED)
for x, s in ((26.8, "PVDD"), (28, "GVDD"), (29.2, "VDD")):
    W((x, 15.8), (x, 16.3), c=RED); T(x, 16.55, s, size=7.5, c=RED)
T(28, 17.05, "48–50 В · 12 В · 12 В", size=7.5, c=RED)
for x, s in ((26.8, "/RESET"), (28, "/FAULT"), (29.2, "/CLIP")):
    W((x, 8.2), (x, 7.7), c=RED); T(x, 7.45, s, size=7.5, c=RED)
T(28, 6.95, "→ к МК", size=7.5, c=RED)

# --- speaker between SPK+ and SPK-
W((38, 14), (38, 13.1)); W((38, 10), (38, 10.9))
ax.add_patch(Rectangle((37.75, 11.4), 0.5, 1.2, fill=False, ec=K, lw=LW))
ax.add_patch(Polygon([(38.25, 11.4), (38.85, 10.8), (38.85, 13.2), (38.25, 12.6)], closed=True, fill=False, ec=K, lw=LW))
W((38, 13.1), (38, 12.6)); W((38, 10.9), (38, 11.4))
T(39.1, 12, "АС\n4–8 Ом", ha="left", size=9)

# --- title & notes
T(19, 31.3, "PFFB для TPA3255 — один канал BTL (полная дифференциальная схема)", size=15, w="bold")
T(19, 30.5, "Петля: срез 23–38 кГц, запас по фазе ≥ 66°, по усилению ≥ 8,8 дБ (4 Ом / 8 Ом / х.х.);  G = Rfb/Rin = 20 (26 дБ)", size=10)
notes = [
    "Примечания",
    "1. Нижняя половина — зеркальная копия верхней. TPA3255 инвертирует: VOUT− → IN+, VOUT+ → IN−; ООС: SPK− → VIN+, SPK+ → VIN−. Полярность проверить до замыкания петли.",
    "2. Rsi/Cffi = Rs/Cff ÷ 20 — зеркальное опережающее звено: петля получает опережение фазы, а АЧХ усиления остаётся плоской.",
    "3. Rin·Cin ≈ Rfb·Cfb (44 / 40 мс): развязка от PVDD/2 без подъёма усиления на инфранизких частотах.",
    "4. Ci, Cff, Cffi — только C0G/NP0. Cfb, Cin, Cc, Cf, Cz — полипропилен; Cff и Cfb на 100 В.",
    "5. Rz: на полной мощности синусом выше ~5 кГц перегревается — ВЧ-тесты на пониженном уровне.",
    "6. /RESET держать в «0» ≈ 0,5 с после подачи питания (заряд Cfb/Cin). Бутстрепы, развязка PVDD/GVDD и разводка — по даташиту/EVM.",
    "7. Ограничитель входа U2 (VD, порог ≈ ±3 В при допустимых 7 В п-п) — после Rout, не поперёк Ci: утечка и ёмкость стабилитрона там исказили бы интегратор.",
    "8. K ≈ 12 и задержка чипа ≈ 1 мкс — допущения расчёта; перед финалом измерить АЧХ/ФЧХ объекта без внешней петли.",
]
for i, s in enumerate(notes):
    T(-1.5, -5.0 - 0.62 * i, s, ha="left", size=9.5 if i else 10.5, w="bold" if i == 0 else "normal")

ax.set_xlim(-2.5, 41); ax.set_ylim(-10.2, 32)
plt.tight_layout()
plt.savefig("pffb_schematic.png", dpi=140)
plt.savefig("pffb_schematic.svg")
plt.savefig("pffb_schematic.pdf")
