import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon, Arc
plt.rcParams.update({"font.family": "DejaVu Sans"})

fig, ax = plt.subplots(figsize=(20.2, 22.6))
ax.set_aspect("equal"); ax.axis("off")
LW = 1.3; K = "#1a1a1a"; BLUE = "#1f4e8c"; RED = "#a3302d"; GR = "#555"

SH = [True]
def sx(x): return x + 3.0 if (SH[0] and x >= 25.95) else x

def W(*pts, c=K):
    xs, ys = zip(*pts); xs = [sx(v) for v in xs]; ax.plot(xs, ys, color=c, lw=LW, solid_capstyle="round")

def dot(x, y, c=K): ax.plot(sx(x), y, "o", ms=5, color=c)

def T(x, y, s, size=9, ha="center", va="center", c=K, w="normal"):
    ax.text(sx(x), y, s, fontsize=size, ha=ha, va=va, color=c, weight=w)

def _frame(p1, p2):
    p1, p2 = np.array(p1, float), np.array(p2, float)
    d = p2 - p1; L = np.hypot(*d); u = d / L; n = np.array([-u[1], u[0]])
    m = (p1 + p2) / 2
    return p1, p2, u, n, m, L

def _local(m, u, n, pts):
    return [tuple(m + a * u + b * n) for a, b in pts]

def element(p1, p2, kind, label="", lab_side=1, body=1.4, c=K, lab_off=0.55, lab_size=8.5):
    p1 = (sx(p1[0]), p1[1]); p2 = (sx(p2[0]), p2[1])
    _sv = SH[0]; SH[0] = False
    try:
        _element(p1, p2, kind, label, lab_side, body, c, lab_off, lab_size)
    finally:
        SH[0] = _sv

def _element(p1, p2, kind, label, lab_side, body, c, lab_off, lab_size):
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
    x = sx(x); _sv = SH[0]; SH[0] = False
    try: _gnd(x, y, up, c)
    finally: SH[0] = _sv

def _gnd(x, y, up=False, c=K):
    s = 1 if up else -1
    W((x, y), (x, y + .25 * s), c=c)
    for i, w in enumerate((.45, .3, .15)):
        yy = y + (.25 + .14 * i) * s; W((x - w, yy), (x + w, yy), c=c)

def rail(x, y, txt, up=True):
    s = 1 if up else -1
    W((x, y), (x, y + .35 * s)); W((x - .3, y + .35 * s), (x + .3, y + .35 * s))
    T(x, y + .7 * s, txt, size=8.5)

def bjt(x, y, kind, label="", c=K, lab_left=False):
    # base at (x-0.6,y); bar at x; collector/emitter ends at (x+0.6, y±0.7)
    W((x - 0.6, y), (x, y), c=c); W((x, y - .38), (x, y + .38), c=c)
    npn = kind == "NPN"
    cy, ey = (y + .7, y - .7) if npn else (y - .7, y + .7)
    W((x, y + (.18 if npn else -.18)), (x + .6, cy), c=c)
    W((x, y + (-.18 if npn else .18)), (x + .6, ey), c=c)
    # emitter arrow
    ex0, ey0 = x, y + (-.18 if npn else .18)
    dx, dy = .6, ey - ey0
    if npn:
        tip = (ex0 + .8 * dx, ey0 + .8 * dy); base_ = (ex0 + .45 * dx, ey0 + .45 * dy)
    else:
        tip = (ex0 + .25 * dx, ey0 + .25 * dy); base_ = (ex0 + .6 * dx, ey0 + .6 * dy)
    vx, vy = tip[0] - base_[0], tip[1] - base_[1]; L = np.hypot(vx, vy); nx, ny = -vy / L * .12, vx / L * .12
    ax.add_patch(Polygon([tip, (base_[0] + nx, base_[1] + ny), (base_[0] - nx, base_[1] - ny)], closed=True, fc=c, ec=c))
    if label:
        T(x - .75 if lab_left else x + .75, y, label, ha="right" if lab_left else "left", size=8)
    return (x + .6, cy), (x + .6, ey)

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

    # --- input network: Rsrc -> (Cdiff) -> Cin -> Ra -> (Cm to gnd) -> Rb -> sum
    T(-0.4, Y(14), vin, ha="right", size=10, w="bold")
    dot(-0.2, Y(14))
    W((-0.2, Y(14)), (-0.2, Y(15.2)))
    T(-0.2, Y(15.55), "FF → U3 низ" if top else "FF → U3 верх", size=8, c=BLUE, w="bold")
    element((-0.2, Y(14)), (1.6, Y(14)), "R", "Rsrc 470", side, body=1.0)
    dot(1.6, Y(14))
    element((1.6, Y(14)), (4.2, Y(14)), "C", "Cin 2,2 мкФ\nплёнка", side)
    element((4.2, Y(14)), (6.6, Y(14)), "R", "Ra 768", -side, body=1.1)
    dot(7.0, Y(14)); W((6.6, Y(14)), (7.0, Y(14)))
    element((7.0, Y(14)), (7.0, Y(15.5)), "C", "", body=.5)
    gnd(7.0, Y(15.5), up=top)
    T(7.55, Y(15.0), "Cm 6,8 нФ", ha="left", size=8.5)
    element((7.0, Y(14)), (11.2, Y(14)), "R", "Rb 768", -side, body=1.1)
    W((11.2, Y(14)), (12, Y(14)))
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
    dot(12, Y(17.0)); dot(xo, Y(17.0))
    element((12, Y(17.0)), (xo, Y(17.0)), "R", "Rdc 1,0 М", side, body=1.5, lab_off=.45)
    dot(12, Y(18.5)); dot(xo, Y(18.5))
    element((12, Y(18.5)), (16, Y(18.5)), "C", "Cd 1,8 н", -side, lab_off=.5)
    element((16, Y(18.5)), (xo, Y(18.5)), "C", "Cd 1,8 н", -side, lab_off=.5)
    dot(16, Y(18.5))
    element((16, Y(18.5)), (16, Y(20.0)), "R", "", body=.9)
    element((16, Y(20.0)), (16, Y(21.2)), "C", "", body=.5)
    gnd(16, Y(21.2), up=top)
    T(16.5, Y(19.3), "Rt 33 к", ha="left", size=8.5)
    T(16.5, Y(20.6), "Cx 22 нФ", ha="left", size=8.5)
    T(13.9, Y(20.6), "Cd — C0G", ha="right", size=7.5, c=GR)
    # connections to limiter copy
    n = "1" if top else "2"
    W((12, Y(21.9)), (11.3, Y(21.9))); dot(12, Y(21.9))
    T(11.2, Y(21.9), f"A{n} → огр.{n}", ha="right", size=8, c=BLUE, w="bold")
    W((xo, Y(15.9)), (20.7, Y(15.9))); dot(xo, Y(15.9))
    T(20.8, Y(15.9), f"B{n} → огр.{n}", ha="left", size=8, c=BLUE, w="bold")

    # --- protection diodes to rails
    W((12, Y(24.7)), (10, Y(24.7))); dot(12, Y(24.7))
    element((10, Y(24.7)), (10, Y(25.9)), "D" if top else "Dr", "", body=0.9)
    element((10, Y(24.7)), (10, Y(23.5)), "Dr" if top else "D", "", body=0.9)
    T(10, Y(26.3), "+4,8 В" if top else "−4,8 В", size=8)
    T(10, Y(23.1), "−4,8 В" if top else "+4,8 В", size=8)
    T(9.4, Y(24.7), "BAV99", ha="right", size=8)

    # --- output to chip: U1 -> U3 (сумматор) -> Rout 470 -> VD -> Cc
    W((xo, Y(14)), (20.6, Y(14)))
    SH[0] = False     # Rout–Cc в абсолютных координатах 22,3…29,0
    W((22.3, Y(14)), (23.2, Y(14)))
    element((23.2, Y(14)), (25.4, Y(14)), "R", "Rout 100", side, body=1.1, lab_off=.4, lab_size=8)
    W((25.4, Y(14)), (26.4, Y(14)))
    element((26.4, Y(14)), (28.4, Y(14)), "C", "Cc 100 мкФ\nнеполярн.", side, lab_off=.45, lab_size=8)
    W((28.4, Y(14)), (29.0, Y(14)))
    SH[0] = True
    T(26.15, Y(14), cin_, ha="left", size=8)

    # --- chip output -> LC -> speaker
    T(29.85, Y(14), cout_, ha="right", size=8)
    W((30, Y(14)), (30.4, Y(14)))
    element((30.4, Y(14)), (33.4, Y(14)), "L", "L 6,8 мкГн", side, body=1.8)
    W((33.4, Y(14)), (38, Y(14)))
    dot(34.2, Y(14)); dot(36.2, Y(14)); dot(38, Y(14))
    element((34.2, Y(14)), (34.2, Y(17.4)), "C", "Cf\n470 нФ", 1, lab_off=.6)
    gnd(34.2, Y(17.4), up=top)
    element((36.2, Y(14)), (36.2, Y(15.9)), "R", "Rz 3,9 Ом\n5 Вт", -1, body=1.0, lab_off=.45)
    element((36.2, Y(15.9)), (36.2, Y(17.4)), "C", "", body=.6)
    T(36.75, Y(16.9), "Cz 470 нФ", ha="left", size=8.5)
    gnd(36.2, Y(17.4), up=top)
    T(38.3, Y(14) + .35 * side, spk, ha="left", size=10, w="bold")

    # --- post-filter path (y=26): Cfb -> 20k -> (1.5n) -> 20k -> bus
    W((38, Y(14)), (38, Y(26)))
    W((38, Y(26)), (36.0, Y(26)))
    element((36.0, Y(26)), (33.0, Y(26)), "C", "Cfb 0,1 мкФ / 100 В", side)
    element((33.0, Y(26)), (28.0, Y(26)), "R", "Rp1 20,0 к", -side, body=1.6)
    dot(27.4, Y(26)); W((28.0, Y(26)), (27.4, Y(26)))
    element((27.4, Y(26)), (27.4, Y(27.5)), "C", "", body=.5)
    gnd(27.4, Y(27.5), up=top)
    T(27.95, Y(27.0), "Cp 3,3 нФ C0G", ha="left", size=8.5)
    element((27.4, Y(26)), (21.5, Y(26)), "R", "Rp2 20,0 к", -side, body=1.6)
    W((21.5, Y(26)), (12, Y(26)))
    T(17.0, Y(26) + .35 * side, "ООС после фильтра (НЧ/СЧ)", size=8, c=GR)

    # --- pre-filter path (y=23.4): from chip OUT pin, Cpre -> 13k3 -> 56p -> 13k3 -> 56p -> 13k3 -> bus
    dot(30.4, Y(14))
    W((30.4, Y(14)), (30.4, Y(23.4)))
    element((30.4, Y(23.4)), (28.9, Y(23.4)), "C", "Cpre\n820 п", -side, lab_off=.55, lab_size=8)
    element((28.9, Y(23.4)), (26.4, Y(23.4)), "R", "13,3 к", side, body=1.1)
    dot(26.0, Y(23.4)); W((26.4, Y(23.4)), (26.0, Y(23.4)))
    element((26.0, Y(23.4)), (26.0, Y(22.3)), "C", "", body=.45)
    gnd(26.0, Y(22.3), up=not top)
    element((26.0, Y(23.4)), (23.6, Y(23.4)), "R", "13,3 к", side, body=1.1)
    dot(23.2, Y(23.4)); W((23.6, Y(23.4)), (23.2, Y(23.4)))
    element((23.2, Y(23.4)), (23.2, Y(22.3)), "C", "", body=.45)
    gnd(23.2, Y(22.3), up=not top)
    T(24.6, Y(22.35), "2 × 56 пФ", size=8)
    element((23.2, Y(23.4)), (20.8, Y(23.4)), "R", "13,3 к", side, body=1.1)
    W((20.8, Y(23.4)), (12, Y(23.4))); dot(12, Y(23.4))
    T(16.2, Y(23.4) + .35 * side, "ООС до фильтра (ВЧ)", size=8, c=GR)

half(1); half(-1)
element((1.6, 14), (1.6, 10), "C", "Cdi\n2,2 нФ", -1, lab_off=.65)

# --- FDA box
ax.add_patch(Rectangle((14, 8.6), 4, 6.8, fill=False, ec=BLUE, lw=1.8))
T(15.6, 12.9, "U1", size=10, c=BLUE, w="bold"); T(15.6, 12.3, "OPA1632", size=10, c=BLUE, w="bold")
T(15.6, 11.6, "FDA", size=8.5, c=BLUE)
W((16, 15.4), (16, 15.9), c=BLUE); T(16.15, 15.75, "+4,8 В", size=8, c=BLUE, ha="left")
W((16, 8.6), (16, 8.1), c=BLUE); T(16.15, 8.25, "−4,8 В", size=8, c=BLUE, ha="left")
W((18, 12), (19, 12), c=BLUE); T(17.85, 12, "VOCM", ha="right", size=7.5, c=BLUE)
element((19, 12), (19, 11.3), "C", "", body=.35, c=BLUE)
gnd(19, 11.3, c=BLUE)
T(19.55, 11.65, "100 н", ha="left", size=7.5, c=BLUE)

# --- U3 summing FDA
ax.add_patch(Rectangle((20.6, 8.6), 1.7, 6.8, fill=False, ec=BLUE, lw=1.8))
T(21.45, 12.6, "U3", size=10, c=BLUE, w="bold"); T(21.45, 12.0, "OPA\n1632", size=8, c=BLUE, w="bold")
T(21.45, 11.1, "Σ ×1\n+FF", size=7.5, c=BLUE)
T(21.45, 9.4, "±4,8 В", size=7, c=BLUE)

# --- TPA3255 box
ax.add_patch(Rectangle((sx(26), 8.2), 4, 7.6, fill=False, ec=RED, lw=1.8))
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
ax.add_patch(Rectangle((sx(37.75), 11.4), 0.5, 1.2, fill=False, ec=K, lw=LW))
ax.add_patch(Polygon([(sx(38.25), 11.4), (sx(38.85), 10.8), (sx(38.85), 13.2), (sx(38.25), 12.6)], closed=True, fill=False, ec=K, lw=LW))
W((38, 13.1), (38, 12.6)); W((38, 10.9), (38, 11.4))
T(39.1, 12, "АС\n4–8 Ом", ha="left", size=9)


SH[0] = False
# ======================= limiter / clip detector (one copy, ×2) =======================
OX, OY = 0.0, -14.4      # inset origin
def P(x, y): return (OX + x, OY + y)
ax.add_patch(Rectangle(P(-1.8, -15.8), 44.6, 16.6, fill=False, ec=GR, lw=.8, ls="--"))
T(*P(-1.4, 0.25), "Ограничитель и детектор клиппинга (по образцу Hypex NC400), ×2:  огр.1 — A1 = VIN+, B1 = VOUT−;  огр.2 — A2 = VIN−, B2 = VOUT+", ha="left", size=10.5, w="bold")
VP, VN = -1.5, -14.5                 # rails
W(P(4, VP), P(22, VP)); T(*P(3.8, VP), "+4,8 В", ha="right", size=8.5)
W(P(4, VN), P(15.5, VN)); T(*P(3.8, VN), "−4,8 В", ha="right", size=8.5)
xi, xo_, xz = 10.6, 7.0, 7.0          # mirror input column, output column
# --- upper PNP mirror T24
ax.add_patch(Rectangle(P(6.2, -3.6), 5.0, 1.4, fill=False, ec=K, lw=LW))
T(*P(8.7, -2.9), "T24 BCV62B\n(PNP-отражатель)", size=8)
W(P(7.0, VP), P(7.0, -2.2)); W(P(10.4, VP), P(10.4, -2.2))
T(*P(6.85, -3.95), "вых", ha="right", size=7.5, c=GR); T(*P(10.75, -3.95), "вх", ha="left", size=7.5, c=GR)
# mirror input: R96 shunt, D15, T26
W(P(10.4, -3.6), P(10.4, -4.4)); dot(*P(10.4, -4.4))
W(P(10.4, -4.4), P(12.2, -4.4)); element(P(12.2, -4.4), P(12.2, VP), "R", "R96\n10 к", -1, body=1.0, lab_off=.4)
element(P(10.4, -4.4), P(10.4, -6.0), "D", "D15 BAS316", -1, body=.9, lab_off=.45)
dot(*P(10.4, -6.0))
# T26 NPN (collector up at 10.4,-6.0)
c26, e26 = bjt(OX + 9.8, OY - 6.7, "NPN")
W(P(10.4, -6.0), c26)
T(*P(8.9, -6.2), "T26\nBC846", ha="right", size=8)
# T28 PNP (emitter up)
c28, e28 = bjt(OX + 9.8, OY - 9.1, "PNP")
T(*P(8.9, -9.6), "T28\nBC856", ha="right", size=8)
W(e26, P(10.4, -7.9)); W(e28, P(10.4, -7.9)); dot(*P(10.4, -7.9))
element(P(10.4, -7.9), P(13.0, -7.9), "R", "R99 100", 1, body=1.0, lab_off=.35)
gnd(*P(13.0, -7.9))
# bases bus
W(P(9.2, -6.7), P(8.6, -6.7)); W(P(9.2, -9.1), P(8.6, -9.1))
W(P(8.6, -6.7), P(8.6, -9.1)); dot(*P(8.6, -8.3))
# B input with divider 10k / 3.0k
T(*P(-1.6, -8.3), "B (от VOUT)", ha="left", size=9, c=BLUE, w="bold")
element(P(2.0, -8.3), P(4.6, -8.3), "R", "Rb1 10 к", -1, body=1.1, lab_off=.35)
dot(*P(4.6, -8.3))
element(P(4.6, -8.3), P(4.6, -11.0), "R", "Rb2\n15 к", 1, body=1.0, lab_off=.35)
gnd(*P(4.6, -11.0))
W(P(4.6, -8.3), P(6.75, -8.3))
_t = np.linspace(np.pi, 0, 20); ax.plot(OX + 7.0 + .25 * np.cos(_t), OY - 8.3 + .25 * np.sin(_t), color=K, lw=LW)
W(P(7.25, -8.3), P(8.6, -8.3))
# lower: T28 collector -> D19 -> NPN mirror T30 input, R102
W(c28, P(10.4, -10.4))
element(P(10.4, -10.4), P(10.4, -11.8), "D", "D19 BAS316", -1, body=.9, lab_off=.45)
dot(*P(10.4, -11.8)); W(P(10.4, -11.8), P(12.2, -11.8))
element(P(12.2, -11.8), P(12.2, VN), "R", "R102\n10 к", -1, body=1.0, lab_off=.4)
W(P(10.4, -11.8), P(10.4, -12.4))
ax.add_patch(Rectangle(P(6.2, -13.8), 5.0, 1.4, fill=False, ec=K, lw=LW))
T(*P(8.7, -13.1), "T30 BCV61B\n(NPN-отражатель)", size=8)
W(P(7.0, -13.8), P(7.0, VN)); W(P(10.4, -13.8), P(10.4, VN))
# output column: D16 / D18 zeners to node A
W(P(7.0, -3.6), P(7.0, -4.6))
element(P(7.0, -4.6), P(7.0, -7.2), "Zr", "D16\n3V3", 1, body=.9, lab_off=.4)
dot(*P(7.0, -7.5)); W(P(7.0, -7.2), P(7.0, -8.6))
element(P(7.0, -8.6), P(7.0, -11.2), "Zr", "D18\n3V3", 1, body=.9, lab_off=.4)
W(P(7.0, -11.2), P(7.0, -12.4))
W(P(7.0, -7.5), P(2.0, -7.5))
T(*P(-1.6, -7.5), "A (к VIN)", ha="left", size=9, c=BLUE, w="bold")
# crossing marker note: B line passes under A column (no junction)
# --- CLIP flag: from T26 collector node via R98 to T25
W(P(10.4, -6.0), P(14.0, -6.0))
element(P(14.0, -6.0), P(16.4, -6.0), "R", "R98 10 к", -1, body=1.0, lab_off=.35)
dot(*P(16.4, -6.0))
element(P(16.4, -6.0), P(16.4, VP), "C", "C67\n1 н", 1, lab_off=.4)
W(P(16.4, -6.0), P(17.8, -6.0)); dot(*P(17.8, -6.0))
element(P(17.8, -6.0), P(17.8, VP), "R", "R97\n22 к", -1, body=1.0, lab_off=.35)
W(P(17.8, -6.0), P(18.6, -6.0))
c25, e25 = bjt(OX + 19.2, OY - 6.0, "PNP")
T(*P(20.0, -6.5), "T25\nBC856", ha="left", size=8)
element(e25, P(19.8, VP), "R", "R103 1 к", -1, body=.9, lab_off=.4)
W(c25, P(19.8, -8.6)); W(P(19.8, -8.6), P(21.6, -8.6))
T(*P(21.7, -8.6), "CLIP+ (огр.)", ha="left", size=9, c=RED, w="bold")
# --- MCU interface
MX = 27.0
T(*P(MX - 1.6, -1.0), "К МК (STM32): клиппинг любого знака", ha="left", size=9.5, w="bold")
T(*P(MX, -3.6), "CLIP+ огр.1", ha="right", size=8.5, c=RED); T(*P(MX, -5.0), "CLIP+ огр.2", ha="right", size=8.5, c=RED)
element(P(MX + .1, -3.6), P(MX + 1.8, -3.6), "D", "", body=.8); element(P(MX + .1, -5.0), P(MX + 1.8, -5.0), "D", "", body=.8)
T(*P(MX + 0.95, -5.75), "BAS316 ×2", size=7.5)
W(P(MX + 1.8, -3.6), P(MX + 1.8, -5.0)); dot(*P(MX + 1.8, -4.3))
W(P(MX + 1.8, -4.3), P(MX + 2.2, -4.3))
element(P(MX + 2.2, -4.3), P(MX + 4.6, -4.3), "R", "10 к", 1, body=1.0, lab_off=.35)
dot(*P(MX + 4.6, -4.3)); element(P(MX + 4.6, -4.3), P(MX + 4.6, -6.6), "R", "10 к", 1, body=.9, lab_off=.35)
gnd(*P(MX + 4.6, -6.6))
W(P(MX + 4.6, -4.3), P(MX + 5.4, -4.3))
c_, e_ = bjt(OX + MX + 6.0, OY - 4.3, "NPN")
T(*P(MX + 6.85, -4.6), "BC847", ha="left", size=8)
gnd(*e_)
W(c_, P(MX + 6.6, -2.9)); dot(*P(MX + 6.6, -2.9))
element(P(MX + 6.6, -2.9), P(MX + 6.6, -1.8), "R", "", body=.6)
T(*P(MX + 6.95, -2.2), "4,7 к → +3,3 В", ha="left", size=8)
W(P(MX + 6.6, -2.9), P(MX + 8.4, -2.9))
T(*P(MX + 8.5, -3.35), "/CLIP → МК\n(акт. «0»)", ha="left", size=8.5, c=RED, w="bold")
T(*P(MX - 1.5, -8.6), "Флаг CLIPNEG (T29) не нужен: отрицательный клиппинг\nвиден как CLIP+ второй копии. С прямой подачей выход\nинтегратора несёт только ошибку (≤ 0,57 В пик на полной\nмощности); порог ≈ 0,9–1,0 В, при перегрузе ≈ 1,2 В.", ha="left", size=8.5, c=GR)

# ======================= U3 detail =======================
QX, QY = 0.0, -31.4
def Q(x, y): return (QX + x, QY + y)
ax.add_patch(Rectangle(Q(-1.8, -9.6), 44.6, 10.4, fill=False, ec=GR, lw=.8, ls="--"))
T(*Q(-1.4, 0.25), "Сумматор U3 (OPA1632) с прямой подачей входа — верхняя половина; нижняя зеркальна (от U1 VOUT+, от Vin+, выход VOUT− → IN− U2)", ha="left", size=10.5, w="bold")
# U1 path
T(*Q(-1.4, -2.5), "от U1 VOUT−", ha="left", size=9, c=BLUE, w="bold")
W(Q(2.6, -2.5), Q(6.0, -2.5)); element(Q(6.0, -2.5), Q(10.0, -2.5), "R", "R1 4,99 к", 1, body=1.4, lab_off=.35)
W(Q(10.0, -2.5), Q(12.6, -2.5)); W(Q(12.6, -2.5), Q(12.6, -4.0))
# FF path
T(*Q(-1.4, -5.5), "от Vin− (вход, до Rsrc)", ha="left", size=9, c=BLUE, w="bold")
W(Q(4.6, -5.5), Q(5.0, -5.5))
element(Q(5.0, -5.5), Q(6.8, -5.5), "C", "Cffc\n1 мкФ", 1, lab_off=.45, lab_size=8)
element(Q(6.8, -5.5), Q(9.0, -5.5), "R", "1,5 к", 1, body=1.0, lab_off=.35)
dot(*Q(9.4, -5.5)); W(Q(9.0, -5.5), Q(9.4, -5.5))
element(Q(9.4, -5.5), Q(9.4, -7.4), "C", "", body=.5); gnd(*Q(9.4, -7.4))
T(*Q(9.85, -6.7), "Cfx 3,3 нФ", ha="left", size=8)
element(Q(9.4, -5.5), Q(11.8, -5.5), "R", "1,5 к", 1, body=1.0, lab_off=.35)
W(Q(11.8, -5.5), Q(12.6, -5.5)); W(Q(12.6, -5.5), Q(12.6, -4.0))
dot(*Q(12.6, -4.0)); W(Q(12.6, -4.0), Q(14.0, -4.0))
T(*Q(12.4, -4.0), "S", ha="right", size=8, c=GR)
# U3 box
ax.add_patch(Rectangle(Q(14.0, -6.6), 3.4, 5.4, fill=False, ec=BLUE, lw=1.6))
T(*Q(15.9, -3.1), "U3 OPA1632", size=9, c=BLUE, w="bold")
T(*Q(14.15, -4.0), "VIN+", ha="left", size=7, c=BLUE)
T(*Q(17.25, -2.0), "VOUT−", ha="right", size=7, c=BLUE); T(*Q(17.25, -5.2), "VOUT+", ha="right", size=7, c=BLUE)
# feedback R3: VOUT− -> S
W(Q(12.6, -4.0), Q(12.6, -0.9))
element(Q(12.6, -0.9), Q(18.4, -0.9), "R", "R3 4,99 к", 1, body=1.4, lab_off=.3)
W(Q(18.4, -0.9), Q(18.4, -2.0)); W(Q(18.4, -2.0), Q(17.4, -2.0))
# output
W(Q(17.4, -5.2), Q(19.4, -5.2))
T(*Q(19.6, -5.2), "→ Rout 100 → Cc → IN+ U2", ha="left", size=9, c=RED, w="bold")
T(*Q(19.6, -1.9), "Выход VOUT+ = U1·VOUT− + (R3/Rff)·(−Vin+)  →  ×1 для интегратора, ×1,66 = G/K для входа.\n"
                  "Знак как у прежней схемы: петля и полярность не меняются.\n"
                  "Прямая подача не входит в петлю: запасы и подавление искажений те же,\n"
                  "а интегратор отрабатывает только ошибку.", ha="left", va="top", size=8.5, c=GR)
T(*Q(-1.4, -8.8), "Rff = 2 × 1,5 к (≈ R3·K/G); Т-звено с Cfx 3,3 нФ — ФНЧ ≈ 65 кГц в ветви прямой подачи (как C50 с R57–R58 в NC400).\n"
                  "Cffc — плёнка. Источник дополнительно нагружен ≈ 2,9 кОм на плечо (параллельно основному пути).", ha="left", size=8.5, c=GR)

# --- title & notes
T(20.7, 31.3, "TPA3255: смешанная ООС, двойной интегратор, прямая подача и ограничитель по образцу NC400", size=15, w="bold")
T(20.7, 30.5, "Петля: срез 46–50 кГц, запас ≥ 46° / ≥ 10 дБ; инфранизкий срез 0,15 Гц, запас 62°; подавление искажений 41 / 14 / 7 дБ на 1 / 10 / 20 кГц; АЧХ 20 Гц–20 кГц ±0,4 дБ (4 / 8 Ом / х.х.); G = 26 дБ", size=10)
notes = [
    "Примечания",
    "1. Нижняя половина — зеркальная копия верхней. TPA3255 инвертирует: VOUT− → IN+, VOUT+ → IN−; ООС: SPK−/OUT− → VIN+, SPK+/OUT+ → VIN−. Проверить фазу до замыкания петли.",
    "2. Два пути ООС складываются в суммирующем узле: после фильтра — Cfb + Т-звено Rp1–Cp–Rp2 (ФНЧ ≈ 5 кГц), до фильтра — Cpre + RC-лестница (ФВЧ ≈ 5 кГц",
    "    и два полюса ≈ 250 кГц против несущей). Стык на 5 кГц выводит LC-резонанс из петли настолько, что хватает Зобеля 3,9 Ом / 470 нФ.",
    "3. Двойной интегратор: Т-цепочка Cd–(Rt+Cx)–Cd, ноль ≈ 1,3 кГц. Cx переводит его в одинарный ниже ~300 Гц, поэтому фаза петли на НЧ не опускается ниже −132°",
    "    (условной устойчивости нет). Rdc 1 МОм задаёт усиление по постоянному току. Cd, Cp, Cpre, 56 пФ — только C0G; Cpre на 100 В.",
    "4. Прямая подача (U3, как R63/R72 в NC400): вход идёт на TPA3255 в обход интегратора с весом G/K, интегратор отрабатывает только ошибку",
    "    (≤ 0,55 В пик вместо 2,0 В). Петля и подавление искажений не меняются. Ограничитель (×2) — по образцу Hypex NC400, порог ≈ 0,9–1,0 В.",
    "5. U1, U3, ограничитель и BAV99 питаются от ±4,8 В (LM317L/LM337L от ±15 В: R1 240 Ом, R2 680 Ом). U3 при этом не выходит за ≈ ±3,2 В —",
    "    это и есть защита входа U2 (допустимо 7 В п-п). Входной ФНЧ: Rsrc + Cdi, Ra–Cm–Rb; в прямой подаче 2 × 1,5 к + Cfx. Несущая на выходе U1 ≈ 50 мВ пик — проверить.",
    "6. Инфранизкие: ФВЧ петли — Cfb·40 кОм ≈ 40 Гц (ООС) и Cc·20 кОм ≈ 0,08 Гц (вход U2) разнесены, срез 0,15 Гц с запасом 62°; Cin и Cffc согласованы с Cfb.",
    "    Rz: синус полной мощности 20 кГц — 3,9 Вт, 10 кГц — 1 Вт, музыка — сотые доли ватта. /RESET держать в «0» ≈ 3 с (τ заряда Cc ≈ 2 с); ±4,8 В раньше PVDD.",
    "7. K ≈ 12 и задержка чипа ≈ 1 мкс — допущения расчёта; перед финалом измерить АЧХ/ФЧХ объекта без внешней петли.",
]
for i, s in enumerate(notes):
    T(-1.5, -4.8 - 0.62 * i, s, ha="left", size=9.5 if i else 10.5, w="bold" if i == 0 else "normal")

ax.set_xlim(-2.5, 44); ax.set_ylim(-41.6, 32)
plt.tight_layout()
plt.savefig("tpa3255_ff_schematic.png", dpi=140)
plt.savefig("tpa3255_ff_schematic.pdf")
