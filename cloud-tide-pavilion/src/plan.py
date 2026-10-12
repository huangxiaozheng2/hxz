"""云上潮汐馆 —— 平面与高程定义（纯几何，不依赖 Blender）。

坐标：x 向东，y 向北，z 向上，单位米。庭院中心位于原点，水面 z = 0。
三片屋顶都是围绕庭院的“月牙”：
    A  主屋顶  —— 北侧，舒展，西端翼尖高高扬起
    B  次屋顶  —— 叠于 A 之上的二层，轻柔回旋，探入庭院上空
    C  第三屋顶 —— 南侧，低伏于水面，覆盖入口门廊与门厅
"""
import math
import numpy as np
import shapely
from shapely.geometry import Polygon, Point, LineString
from scipy.interpolate import PchipInterpolator

FFL = 0.45          # 室内/平台完成面标高
DECK = 6.30         # 二层屋顶平台完成面
RIB_DEPTH = 0.10    # 木肋高度
EDGE_T = 0.15       # 屋檐最薄处厚度
REVEAL = 0.12       # 檐下阴影缝宽度


def smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def resample_ring(coords, step):
    """沿闭合折线按弧长等距重采样，返回 (N,2)，不重复首点。"""
    pts = np.asarray(coords, float)
    if np.allclose(pts[0], pts[-1]):
        pts = pts[:-1]
    closed = np.vstack([pts, pts[:1]])
    seg = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    L = s[-1]
    n = max(8, int(round(L / step)))
    si = np.linspace(0, L, n, endpoint=False)
    x = np.interp(si, s, closed[:, 0])
    y = np.interp(si, s, closed[:, 1])
    return np.stack([x, y], 1)


def resample_open(coords, step=None, n=None):
    pts = np.asarray(coords, float)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    if n is None:
        n = max(2, int(round(s[-1] / step)) + 1)
    si = np.linspace(0, s[-1], n)
    return np.stack([np.interp(si, s, pts[:, 0]), np.interp(si, s, pts[:, 1])], 1)


def ccw(ring):
    a = 0.5 * np.sum(ring[:, 0] * np.roll(ring[:, 1], -1) - np.roll(ring[:, 0], -1) * ring[:, 1])
    return ring if a > 0 else ring[::-1]


class Crescent:
    """以庭院中心为极点的月牙形屋顶。

    theta_s/theta_e：起止角（度，逆时针，可超过 360）
    rin_keys / w_keys：沿 t∈[0,1] 的内缘半径与宽度控制点
    h_keys：沿 t 的屋面标高（中线顶面）
    """

    def __init__(self, name, theta_s, theta_e, rin_keys, w_keys, h_keys,
                 t_core, curl=0.0, crown=0.06, round_r=0.7, flat=None):
        self.name = name
        self.ts, self.te = theta_s, theta_e
        self.span = theta_e - theta_s
        self.rin = PchipInterpolator(*zip(*rin_keys))
        self.wid = PchipInterpolator(*zip(*w_keys))
        self.h = PchipInterpolator(*zip(*h_keys))
        self.t_core = t_core
        self.curl = curl
        self.crown = crown
        self.flat = flat  # (t0,t1) 区间内保持平整（屋顶平台）
        tt = np.linspace(0, 1, 900)
        th = np.radians(self.ts + tt * self.span)
        ri = self.rin(tt)
        ro = ri + np.maximum(self.wid(tt), 0)
        outer = np.stack([ro * np.cos(th), ro * np.sin(th)], 1)
        inner = np.stack([ri * np.cos(th), ri * np.sin(th)], 1)[::-1]
        poly = Polygon(np.vstack([outer, inner])).buffer(0)
        poly = poly.buffer(-round_r, 64).buffer(round_r, 64)
        if poly.geom_type != "Polygon":
            poly = max(poly.geoms, key=lambda g: g.area)
        self.poly = shapely.simplify(poly, 0.005)
        shapely.prepare(self.poly)
        self.boundary = self.poly.exterior

    # ---- 参数化 ----
    def tv(self, x, y):
        th = np.degrees(np.arctan2(y, x))
        t = ((th - self.ts) % 360.0) / self.span
        t = np.where(t > 1.0 + (360 - self.span) / self.span * 0.5, t - 360.0 / self.span, t)
        t = np.clip(t, 0, 1)
        r = np.hypot(x, y)
        ri = self.rin(t)
        w = np.maximum(self.wid(t), 0.5)
        v = np.clip((r - ri) / w, 0, 1)
        return t, v

    def dist_edge(self, x, y):
        pts = shapely.points(np.stack([np.atleast_1d(x), np.atleast_1d(y)], 1))
        return shapely.distance(self.boundary, pts)

    def thickness(self, d):
        return EDGE_T + (self.t_core - EDGE_T) * smoothstep(0.0, 4.5, d) ** 0.85

    def gutter(self, d):
        # 屋面排水浅槽：距边 0.45~0.75 m 的余弦凹槽
        u = np.clip((d - 0.45) / 0.30, 0, 1)
        return -0.035 * np.sin(np.pi * u) * ((d > 0.45) & (d < 0.75))

    def surfaces(self, x, y, d=None):
        """返回 (z_top, z_bot)。"""
        x = np.atleast_1d(x); y = np.atleast_1d(y)
        if d is None:
            d = self.dist_edge(x, y)
        t, v = self.tv(x, y)
        base = self.h(t)
        flatmask = np.zeros_like(t)
        if self.flat is not None:
            a, b = self.flat
            flatmask = smoothstep(a - 0.08, a, t) * (1 - smoothstep(b, b + 0.08, t))
        lift = self.curl * v ** 2.2 * (1 - flatmask)
        crown = self.crown * 4 * v * (1 - v)
        th = self.thickness(d)
        top = base + lift + crown - 0.32 * (self.t_core - th) + self.gutter(d)
        bot = base + lift + crown - 0.32 * (self.t_core - th) - th
        return top, bot


# ---------------------------------------------------------------- 三片屋顶
ROOF_A = Crescent(
    "Roof_A_Main", -35, 250,
    rin_keys=[(0, 12.8), (0.12, 9.3), (0.28, 8.1), (0.55, 8.0), (0.72, 8.9), (0.86, 12.2), (1, 19.0)],
    w_keys=[(0, 0.0), (0.07, 4.2), (0.25, 9.6), (0.50, 12.2), (0.72, 10.2), (0.88, 5.2), (1, 0.0)],
    h_keys=[(0, 4.95), (0.12, 5.75), (0.22, 5.95), (0.64, 5.95), (0.75, 6.95), (0.88, 8.6), (1, 11.2)],
    t_core=0.85, curl=1.3, crown=0.05, flat=(0.22, 0.64),
)

ROOF_B = Crescent(
    "Roof_B_Upper", -25, 200,
    rin_keys=[(0, 14.2), (0.12, 9.2), (0.35, 6.9), (0.65, 6.9), (0.88, 8.7), (1, 11.8)],
    w_keys=[(0, 0.0), (0.10, 3.4), (0.40, 8.4), (0.62, 8.6), (0.90, 3.6), (1, 0.0)],
    h_keys=[(0, 8.55), (0.18, 10.35), (0.5, 10.95), (0.8, 11.3), (1, 12.1)],
    t_core=0.62, curl=0.45, crown=0.04,
)

ROOF_C = Crescent(
    "Roof_C_Low", 188, 398,
    rin_keys=[(0, 10.6), (0.15, 9.5), (0.50, 9.4), (0.72, 10.0), (0.86, 13.4), (1, 19.5)],
    w_keys=[(0, 0.0), (0.08, 3.0), (0.35, 8.0), (0.55, 9.0), (0.80, 6.2), (0.92, 3.0), (1, 0.0)],
    h_keys=[(0, 5.05), (0.25, 4.0), (0.5, 3.78), (0.66, 3.72), (0.85, 3.05), (1, 2.15)],
    t_core=0.55, curl=0.35, crown=0.04,
)

ROOFS = [ROOF_A, ROOF_B, ROOF_C]


def roof_soffit(roof, x, y):
    _, bot = roof.surfaces(x, y)
    return bot


# ---------------------------------------------------------------- 玻璃围合体
def polar_band(theta_s, theta_e, r_in_fn, r_out_fn, n=400, round_r=0.9):
    tt = np.linspace(0, 1, n)
    th = np.radians(theta_s + tt * (theta_e - theta_s))
    ri = r_in_fn(tt); ro = r_out_fn(tt)
    outer = np.stack([ro * np.cos(th), ro * np.sin(th)], 1)
    inner = np.stack([ri * np.cos(th), ri * np.sin(th)], 1)[::-1]
    p = Polygon(np.vstack([outer, inner])).buffer(0)
    return p.buffer(-round_r, 64).buffer(round_r, 64)


def _safe_inset(poly, roof, inset):
    return poly.intersection(roof.poly.buffer(-inset, 64))


# 一层主展廊：A 屋顶下的月牙形玻璃体
E1 = polar_band(
    8, 180,
    lambda t: 9.4 + 0 * t,
    lambda t: 9.4 + PchipInterpolator([0, .12, .35, .6, .82, 1], [3.0, 6.0, 8.6, 9.4, 8.2, 4.4])(t),
)
E1 = _safe_inset(E1, ROOF_A, 1.9).buffer(-0.5, 64).buffer(0.5, 64)

# 门厅：C 屋顶下，入口门廊自南侧凹入
_lobby = polar_band(
    232, 318,
    lambda t: 10.6 + 0 * t,
    lambda t: 10.6 + PchipInterpolator([0, .2, .5, .8, 1], [2.6, 5.0, 5.6, 5.0, 2.6])(t),
)
_porch_cut = Polygon([(-2.3, -30), (2.3, -30), (2.3, -14.0), (-2.3, -14.0)]).buffer(1.0, 64)
E2 = _safe_inset(_lobby, ROOF_C, 1.9).difference(_porch_cut).buffer(-0.4, 64).buffer(0.4, 64)

# 二层观景厅：A 屋面平台上、B 屋顶下
E3 = polar_band(
    56, 124,
    lambda t: 10.0 + 0 * t,
    lambda t: 10.0 + PchipInterpolator([0, .3, .7, 1], [3.7, 4.4, 4.4, 3.7])(t),
    round_r=0.6,
)
E3 = _safe_inset(E3, ROOF_B, 1.8)

# 屋面平台（A 顶面上的铺装层）
DECK_POLY = polar_band(
    24, 146,
    lambda t: 8.75 + 0 * t,
    lambda t: 8.75 + PchipInterpolator([0, .25, .6, 1], [5.5, 10.0, 10.5, 6.5])(t),
    round_r=1.5,
).intersection(ROOF_A.poly.buffer(-1.25, 64))

# 庭院
COURT_C = (0.0, 0.0)
TREE_POS = (1.45, 1.05)          # 略偏离中心
SCULPT_POS = (-2.55, -1.95)      # 与树形成平衡的磨砂玻璃雕塑

# 绕庭院上升的弧形坡道（逆时针，西侧起步，北侧抵达屋面平台）
RAMP_R = 6.35
RAMP_W = 1.6
RAMP_T0, RAMP_T1 = 140.0, 448.0

# 一层平台（浮于水面的石材基座）
PLINTH = shapely.unary_union([
    E1.buffer(3.6, 64), E2.buffer(3.4, 64), Point(0, 0).buffer(9.8, 64),
]).buffer(1.5, 64).buffer(-1.5, 64)

# 入口栈桥：S 形，自西南岸线来
BRIDGE_PATH = np.array([
    (-58, -66), (-46, -60), (-32, -52), (-20, -44), (-10, -37), (-3, -31), (0, -26.5), (0, -17.0),
])

def _bridge_poly():
    from scipy.ndimage import gaussian_filter1d
    path = resample_open(BRIDGE_PATH, 0.4)
    path = np.stack([gaussian_filter1d(path[:, 0], 4, mode="nearest"),
                     gaussian_filter1d(path[:, 1], 4, mode="nearest")], 1)
    return LineString(path).buffer(1.6, cap_style="flat", join_style="round")


BRIDGE = _bridge_poly()
# 基座与栈桥合为一体，交接处以 1.2 m 圆角过渡，避免拼缝
PLINTH_ALL = PLINTH.union(BRIDGE).buffer(1.2, 64).buffer(-1.2, 64)

# 柱下室内雕塑与螺旋楼梯（一层西厅）
STAIR_C = (12.3 * math.cos(math.radians(166)), 12.3 * math.sin(math.radians(166)))
STAIR_PLATFORM_Z = 3.0


def plot(path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11, 11))
    for poly, c, lab in [(PLINTH, "#cfd8dc", "plinth"), (ROOF_C.poly, "#ffcc80", "C"), (ROOF_A.poly, "#90caf9", "A"),
                         (ROOF_B.poly, "#ce93d8", "B")]:
        x, y = poly.exterior.xy
        ax.fill(x, y, alpha=0.35, fc=c, ec="k", lw=0.6, label=lab)
    for poly, c in [(E1, "b"), (E2, "r"), (E3, "m"), (DECK_POLY, "g")]:
        x, y = poly.exterior.xy
        ax.plot(x, y, c, lw=1.4)
    th = np.radians(np.linspace(RAMP_T0, RAMP_T1, 200))
    for r in (RAMP_R - RAMP_W / 2, RAMP_R + RAMP_W / 2):
        ax.plot(r * np.cos(th), r * np.sin(th), "c")
    ax.plot(*BRIDGE_PATH.T, "k--")
    ax.plot(*TREE_POS, "go", ms=12); ax.plot(*SCULPT_POS, "ks")
    ax.plot(*STAIR_C, "r*", ms=14)
    ax.set_aspect("equal"); ax.grid(alpha=.3); ax.legend()
    ax.set_xlim(-32, 32); ax.set_ylim(-34, 30)
    fig.savefig(path, dpi=80)

    # 沿角度的高程展开图
    fig, ax = plt.subplots(figsize=(14, 4))
    for roof, c in [(ROOF_A, "b"), (ROOF_B, "m"), (ROOF_C, "orange")]:
        tt = np.linspace(0, 1, 300)
        th = roof.ts + tt * roof.span
        for vv in (0.05, 0.5, 0.95):
            r = roof.rin(tt) + vv * np.maximum(roof.wid(tt), 0)
            x = r * np.cos(np.radians(th)); y = r * np.sin(np.radians(th))
            top, bot = roof.surfaces(x, y)
            ax.plot(th % 360, top, c, lw=0.8); ax.plot(th % 360, bot, c, lw=0.5, ls=":")
    ax.axhline(DECK, c="g"); ax.axhline(FFL, c="k")
    fig.savefig(path.replace(".png", "_elev.png"), dpi=70)


if __name__ == "__main__":
    import sys
    plot(sys.argv[1] if len(sys.argv) > 1 else "plan.png")
    for n, p in [("A", ROOF_A.poly), ("B", ROOF_B.poly), ("C", ROOF_C.poly), ("E1", E1), ("E2", E2), ("E3", E3), ("deck", DECK_POLY)]:
        print(n, p.geom_type, round(p.area, 1))
