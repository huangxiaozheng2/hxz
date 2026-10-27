"""功能与疏散（纯几何）：房间面积、安全出口、任一点到最近出口的距离。diagrams.py 与 boards.py 共用。

疏散距离按 0.25 m 网格计算：可走区域为室内与平台，玻璃与实墙不可穿越（门洞除外），
楼梯间、电梯筒、月洞墙为障碍；同时给出直线距离，便于对照 GB 50016 第 5.5.17 条
（展览厅内任一点至最近疏散门的直线距离不大于 30 m）。
"""
import heapq
import math

import numpy as np
import shapely
from shapely.geometry import Point, LineString, Polygon

import plan as P
from facade import envelope_bays, e1_solid, e1_doors, e2_doors, e3_solid, e3_doors

CORE = Point(*P.CORE_C).buffer(P.CORE_R, 64)
MOON_WALL = LineString([P.MOON_C - P.AXIS_N * P.MOON_HALF, P.MOON_C + P.AXIS_N * P.MOON_HALF]).buffer(0.17)


def bay_segments(poly, solid_fn=None, doors_fn=None):
    """围合体逐分格：[(类型 solid/door/glass, 折线, 室内方向)]。"""
    dense, s, L, sm, mp, nrm = envelope_bays(poly)
    closed = np.vstack([dense, dense[:1]])
    n = len(mp)
    sol = solid_fn(mp, nrm) if solid_fn else []
    drs = doors_fn(mp, nrm) if doors_fn else []
    out = []
    for i in range(n):
        a, b = sm[i], (sm[i + 1] if i + 1 < n else L)
        sel = (s > a) & (s < b)
        pa = [np.interp(a, s, closed[:, 0]), np.interp(a, s, closed[:, 1])]
        pb = [np.interp(b, s, closed[:, 0]), np.interp(b, s, closed[:, 1])]
        pts = np.vstack([pa, closed[sel], pb])
        kind = "solid" if any(x <= i < y for x, y in sol) else "door" if i in drs else "glass"
        out.append((kind, pts, -nrm[i]))
    return out


def _wedge(a0, a1, r=40.0):
    th = np.radians(np.linspace(a0, a1, 64))
    return Polygon(np.vstack([[[0.0, 0.0]], np.stack([r * np.cos(th), r * np.sin(th)], 1)]))


def service_partition():
    """卫生间隔墙（θ = WC_THETA）与门洞的径向范围。"""
    a = math.radians(P.WC_THETA)
    d = np.array([math.cos(a), math.sin(a)])
    seg = LineString([d * 5, d * 30]).intersection(P.E1.buffer(-0.05))
    r0, r1 = sorted(float(np.hypot(*q)) for q in seg.coords)
    return d, r0, r1, (r0 + 0.6, r0 + 1.6)


def stair2_door():
    """楼梯间西端门洞中点（墙外 0.3 m）。"""
    c, e_r, e_t = P.stair2_frame()
    return c + e_r * 0.6 + e_t * (P.STAIR2_HALF + 0.3)


# ----------------------------------------------------------------- 房间
WC = P.E1.intersection(_wedge(P.WC_SPLIT_THETA, P.WC_THETA))
STORE = P.E1.intersection(_wedge(0.0, P.WC_SPLIT_THETA))
GALLERY = P.E1.difference(P.SERVICE).difference(P.STAIR2).difference(CORE)
TERRACE = P.DECK_POLY.difference(P.E3).difference(P.STAIR2).difference(CORE).difference(MOON_WALL)


def ramp_landing():
    hw = P.RAMP_W / 2
    ang = np.radians(np.linspace(P.RAMP_END_DEG - 7.5, P.RAMP_END_DEG + 6.5, 30))
    return Polygon(np.vstack([np.stack([(P.RAMP_R - hw) * np.cos(ang), (P.RAMP_R - hw) * np.sin(ang)], 1),
                              np.stack([9.2 * np.cos(ang[::-1]), 9.2 * np.sin(ang[::-1])], 1)]))


def rooms():
    """[(名称, 面积 m², 所在层)]"""
    return [
        ("展廊", GALLERY.area, "首层"),
        ("门厅（含接待）", P.E2.area, "首层"),
        ("卫生间（含无障碍）", WC.area, "首层"),
        ("储藏 / 设备检修口", STORE.area, "首层"),
        ("疏散楼梯间", P.STAIR2.area, "首层—平台"),
        ("电梯", CORE.area, "首层—平台"),
        ("设备间（基座内）", P.PLANT_ROOM.area, "基座"),
        ("观景厅", P.E3.area, "屋面平台"),
        ("屋面平台（室外）", TERRACE.area, "屋面平台"),
    ]


# ----------------------------------------------------------------- 疏散距离
def _barriers(segments):
    lines = [LineString(pts).buffer(0.13, cap_style="flat") for kind, pts, _ in segments if kind != "door"]
    return shapely.unary_union(lines)


def _distance_field(walk, sources, step=0.25):
    minx, miny, maxx, maxy = walk.bounds
    xs = np.arange(minx, maxx, step); ys = np.arange(miny, maxy, step)
    X, Y = np.meshgrid(xs, ys)
    ok = shapely.contains_xy(walk, X, Y)
    src = shapely.contains_xy(sources, X, Y) & ok
    D = np.full(ok.shape, np.inf)
    pq = [(0.0, i, j) for i, j in zip(*np.where(src))]
    for _, i, j in pq:
        D[i, j] = 0.0
    heapq.heapify(pq)
    nb = [(-1, 0, 1), (1, 0, 1), (0, -1, 1), (0, 1, 1), (-1, -1, 1.4142), (-1, 1, 1.4142), (1, -1, 1.4142),
          (1, 1, 1.4142)]
    H, Wd = ok.shape
    while pq:
        d, i, j = heapq.heappop(pq)
        if d > D[i, j]:
            continue
        for di, dj, c in nb:
            a, b = i + di, j + dj
            if 0 <= a < H and 0 <= b < Wd and ok[a, b]:
                nd = d + c * step
                if nd < D[a, b]:
                    D[a, b] = nd
                    heapq.heappush(pq, (nd, a, b))
    return X, Y, ok, D


def ground_egress():
    """首层：展廊、门厅、卫生间到最近对外门（庭院门、门厅南门）的距离。"""
    seg1 = bay_segments(P.E1, e1_solid, e1_doors)
    seg2 = bay_segments(P.E2, None, e2_doors)
    d, r0, r1, (dr0, dr1) = service_partition()
    part = shapely.unary_union([LineString([d * r0, d * dr0]), LineString([d * dr1, d * r1])]).buffer(0.08)
    walk = shapely.unary_union([P.E1, P.E2]).difference(_barriers(seg1 + seg2)).difference(P.STAIR2.buffer(0.05))
    walk = walk.difference(CORE).difference(part)
    doors = [pts for kind, pts, _ in seg1 + seg2 if kind == "door"]
    sources = shapely.unary_union([LineString(p).buffer(0.3) for p in doors])
    X, Y, ok, D = _distance_field(walk, sources)
    dpts = np.vstack([p.mean(0) for p in doors])
    straight = np.min(np.hypot(X[ok][:, None] - dpts[None, :, 0], Y[ok][:, None] - dpts[None, :, 1]), axis=1)
    return dict(X=X, Y=Y, ok=ok, D=D, exits=dpts, walk_max=float(np.nanmax(np.where(np.isfinite(D[ok]), D[ok], np.nan))),
                straight_max=float(straight.max()))


def deck_egress():
    """屋面平台与观景厅：到坡道到达点、疏散楼梯间门的距离。"""
    seg3 = bay_segments(P.E3, e3_solid, e3_doors)
    land = ramp_landing()
    walk = shapely.unary_union([P.DECK_POLY.buffer(-0.3), land, P.E3.buffer(-0.05)])
    walk = walk.difference(_barriers(seg3)).difference(P.STAIR2.buffer(0.05)).difference(CORE).difference(MOON_WALL)
    for kind, pts, _ in seg3:
        if kind == "door":
            walk = walk.union(LineString(pts).buffer(0.35, cap_style="flat"))
    ramp_exit = land.intersection(Point(0, 0).buffer(8.4))
    stair_exit = Point(*stair2_door()).buffer(0.45)
    X, Y, ok, D = _distance_field(walk, shapely.unary_union([ramp_exit, stair_exit]))
    ex = np.array([np.array(ramp_exit.centroid.coords[0]), stair2_door()])
    straight = np.min(np.hypot(X[ok][:, None] - ex[None, :, 0], Y[ok][:, None] - ex[None, :, 1]), axis=1)
    fin = np.isfinite(D[ok])
    return dict(X=X, Y=Y, ok=ok, D=D, exits=ex, walk_max=float(D[ok][fin].max()), straight_max=float(straight.max()),
                unreached=int((~fin).sum()), exit_gap=float(np.linalg.norm(ex[0] - ex[1])))


def farthest_points(eg, k=3, sep=8.0):
    """距出口最远的 k 个点（彼此相距不小于 sep）。"""
    X, Y, ok, D = eg["X"], eg["Y"], eg["ok"], eg["D"]
    m = ok & np.isfinite(D)
    order = np.argsort(-D[m])
    xs, ys, ds = X[m][order], Y[m][order], D[m][order]
    out = []
    for x, y, d in zip(xs, ys, ds):
        if all(math.hypot(x - a, y - b) >= sep for a, b, _ in out):
            out.append((x, y, d))
        if len(out) == k:
            break
    return out


def trace_path(eg, start):
    """沿距离场下降，得到从 start 到最近出口的步行路线。"""
    X, Y, ok, D = eg["X"], eg["Y"], eg["ok"], eg["D"]
    i, j = np.unravel_index(np.argmin(np.where(ok, (X - start[0]) ** 2 + (Y - start[1]) ** 2, np.inf)), X.shape)
    path = [(X[i, j], Y[i, j])]
    H, Wd = ok.shape
    while D[i, j] > 0:
        best = None
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                a, b = i + di, j + dj
                if (di or dj) and 0 <= a < H and 0 <= b < Wd and ok[a, b] and (best is None or D[a, b] < D[best]):
                    best = (a, b)
        if best is None or D[best] >= D[i, j]:
            break
        i, j = best
        path.append((X[i, j], Y[i, j]))
    path = np.array(path)
    if len(path) > 6:                                          # 轻度平滑
        from scipy.ndimage import uniform_filter1d
        sm = np.stack([uniform_filter1d(path[:, 0], 7, mode="nearest"), uniform_filter1d(path[:, 1], 7, mode="nearest")], 1)
        sm[0], sm[-1] = path[0], path[-1]
        path = sm
    return path


if __name__ == "__main__":
    for n, a, lv in rooms():
        print(f"{lv:6s} {n:12s} {a:6.1f} m²")
    g = ground_egress()
    print("首层：步行最远 %.1f m，直线最远 %.1f m，出口 %d 处" % (g["walk_max"], g["straight_max"], len(g["exits"])))
    k = deck_egress()
    print("平台：步行最远 %.1f m，直线最远 %.1f m，两出口相距 %.1f m，未连通格 %d" % (
        k["walk_max"], k["straight_max"], k["exit_gap"], k["unreached"]))
