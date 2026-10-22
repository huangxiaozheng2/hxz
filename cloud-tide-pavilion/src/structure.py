"""结构体系（纯几何）：所有竖向支承的位置，以及屋面“到最近支承距离”的检查。

荷载路径：
    屋面板（轻质龙骨 + GRC 面层）→ 次肋（100 mm 木格栅）→ 主梁（胶合木，约 1.5 m 一根，与幕墙分格对齐）
    → 柱顶环梁（钢箱梁）→ 柱 / 承重墙 / 电梯筒 / 斜柱 → 基座下的灌注桩
次屋顶 B 的柱子立在主屋顶 A 的结构层上，并与下层柱上下对齐，使荷载直接落到基础。

本模块只做几何层面的合理性检查（悬挑与跨度），不替代结构计算。
"""
import math

import numpy as np
import shapely
from shapely.geometry import Point, LineString

import plan as P
from facade import envelope_bays, outer_side, angle_deg, e1_solid, e3_solid

COLUMN_EVERY = 3            # 每 3 个幕墙分格（约 4.5 m）一根柱
COLUMN_INSET = 0.85         # 柱心距玻璃线
CAPITAL_R = 0.25            # 柱头顶板半径（作为支承区域）


def _polar(r, deg):
    a = math.radians(deg)
    return np.array([r * math.cos(a), r * math.sin(a)])


def _envelope_columns(poly, side, skip_solid=None):
    """沿围合体每 3 个分格取一根柱：side = inner / outer / both（相对庭院中心）。"""
    dense, s, L, sm, mp, nrm = envelope_bays(poly)
    ro = outer_side(mp, nrm)
    solid = skip_solid(mp, nrm) if skip_solid else []
    out = []
    for i in range(0, len(mp), COLUMN_EVERY):
        if any(a <= i < b for a, b in solid):
            continue
        if side == "outer" and ro[i] < 0.35:
            continue
        if side == "inner" and ro[i] > -0.35:
            continue
        if side == "both" and abs(ro[i]) <= 0.35:
            continue
        c = mp[i] - nrm[i] * COLUMN_INSET
        if poly.buffer(-0.4).contains(Point(*c)):
            out.append(c)
    return np.array(out)


def _solid_lines(poly, rule):
    """承重实墙段（竖梃编号区间）转成平面折线。"""
    dense, s, L, sm, mp, nrm = envelope_bays(poly)
    n = len(mp)
    lines = []
    for a, b in rule(mp, nrm):
        sa, sb = sm[a % n], (sm[b] if b < n else L)
        sel = (s[:-1] >= sa - 1e-6) & (s[:-1] <= sb + 1e-6)
        if sel.sum() >= 2:
            lines.append(LineString(dense[sel]))
    return lines


# ----------------------------------------------------------------- 一层柱网
# 展廊：内侧（庭院一侧）柱列 + 西端高厅外侧柱列（外侧其余部分为承重实墙）
E1_COLS_INNER = _envelope_columns(P.E1, "inner")
E1_COLS_OUTER = _envelope_columns(P.E1, "outer", skip_solid=e1_solid)
E1_WALLS = _solid_lines(P.E1, e1_solid)
# 门厅：内外两侧柱列
E2_COLS = _envelope_columns(P.E2, "both")

# 主屋顶东翼尖下的柱（立于基座）
A_EAST_COLS = np.array([_polar(11.7, -8.0)])
# 低檐 C：西端一根柱（避开水口）；东翼尖由两处水中石墩承托（plan.C_PIERS）
C_EXTRA_COLS = np.array([_polar(12.4, 220.0)])

# ----------------------------------------------------------------- 二层柱（立于主屋顶结构层）
# 次屋顶 B 的柱与一层内侧柱上下对齐；另在主屋顶东翼尖柱之上再立一根
_b_zone = P.ROOF_B.poly.buffer(-0.6)
B_COLS = np.array([c for c in np.vstack([E1_COLS_INNER, A_EAST_COLS]) if _b_zone.contains(Point(*c))])
E3_WALLS = _solid_lines(P.E3, e3_solid)
MOON_LINE = LineString([P.MOON_C - P.AXIS_N * P.MOON_HALF, P.MOON_C + P.AXIS_N * P.MOON_HALF])


def _ycol_tops():
    return np.array([_polar(r, a) for r, a in P.YCOL_TOPS])


def supports(roof):
    """返回 [(类型, shapely 几何)]：该屋面的全部竖向支承。"""
    core = Point(*P.CORE_C).buffer(P.CORE_R)
    pts = lambda arr, kind: [(kind, Point(*p).buffer(CAPITAL_R)) for p in arr]
    if roof is P.ROOF_A:
        out = pts(E1_COLS_INNER, "柱") + pts(E1_COLS_OUTER, "柱") + pts(A_EAST_COLS, "柱")
        out += [("承重墙", w) for w in E1_WALLS] + [("电梯筒", core)] + pts(_ycol_tops(), "Y 形斜柱")
        return out
    if roof is P.ROOF_B:
        out = pts(B_COLS, "柱") + [("电梯筒", core), ("月洞墙", MOON_LINE)]
        out += [("承重墙", w) for w in E3_WALLS]
        return out
    out = pts(E2_COLS, "柱") + pts(C_EXTRA_COLS, "柱") + pts(P.C_PIERS, "水中石墩")
    return out


def support_distance(roof, step=0.25):
    """屋面采样点到最近支承的距离（m）。返回 (xy, dist)。"""
    minx, miny, maxx, maxy = roof.poly.bounds
    X, Y = np.meshgrid(np.arange(minx, maxx, step), np.arange(miny, maxy, step))
    inside = shapely.contains_xy(roof.poly, X, Y)
    xy = np.stack([X[inside], Y[inside]], 1)
    geoms = [g for _, g in supports(roof)]
    union = shapely.unary_union(geoms)
    d = shapely.distance(union, shapely.points(xy))
    return xy, d


# ----------------------------------------------------------------- 初步校核（方案阶段手算）
GLULAM_E = 12600.0          # GL28h 弹性模量 MPa
GLULAM_FM = 19.0            # 设计抗弯强度（约值）MPa
STEEL_FY = 355.0
PRIMARY_B = 0.16            # 主梁宽 m
PRIMARY_S = 1.5             # 主梁间距 m（与幕墙分格对齐）


def prelim_checks():
    """主梁悬挑、平台主梁、柱的初步校核。荷载：屋面 gk 1.5 / qk 0.5 kPa；屋面平台 gk 3.0 / qk 3.5 kPa；
    分项系数 1.3 / 1.5。仅用于判断构件尺度是否合理，不替代结构计算。"""
    out = {}
    w_roof_d, w_roof_k = 1.3 * 1.5 + 1.5 * 0.5, 1.5 + 0.5
    w_deck_d, w_deck_k = 1.3 * 3.0 + 1.5 * 3.5, 3.0 + 3.5
    for roof in P.ROOFS:                                               # 各屋面最大悬挑（按悬挑计，偏保守）
        L = float(support_distance(roof)[1].max())
        h = roof.beam_core
        W = PRIMARY_B * h ** 2 / 6 * 1e9                               # mm³
        I = PRIMARY_B * h ** 3 / 12 * 1e12                             # mm⁴
        M = w_roof_d * PRIMARY_S * L ** 2 / 2                          # kN·m
        sigma = M * 1e6 / W
        delta = (w_roof_k * PRIMARY_S) * (L * 1000) ** 4 / (8 * GLULAM_E * I)
        out[roof.name] = dict(L=L, h=h, M=M, sigma=sigma, util=sigma / GLULAM_FM, delta=delta, ratio=L * 1000 / delta)
    h = P.ROOF_A.beam_core
    W = PRIMARY_B * h ** 2 / 6 * 1e9
    I = PRIMARY_B * h ** 3 / 12 * 1e12
    # 屋面平台主梁：内侧柱列到承重外墙
    th = np.radians(90)
    r_out = max(r for r in np.arange(10, 25, 0.05) if P.E1.contains(Point(r * math.cos(th), r * math.sin(th))))
    span = r_out - float(np.hypot(*E1_COLS_INNER[np.argmin(np.abs(np.degrees(np.arctan2(E1_COLS_INNER[:, 1], E1_COLS_INNER[:, 0])) - 90))]))
    Md = w_deck_d * PRIMARY_S * span ** 2 / 8
    sig2 = Md * 1e6 / W
    d2 = 5 * (w_deck_k * PRIMARY_S) * (span * 1000) ** 4 / (384 * GLULAM_E * I)
    out["deck_beam"] = dict(L=span, M=Md, sigma=sig2, util=sig2 / GLULAM_FM, delta=d2, ratio=span * 1000 / d2)
    # 柱：一层内侧柱（叠加二层柱荷载）。钢管 φ193.7×10
    A_s = math.pi / 4 * (193.7 ** 2 - 173.7 ** 2)
    i = math.sqrt((193.7 ** 2 + 173.7 ** 2) / 16)
    Lc = 5.0
    lam = Lc * 1000 / i
    lam_bar = lam / (math.pi * math.sqrt(210000 / STEEL_FY))
    phi = 0.5 * (1 + 0.21 * (lam_bar - 0.2) + lam_bar ** 2)
    chi = 1 / (phi + math.sqrt(phi ** 2 - lam_bar ** 2))
    Nb = chi * A_s * STEEL_FY / 1000
    trib_a = 4.5 * (span / 2 + 2.3)
    N = trib_a * w_deck_d + 4.5 * 5.0 * w_roof_d
    out["column"] = dict(N=N, Nb=Nb, util=N / Nb, lam=lam)
    return out


def report():
    for roof in P.ROOFS:
        xy, d = support_distance(roof)
        i = int(np.argmax(d))
        th = math.degrees(math.atan2(xy[i, 1], xy[i, 0])) % 360
        print(f"{roof.name}: 最大距离 {d.max():.2f} m @ θ={th:.0f}° r={np.hypot(*xy[i]):.1f}; "
              f"95% 分位 {np.percentile(d, 95):.2f} m; 支承 {len(supports(roof))} 处")


if __name__ == "__main__":
    print("E1 内侧柱", len(E1_COLS_INNER), "外侧柱", len(E1_COLS_OUTER), "承重墙段", len(E1_WALLS))
    print("E2 柱", len(E2_COLS), "B 柱", len(B_COLS))
    for c in np.vstack([A_EAST_COLS, C_EXTRA_COLS, P.C_PIERS]):
        print("extra", np.round(c, 2), "on plinth:", P.PLINTH_CUT.contains(Point(*c)),
              "in channel:", P.CHANNEL.contains(Point(*c)))
    report()
    for k, v in prelim_checks().items():
        print(k, {a: round(b, 2) for a, b in v.items()})
