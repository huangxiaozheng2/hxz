"""分析图：总平面与日落轴线、沿轴线长剖面、潮位曲线。全部由 plan.py 的同一套几何生成。
用法：python diagrams.py 输出目录
"""
import math
import os
import sys

import numpy as np
import shapely
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Circle, Polygon as MPoly, FancyArrowPatch
from shapely.geometry import Point

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plan as P  # noqa: E402

FONT = "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"
font_manager.fontManager.addfont(FONT)
plt.rcParams["font.family"] = font_manager.FontProperties(fname=FONT).get_name()
plt.rcParams["axes.unicode_minus"] = False

INK = "#2b2a27"
MUTED = "#8a837a"
POCHE = "#33312e"
STONE = "#efe9df"
STONE_BEYOND = "#dcd5ca"
WATER = "#dfe7ec"
WATER_HT = "#b8ccd8"
GLASS = "#6f8794"
OAK = "#b48a5e"
SUN = "#c8743a"
PAPER = "#f7f4ee"


def fill_poly(ax, g, **kw):
    parts = [g] if g.geom_type == "Polygon" else list(getattr(g, "geoms", []))
    for p in parts:
        if p.is_empty:
            continue
        ax.add_patch(MPoly(np.asarray(p.exterior.coords), closed=True, **kw))
        for i in p.interiors:
            ax.add_patch(MPoly(np.asarray(i.coords), closed=True, fc=kw.get("hole_fc", WATER), ec="none"))


def line_poly(ax, g, **kw):
    parts = [g] if g.geom_type == "Polygon" else list(getattr(g, "geoms", []))
    for p in parts:
        x, y = p.exterior.xy
        ax.plot(x, y, **kw)


def scale_bar(ax, x0, y0, units=(0, 5, 10, 20), h=0.5, label_size=9):
    for a, b in zip(units[:-1], units[1:]):
        k = units.index(a)
        ax.add_patch(MPoly([(x0 + a, y0), (x0 + b, y0), (x0 + b, y0 + h), (x0 + a, y0 + h)], closed=True,
                           fc=INK if k % 2 == 0 else "white", ec=INK, lw=0.8))
    for u in units:
        ax.text(x0 + u, y0 - 0.6, f"{u}", ha="center", va="top", fontsize=label_size, color=INK)
    ax.text(x0 + units[-1] + 1.0, y0 - 0.6, "m", ha="left", va="top", fontsize=label_size, color=INK)


# ======================================================================= 总平面
def site_plan(path):
    fig = plt.figure(figsize=(11, 10.5), dpi=200, facecolor=PAPER)
    ax = fig.add_axes([0.02, 0.02, 0.96, 0.9])
    ax.set_facecolor(WATER)
    ax.set_xlim(-36, 30); ax.set_ylim(-31, 32); ax.set_aspect("equal"); ax.axis("off")
    ax.add_patch(MPoly([(-36, -31), (30, -31), (30, 32), (-36, 32)], closed=True, fc=WATER, ec="none", zorder=0))
    # 基座与潮池
    fill_poly(ax, P.PLINTH_CUT, fc=STONE, ec="#8d877e", lw=0.8, zorder=1)
    for r_out, r_in, z in P.BASIN_STEPS:
        ring = Point(0, 0).buffer(r_out, 128).difference(Point(0, 0).buffer(r_in, 128)).difference(P.CHANNEL)
        fill_poly(ax, ring, fc=WATER_HT if z < P.HIGH_TIDE else "#e6dfd3", ec="#a59e93", lw=0.5, zorder=2)
    fill_poly(ax, Point(0, 0).buffer(P.BASIN_STEPS[-1][1], 128).union(P.CHANNEL.intersection(P.PLINTH_ALL.buffer(0.4))),
              fc=WATER_HT, ec="#a59e93", lw=0.5, zorder=2)
    for r in (P.RAMP_R - P.RAMP_W / 2, P.RAMP_R + P.RAMP_W / 2):
        th = np.radians(np.linspace(0, 360, 361))
        ax.plot(r * np.cos(th), r * np.sin(th), color="#7d766d", lw=0.7, zorder=3)
    # 树岛 / 雕塑
    ax.add_patch(Circle(P.TREE_POS, P.TREE_ISLAND_R, fc="#c9cdb2", ec="#7f8466", lw=0.6, zorder=3))
    ax.add_patch(Circle(P.TREE_POS, 3.2, fc="none", ec="#8f9a70", lw=0.6, ls=(0, (2, 2)), zorder=3))
    ax.add_patch(Circle(P.SCULPT_POS, 0.95, fc="white", ec=INK, lw=0.6, zorder=3))
    # 围合体
    for g in (P.E1, P.E2, P.E3):
        line_poly(ax, g, color=GLASS, lw=1.0, zorder=4)
    for sk in P.SKYLIGHTS:
        fill_poly(ax, sk, fc="#f2dfae", ec="#c9a85a", lw=0.5, zorder=4)
    # 屋顶
    for roof, ls, lab, pos in ((P.ROOF_A, "-", "A  主屋顶", (-21.5, 15.5)), (P.ROOF_B, "--", "B  次屋顶", (2.5, 16.6)),
                               (P.ROOF_C, "-.", "C  低檐", (9.0, -19.5))):
        fill_poly(ax, roof.poly, fc="#8a7f70", alpha=0.07, ec="none", zorder=5)
        line_poly(ax, roof.poly, color=INK, lw=1.5, ls=ls, zorder=5)
        ax.text(*pos, lab, fontsize=12, color=INK, zorder=8)
    # 结构与竖向交通
    ax.add_patch(Circle(P.CORE_C, P.CORE_R, fc=POCHE, ec="none", zorder=6))
    a = P.MOON_C - P.AXIS_N * (P.MOON_HALF + 0.9); b = P.MOON_C + P.AXIS_N * (P.MOON_HALF + 0.9)
    ax.plot([a[0], b[0]], [a[1], b[1]], color=POCHE, lw=4.5, solid_capstyle="round", zorder=6)
    ax.add_patch(Circle(P.YCOL_BASE, 1.05, fc=STONE_BEYOND, ec=INK, lw=0.6, zorder=6))
    for r, ang in P.YCOL_TOPS:
        t = (r * math.cos(math.radians(ang)), r * math.sin(math.radians(ang)))
        ax.plot([P.YCOL_BASE[0], t[0]], [P.YCOL_BASE[1], t[1]], color=INK, lw=1.6, zorder=6)
    ax.add_patch(Circle(P.C_PIER, 0.62, fc=STONE_BEYOND, ec=INK, lw=0.6, zorder=6))
    # 日照方位
    for az, lab, col, lw in ((270.0, "春秋分日落 270°", MUTED, 0.9), (297.4, "夏至日落 297.4°", MUTED, 0.9)):
        d = np.array([math.sin(math.radians(az)), math.cos(math.radians(az))])
        ax.plot([0, d[0] * 34], [0, d[1] * 34], color=col, lw=lw, ls=(0, (4, 3)), zorder=7)
        tp = d * 29 + np.array([-d[1], d[0]]) * 1.3
        ax.text(tp[0], tp[1], lab, fontsize=9, color=col, ha="center", va="center", zorder=8,
                rotation=math.degrees(math.atan2(d[1], d[0])) - 180,
                bbox=dict(fc=WATER, ec="none", pad=0.5))
    ax.plot([-P.AXIS[0] * 20, P.AXIS[0] * 40], [-P.AXIS[1] * 20, P.AXIS[1] * 40], color=SUN, lw=2.0,
            ls=(0, (7, 3)), zorder=7)
    sp = P.AXIS * 31.5
    ax.add_patch(Circle(sp, 1.1, fc=SUN, ec="none", zorder=8))
    for k in range(8):
        a_ = k * math.pi / 4
        ax.plot([sp[0] + 1.5 * math.cos(a_), sp[0] + 2.2 * math.cos(a_)],
                [sp[1] + 1.5 * math.sin(a_), sp[1] + 2.2 * math.sin(a_)], color=SUN, lw=1.2, zorder=8)
    ax.text(sp[0] + 0.5, sp[1] + 3.0, "冬至日落 242.6°", fontsize=11, color=SUN, ha="center", zorder=8)

    def note(xy, txt, xytext, ha="left"):
        ax.annotate(txt, xy=xy, xytext=xytext, fontsize=9.5, color=INK, ha=ha, zorder=9,
                    arrowprops=dict(arrowstyle="-", color=INK, lw=0.6, shrinkA=0, shrinkB=2))
    note(P.AXIS * 11, "水口", (-27, -4))
    note((0.6, -0.4), "潮池", (6.5, -2.6))
    note(P.TREE_POS, "树岛", (-8, 3.5), ha="right")
    note(P.SCULPT_POS, "磨砂玻璃雕塑", (-7, -7.5), ha="right")
    note((P.RAMP_R * math.cos(math.radians(330)), P.RAMP_R * math.sin(math.radians(330))), "1:12 坡道（约 1.96 圈）",
         (11.5, -9.5))
    note(P.CORE_C, "电梯筒", (11.5, 14.5))
    note(tuple(P.MOON_C), "月洞墙", (-24, 10.5), ha="right")
    note(P.YCOL_BASE, "Y 形斜柱", (-21, -24.5), ha="center")
    note(P.C_PIER, "水中石墩", (24, 6.5))
    east = max(P.SKYLIGHTS, key=lambda g: g.centroid.x)
    note(east.representative_point().coords[0], "屋面光缝", (21, 14))
    note((14.5, 6.2), "展廊", (21, 10.5))
    note((0.0, -13.0), "门厅", (6.0, -26.0))
    note((-4.0, 12.0), "观景厅", (-8, 22.5), ha="right")
    # 指北针
    nx, ny = 26.5, 27.0
    ax.add_patch(MPoly([(nx, ny + 2.6), (nx - 0.9, ny - 0.6), (nx, ny + 0.2), (nx + 0.9, ny - 0.6)], closed=True,
                       fc=INK, ec="none", zorder=9))
    ax.text(nx, ny + 3.0, "N", ha="center", va="bottom", fontsize=11, color=INK)
    scale_bar(ax, -33, -28.5)
    fig.text(0.04, 0.955, "总平面 · 冬至日落轴线", fontsize=18, color=INK)
    fig.text(0.04, 0.932, "北纬 30°  ·  图示为设计高潮位  ·  水口、潮池、月洞同指冬至日落方位", fontsize=10.5, color=MUTED)
    fig.savefig(path, facecolor=PAPER)
    plt.close(fig)


# ======================================================================= 长剖面
def section(path):
    s = np.arange(-24.0, 34.0, 0.02)
    xy = P.AXIS[None] * s[:, None]
    fig = plt.figure(figsize=(16, 7.4), dpi=200, facecolor=PAPER)
    ax = fig.add_axes([0.03, 0.08, 0.94, 0.78])
    ax.set_xlim(-24, 34); ax.set_ylim(-2.3, 13.5); ax.set_aspect("equal"); ax.axis("off")
    # 水体与潮位
    ax.fill_between(s, -2.3, P.HIGH_TIDE, color=WATER_HT, lw=0, zorder=0)
    ax.plot(s, np.full_like(s, P.HIGH_TIDE), color="#5d7f93", lw=0.9, zorder=1)
    ax.plot(s, np.full_like(s, P.LOW_TIDE), color="#5d7f93", lw=0.9, ls=(0, (5, 3)), zorder=6)
    # 剖切线以外的基座（水口两侧池壁之后）
    for off in (1.0, -1.0):
        q = xy + P.AXIS_N[None] * off * (P.CHANNEL_W / 2 + 0.3)
        inside = shapely.contains_xy(P.PLINTH_CUT, q[:, 0], q[:, 1])
        ax.fill_between(s, P.BASIN_FLOOR, P.FFL, where=inside & (s > 2.0), color=STONE_BEYOND, lw=0, zorder=2)
    # 基座 / 潮池（剖切）
    r = np.abs(s)
    zg = np.full_like(s, np.nan)
    plinth = shapely.contains_xy(P.PLINTH_CUT, xy[:, 0], xy[:, 1])
    zg[plinth] = P.FFL
    for r_out, r_in, z in P.BASIN_STEPS:
        m = (r < r_out) & (r >= r_in) & (s < 0)
        zg[m] = z
    floor = ((r < P.BASIN_STEPS[-1][1]) | ((s > 0) & shapely.contains_xy(P.PLINTH_ALL.buffer(0.4), xy[:, 0], xy[:, 1])))
    floor &= ~plinth
    zg[floor & np.isnan(zg)] = P.BASIN_FLOOR
    solid = ~np.isnan(zg)
    ax.fill_between(s, -1.7, np.where(solid, zg, -1.7), where=solid, color=POCHE, lw=0, zorder=4)
    # 树岛与雕塑（剖切面以外，立面）
    ts = float(np.dot(P.TREE_POS, P.AXIS))
    ax.add_patch(MPoly([(ts - P.TREE_ISLAND_R, P.BASIN_FLOOR), (ts + P.TREE_ISLAND_R, P.BASIN_FLOOR),
                        (ts + P.TREE_ISLAND_R, P.FFL + 0.45), (ts - P.TREE_ISLAND_R, P.FFL + 0.45)], closed=True,
                       fc=STONE_BEYOND, ec="#9b948a", lw=0.6, zorder=3))
    ax.plot([ts, ts - 0.3, ts - 0.1], [P.FFL + 0.33, 3.6, 4.6], color="#7a6a58", lw=3, zorder=3)
    ax.add_patch(matplotlib.patches.Ellipse((ts - 0.2, 6.2), 6.6, 4.0, fc="#c3c7a4", ec="none", alpha=0.75, zorder=3))
    ss = float(np.dot(P.SCULPT_POS, P.AXIS))
    ax.add_patch(matplotlib.patches.Ellipse((ss, -0.40 + 1.05), 1.6, 2.0, fc="none", ec="#8aa3b0", lw=2.2, zorder=3))
    # 屋顶（剖切）与胶合木梁
    labels = {}
    for roof in P.ROOFS:
        ins = shapely.contains_xy(roof.poly, xy[:, 0], xy[:, 1])
        if not ins.any():
            continue
        d = roof.dist_edge(xy[ins, 0], xy[ins, 1])
        top, bot = roof.surfaces(xy[ins, 0], xy[ins, 1], d)
        T = np.full_like(s, np.nan); B = np.full_like(s, np.nan); U = np.full_like(s, np.nan)
        T[ins] = top; B[ins] = bot; U[ins] = bot - roof.rib_depth(d)
        ax.fill_between(s, B, T, where=ins, color=POCHE, lw=0, zorder=5)
        ax.fill_between(s, U, B, where=ins, color=OAK, lw=0, zorder=5)
        labels[roof.name] = (s[ins], T[ins])
        labels[roof.name + "_U"] = (s[ins], U[ins])
    # 屋面平台
    deck = shapely.contains_xy(P.DECK_POLY, xy[:, 0], xy[:, 1])
    a_top, _ = P.ROOF_A.surfaces(xy[:, 0], xy[:, 1])
    ax.fill_between(s, np.minimum(a_top, P.DECK - 0.2), P.DECK, where=deck, color=POCHE, lw=0, zorder=5)
    # 坡道（剖切的各圈）
    for side, ang in ((-1, (P.AXIS_DEG + 180) % 360), (1, P.AXIS_DEG)):
        for ph in (ang - 720, ang - 360, ang, ang + 360):
            if not (P.RAMP_START_DEG < ph < P.RAMP_END_DEG):
                continue
            zt = float(P.ramp_z_of_deg(ph))
            x0, x1 = side * (P.RAMP_R - P.RAMP_W / 2), side * (P.RAMP_R + P.RAMP_W / 2)
            ax.add_patch(MPoly([(x0, zt - 0.26), (x1, zt - 0.26), (x1, zt), (x0, zt)], closed=True, fc=POCHE, ec="none",
                               zorder=6))
            for xe in (x0, x1):
                ax.plot([xe, xe], [zt, zt + 1.05], color=GLASS, lw=0.9, zorder=6)
    # 电梯筒（剖切面后）：贯通至次屋顶，承托 B
    cs = float(np.dot(P.CORE_C, P.AXIS))
    zc = float(P.ROOF_B.underside(np.array([P.CORE_C[0]]), np.array([P.CORE_C[1]]))[0])
    ax.add_patch(MPoly([(cs - P.CORE_R, P.FFL), (cs + P.CORE_R, P.FFL), (cs + P.CORE_R, zc), (cs - P.CORE_R, zc)],
                       closed=True, fc=STONE_BEYOND, ec="#9b948a", lw=0.6, zorder=2))
    note_core = (cs, 8.5)
    # 外侧细柱列（立面）
    for xe in (-(P.RAMP_R + P.RAMP_W / 2 + 0.16),):
        ax.plot([xe, xe], [P.FFL, 5.4], color=MUTED, lw=0.8, zorder=3)
    # 展廊玻璃与端墙
    ins = shapely.contains_xy(P.E1, xy[:, 0], xy[:, 1]).astype(int)
    for k in np.where(np.diff(ins) != 0)[0]:
        xe = s[k]
        zt = float(P.ROOF_A.underside(np.array([xy[k, 0]]), np.array([xy[k, 1]]))[0])
        ax.plot([xe, xe], [P.FFL, zt], color=GLASS if xe > -12 else POCHE, lw=1.2 if xe > -12 else 3.0, zorder=6)
    # 人物
    def person(x, z, h=1.72):
        ax.add_patch(MPoly([(x - 0.18, z), (x + 0.18, z), (x + 0.2, z + 0.82 * h), (x - 0.2, z + 0.82 * h)], closed=True,
                           fc=INK, ec="none", zorder=7))
        ax.add_patch(Circle((x, z + 0.92 * h), 0.12, fc=INK, ec="none", zorder=7))
    person(-8.2, P.FFL); person(8.3, P.FFL); person(-11.2, P.FFL); person(-11.8, P.DECK)
    ph = (P.AXIS_DEG + 180) % 360 + 360
    person(-6.35, float(P.ramp_z_of_deg(ph)))
    person(-4.85, P.BASIN_STEPS[0][2])
    # 冬至夕阳
    tan = math.tan(math.radians(4.0))
    for z0 in (0.35, 1.2, 2.1):
        x0, x1 = 28.5, -1.5
        ax.annotate("", xy=(x1, z0 + (x1 - x0) * tan), xytext=(x0, z0),
                    arrowprops=dict(arrowstyle="-|>", color=SUN, lw=1.1, mutation_scale=9), zorder=8)
    ax.text(28.5, 2.75, "冬至夕阳 · 高度角 4°", color=SUN, fontsize=10.5, ha="right", zorder=9)

    def note(xy_, txt, xytext, ha="left"):
        ax.annotate(txt, xy=xy_, xytext=xytext, fontsize=10, color=INK, ha=ha, zorder=9,
                    arrowprops=dict(arrowstyle="-", color=INK, lw=0.6, shrinkA=0, shrinkB=1))
    for name, txt, pick, xyt in (("Roof_A_Main", "A  主屋顶 · 西翼尖", 0.85, (20, 12.4)),
                                 ("Roof_A_Main", "A  主屋顶", 0.12, (-23.5, 9.3)),
                                 ("Roof_B_Upper", "B  次屋顶", 0.5, (-10, 12.6)),
                                 ("Roof_C_Low", "C  低檐", 0.5, (16.5, 6.6))):
        if name not in labels:
            continue
        xs, zs = labels[name]
        if name == "Roof_C_Low" and txt == "C  低檐":
            sel = xs > 0
            xs, zs = xs[sel], zs[sel]
        if name == "Roof_A_Main":
            sel = xs > 0 if "西翼" in txt else xs < 0
            xs, zs = xs[sel], zs[sel]
        if len(xs):
            i = int(len(xs) * pick)
            note((xs[i], zs[i]), txt, xyt)
    xs, zs = labels.get("Roof_C_Low", (np.array([]), np.array([])))
    sel = xs < 0
    if sel.any():
        note((xs[sel][len(xs[sel]) // 2], zs[sel][len(xs[sel]) // 2]), "C  低檐 · 东翼尖", (-23.5, 4.6))
    xu, zu = labels["Roof_A_Main_U"]
    sel = xu < 0
    k = int(sel.sum() * 0.35)
    note((xu[sel][k], zu[sel][k]), "胶合木径向梁  檐口 100 → 支座 340 mm", (-23.5, 8.0))
    note((-1.5, P.BASIN_FLOOR), "潮池", (-3.0, -1.95), ha="right")
    note((12.0, P.BASIN_FLOOR), "水口", (12.5, -1.95))
    note((-6.35, 3.2), "1:12 坡道", (-4.2, 4.9))
    note((-11.2, 2.6), "展廊", (-15.0, 2.0), ha="right")
    note((-12.6, P.DECK), "屋面平台 +6.30", (-23.5, 10.6))
    note((ts, 5.2), "树岛（剖切面后）", (3.2, 9.6))
    note(note_core, "电梯筒（剖切面后）· 承托次屋顶", (-2.0, 11.4))
    for z, txt in ((P.DECK, "+6.30"), (P.FFL, "+0.85 首层"), (P.HIGH_TIDE, "±0.00 设计高潮位"),
                   (P.LOW_TIDE, "-0.95 设计低潮位")):
        ax.plot([33.2, 33.9], [z, z], color=INK, lw=0.8, zorder=9)
        ax.text(33.0, z + 0.12, txt, fontsize=9.5, color=INK, ha="right", va="bottom", zorder=9)
    scale_bar(ax, -23.5, -2.05, units=(0, 2, 5, 10), h=0.25, label_size=8)
    ax.text(-24, 13.2, "东北  ←", fontsize=10, color=MUTED)
    ax.text(34, 13.2, "→  西南（冬至日落方向）", fontsize=10, color=MUTED, ha="right")
    fig.text(0.03, 0.93, "1-1 剖面 · 沿冬至日落轴线", fontsize=18, color=INK)
    fig.text(0.03, 0.895, "从潮到云：水口引潮入池，坡道沿池上升至屋面平台，夕阳顺轴线照入潮池", fontsize=11, color=MUTED)
    fig.savefig(path, facecolor=PAPER)
    plt.close(fig)


# ======================================================================= 潮位曲线
def tide_curve(path):
    """一日两次涨落（半日潮，周期 12 h 25 min），潮差 0.95 m。"""
    t = np.linspace(0, 24, 600)
    mean = (P.HIGH_TIDE + P.LOW_TIDE) / 2
    amp = (P.HIGH_TIDE - P.LOW_TIDE) / 2
    z = mean + amp * np.cos(2 * np.pi * (t - 4.5) / 12.42)
    fig = plt.figure(figsize=(8, 4.2), dpi=200, facecolor=PAPER)
    ax = fig.add_axes([0.1, 0.16, 0.66, 0.68])
    ax.set_facecolor(PAPER)
    refs = [(P.FFL, "首层 +0.85"), (P.BASIN_STEPS[0][2], "一级台阶 +0.55"), (P.BASIN_STEPS[1][2], "二级 +0.25"),
            (P.BASIN_STEPS[2][2], "三级 -0.05"), (P.BASIN_STEPS[3][2], "四级 -0.35"), (P.BASIN_FLOOR, "池底 -0.65")]
    for zr, lab in refs:
        ax.axhline(zr, color="#cfc8be", lw=0.8, zorder=1)
        ax.text(24.4, zr, lab, va="center", fontsize=9, color=MUTED)
    ax.fill_between(t, P.BASIN_FLOOR, z, where=z > P.BASIN_FLOOR, color="#d6e3ea", lw=0, zorder=2)
    ax.plot(t, z, color="#2f6f8f", lw=2.0, zorder=3)
    for th in (4.5, 4.5 + 12.42):
        ax.plot(th, P.HIGH_TIDE, "o", ms=6, color="#2f6f8f", zorder=4)
        ax.text(th, P.HIGH_TIDE + 0.09, "高潮 ±0.00", ha="center", fontsize=9, color=INK)
    for tl in (4.5 + 6.21, 4.5 + 6.21 + 12.42 - 24 if 4.5 + 6.21 + 12.42 > 24 else 4.5 + 6.21 + 12.42):
        ax.plot(tl, P.LOW_TIDE, "o", ms=6, color="#2f6f8f", zorder=4)
        ax.text(tl, P.LOW_TIDE - 0.17, "低潮 -0.95", ha="center", fontsize=9, color=INK)
    ax.set_xlim(0, 24); ax.set_ylim(-1.25, 1.0)
    ax.set_xticks([0, 6, 12, 18, 24]); ax.set_xticklabels(["0:00", "6:00", "12:00", "18:00", "24:00"])
    ax.set_yticks([])
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color("#b9b2a8")
    ax.tick_params(colors=MUTED, labelsize=9)
    fig.text(0.1, 0.92, "一日两涨两落：潮池水位", fontsize=13, color=INK)
    fig.text(0.1, 0.875, "浅蓝为池中有水的部分；高潮时池水漫至第三级台阶，低潮时整池露底", fontsize=9, color=MUTED)
    fig.savefig(path, facecolor=PAPER)
    plt.close(fig)


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(out, exist_ok=True)
    site_plan(os.path.join(out, "plan_axis.png"))
    section(os.path.join(out, "section.png"))
    tide_curve(os.path.join(out, "tide_curve.png"))
    print("diagrams ->", out)
