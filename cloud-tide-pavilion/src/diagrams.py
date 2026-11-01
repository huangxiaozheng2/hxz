"""分析图：总平面与日落轴线、沿轴线长剖面、潮位曲线。全部由 plan.py 的同一套几何生成。
用法：python diagrams.py 输出目录
"""
import math
import os
import sys

import numpy as np
import shapely
import shapely.ops
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Circle, Polygon as MPoly, FancyArrowPatch
from shapely.geometry import Point, box, LineString, Polygon

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plan as P  # noqa: E402
from plan import ccw, resample_ring  # noqa: E402

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


def scale_bar(ax, x0, y0, units=(0, 5, 10, 20), h=0.5, label_size=9, unit_gap=1.0, label_gap=0.6):
    for a, b in zip(units[:-1], units[1:]):
        k = units.index(a)
        ax.add_patch(MPoly([(x0 + a, y0), (x0 + b, y0), (x0 + b, y0 + h), (x0 + a, y0 + h)], closed=True,
                           fc=INK if k % 2 == 0 else "white", ec=INK, lw=0.8))
    for u in units:
        ax.text(x0 + u, y0 - label_gap, f"{u}", ha="center", va="top", fontsize=label_size, color=INK)
    ax.text(x0 + units[-1] + unit_gap, y0 - label_gap, "m", ha="left", va="top", fontsize=label_size, color=INK)


def tight_layout(fig, ax, title, subtitle, head=0.85, foot=0.15, width=16.0):
    """图高由横向比例决定：标题紧贴图上方，底部只留 foot 英寸（图例等另加）。返回图高（英寸）。"""
    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
    hi = width * 0.94 * (y1 - y0) / (x1 - x0)
    Hf = head + hi + foot
    fig.set_size_inches(width, Hf)
    ax.set_position([0.03, foot / Hf, 0.94, hi / Hf])
    fig.text(0.03, (Hf - 0.36) / Hf, title, fontsize=18, color=INK, va="center")
    fig.text(0.03, (Hf - 0.66) / Hf, subtitle, fontsize=11, color=MUTED, va="center")
    return Hf


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
    for q in P.C_PIERS:
        ax.add_patch(Circle(q, 0.62, fc=STONE_BEYOND, ec=INK, lw=0.6, zorder=6))
    # 柱位
    import structure as ST
    allc = np.vstack([ST.E1_COLS_INNER, ST.E1_COLS_OUTER, ST.E2_COLS, ST.A_EAST_COLS, ST.C_EXTRA_COLS, P.C_PIERS])
    ax.scatter(allc[:, 0], allc[:, 1], s=9, color=INK, zorder=7)
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
    note(P.C_PIERS[1], "水中石墩", (24, 6.5))
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


# ======================================================================= 区位
def location_map(path):
    """区位（设定场地）：湖面、西南岸湖岸公园与滨水步道、入口广场、栈桥、展馆、湖中小岛、滨湖路与城区。"""
    fig = plt.figure(figsize=(8.4, 8.4), dpi=220, facecolor=PAPER)
    ax = fig.add_axes([0.03, 0.03, 0.94, 0.84])
    x0, x1, y0, y1 = -235, 125, -265, 95
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_aspect("equal"); ax.axis("off")
    ax.add_patch(MPoly([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], closed=True, fc=WATER, ec="none", zorder=0))
    frame = box(x0, y0, x1, y1)
    edge = LineString(P.SHORE_LINE)
    land = Polygon(np.vstack([P.SHORE_LINE, [(-400, -420), (260, -420)]])).buffer(0).intersection(frame)
    road_line = P.ROAD_LINE
    fill_poly(ax, land, fc="#e6dfd3", ec="none", zorder=1)
    # 公园（岸线与滨湖路之间，与模型里的种植范围一致）
    park = land.difference(Polygon(np.vstack([np.asarray(road_line.coords), [(260, -420), (-400, -420)]])).buffer(0))
    fill_poly(ax, park, fc="#d5dcc2", ec="none", zorder=2)
    rl = road_line.intersection(frame)
    for g in ([rl] if rl.geom_type == "LineString" else list(rl.geoms)):
        x, y = g.xy
        ax.plot(x, y, color="white", lw=6.5, solid_capstyle="butt", zorder=3)
        ax.plot(x, y, color="#b8b0a4", lw=0.6, zorder=3)
    # 城区（与模型同一份布局）：按层数由浅到深，江南白墙灰瓦的临路小楼单独着色
    for b in P.TOWN:
        if not frame.intersects(b["poly"]):
            continue
        if b["kind"] == "jiangnan":
            fc = "#c9c3b8"
        else:
            k = min(1.0, b["floors"] / 16)
            fc = tuple(np.array([0.86, 0.83, 0.78]) * (1 - 0.18 * k))
        fill_poly(ax, b["poly"].intersection(frame), fc=fc, ec="#a8a094", lw=0.4, zorder=3)
    # 种植（build.py 导出的实际位置）：垂柳、阔叶树、松、灌木
    pj = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "renders", "planting.json")
    if os.path.exists(pj):
        import json
        with open(pj) as f:
            plant = json.load(f)
        style = {"willow": (4.2, "#b9c79a"), "tree": (5.2, "#a9b98a"), "pine": (3.2, "#7f9270"), "shrub": (1.3, "#93a77c")}
        for t in sorted(plant, key=lambda t: t["kind"] != "shrub"):
            r, c = style.get(t["kind"], (4.0, "#a9b98a"))
            ax.add_patch(Circle((t["x"], t["y"]), r * t["scale"], fc=c, ec="#7f8466", lw=0.3, alpha=0.95, zorder=4.5))
    # 驳岸与滨水步道、入口广场、园路
    ex, ey = edge.intersection(frame).xy
    ax.plot(ex, ey, color="#8d877e", lw=1.1, zorder=4)
    fill_poly(ax, P.PROMENADE.intersection(frame), fc="#f3eee5", ec="none", zorder=4)
    fill_poly(ax, P.LANDING, fc="#f3eee5", ec="none", zorder=4)
    fill_poly(ax, P.PARK_PATH.intersection(land), fc="#f3eee5", ec="none", zorder=4)
    # 湖中小岛
    for x, y, r in [(-36, -5, 4.5), (-28, -32, 3.2), (32, -26, 4.0), (40, 2, 3.0), (-40, 34, 5.0), (8, 50, 3.6),
                    (-16, 44, 3.2), (48, -42, 4.0), (-56, -40, 6.5)]:
        ax.add_patch(Circle((x, y), r, fc="#c9cdb2", ec="#7f8466", lw=0.4, zorder=4))
    # 展馆与栈桥
    fill_poly(ax, P.PLINTH_ALL, fc="white", ec=INK, lw=0.6, zorder=5)
    fill_poly(ax, shapely.unary_union([r.poly for r in P.ROOFS]), fc=POCHE, ec="none", zorder=6)
    # 冬至日落方向
    ax.plot([0, P.AXIS[0] * 330], [0, P.AXIS[1] * 330], color=SUN, lw=1.2, ls=(0, (6, 3)), zorder=7)
    ax.text(*(P.AXIS * 200 + np.array([-2, 11])), "冬至日落方向", color=SUN, fontsize=14, ha="center", zorder=8,
            rotation=math.degrees(math.atan2(P.AXIS[1], P.AXIS[0])) - 180)

    def note(xy, txt, xytext, ha="left", size=14.5):
        ax.annotate(txt, xy=xy, xytext=xytext, fontsize=size, color=INK, ha=ha, va="center", zorder=9,
                    arrowprops=dict(arrowstyle="-", color=INK, lw=0.6, shrinkA=0, shrinkB=2))
    gap = float(edge.distance(P.PLINTH.buffer(0)))
    note((6, 10), "云上潮汐馆", (34, 58), size=16)
    note(tuple(P.BRIDGE_PATH[2]), f"栈桥 约 {LineString(P.resample_open(P.BRIDGE_PATH, 0.4)).length:.0f} m", (-5, -95))
    note(tuple(P.LANDING_C), "入口广场", (-150, -26), ha="right")
    note((-112.0, -86.5), "滨水步道", (-222, -42))
    ax.text(-95, -116, "湖岸公园", fontsize=16, color="#5f6a46", ha="center", zorder=9)
    q = road_line.interpolate(0.36, normalized=True)
    ax.text(q.x, q.y - 13, "滨湖路", fontsize=14, color=MUTED, ha="center", zorder=9)
    ax.text(-150, -238, "城区（设定）", fontsize=16, color=MUTED, ha="center", zorder=9,
            bbox=dict(fc="#e6dfd3", ec="none", pad=1.5))
    ax.text(60, 78, "湖面", fontsize=17, color="#5d7f93", ha="center", zorder=9)
    # 指北针、比例尺
    nx, ny = 108, 72
    ax.add_patch(MPoly([(nx, ny + 10), (nx - 3.5, ny - 2.5), (nx, ny + 0.8), (nx + 3.5, ny - 2.5)], closed=True,
                       fc=INK, ec="none", zorder=9))
    ax.text(nx, ny + 12, "N", ha="center", va="bottom", fontsize=14, color=INK)
    scale_bar(ax, -225, 78, units=(0, 25, 50, 100), h=2.6, label_size=12, unit_gap=11.0)
    fig.text(0.04, 0.95, "区位 · 设定场地", fontsize=24, color=INK)
    fig.text(0.04, 0.905, f"北纬 30° 湖泊  ·  展馆离岸约 {gap:.0f} m，自西南岸湖岸公园入口广场经栈桥到达",
             fontsize=13.5, color=MUTED)
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
        T[ins] = top; B[ins] = bot; U[ins] = bot - roof.beam_depth(d)
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
    # 疏散楼梯间（剖切面后）：首层直达次屋顶
    corners = np.array(P.STAIR2.exterior.coords)[:-1]
    s_st = corners @ P.AXIS
    z_st = float(np.min(P.ROOF_B.underside(corners[:, 0], corners[:, 1])))
    ax.add_patch(MPoly([(s_st.min(), P.FFL), (s_st.max(), P.FFL), (s_st.max(), z_st), (s_st.min(), z_st)], closed=True,
                       fc=STONE_BEYOND, ec="#9b948a", lw=0.6, zorder=1.6))
    note_stair = (float(s_st.max()) - 1.5, z_st - 1.8)
    # 剖切面前后 12 m 内的竖向支承：面后为细实线（立面），面前为虚线（投影），让每片屋顶都看得到支点
    import structure as ST
    beyond = np.sign(np.dot(P.TREE_POS, P.AXIS_N))

    def und(roof, c):
        return float(roof.underside(np.array([c[0]]), np.array([c[1]]))[0])

    def vline(c, z0, z1, lw=1.3, allow_front=False):
        sc, nc = float(np.dot(c, P.AXIS)), float(np.dot(c, P.AXIS_N))
        front = np.sign(nc) != beyond
        if abs(nc) > 12 or abs(nc) < 0.3 or (front and not allow_front):
            return None
        ax.plot([sc, sc], [z0, z1], color="#b3aba0" if front else "#958d82", lw=lw,
                ls=(0, (4, 3)) if front else "-", zorder=2.5)
        return sc, front
    # 剖切面后的屋顶轮廓（立面，浅色）
    for roof in P.ROOFS:
        minx, miny, maxx, maxy = roof.poly.bounds
        X, Y = np.meshgrid(np.arange(minx, maxx, 0.15), np.arange(miny, maxy, 0.15))
        X, Y = X.ravel(), Y.ravel()
        m = shapely.contains_xy(roof.poly, X, Y) & (np.sign(X * P.AXIS_N[0] + Y * P.AXIS_N[1]) == beyond)
        X, Y = X[m], Y[m]
        d = roof.dist_edge(X, Y)
        top, bot = roof.surfaces(X, Y, d)
        und_ = bot - roof.beam_depth(d)
        sv = X * P.AXIS[0] + Y * P.AXIS[1]
        bins = np.round(sv / 0.1).astype(int)
        ub = np.unique(bins)
        lo = np.array([und_[bins == b].min() for b in ub]); hi = np.array([top[bins == b].max() for b in ub])
        xs_ = ub * 0.1
        gaps = np.where(np.diff(ub) > 2)[0] + 1
        for seg in np.split(np.arange(len(ub)), gaps):
            ax.fill_between(xs_[seg], lo[seg], hi[seg], color="#e8e2d8", ec="#cdc5b9", lw=0.6, zorder=1.5)
    for c in np.vstack([ST.E1_COLS_INNER, ST.E1_COLS_OUTER, ST.A_EAST_COLS]):
        vline(c, P.FFL, und(P.ROOF_A, c))
    for c in np.vstack([ST.E2_COLS, ST.C_EXTRA_COLS]):
        vline(c, P.FFL, und(P.ROOF_C, c))
    for c in ST.B_COLS:
        a_top, _ = P.ROOF_A.surfaces(np.array([c[0]]), np.array([c[1]]))
        z0 = P.DECK if P.DECK_POLY.contains(Point(*c)) else float(a_top[0])
        vline(c, z0, und(P.ROOF_B, c))
    for c in P.C_PIERS:                                              # 水中石墩 + 钢柱
        r = vline(c, 0.6, und(P.ROOF_C, c), lw=1.6, allow_front=True)
        if r:
            ax.add_patch(MPoly([(r[0] - 0.62, -1.7), (r[0] + 0.62, -1.7), (r[0] + 0.62, 0.6), (r[0] - 0.62, 0.6)],
                               closed=True, fc="none", ec="#b3aba0", lw=1.0, ls=(0, (4, 3)), zorder=2.5))
    yb = np.array(P.YCOL_BASE)                                       # Y 形斜柱：石墩 + 两肢
    sb = float(np.dot(yb, P.AXIS))
    ax.add_patch(MPoly([(sb - 1.05, -1.7), (sb + 1.05, -1.7), (sb + 1.05, P.FFL), (sb - 1.05, P.FFL)], closed=True,
                       fc="none", ec="#b3aba0", lw=1.0, ls=(0, (4, 3)), zorder=2.5))
    y_tops = []
    for rr, aa in P.YCOL_TOPS:
        t = np.array([rr * math.cos(math.radians(aa)), rr * math.sin(math.radians(aa))])
        st, zt = float(np.dot(t, P.AXIS)), und(P.ROOF_A, t)
        ax.plot([sb, st], [P.FFL + 0.55, zt], color="#b3aba0", lw=1.8, ls=(0, (4, 3)), zorder=2.5)
        y_tops.append((st, zt))
    fig.text(0.97, 0.035, "浅色：剖切面后的屋顶与柱（立面）    虚线：剖切面前的 Y 形斜柱与水中石墩（投影）",
             fontsize=9, color=MUTED, ha="right")
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
    note((xu[sel][k], zu[sel][k]), "胶合木主梁  檐口 200 → 支座 600 mm", (-23.5, 8.0))
    note((-1.5, P.BASIN_FLOOR), "潮池", (-3.0, -1.95), ha="right")
    note((12.0, P.BASIN_FLOOR), "水口", (12.5, -1.95))
    note((-6.35, 3.2), "1:12 坡道", (-4.2, 4.9))
    note((-11.2, 2.6), "展廊", (-15.0, 2.0), ha="right")
    note((-12.6, P.DECK), "屋面平台 +6.30", (-23.5, 10.6))
    note((ts, 5.2), "树岛（剖切面后）", (3.2, 9.6))
    note(note_core, "电梯筒（剖切面后）· 承托次屋顶", (-2.0, 11.4))
    ax.text(note_stair[0] + 1.2, note_stair[1] + 0.05, "疏散楼梯间（剖切面后）", fontsize=9, color=MUTED, ha="right",
            va="center", zorder=9)
    ym = ((sb + y_tops[0][0]) / 2, (P.FFL + 0.55 + y_tops[0][1]) / 2)
    note(ym, "Y 形斜柱（投影）", (21.5, 4.6))
    cp = min(P.C_PIERS, key=lambda c: abs(float(np.dot(c, P.AXIS_N))))
    note((float(np.dot(cp, P.AXIS)), 0.3), "水中石墩（投影）", (-23.5, 1.0))
    for z, txt in ((P.DECK, "+6.30"), (P.FFL, "+0.85 首层"), (P.HIGH_TIDE, "±0.00 设计高潮位"),
                   (P.LOW_TIDE, "-0.95 设计低潮位")):
        level_mark(ax, 33.8, z, txt, side="left", run=3.8)
    scale_bar(ax, -23.5, -2.05, units=(0, 2, 5, 10), h=0.25, label_size=8)
    ax.text(-24, 13.2, "东北  ←", fontsize=10, color=MUTED)
    ax.text(34, 13.2, "→  西南（冬至日落方向）", fontsize=10, color=MUTED, ha="right")
    fig.text(0.03, 0.93, "1-1 剖面 · 沿冬至日落轴线", fontsize=18, color=INK)
    fig.text(0.03, 0.895, "从潮到云：水口引潮入池，坡道沿池上升至屋面平台，夕阳顺轴线照入潮池", fontsize=11, color=MUTED)
    fig.savefig(path, facecolor=PAPER)
    plt.close(fig)


# ======================================================================= 2-2 横剖面
SEC2_DEG = 105.0                                   # 剖切线方位：疏散楼梯间中心线（北）—门厅（南）


def _sec2_frame():
    a = math.radians(SEC2_DEG)
    D = np.array([math.cos(a), math.sin(a)])       # 剖切线方向（指向北侧展廊）
    W = np.array([-math.sin(a), -math.cos(a)]) if False else np.array([D[1], -D[0]]) * -1
    return D, W                                    # W：看向剖切面后方（西侧）


def _cut_crossings(poly, D, s0=-30, s1=30):
    """剖切线与多边形边界的交点（沿 s 的位置）。"""
    line = LineString([D * s0, D * s1])
    inter = line.intersection(poly.boundary)
    pts = [inter] if inter.geom_type == "Point" else list(getattr(inter, "geoms", []))
    return sorted(float(np.dot([p.x, p.y], D)) for p in pts if p.geom_type == "Point")


def _bay_kind(poly, q, solid_fn, doors_fn=None):
    """边界上一点 q 属于实墙、门还是玻璃。"""
    import program as PG
    segs = PG.bay_segments(poly, solid_fn, doors_fn)
    best = min(segs, key=lambda sg: LineString(sg[1]).distance(Point(*q)))
    return best[0]


def section_cross(path, env=False):
    """2-2 剖面：南侧门厅—潮池与坡道—展廊—疏散楼梯间—屋面平台与观景厅—北侧临水边。
    env=True 时以浅灰为底，叠加环境策略（遮阳、通风、地源 / 湖水源热泵、雨水、天光）。"""
    import structure as ST
    from facade import e1_solid, e1_doors, e2_doors, e3_solid, e3_doors
    D, W = _sec2_frame()
    s0, s1 = -27.0, 27.0
    sv = np.arange(s0, s1, 0.02)
    xy = D[None] * sv[:, None]
    poche = "#8c8780" if env else POCHE
    oak = "#cbb59a" if env else OAK
    fig = plt.figure(figsize=(16, 7.6), dpi=200, facecolor=PAPER)
    ax = fig.add_axes([0.03, 0.06, 0.94, 0.8])
    ax.set_xlim(s0, s1); ax.set_ylim(-2.3, 14.2); ax.set_aspect("equal"); ax.axis("off")
    ax.fill_between(sv, -2.3, P.HIGH_TIDE, color=WATER_HT, lw=0, zorder=0)
    ax.plot(sv, np.zeros_like(sv), color="#5d7f93", lw=0.9, zorder=1)
    ax.plot(sv, np.full_like(sv, P.LOW_TIDE), color="#5d7f93", lw=0.9, ls=(0, (5, 3)), zorder=6)
    beyond_sign = 1.0

    def nb(p):                                     # 到剖切面的距离（>0 在剖切面后）
        return float(np.dot(p, W)) * beyond_sign
    # 剖切面后的屋顶轮廓
    for roof in P.ROOFS:
        minx, miny, maxx, maxy = roof.poly.bounds
        X, Y = np.meshgrid(np.arange(minx, maxx, 0.15), np.arange(miny, maxy, 0.15))
        X, Y = X.ravel(), Y.ravel()
        m = shapely.contains_xy(roof.poly, X, Y) & ((X * W[0] + Y * W[1]) > 0)
        X, Y = X[m], Y[m]
        dd = roof.dist_edge(X, Y)
        top, bot = roof.surfaces(X, Y, dd)
        und_ = bot - roof.beam_depth(dd)
        svv = X * D[0] + Y * D[1]
        bins = np.round(svv / 0.1).astype(int)
        ub = np.unique(bins)
        lo = np.array([und_[bins == b].min() for b in ub]); hi = np.array([top[bins == b].max() for b in ub])
        xs_ = ub * 0.1
        gaps = np.where(np.diff(ub) > 2)[0] + 1
        for seg in np.split(np.arange(len(ub)), gaps):
            ax.fill_between(xs_[seg], lo[seg], hi[seg], color="#e8e2d8", ec="#cdc5b9", lw=0.6, zorder=1.5)
    # 剖切面后：月洞墙（带圆洞）、树岛、雕塑
    ms = float(np.dot(P.MOON_C, D))
    wl = np.array([P.MOON_C - P.AXIS_N * P.MOON_HALF, P.MOON_C + P.AXIS_N * P.MOON_HALF]) @ D
    zt = float(P.ROOF_B.underside(np.array([P.MOON_C[0]]), np.array([P.MOON_C[1]]))[0])
    ax.add_patch(MPoly([(wl.min(), P.DECK), (wl.max(), P.DECK), (wl.max(), zt), (wl.min(), zt)], closed=True,
                       fc=STONE_BEYOND, ec="#9b948a", lw=0.6, zorder=1.7))
    hr = P.MOON_HOLE_R * abs(float(np.dot(P.AXIS_N, D)))
    ax.add_patch(matplotlib.patches.Ellipse((ms, P.DECK + 0.5 + 1.1), 2 * hr, 2 * 1.1, fc=PAPER, ec="#9b948a", lw=0.6,
                                            zorder=1.8))
    ts = float(np.dot(P.TREE_POS, D))
    ax.add_patch(MPoly([(ts - P.TREE_ISLAND_R, P.BASIN_FLOOR), (ts + P.TREE_ISLAND_R, P.BASIN_FLOOR),
                        (ts + P.TREE_ISLAND_R, P.FFL + 0.45), (ts - P.TREE_ISLAND_R, P.FFL + 0.45)], closed=True,
                       fc=STONE_BEYOND, ec="#9b948a", lw=0.6, zorder=2))
    ax.plot([ts, ts - 0.3, ts - 0.1], [P.FFL + 0.33, 3.6, 4.6], color="#7a6a58", lw=3, zorder=2)
    ax.add_patch(matplotlib.patches.Ellipse((ts - 0.2, 6.2), 6.6, 4.0, fc="#c3c7a4", ec="none", alpha=0.75, zorder=2))
    scs = float(np.dot(P.SCULPT_POS, D))
    ax.add_patch(matplotlib.patches.Ellipse((scs, -0.40 + 1.05), 1.6, 2.0, fc="none", ec="#8aa3b0", lw=2.2, zorder=2))
    # 剖切面后 10 m 内的柱（立面）
    for c in np.vstack([ST.E1_COLS_INNER, ST.E1_COLS_OUTER, ST.E2_COLS, ST.C_EXTRA_COLS]):
        n_ = nb(c)
        if 0.3 < n_ < 10:
            roof = P.ROOF_C if any(np.allclose(c, q) for q in np.vstack([ST.E2_COLS, ST.C_EXTRA_COLS])) else P.ROOF_A
            ax.plot([np.dot(c, D)] * 2, [P.FFL, float(roof.underside(np.array([c[0]]), np.array([c[1]]))[0])],
                    color="#958d82", lw=1.2, zorder=2.5)
    for c in ST.B_COLS:
        n_ = nb(c)
        if 0.3 < n_ < 10:
            ax.plot([np.dot(c, D)] * 2, [P.DECK, float(P.ROOF_B.underside(np.array([c[0]]), np.array([c[1]]))[0])],
                    color="#958d82", lw=1.2, zorder=2.5)
    # 基座与潮池（剖切）
    r = np.abs(sv)
    zg = np.full_like(sv, np.nan)
    plinth = shapely.contains_xy(P.PLINTH_CUT, xy[:, 0], xy[:, 1])
    zg[plinth] = P.FFL
    for r_out, r_in, z in P.BASIN_STEPS:
        zg[(r < r_out) & (r >= r_in)] = z
    zg[(r < P.BASIN_STEPS[-1][1])] = P.BASIN_FLOOR
    solid = ~np.isnan(zg)
    ax.fill_between(sv, -1.7, np.where(solid, zg, -1.7), where=solid, color=poche, lw=0, zorder=4)
    # 屋顶（剖切）与胶合木梁
    cuts = {}
    for roof in P.ROOFS:
        ins = shapely.contains_xy(roof.poly, xy[:, 0], xy[:, 1])
        if not ins.any():
            continue
        dd = roof.dist_edge(xy[ins, 0], xy[ins, 1])
        top, bot = roof.surfaces(xy[ins, 0], xy[ins, 1], dd)
        T = np.full_like(sv, np.nan); B = np.full_like(sv, np.nan); U = np.full_like(sv, np.nan)
        T[ins] = top; B[ins] = bot; U[ins] = bot - roof.beam_depth(dd)
        ax.fill_between(sv, B, T, where=ins, color=poche, lw=0, zorder=5)
        ax.fill_between(sv, U, B, where=ins, color=oak, lw=0, zorder=5)
        cuts[roof.name] = (sv[ins], T[ins], U[ins])
    deck = shapely.contains_xy(P.DECK_POLY, xy[:, 0], xy[:, 1])
    a_top, _ = P.ROOF_A.surfaces(xy[:, 0], xy[:, 1])
    ax.fill_between(sv, np.minimum(a_top, P.DECK - 0.2), P.DECK, where=deck, color=poche, lw=0, zorder=5)
    # 屋面光缝（剖到处开口，顶部为玻璃）
    sky = []
    for sk in P.SKYLIGHTS:
        it = LineString([D * s0, D * s1]).intersection(sk)
        if it.is_empty:
            continue
        cs = np.asarray(it.coords) @ D
        a_, b_ = float(cs.min()), float(cs.max())
        m_ = (sv >= a_) & (sv <= b_)
        top_a, bot_a = P.ROOF_A.surfaces(xy[m_, 0], xy[m_, 1])
        ax.fill_between(sv[m_], bot_a - 0.6, top_a + 0.02, color=PAPER, lw=0, zorder=5.2)
        ax.plot(sv[m_], top_a + 0.02, color=GLASS, lw=1.2, zorder=5.3)
        sky.append((a_, b_))
    # 平台栏杆、首层临水栏杆
    for q_s in _cut_crossings(P.DECK_POLY.buffer(-0.25), D):
        q = D * q_s
        if not P.E3.buffer(0.4).contains(Point(*q)):
            ax.plot([q_s, q_s], [P.DECK, P.DECK + 1.05], color=GLASS, lw=1.0, zorder=6)
    for q_s in _cut_crossings(P.PLINTH_CUT.buffer(-0.25), D):
        q = D * q_s
        if not P.WATER_CUT.buffer(0.6).contains(Point(*q)) and abs(q_s) > 8:
            ax.plot([q_s, q_s], [P.FFL, P.FFL + 1.05], color=GLASS, lw=1.0, zorder=6)
    # 围合体（剖切）：实墙涂黑，玻璃细线
    for poly, roof, z0, sfn, dfn in ((P.E1, P.ROOF_A, P.FFL, e1_solid, e1_doors), (P.E2, P.ROOF_C, P.FFL, None, e2_doors),
                                     (P.E3, P.ROOF_B, P.DECK, e3_solid, e3_doors)):
        for q_s in _cut_crossings(poly, D):
            q = D * q_s
            zt = float(roof.underside(np.array([q[0]]), np.array([q[1]]))[0])
            kind = _bay_kind(poly, q, sfn, dfn) if sfn else "glass"
            if kind == "solid":
                ax.add_patch(MPoly([(q_s - 0.17, z0), (q_s + 0.17, z0), (q_s + 0.17, zt), (q_s - 0.17, zt)], closed=True,
                                   fc=poche, ec="none", zorder=6))
            else:
                for off in (-0.04, 0.04):
                    ax.plot([q_s + off] * 2, [z0, zt], color=GLASS, lw=0.9, zorder=6)
    # 疏散楼梯间（剖切）：墙、两跑梯段的剖面，平台层平台在剖切面后
    c2, e_r, e_t = P.stair2_frame()
    s_c = float(np.dot(c2, D))
    hd, t = (P.STAIR2_R_OUT - P.STAIR2_R_IN) / 2, P.STAIR2_WALL
    z_st = float(np.min(P.ROOF_B.underside(*np.array(P.STAIR2.exterior.coords)[:-1].T))) - 0.03
    for a_, b_ in ((-hd, -hd + t), (hd - t, hd)):
        ax.add_patch(MPoly([(s_c + a_, P.FFL), (s_c + b_, P.FFL), (s_c + b_, z_st), (s_c + a_, z_st)], closed=True,
                           fc=poche, ec="none", zorder=6))
    ax.add_patch(MPoly([(s_c - hd, z_st), (s_c + hd, z_st), (s_c + hd, z_st + 0.18), (s_c - hd, z_st + 0.18)],
                       closed=True, fc=poche, ec="none", zorder=6))
    r1 = P.STAIR2_RISER
    for (u0, u1), zc in (((-hd + t, -0.05), P.FFL + 0.5 * 16 * r1 * (2.05 / 4.05) * 2), ((0.05, hd - t), P.FFL + 16 * r1 + (2.0 / 4.05) * 15 * r1)):
        ax.add_patch(MPoly([(s_c + u0, zc - 0.3), (s_c + u1, zc - 0.3), (s_c + u1, zc), (s_c + u0, zc)], closed=True,
                           fc=poche, ec="none", zorder=6))
    ax.plot([s_c - hd + t, s_c + hd - t], [P.DECK, P.DECK], color="#958d82", lw=1.2, zorder=5.5)
    for k in range(9):                                     # 剖切面后的梯段（踏步线）
        z_ = P.FFL + k * r1
        ax.plot([s_c - hd + t, s_c - 0.05], [z_, z_], color="#b3aba0", lw=0.6, zorder=5.5)
        z2 = P.FFL + 16 * r1 + (8 + k) * r1
        ax.plot([s_c + 0.05, s_c + hd - t], [z2, z2], color="#b3aba0", lw=0.6, zorder=5.5)
    # 坡道：剖到的各圈
    for side, ang in ((1, SEC2_DEG), (-1, SEC2_DEG + 180)):
        for ph in (ang - 720, ang - 360, ang, ang + 360):
            if not (P.RAMP_START_DEG < ph < P.RAMP_END_DEG):
                continue
            zt_ = float(P.ramp_z_of_deg(ph))
            x0_, x1_ = side * (P.RAMP_R - P.RAMP_W / 2), side * (P.RAMP_R + P.RAMP_W / 2)
            ax.add_patch(MPoly([(x0_, zt_ - 0.26), (x1_, zt_ - 0.26), (x1_, zt_), (x0_, zt_)], closed=True, fc=poche,
                               ec="none", zorder=6))
            for xe in (x0_, x1_):
                ax.plot([xe, xe], [zt_, zt_ + 1.05], color=GLASS, lw=0.9, zorder=6)
    # 人物
    def person(x, z, h=1.72):
        ax.add_patch(MPoly([(x - 0.18, z), (x + 0.18, z), (x + 0.2, z + 0.82 * h), (x - 0.2, z + 0.82 * h)], closed=True,
                           fc="#6d6a64" if env else INK, ec="none", zorder=7))
        ax.add_patch(Circle((x, z + 0.92 * h), 0.12, fc="#6d6a64" if env else INK, ec="none", zorder=7))
    person(-13.5, P.FFL); person(-11.2, P.FFL, 1.2); person(11.4, P.FFL); person(17.2, P.FFL); person(11.2, P.DECK)
    person(-8.3, P.FFL)
    info = dict(D=D, W=W, s_c=s_c, z_st=z_st, cuts=cuts, ms=ms, sky=sky)
    if not env:
        def note(xy_, txt, xytext, ha="left"):
            ax.annotate(txt, xy=xy_, xytext=xytext, fontsize=10, color=INK, ha=ha, zorder=9,
                        arrowprops=dict(arrowstyle="-", color=INK, lw=0.6, shrinkA=0, shrinkB=1))
        note((s_c, 4.0), "疏散楼梯间（剖切）", (19.0, 3.0))
        note((ms - 0.6, P.DECK + 2.9), "月洞墙（剖切面后）", (1.5, 12.9))
        note((12.0, P.DECK + 1.2), "观景厅", (14.0, 9.0))
        note((14.0, P.DECK), "屋面平台 +6.30", (19.0, 7.3))
        note((14.2, 2.4), "展廊", (19.0, 1.6))
        note((-13.0, 2.4), "门厅", (-21.5, 2.6), ha="right")
        note((-1.6, 0.0), "潮池", (-1.0, 3.0))
        note((-6.35, 2.6), "1:12 坡道", (-9.0, 5.4), ha="right")
        note((ts - 1.5, 6.0), "树岛（剖切面后）", (-6.5, 9.9), ha="right")
        for name, txt, xyt in (("Roof_B_Upper", "B  次屋顶", (-4.0, 13.4)), ("Roof_A_Main", "A  主屋顶", (21.0, 10.2)),
                               ("Roof_C_Low", "C  低檐", (-24.5, 7.4))):
            if name in cuts:
                xs_, T_, _ = cuts[name]
                i = len(xs_) // 2
                note((xs_[i], T_[i]), txt, xyt)
        for z, txt in ((P.DECK, "+6.30"), (P.FFL, "+0.85 首层"), (P.HIGH_TIDE, "±0.00 设计高潮位"), (P.LOW_TIDE, "-0.95 设计低潮位")):
            level_mark(ax, s1 - 0.4, z, txt, side="left", run=3.8)
        ax.set_ylim(-3.9, 14.2)
        scale_bar(ax, s0 + 0.5, -2.95, units=(0, 2, 5, 10), h=0.2, label_size=8, unit_gap=0.6)
        ax.text(s0, 13.8, "南  ←", fontsize=10, color=MUTED)
        ax.text(s1, 13.8, "→  北", fontsize=10, color=MUTED, ha="right")
        tight_layout(fig, ax, "2-2 剖面 · 门厅—潮池—展廊—疏散楼梯间—观景厅",
                     "沿疏散楼梯间中心线（方位 105° / 285°）剖切；浅色为剖切面后的屋顶、柱、月洞墙与树岛")
    if path is None:
        return fig, ax, info
    fig.savefig(path, facecolor=PAPER)
    plt.close(fig)


# ======================================================================= 1:20 檐口节点
DETAIL_DEG = 300.0                                 # 门厅外檐（与“檐口细部”效果图同一处）


def detail_eave(path, portrait=False):
    """1:20 节点：门厅外檐。屋面构造、楔形挑檐与包边、阴影缝、檐沟与虹吸排水、胶合木主梁与木格栅、
    钢箱环梁与柱头、幕墙收口、地面构造、基座挑檐与临水玻璃栏板。尺寸取自模型。
    portrait=True 时为竖版（说明列在图下方），供图板右栏使用。"""
    import program as PG
    th = math.radians(DETAIL_DEG)
    d = np.array([math.cos(th), math.sin(th)])
    line = LineString([d * 5, d * 30])

    def cross(poly):
        i = line.intersection(poly.boundary)
        pts = [i] if i.geom_type == "Point" else list(i.geoms)
        return sorted(float(np.hypot(q.x, q.y)) for q in pts)
    r_g = cross(P.E2)[-1]                           # 外侧玻璃
    r_e = cross(P.ROOF_C.poly)[-1]                  # 檐口
    r_p = cross(P.PLINTH_CUT)[-1]                   # 基座边
    r_c = r_g - 0.85                                # 柱心
    R = np.linspace(r_g - 1.9, r_e, 400)
    XY = d[None] * R[:, None]
    de = P.ROOF_C.dist_edge(XY[:, 0], XY[:, 1])
    top, bot = P.ROOF_C.surfaces(XY[:, 0], XY[:, 1], de)
    und = bot - P.ROOF_C.beam_depth(de)
    gut = (de > 0.45) & (de < 0.75)
    top = top - np.where(gut, 0.04 * np.sin(np.pi * (de - 0.45) / 0.30), 0.0)   # 余弦形檐沟
    zu = lambda r: float(np.interp(r, R, und))
    zb = lambda r: float(np.interp(r, R, bot))
    zt = lambda r: float(np.interp(r, R, top))
    fig = plt.figure(figsize=(12.5, 11.2), dpi=200, facecolor=PAPER)
    ax = fig.add_axes([0.04, 0.05, 0.62, 0.86])
    x0, x1, y0, y1 = r_g - 1.95, r_p + 0.75, -1.25, 5.15
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_aspect("equal"); ax.axis("off")
    LW = 0.7
    steel, wood, glass_c, ins = "#7d8086", "#c9a77c", "#9fb6c2", "#efe2b8"
    # 基座：钢筋混凝土箱体顶板 + 外墙 + 挑檐；水只在箱体外侧
    slab_t, slab_b = P.FFL - 0.15, P.FFL - 0.40
    wall_r = r_p - 0.6
    ax.add_patch(MPoly([(wall_r, y0), (x1, y0), (x1, 0), (wall_r, 0)], closed=True, fc=WATER_HT, ec="none", zorder=0))
    ax.plot([wall_r, x1], [0, 0], color="#5d7f93", lw=0.9, zorder=1)
    ax.plot([wall_r, x1], [P.LOW_TIDE] * 2, color="#5d7f93", lw=0.9, ls=(0, (5, 3)), zorder=1)
    ax.text((x0 + wall_r - 0.3) / 2, -0.35, "箱体内部：设备间 · 雨水蓄水池", ha="center", va="center", fontsize=9,
            color=MUTED, zorder=2)
    conc = dict(fc="#d9d4cb", ec=INK, lw=LW, hatch="..", zorder=3)
    ax.add_patch(MPoly([(x0, slab_b), (wall_r, slab_b), (wall_r, y0), (wall_r - 0.3, y0), (wall_r - 0.3, slab_b - 0.0),
                        (x0, slab_b)], closed=True, **conc))
    ax.add_patch(MPoly([(x0, slab_b), (r_p, slab_b), (r_p, slab_t), (x0, slab_t)], closed=True, **conc))
    ax.add_patch(MPoly([(wall_r - 0.3, y0), (wall_r, y0), (wall_r, slab_b), (wall_r - 0.3, slab_b)], closed=True, **conc))
    # 地面构造：30 石材 + 30 结合层 + 50 地暖 + 40 保温
    layers = [(slab_t, slab_t + 0.04, ins, "xx"), (slab_t + 0.04, slab_t + 0.09, "#e9e4dc", None),
              (slab_t + 0.09, slab_t + 0.12, "#e3ddd2", None), (slab_t + 0.12, P.FFL, "#cfc6b8", None)]
    for za, zb_, fc, hatch in layers:
        ax.add_patch(MPoly([(x0, za), (r_p, za), (r_p, zb_), (x0, zb_)], closed=True, fc=fc, ec=INK, lw=0.4,
                           hatch=hatch, zorder=3))
    for xx in np.arange(x0 + 0.1, r_p - 0.1, 0.2):                    # 地暖盘管
        ax.add_patch(Circle((xx, slab_t + 0.065), 0.01, fc="#c8743a", ec="none", zorder=4))
    ax.add_patch(MPoly([(r_p - 0.01, P.FFL), (r_p + 0.01, P.FFL), (r_p + 0.01, P.FFL - 0.30), (r_p - 0.01, P.FFL - 0.30)],
                       closed=True, fc=INK, ec="none", zorder=4))
    ax.add_patch(MPoly([(r_p - 0.16, P.FFL - 0.32), (r_p - 0.08, P.FFL - 0.32), (r_p - 0.08, P.FFL - 0.30),
                        (r_p - 0.16, P.FFL - 0.30)], closed=True, fc="#f3c46b", ec="none", zorder=4))
    # 玻璃栏板 1.05 m
    rb = r_p - 0.25
    ax.add_patch(MPoly([(rb - 0.035, P.FFL), (rb + 0.035, P.FFL), (rb + 0.035, P.FFL + 0.09), (rb - 0.035, P.FFL + 0.09)],
                       closed=True, fc=steel, ec=INK, lw=0.4, zorder=5))
    ax.add_patch(MPoly([(rb - 0.012, P.FFL + 0.08), (rb + 0.012, P.FFL + 0.08), (rb + 0.012, P.FFL + 1.015),
                        (rb - 0.012, P.FFL + 1.015)], closed=True, fc=glass_c, ec=INK, lw=0.4, alpha=0.9, zorder=5))
    ax.add_patch(Circle((rb, P.FFL + 1.05), 0.024, fc=steel, ec=INK, lw=0.4, zorder=5))
    # 柱：钢管 φ194、外张柱头、顶板 φ500
    zc_top = zu(r_c) - 0.01
    ax.add_patch(MPoly([(r_c - 0.097, P.FFL), (r_c + 0.097, P.FFL), (r_c + 0.097, zc_top - 0.62),
                        (r_c - 0.097, zc_top - 0.62)], closed=True, fc=steel, ec=INK, lw=LW, zorder=5))
    u = np.linspace(0, 1, 12)
    rr = 0.097 + (0.21 - 0.097) * u ** 2.2
    zz = zc_top - 0.62 + 0.6 * u
    ax.add_patch(MPoly(np.vstack([np.stack([r_c - rr, zz], 1), np.stack([r_c + rr[::-1], zz[::-1]], 1)]), closed=True,
                       fc=steel, ec=INK, lw=LW, zorder=5))
    ax.add_patch(MPoly([(r_c - 0.25, zc_top - 0.025), (r_c + 0.25, zc_top - 0.025), (r_c + 0.25, zc_top),
                        (r_c - 0.25, zc_top)], closed=True, fc="#5f6268", ec=INK, lw=LW, zorder=6))
    ax.add_patch(MPoly([(r_c - 0.30, P.FFL - 0.02), (r_c + 0.30, P.FFL - 0.02), (r_c + 0.30, P.FFL + 0.02),
                        (r_c - 0.30, P.FFL + 0.02)], closed=True, fc="#5f6268", ec=INK, lw=0.4, zorder=6))
    for off in (-0.16, 0.16):                                         # 抗拔螺栓
        ax.plot([r_c + off] * 2, [zc_top, zc_top + 0.16], color=INK, lw=1.2, zorder=7)
    # 钢箱环梁（剖切，220 宽）
    ax.add_patch(MPoly([(r_c - 0.11, zc_top), (r_c + 0.11, zc_top), (r_c + 0.11, zb(r_c)), (r_c - 0.11, zb(r_c))],
                       closed=True, fc="#9a9ca1", ec=INK, lw=LW, hatch="////", zorder=6))
    # 胶合木主梁（剖切面后，立面）：底面随悬挑弯矩变化，端部收于阴影缝内
    r_end = r_e - P.REVEAL - 0.04
    m = (R >= r_c - 0.11) & (R <= r_end)
    ax.fill_between(R[m], und[m], bot[m], color=wood, ec=INK, lw=0.4, zorder=4)
    # 木格栅次肋（剖到一根，100 高）
    m2 = R <= r_end
    ax.fill_between(R[m2], bot[m2] - P.RIB_DEPTH, bot[m2], color="#dcc3a0", ec=INK, lw=0.4, zorder=4.5)
    # 屋面：GRC 面板 + 防水 + 保温 + 轻钢龙骨；挑檐端部为实心 GRC 楔形板
    thick = top - bot
    keel = thick > 0.30
    ax.fill_between(R, bot, top, color="#e9e5de", ec="none", zorder=4.2)
    ax.fill_between(R[keel], np.maximum(bot[keel], top[keel] - 0.142), top[keel] - 0.022, color=ins, hatch="xx",
                    ec="#b9a96f", lw=0.3, zorder=4.3)
    ax.fill_between(R, top - 0.02, top, color="#cfc9bf", ec=INK, lw=0.4, zorder=4.4)
    ax.plot(R, bot, color=INK, lw=LW, zorder=4.6); ax.plot(R, top, color=INK, lw=LW, zorder=4.6)
    for xx in np.arange(r_c + 0.3, r_e - 1.2, 0.6):                   # 龙骨（C 型钢）
        z_a, z_b = zb(xx) + 0.02, zt(xx) - 0.16
        if z_b - z_a > 0.08:
            ax.plot([xx - 0.03, xx - 0.03, xx + 0.03], [z_b, z_a, z_a], color=steel, lw=1.0, zorder=4.4)
            ax.plot([xx - 0.03, xx + 0.03], [z_b, z_b], color=steel, lw=1.0, zorder=4.4)
    # 檐沟格栅 + 虹吸雨水斗 + 层内水平管 → 柱内立管
    rg_ = R[gut]
    if len(rg_):
        ax.plot([rg_.min(), rg_.max()], [zt(rg_.min()) - 0.005] * 2, color=INK, lw=0.6, ls=(0, (1, 1)), zorder=5)
        ro = (rg_.min() + rg_.max()) / 2
        ax.add_patch(MPoly([(ro - 0.04, zt(ro) - 0.10), (ro + 0.04, zt(ro) - 0.10), (ro + 0.04, zt(ro)),
                            (ro - 0.04, zt(ro))], closed=True, fc="#5f6268", ec="none", zorder=5))
        zp = zt(ro) - 0.12
        ax.plot([ro, r_c + 0.02, r_c + 0.02], [zp, zb(r_c) + 0.06, P.FFL + 0.2], color="#3d6f9c", lw=1.0,
                ls=(0, (4, 2)), zorder=7)
    # 檐口包边（半圆形拉丝金属）与阴影缝
    ze = (zt(r_e - 0.01) + zb(r_e - 0.01)) / 2
    tt = np.linspace(-np.pi / 2, np.pi / 2, 30)
    ax.add_patch(MPoly(np.stack([r_e - 0.02 + 0.08 * np.cos(tt), ze + 0.08 * np.sin(tt)], 1), closed=True,
                       fc="#c8c4bd", ec=INK, lw=LW, zorder=6))
    # 幕墙：玻璃（8+12A+8）、竖梃（剖切面后）、顶部收口带、底槛、暗藏灯带
    z_head = zu(r_g) - 0.04
    for off in (-0.014, 0.014):
        ax.plot([r_g + off] * 2, [P.FFL + 0.07, z_head], color="#4c6f80", lw=0.8, zorder=6)
    ax.add_patch(MPoly([(r_g - 0.025, P.FFL + 0.07), (r_g + 0.025, P.FFL + 0.07), (r_g + 0.025, z_head),
                        (r_g - 0.025, z_head)], closed=True, fc=glass_c, ec="none", alpha=0.35, zorder=5.5))
    ax.add_patch(MPoly([(r_g - 0.09, P.FFL - 0.01), (r_g + 0.05, P.FFL - 0.01), (r_g + 0.05, P.FFL + 0.07),
                        (r_g - 0.09, P.FFL + 0.07)], closed=True, fc=steel, ec=INK, lw=0.4, zorder=6))
    ax.add_patch(MPoly([(r_g - 0.10, z_head), (r_g + 0.07, z_head), (r_g + 0.07, zb(r_g) + 0.04),
                        (r_g - 0.10, zb(r_g) + 0.04)], closed=True, fc="#c8c4bd", ec=INK, lw=0.4, zorder=6))
    ax.add_patch(MPoly([(r_g - 0.16, z_head - 0.02), (r_g - 0.10, z_head - 0.02), (r_g - 0.10, z_head),
                        (r_g - 0.16, z_head)], closed=True, fc="#f3c46b", ec="none", zorder=6))
    # 尺寸标注
    def dim_h(xa, xb, y, txt, up=True):
        ax.annotate("", xy=(xa, y), xytext=(xb, y), arrowprops=dict(arrowstyle="<->", color=INK, lw=0.6,
                                                                    shrinkA=0, shrinkB=0), zorder=8)
        ax.text((xa + xb) / 2, y + (0.06 if up else -0.06), txt, ha="center", va="bottom" if up else "top",
                fontsize=8.5, color=INK, zorder=8)

    def dim_v(x, ya, yb, txt):
        ax.annotate("", xy=(x, ya), xytext=(x, yb), arrowprops=dict(arrowstyle="<->", color=INK, lw=0.6,
                                                                    shrinkA=0, shrinkB=0), zorder=8)
        ax.text(x + 0.05, (ya + yb) / 2, txt, ha="left", va="center", fontsize=8.5, color=INK, rotation=90, zorder=8)
    dim_h(r_g, r_e, zt(r_e) + 0.32, f"挑檐 {r_e - r_g:.2f} m")
    dim_v(r_g - 0.55, P.FFL, zu(r_g), f"净高 {zu(r_g) - P.FFL:.2f} m")
    dim_v(rb + 0.18, P.FFL, P.FFL + 1.05, "1.05 m")
    for z, txt in ((P.FFL, "+0.85"), (0.0, "±0.00 设计高潮位"), (P.LOW_TIDE, "-0.95 设计低潮位")):
        level_mark(ax, x1 - 0.08, z, txt, side="left", h=0.055, run=0.62)
    # 编号引线 + 右侧说明
    items = [
        ((r_e - 1.6, zt(r_e - 1.6) - 0.01), "20 厚 GRC 屋面板，象牙白矿物涂层"),
        ((r_c + 0.8, zt(r_c + 0.8) - 0.08), "防水卷材 + 120 厚挤塑保温"),
        ((r_c + 0.31, zb(r_c + 0.31) + 0.09), "轻钢龙骨层，屋面板下设钢拉杆水平支撑"),
        ((r_c + 1.0, zu(r_c + 1.0) + 0.05), f"胶合木主梁 GL28h 160 宽，高 {P.BEAM_EDGE * 1000:.0f} → {P.ROOF_C.beam_core * 1000:.0f}"),
        ((r_g + 0.6, zb(r_g + 0.6) - 0.05), "100 厚木格栅次肋，间距约 240"),
        ((r_e - 0.02, ze), "半圆形拉丝金属包边，檐口厚 150"),
        ((r_e - P.REVEAL / 2 - 0.02, zb(r_e - 0.1) - 0.04), "120 阴影缝，梁与格栅端部收于缝内"),
        ((ro if len(rg_) else r_e - 0.6, zt(r_e - 0.6) - 0.03), "余弦形檐沟 + 金属格栅 + 虹吸雨水斗"),
        ((r_c, zb(r_c) - 0.15), "钢箱环梁 220 宽，梁底与主梁底平"),
        ((r_c + 0.12, zc_top - 0.3), "钢管柱 φ194×10，外张柱头，顶板 φ500，抗拔螺栓"),
        ((r_c + 0.02, P.FFL + 1.2), "雨水立管 DN100 藏于柱内 → 基座蓄水池"),
        ((r_g, z_head + 0.08), "顶部金属收口带封至梁底，内侧暗藏灯带"),
        ((r_g, P.FFL + 1.6), "幕墙：50×130 竖梃，8+12A+8 Low-E 中空夹胶玻璃"),
        ((r_g - 1.2, P.FFL - 0.03), "30 石材 + 30 结合层 + 50 地暖 + 40 保温"),
        ((r_g - 1.0, slab_b + 0.12), "250 厚钢筋混凝土箱体顶板（箱内为设备间、蓄水池）"),
        ((rb, P.FFL + 0.6), "玻璃栏板 12+12 夹胶，φ48 扶手，高 1.05 m"),
        ((r_p - 0.12, P.FFL - 0.31), "基座挑檐 300，檐下灯带"),
    ]
    # 编号标签：贴在构件旁（不拉长引线），右侧列出说明
    offsets = [(0.0, 0.16), (-0.05, 0.22), (0.0, -0.20), (0.15, -0.2), (0.0, -0.18), (0.2, 0.05), (0.05, -0.2),
               (0.0, 0.18), (-0.28, 0.0), (0.26, -0.1), (0.22, 0.0), (-0.2, 0.08), (0.18, 0.0), (0.0, 0.18),
               (0.0, -0.12), (-0.2, 0.0), (0.0, -0.17)]
    ys = np.linspace(0.90, 0.12, len(items))
    for k, ((xa, ya), txt) in enumerate(items):
        ox, oy = offsets[k]
        tx, ty = xa + ox, ya + oy
        if ox or oy:
            ax.plot([xa, tx], [ya, ty], color="#6d6a64", lw=0.5, zorder=9)
        ax.add_patch(Circle((xa, ya), 0.018, fc=SUN, ec="none", zorder=10))
        ax.add_patch(Circle((tx, ty), 0.075, fc=PAPER, ec=SUN, lw=0.8, zorder=10))
        ax.text(tx, ty, f"{k + 1}", ha="center", va="center", fontsize=7.5, color=SUN, zorder=11)
    scale_bar(ax, x0 + 0.1, y0 + 0.18, units=(0, 0.5, 1), h=0.04, label_size=8, unit_gap=0.08, label_gap=0.06)
    title = "节点大样 1:20 · 门厅外檐"
    sub = ["楔形挑檐、阴影缝与檐沟排水；屋面荷载经胶合木主梁 → 钢箱环梁 → 柱", "尺寸取自模型（方位 300° 径向剖切）"]
    if portrait:                                         # 竖版：图在上，说明在下
        Wf = 8.4
        dh = 0.92 * Wf * (y1 - y0) / (x1 - x0)
        head, gap, row = 1.15, 0.25, 0.26
        Hf = head + dh + gap + row * len(items) + 0.15
        fig.set_size_inches(Wf, Hf)
        ax.set_position([0.04, (Hf - head - dh) / Hf, 0.92, dh / Hf])
        fig.text(0.04, (Hf - 0.36) / Hf, title, fontsize=18, color=INK, va="center")
        for k, t in enumerate(sub):
            fig.text(0.04, (Hf - 0.66 - 0.21 * k) / Hf, t, fontsize=10, color=MUTED, va="center")
        for k, (_, txt) in enumerate(items):
            fy = (Hf - head - dh - gap - row * (k + 0.5)) / Hf
            fig.text(0.05, fy, f"{k + 1:>2}", fontsize=9.5, color=SUN, va="center")
            fig.text(0.09, fy, txt, fontsize=9.5, color=INK, va="center")
    else:
        for k, (_, txt) in enumerate(items):
            fig.text(0.70, ys[k], f"{k + 1:>2}", fontsize=9.5, color=SUN, va="center")
            fig.text(0.725, ys[k], txt, fontsize=9.5, color=INK, va="center")
        fig.text(0.04, 0.955, title, fontsize=18, color=INK)
        fig.text(0.04, 0.93, "；".join(sub), fontsize=10, color=MUTED)
    fig.savefig(path, facecolor=PAPER)
    plt.close(fig)


# ======================================================================= 形体生成
def _axo(x, y, z, a=-45.0, el=32.0):
    """轴测投影：自东南上方看，屏幕上方为西北。"""
    ca, sa = math.cos(math.radians(a)), math.sin(math.radians(a))
    xr = x * ca - y * sa
    yr = x * sa + y * ca
    return xr, yr * math.sin(math.radians(el)) + z * math.cos(math.radians(el))


def _axo_poly(ax, ring, z, **kw):
    ring = np.asarray(ring)
    zz = np.broadcast_to(z, (len(ring),))
    X, Y = _axo(ring[:, 0], ring[:, 1], zz)
    ax.add_patch(MPoly(np.stack([X, Y], 1), closed=True, **kw))


def form_diagram(path):
    """形体生成：潮池 → 三片月牙 → 错落升起 → 对准冬至日落。"""
    fig = plt.figure(figsize=(18, 3.9), dpi=200, facecolor=PAPER)
    clip = Point(0, 0).buffer(24, 128)
    plinth = P.PLINTH_ALL.intersection(clip)
    cols = {"Roof_A_Main": "#efe9df", "Roof_B_Upper": "#f8f4ee", "Roof_C_Low": "#e7e0d4"}
    steps = [("1  潮池", "一圈会涨落的水：经水口与湖面相通，一日两涨两落"),
             ("2  三片月牙", "三片月牙屋顶环抱潮池，彼此交错，围出庭院"),
             ("3  错落升起", "主屋顶西翼尖扬起，次屋顶叠于其上，低檐伏于水面"),
             ("4  对准冬至日落", "水口、潮池与月洞墙同指冬至日落方位 242.6°")]

    def roof_ring(roof, n=360):
        ring = ccw_ring(roof.poly)
        tt = np.linspace(0, len(ring) - 1, n).astype(int)
        ring = ring[tt]
        top, _ = roof.surfaces(ring[:, 0], ring[:, 1])
        return ring, top
    for k, (title, cap) in enumerate(steps):
        ax = fig.add_axes([0.01 + k * 0.248, 0.215, 0.235, 0.635])
        ax.set_aspect("equal"); ax.axis("off")
        ax.set_xlim(-26, 26); ax.set_ylim(-10.8, 19.2)
        base = P.PLINTH_CUT.intersection(clip)              # 台基上的水口：潮池经此与湖面相通
        parts = [base] if base.geom_type == "Polygon" else list(base.geoms)
        for g in parts:
            _axo_poly(ax, np.asarray(g.exterior.coords), 0.0, fc="#e4ddd2", ec="#a59e93", lw=0.6, zorder=1)
        inlet = P.CHANNEL.intersection(plinth).difference(P.BASIN)
        for g in ([inlet] if inlet.geom_type == "Polygon" else list(inlet.geoms)):
            _axo_poly(ax, np.asarray(g.exterior.coords), 0.0, fc=WATER_HT, ec="#5d7f93", lw=0.6, zorder=2)
        circ = np.stack([P.BASIN_R * np.cos(np.linspace(0, 2 * np.pi, 120)),
                         P.BASIN_R * np.sin(np.linspace(0, 2 * np.pi, 120))], 1)
        _axo_poly(ax, circ, 0.0, fc=WATER_HT, ec="#5d7f93", lw=0.8, zorder=2)
        if k == 0:
            for r_out, _, _ in P.BASIN_STEPS:
                c = np.stack([r_out * np.cos(np.linspace(0, 2 * np.pi, 120)), r_out * np.sin(np.linspace(0, 2 * np.pi, 120))], 1)
                X, Y = _axo(c[:, 0], c[:, 1], np.zeros(len(c)))
                ax.plot(X, Y, color="#8aa3b0", lw=0.5, zorder=2)
            for zz_ in (0.0, -0.95):
                X, Y = _axo(circ[:, 0] * 0.6, circ[:, 1] * 0.6, np.full(len(circ), zz_ * 1.5))
                ax.plot(X, Y, color="#5d7f93", lw=0.8, ls="-" if zz_ == 0 else (0, (3, 2)), zorder=3)
        roofs = sorted(P.ROOFS, key=lambda r: float(np.mean(roof_ring(r, 60)[1])))
        if k >= 2:                                           # 屋顶在水面上的投影，提示“升起”
            for roof in roofs:
                ring, top = roof_ring(roof)
                X, Y = _axo(ring[:, 0], ring[:, 1], np.zeros(len(ring)))
                ax.add_patch(MPoly(np.stack([X, Y], 1), closed=True, fc="#8a7f70", ec="none", alpha=0.16, zorder=3))
        if k >= 1:
            for roof in roofs:
                ring, top = roof_ring(roof)
                zz = np.zeros(len(ring)) if k == 1 else top
                zo = 5 + 2 * roofs.index(roof)
                ghost = k == 3 and roof.name == "Roof_B_Upper"      # 第 4 步揭开次屋顶，露出其下的月洞墙
                if k >= 2 and not ghost:                         # 屋檐厚度
                    Xb, Yb = _axo(ring[:, 0], ring[:, 1], zz - 0.55)
                    ax.add_patch(MPoly(np.stack([Xb, Yb], 1), closed=True, fc="#b9b0a3", ec=INK, lw=0.6, zorder=zo))
                X, Y = _axo(ring[:, 0], ring[:, 1], zz)
                if ghost:
                    ax.add_patch(MPoly(np.stack([X, Y], 1), closed=True, fc=cols[roof.name], alpha=0.45, ec="none",
                                       zorder=zo + 1))
                    ax.plot(np.r_[X, X[0]], np.r_[Y, Y[0]], color=INK, lw=0.8, ls=(0, (4, 2.5)), zorder=zo + 1)
                    continue
                ax.add_patch(MPoly(np.stack([X, Y], 1), closed=True, fc=cols[roof.name] if k > 1 else "none",
                                   ec=INK, lw=1.0 if k > 1 else 1.2, zorder=zo + 1))
                if k >= 2:
                    for i in range(0, len(ring), 45):
                        Xb, Yb = _axo(np.array([ring[i, 0]] * 2), np.array([ring[i, 1]] * 2), np.array([0.0, top[i]]))
                        ax.plot(Xb, Yb, color="#9b948a", lw=0.5, ls=(0, (2, 2)), zorder=4)
                if k == 1:                                       # 标注放在不与其他屋顶重叠的部分
                    others = shapely.union_all([r.poly for r in roofs if r is not roof])
                    own = roof.poly.difference(others)
                    own = max(own.geoms, key=lambda g: g.area) if own.geom_type != "Polygon" else own
                    c = shapely.ops.polylabel(own, tolerance=0.05)
                    Xc, Yc = _axo(np.array([c.x]), np.array([c.y]), np.array([0.0]))
                    ax.text(Xc[0], Yc[0], roof.name.split("_")[1], ha="center", va="center", fontsize=12, color=INK,
                            zorder=9)
        if k == 3:
            import matplotlib.patheffects as pe
            halo = [pe.withStroke(linewidth=3.2, foreground=PAPER)]
            # 月洞墙：立于屋面平台西端，墙面垂直于轴线（厚 0.42 m，圆洞半径 1.42 m）
            hr, hz, zt = P.MOON_HOLE_R, P.DECK + 0.5 + P.MOON_HOLE_R, P.DECK + 4.3
            u = np.array([-P.MOON_HALF, P.MOON_HALF, P.MOON_HALF, -P.MOON_HALF])
            wz = np.array([P.DECK, P.DECK, zt, zt])
            tt = np.linspace(0, 2 * np.pi, 90)
            for off, fc in ((0.21, "#cfc6b8"), (-0.21, "#f4efe7")):      # 先画背面，再画朝向观者的一面
                q = P.MOON_C[None] + P.AXIS[None] * off + P.AXIS_N[None] * u[:, None]
                Xw, Yw = _axo(q[:, 0], q[:, 1], wz)
                ax.add_patch(MPoly(np.stack([Xw, Yw], 1), closed=True, fc=fc, ec=INK, lw=0.8, zorder=11))
                qh = P.MOON_C[None] + P.AXIS[None] * off + P.AXIS_N[None] * (hr * np.cos(tt))[:, None]
                Xh, Yh = _axo(qh[:, 0], qh[:, 1], hz + hr * np.sin(tt))
                ax.add_patch(MPoly(np.stack([Xh, Yh], 1), closed=True, fc=cols["Roof_A_Main"], ec=INK, lw=0.7,
                                   zorder=11))
            # 轴线：潮池中心 → 水口 → 冬至日落；月洞的视线与之平行
            ln = np.stack([P.AXIS * -4, P.AXIS * 24.0], 0)
            X, Y = _axo(ln[:, 0], ln[:, 1], np.array([0.6, 0.6]))
            ax.plot(X, Y, color=SUN, lw=1.6, ls=(0, (6, 3)), zorder=12, path_effects=halo)
            q0 = P.MOON_C; q1 = P.MOON_C + P.AXIS * 17.0
            Xv, Yv = _axo(np.array([q0[0], q1[0]]), np.array([q0[1], q1[1]]), np.array([hz, hz]))
            ax.add_patch(FancyArrowPatch((Xv[0], Yv[0]), (Xv[1], Yv[1]), arrowstyle="-|>", mutation_scale=11, color=SUN,
                                         lw=1.4, ls=(0, (6, 3)), zorder=12))
            sp = P.AXIS * 25.5
            Xs, Ys = _axo(np.array([sp[0]]), np.array([sp[1]]), np.array([0.6]))
            ax.add_patch(Circle((Xs[0], Ys[0]), 1.1, fc=SUN, ec="none", zorder=12))
            ax.text(Xs[0], Ys[0] - 1.7, "冬至日落\n242.6°", ha="center", va="top", fontsize=9.5, color=SUN, zorder=12,
                    linespacing=1.15)
            ax.text(Xv[0] + 1.5, Yv[0] - 0.2, "月洞墙", ha="left", va="center", fontsize=9.5, color=INK, zorder=12,
                    path_effects=halo)
        fig.text(0.02 + k * 0.248, 0.125, title, fontsize=15, color=INK)
        fig.text(0.02 + k * 0.248, 0.045, cap, fontsize=10, color=MUTED)
    fig.text(0.01, 0.905, "形体生成", fontsize=20, color=INK)
    fig.savefig(path, facecolor=PAPER)
    plt.close(fig)


def ccw_ring(poly):
    return ccw(resample_ring(poly.exterior.coords, 0.3))


# ======================================================================= 环境策略
ENV_LEGEND = [("日照", "冬至正午 36.6° 的阳光越过低檐照进潮池，在门厅玻璃内约 2 m；夏至正午 83.4° 被屋面与 2.8 m 挑檐挡住"),
              ("庭院拔风", "湖面凉风自南侧低位进入，经门厅与潮池上空升起，自庭院上方排出"),
              ("展廊排风", "庭院侧玻璃可开启，热空气经屋面光缝排出"),
              ("湖水源热泵", "湖底换热盘管 + 地板辐射供暖 / 供冷，设备间在基座箱体内"),
              ("雨水", "檐沟虹吸雨水斗 → 柱内立管 → 基座蓄水池，回用于冲厕与潮池补水"),
              ("天光", "屋面光缝顺北侧实墙洗下，展厅不受直射光"),
              ("西晒", "展廊朝西、北为实墙；观景厅西侧由月洞墙与深挑檐遮挡")]


def env_diagram(path, legend=True):
    """环境策略：以 2-2 剖面为底。冬夏日照与挑檐遮阳、庭院拔风、展廊经屋面光缝排风、湖水源热泵与地板辐射、
    屋面雨水回用、天光洗墙。legend=False 时不画图例（图板右栏另列）。"""
    from matplotlib.path import Path as MPath
    fig, ax, info = section_cross(None, env=True)
    blue, warm, rain, sun_w, sun_s, day = "#3d7fb0", "#d06a34", "#2f6fa1", "#e3a23b", "#c8743a", "#e8c45a"

    def flow(pts, col, lw=2.2, ls="-"):
        pts = np.asarray(pts, float)
        codes = [MPath.MOVETO] + [MPath.CURVE3] * (len(pts) - 1) if len(pts) % 2 == 1 else [MPath.MOVETO] + [MPath.LINETO] * (len(pts) - 1)
        ax.add_patch(FancyArrowPatch(path=MPath(pts, codes), arrowstyle="-|>", mutation_scale=14, color=col, lw=lw,
                                     ls=ls, zorder=12))

    def tag(x, y, n):
        ax.add_patch(Circle((x, y), 0.55, fc=PAPER, ec=SUN, lw=1.0, zorder=13))
        ax.text(x, y, str(n), ha="center", va="center", fontsize=10, color=SUN, zorder=14)
    # 1 冬至正午低角度阳光进入门厅与庭院；夏至正午高角度阳光被屋面与挑檐挡住
    for alt, col, ends in ((36.56, sun_w, [(-14.1, P.FFL), (-3.2, 0.25), (0.8, P.BASIN_FLOOR + 0.6)]),
                           (83.44, sun_s, [(-15.5, 4.8), (-12.5, 4.65), (10.5, 6.0), (14.0, 6.05)])):
        dv = np.array([math.cos(math.radians(alt)), -math.sin(math.radians(alt))])
        for ex, ez in ends:
            t0 = (13.6 - ez) / -dv[1]
            st = np.array([ex, ez]) - dv * t0
            ax.add_patch(FancyArrowPatch(tuple(st), (ex, ez), arrowstyle="-|>", mutation_scale=10, color=col, lw=1.3,
                                         zorder=11))
    tag(-24.6, 12.2, 1)
    # 2 庭院拔风：湖面凉风自南侧低位进入，经门厅、潮池上空升起，自庭院上方排出
    flow([(-26.5, 1.5), (-20.0, 1.7), (-13.0, 2.0)], blue)
    flow([(-11.5, 2.2), (-6.0, 2.4), (-2.0, 4.5)], blue)
    flow([(-1.5, 5.2), (0.5, 8.8), (2.0, 13.4)], warm, lw=2.6)
    tag(-20.0, 3.3, 2)
    # 3 展廊：庭院侧玻璃可开启，热空气经屋面光缝排出
    if info["sky"]:
        a_, b_ = info["sky"][0]
        sx = (a_ + b_) / 2
        flow([(8.0, 1.8), (13.0, 2.4), (sx, 4.2)], blue, lw=1.8)
        flow([(sx, 4.6), (sx + 0.2, 6.6), (sx + 0.4, 8.6)], warm, lw=1.8)
        tag(sx + 1.6, 8.2, 3)
        # 6 天光：光缝顺北墙洗下
        for k in range(3):
            ax.add_patch(FancyArrowPatch((sx - 0.5 + 0.4 * k, 9.6), (sx + 0.6 + 0.25 * k, 1.4 + 0.5 * k),
                                         arrowstyle="-|>", mutation_scale=9, color=day, lw=1.2, alpha=0.9, zorder=11))
        tag(sx - 1.7, 9.4, 6)
    # 4 湖水源热泵 + 地板辐射供暖供冷
    ax.add_patch(MPoly([(-15.5, -1.55), (-9.5, -1.55), (-9.5, 0.25), (-15.5, 0.25)], closed=True, fc="#f3eee5",
                       ec="#6d6a64", lw=0.8, zorder=10))
    ax.text(-12.5, -0.45, "设备间\n热泵 · 水箱", ha="center", va="center", fontsize=8.5, color=INK, zorder=11)
    xx = np.linspace(-26.5, -20.5, 120)
    ax.plot(xx, -1.95 + 0.12 * np.sin(xx * 6), color=blue, lw=1.4, zorder=11)
    ax.plot([-20.5, -18.0, -15.5], [-1.95, -1.6, -1.2], color=blue, lw=1.4, zorder=11)
    for xa_, xb_ in ((-17.6, -10.6), (9.8, 18.0)):
        xx = np.linspace(xa_, xb_, 200)
        ax.plot(xx, P.FFL - 0.07 + 0.03 * np.sin(xx * 9), color=warm, lw=1.0, zorder=11)
    tag(-23.4, -1.0, 4)
    # 5 雨水：檐沟 → 柱内立管 → 基座蓄水池 → 冲厕、潮池补水
    ax.add_patch(MPoly([(-8.8, -1.55), (-6.4, -1.55), (-6.4, -0.2), (-8.8, -0.2)], closed=True, fc="#dfe9f0",
                       ec=rain, lw=0.8, zorder=10))
    ax.text(-7.6, -0.85, "蓄水池", ha="center", va="center", fontsize=8.5, color=rain, zorder=11)
    flow([(-18.0, 4.62), (-16.5, 4.4), (-14.8, 4.1)], rain, lw=1.4, ls=(0, (3, 2)))
    ax.plot([-14.8, -14.8, -8.8], [4.0, -1.3, -1.3], color=rain, lw=1.4, ls=(0, (3, 2)), zorder=11)
    flow([(-6.4, -0.6), (-5.6, -0.4), (-4.9, -0.2)], rain, lw=1.4, ls=(0, (3, 2)))
    tag(-18.6, 5.9, 5)
    # 7 西晒：展廊外墙为实墙；观景厅西侧有月洞墙与深挑檐
    tag(info["ms"] - 2.2, P.DECK + 4.6, 7)
    ax.set_ylim(-2.9, 14.2)
    foot = 1.25 if legend else 0.2                         # 图例在图下方
    Hf = tight_layout(fig, ax, "环境策略 · 以 2-2 剖面示意", "北纬 30°：遮阳、通风、湖水源热泵、雨水回用与天光",
                      foot=foot)
    if legend:
        for k, (name, t) in enumerate(ENV_LEGEND):
            fig.text(0.035 + (k // 4) * 0.47, (foot - 0.3 - (k % 4) * 0.25) / Hf, f"{k + 1}  {name}：{t}", fontsize=9.5,
                     color=INK, va="center")
    fig.savefig(path, facecolor=PAPER)
    plt.close(fig)


# ======================================================================= 潮位曲线
def level_mark(ax, x, z, txt, side="right", size=8.5, color=INK, h=0.3, run=3.6):
    """标高符号：倒三角尖端落在标高处，数字写在三角形上方的横线上。"""
    ax.add_patch(MPoly([(x, z), (x - h, z + h), (x + h, z + h)], closed=True, fc="none", ec=color, lw=0.8, zorder=9))
    x1 = x + h + run if side == "right" else x - h - run
    ax.plot([x - h if side == "right" else x + h, x1], [z + h, z + h], color=color, lw=0.8, zorder=9)
    ax.text(x - h + 0.1 if side == "right" else x + h - 0.1, z + h + 0.06, txt, fontsize=size, color=color,
            ha="left" if side == "right" else "right", va="bottom", zorder=9)


ELEV_NOTES = [  # (要素, 指向点 (x, z), 文字位置 (x, z), 对齐)
    ("B  次屋顶", (2.0, 10.9), (6.5, 13.4), "left"),
    ("A  主屋顶 · 西翼尖", (-17.5, 9.3), (-19.0, 11.4), "right"),
    ("C  低檐", (14.0, 4.6), (16.5, 8.6), "left"),
    ("Y 形斜柱", (-11.3, 3.2), (-18.6, 4.4), "right"),
    ("月洞墙", (-8.4, 8.3), (-16.0, 7.6), "right"),
    ("栈桥（剖切）", (0.0, 0.55), (4.0, -3.0), "left"),
]


def elevation_drawing(raw, path, notes=ELEV_NOTES):
    """南立面：elevation.py 的线稿渲染 + 水体、标高、比例尺与要素标注。"""
    import json
    with open(raw[:-4] + ".json") as f:
        meta = json.load(f)
    img = plt.imread(raw)
    u0, u1, z0, z1 = meta["extent"]
    xa, xb, za, zb = -23.0, 29.0, -4.4, 14.4
    fig = plt.figure(figsize=(16, 6), dpi=200, facecolor=PAPER)
    ax = fig.add_axes([0.03, 0.03, 0.94, 0.8])
    ax.set_xlim(xa, xb); ax.set_ylim(za, zb); ax.set_aspect("equal"); ax.axis("off")
    xr = 21.5                                              # 右侧只到标高线起点：其外只有水面（旧底图里水面边缘有一道线）
    img = img[:, :int(round((xr - u0) * meta["px_per_m"]))]
    ax.imshow(img, extent=(u0, xr, z0, z1), zorder=3, interpolation="antialiased")
    # 水体在立面前方：设计高潮位以下以半透明浅蓝覆盖，没入水中的台基仍隐约可见
    zw = -2.5
    ax.add_patch(MPoly([(xa, zw), (21.5, zw), (21.5, P.HIGH_TIDE), (xa, P.HIGH_TIDE)], closed=True, fc=WATER_HT,
                       alpha=0.55, ec="none", zorder=4))
    ax.plot([xa, 21.5], [P.HIGH_TIDE] * 2, color="#5d7f93", lw=1.0, zorder=5)
    ax.plot([xa, 21.5], [P.LOW_TIDE] * 2, color="#5d7f93", lw=0.8, ls=(0, (5, 3)), zorder=5)
    for z, txt in ((P.DECK, "+6.30 屋面平台"), (P.FFL, "+0.85 首层"), (P.HIGH_TIDE, "±0.00 设计高潮位"),
                   (P.LOW_TIDE, "-0.95 设计低潮位")):
        ax.plot([20.0, 22.9], [z, z], color=INK, lw=0.6, zorder=9)
        level_mark(ax, 22.9, z, txt)
    zmax, xmax = 0.0, 0.0
    for r in P.ROOFS:                                      # 最高点（次屋顶）
        xs = np.linspace(*r.poly.bounds[0::2], 200); ys = np.linspace(*r.poly.bounds[1::2], 200)
        X, Y = np.meshgrid(xs, ys); X, Y = X.ravel(), Y.ravel()
        m = shapely.contains_xy(r.poly, X, Y)
        top, _ = r.surfaces(X[m], Y[m])
        if top.max() > zmax:
            zmax, xmax = float(top.max()), float(X[m][np.argmax(top)])
    level_mark(ax, xmax, zmax, f"+{zmax:.2f} 最高点")
    for txt, xy, xyt, ha in notes:
        ax.annotate(txt, xy=xy, xytext=xyt, fontsize=10, color=INK, ha=ha, va="center", zorder=9,
                    arrowprops=dict(arrowstyle="-", color=INK, lw=0.6, shrinkA=2, shrinkB=1))
    scale_bar(ax, xa + 0.6, -3.5, units=(0, 2, 5, 10), h=0.18, label_size=8, unit_gap=0.6)
    ax.text(xa + 0.3, zb - 0.6, "西  ←", fontsize=10, color=MUTED)
    ax.text(27.0, zb - 0.6, "→  东", fontsize=10, color=MUTED, ha="right")
    tight_layout(fig, ax, "南立面", "自湖面南侧看向门厅；栈桥在台基南缘外剖切（黑色）；浅蓝为设计高潮位以下的水体")
    fig.savefig(path, facecolor=PAPER)
    plt.close(fig)


SOLAR_DAY = {"winter": ("冬至", "#c8743a"), "summer": ("夏至", "#3d7fb0")}
SUN_HOURS_RAMP = ["#c9a67b", "#d0853c", "#b8621c", "#92440c", "#6c2b04"]      # 单色相由浅到深（已用色板校验脚本检查）
SUN_HOURS_BINS = [0, 1, 2, 4, 6, 24]


def solar_diagram(src, path):
    """日照分析：太阳轨迹图（北纬 30°）、潮池与门厅被晒到的面积比例、冬夏至首层日照时数。数据来自 solar.py 的光线追踪。"""
    import json
    from matplotlib.colors import ListedColormap, BoundaryNorm
    with open(src) as f:
        D = json.load(f)
    X, Y = np.array(D["x"]), np.array(D["y"])
    pool, lobby = np.array(D["pool"]), np.array(D["lobby"])
    fig = plt.figure(figsize=(16, 7.9), dpi=200, facecolor=PAPER)
    # ---- 太阳轨迹（等距方位投影：半径 = 90° - 高度角，北在上）
    ax = fig.add_axes([0.02, 0.07, 0.31, 0.74])
    ax.set_aspect("equal"); ax.axis("off"); ax.set_xlim(-104, 104); ax.set_ylim(-104, 104)

    def xy(alt, az):
        r = 90 - np.asarray(alt)
        th = np.radians(90 - np.asarray(az))
        return r * np.cos(th), r * np.sin(th)
    for a in (0, 15, 30, 45, 60, 75):
        t = np.linspace(0, 2 * np.pi, 200)
        ax.plot((90 - a) * np.cos(t), (90 - a) * np.sin(t), color="#c9c2b6", lw=0.8 if a else 1.2, zorder=1)
        if a:
            ax.text(2, 90 - a + 1.5, f"{a}°", fontsize=11, color=MUTED, zorder=2)
    for az, lab in ((0, "北"), (90, "东"), (180, "南"), (270, "西")):
        x0, y0 = xy(0, az)
        ax.plot([0, x0], [0, y0], color="#d8d2c7", lw=0.6, zorder=1)
        x1, y1 = xy(-9, az)
        ax.text(x1, y1, lab, ha="center", va="center", fontsize=11, color=INK)
    for name, (lab, col) in list(SOLAR_DAY.items()) + [("equinox", ("春秋分", "#9b948a"))]:
        dec = D["days"][name]["dec"]
        t0, t1 = P.day_length(P.LATITUDE, dec)
        tt = np.linspace(t0, t1, 160)
        A = np.array([P.sun_vector(P.LATITUDE, dec, t)[1:] for t in tt])
        px, py = xy(A[:, 0], A[:, 1])
        ax.plot(px, py, color=col, lw=2.0 if name != "equinox" else 1.0, zorder=4,
                ls="-" if name != "equinox" else (0, (4, 3)))
        for hr in range(int(np.ceil(t0)), int(t1) + 1):
            _, al, az_ = P.sun_vector(P.LATITUDE, dec, hr)
            hx, hy = xy(al, az_)
            if name != "equinox":
                ax.add_patch(Circle((hx, hy), 1.6, fc=PAPER, ec=col, lw=1.2, zorder=5))
                if hr in (8, 12, 16):
                    ax.text(hx, hy + 4.2, f"{hr}:00", fontsize=10.5, color=MUTED, ha="center", zorder=6)
        _, al, _ = P.sun_vector(P.LATITUDE, dec, 12.0)
        nx, ny = xy(al, 180)
        ax.text(nx + 3.5, ny - (5 if name == "winter" else 2), f"{lab} 正午 {al:.1f}°" if name != "equinox" else lab,
                fontsize=10.5, color=col if name != "equinox" else MUTED, ha="left", va="center", zorder=6)
    sx, sy = xy(0, P.SOLSTICE_SUNSET_AZ)
    ax.plot([-sx * 0.95, sx], [-sy * 0.95, sy], color=SUN, lw=1.4, ls=(0, (6, 3)), zorder=3)
    ax.add_patch(Circle((sx, sy), 3.4, fc=SUN, ec="none", zorder=6))
    ax.text(sx + 5, sy - 6, f"冬至日落 {P.SOLSTICE_SUNSET_AZ}°\n水口 · 潮池 · 月洞墙同向", fontsize=10.5, color=SUN,
            ha="left", va="top", zorder=6)
    fig.text(0.03, 0.83, "太阳轨迹 · 北纬 30°（真太阳时）", fontsize=13.5, color=INK)
    # ---- 被晒到的面积比例（小多图：潮池水面、门厅地面）
    for k, (key, title, idx) in enumerate((("pool", "潮池水面被晒到的面积", 3), ("lobby", "门厅地面被晒到的面积", 4))):
        a2 = fig.add_axes([0.38, 0.49 - k * 0.38, 0.27, 0.27])
        a2.set_facecolor(PAPER)
        for name, (lab, col) in SOLAR_DAY.items():
            ser = np.array(D["days"][name]["series"])
            a2.plot(ser[:, 0], ser[:, idx] * 100, color=col, lw=2.0, solid_capstyle="round", label=lab)
            i = int(np.argmax(ser[:, idx]))                       # 直接标注放在峰值上方
            a2.text(ser[i, 0], ser[i, idx] * 100 + 5, f"{lab} {ser[i, idx] * 100:.0f}%", fontsize=10, color=INK,
                    ha="center", va="bottom")
        if k == 0:
            a2.legend(loc="upper right", fontsize=10, frameon=False, handlelength=1.6, labelcolor=INK)
        a2.set_xlim(4.5, 20); a2.set_ylim(0, 108)
        a2.set_xticks([6, 9, 12, 15, 18]); a2.set_xticklabels(["6:00", "9:00", "12:00", "15:00", "18:00"], fontsize=11)
        a2.set_yticks([0, 50, 100]); a2.set_yticklabels(["0", "50%", "100%"], fontsize=11)
        for sp in ("top", "right"):
            a2.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            a2.spines[sp].set_color("#b9b2a6"); a2.spines[sp].set_linewidth(0.8)
        a2.tick_params(colors=MUTED, length=3)
        a2.grid(axis="y", color="#e2dcd2", lw=0.6)
        a2.set_axisbelow(True)
        a2.set_title(title, fontsize=12.5, color=INK, loc="left", pad=6)
    fig.text(0.38, 0.83, "一天中被直射阳光照到的面积（光线追踪）", fontsize=13.5, color=INK)
    # ---- 首层日照时数（冬至 / 夏至）
    cmap = ListedColormap(SUN_HOURS_RAMP)
    norm = BoundaryNorm(SUN_HOURS_BINS, cmap.N)
    for k, name in enumerate(("winter", "summer")):
        a3 = fig.add_axes([0.70 + k * 0.148, 0.2, 0.14, 0.58])
        a3.set_aspect("equal"); a3.axis("off"); a3.set_xlim(-21.5, 21.5); a3.set_ylim(-21.5, 21.5)
        a3.set_anchor("N")
        hrs = np.array(D["days"][name]["hours"])
        st = D["step"]
        from matplotlib.collections import PatchCollection
        from matplotlib.patches import Rectangle
        cells = [Rectangle((x - st / 2, y - st / 2), st, st) for x, y in zip(X, Y)]
        pc = PatchCollection(cells, cmap=cmap, norm=norm, edgecolor="none", zorder=1)
        pc.set_array(hrs)
        a3.add_collection(pc)
        for g in (P.E1, P.E2):
            line_poly(a3, g, color=INK, lw=0.7, zorder=3)
        a3.add_patch(Circle((0, 0), P.BASIN_R, fc="none", ec="#2f6fa1", lw=0.8, zorder=3))
        for roof in P.ROOFS:
            line_poly(a3, roof.poly, color="#6d6a64", lw=0.5, ls=(0, (3, 2)), zorder=2)
        lab, col = SOLAR_DAY[name]
        a3.set_title(f"{lab} · 日照时数", fontsize=12.5, color=INK, loc="left", pad=4)
        mp, ml = hrs[pool].mean(), hrs[lobby].mean()
        a3.text(-21, -24.5, f"潮池平均 {mp:.1f} h\n门厅地面平均 {ml:.1f} h", fontsize=10, color=INK, va="top")
    cax = fig.add_axes([0.70, 0.36, 0.28, 0.022])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal",
                      ticks=SUN_HOURS_BINS[:-1])
    cb.ax.set_xticklabels(["0", "1", "2", "4", "6 h 以上"], fontsize=11)
    cb.outline.set_visible(False)
    cb.ax.tick_params(colors=MUTED, length=0)
    fig.text(0.70, 0.83, "首层地面与潮池的直射日照时数", fontsize=13.5, color=INK)
    hw_, hs_ = (np.array(D["days"][n]["hours"]) for n in ("winter", "summer"))
    fig.text(0.70, 0.29, f"读图：门厅地面冬至平均 {hw_[lobby].mean():.1f} h 直射阳光，夏至只有 {hs_[lobby].mean():.1f} h，\n"
             f"集中在清晨与傍晚的低角度斜照。潮池夏至反而更晒（{hs_[pool].mean():.1f} h）：\n"
             f"庭院中央向天空敞开，高角度阳光直落池面。", fontsize=10.5, color=INK, va="top", linespacing=1.6)
    # ---- 结论（由数据算出）
    W_, S_ = (np.array(D["days"][n]["series"]) for n in ("winter", "summer"))
    noon = lambda ser, i: float(ser[np.argmin(abs(ser[:, 0] - 12)), i]) * 100
    fig.text(0.38, 0.045, f"正午门厅地面：冬至 {noon(W_, 4):.0f}% 有阳光，夏至 {noon(S_, 4):.0f}%（挑檐挡住）；"
             f"潮池水面：冬至 {noon(W_, 3):.0f}%，夏至 {noon(S_, 3):.0f}%。\n玻璃按透光计，庭院的树参与遮挡，人物与湖岸不计。",
             fontsize=11, color=INK, va="center", linespacing=1.6)
    fig.text(0.03, 0.955, "日照分析 · 冬至与夏至", fontsize=18, color=INK, va="center")
    fig.text(0.03, 0.915, "在模型上逐点向太阳做光线追踪，每 10 分钟一次；首层地面按 0.5 m 网格取样，共 "
             f"{len(X)} 点", fontsize=11, color=MUTED, va="center")
    fig.savefig(path, facecolor=PAPER)
    plt.close(fig)


def _exp_projector():
    C = P.EXPLODE_CAMERA
    phi, el = math.radians(C["phi"]), math.radians(C["el"])
    f = -np.array([math.cos(el) * math.cos(phi), math.cos(el) * math.sin(phi), math.sin(el)])
    up = np.array([0, 0, 1.0]) - f[2] * f
    up /= np.linalg.norm(up)
    X = np.cross(f, up)
    W, H = C["size"]
    k = H / C["ortho_scale"]
    T = np.array(C["target"])

    def px(p):
        q = np.asarray(p, float) - T
        return np.array([W / 2 + q @ X * k, H / 2 - q @ up * k])
    return px, X, up


EXPLODE_LABELS = [
    ("roofB", "B  次屋顶", "胶合木主梁约 1.5 m 一根，其间木格栅；\n楔形挑檐与金属包边"),
    ("upper", "观景厅 · 月洞墙", "二层柱立在主屋顶结构层上，\n与首层内侧柱上下对齐"),
    ("roofs", "A  主屋顶 · C  低檐 · 屋面平台", "屋面光缝 8 段；平台四周玻璃栏板"),
    ("ground", "首层", "展廊 · 门厅 · 1:12 坡道\n电梯筒 · 疏散楼梯间 · Y 形斜柱"),
    ("base", "基座 · 潮池 · 水口", "四级台阶的潮池经水口与湖面相通；\n栈桥与水中石墩"),
]


def exploded_drawing(raw, path):
    """爆炸轴测的标注：每层在东侧取锚点引出说明，层间以虚线对位。"""
    img = plt.imread(raw)
    H, W = img.shape[:2]
    px, X, up = _exp_projector()
    off = P.EXPLODE_OFFSETS
    geo = {"roofB": [(P.ROOF_B.poly, None)], "roofs": [(P.ROOF_C.poly, None), (P.ROOF_A.poly, None)],
           "upper": [(P.E3, P.DECK + 1.8)], "ground": [(P.E1, P.FFL + 2.6), (P.E2, P.FFL + 2.6)],
           "base": [(P.PLINTH_CUT.intersection(Point(0, 0).buffer(22)), P.FFL)]}

    def anchor(key):
        best = None
        for poly, z in geo[key]:
            parts = [poly] if poly.geom_type == "Polygon" else list(poly.geoms)
            for g in parts:
                pts = np.asarray(g.exterior.coords)[:, :2]
                c = np.array(g.centroid.coords[0])
                for q in pts[::3]:
                    q2 = q + (c - q) / (np.linalg.norm(c - q) + 1e-9) * 1.6
                    if not g.contains(Point(*q2)):
                        continue
                    if z is None:
                        roof = next(r for r in P.ROOFS if r.poly.equals(poly))
                        zz = float(roof.surfaces(np.array([q2[0]]), np.array([q2[1]]))[0][0])
                    else:
                        zz = z
                    p3 = np.array([q2[0], q2[1], zz + off[key]])
                    score = p3 @ X
                    if best is None or score > best[0]:
                        best = (score, px(p3))
        return best[1]
    rows = np.where(img[..., 3].max(axis=1) > 0.05)[0]          # 按线稿的实际范围取图幅
    y0, y1 = rows[0] - 40, rows[-1] + 40
    x0, x1 = 120, W + 700
    Hf = 12 * (y1 - y0) / (x1 - x0) / 0.92 + 0.1
    fig = plt.figure(figsize=(12, Hf), dpi=200, facecolor=PAPER)
    ax = fig.add_axes([0.0, 0.0, 1.0, (12 * (y1 - y0) / (x1 - x0)) / Hf])
    ax.set_xlim(x0, x1); ax.set_ylim(y1, y0); ax.set_aspect("equal"); ax.axis("off")
    ax.imshow(img, extent=(0, W, H, 0), interpolation="antialiased", zorder=2)
    # 层间对位虚线：同一平面位置在相邻两层的高度
    pairs = [("base", "ground", (14.0, -5.0, P.FFL)), ("base", "ground", (-11.6, -10.4, P.FFL)),
             ("ground", "roofs", (P.CORE_C[0], P.CORE_C[1], 6.0)), ("ground", "roofs", (-14.9, -6.9, 6.5)),
             ("roofs", "upper", (P.MOON_C[0], P.MOON_C[1], P.DECK)), ("upper", "roofB", (P.CORE_C[0], P.CORE_C[1], 9.6))]
    for l0, l1, p in pairs:
        a = px(np.array(p) + [0, 0, off[l0]]); b = px(np.array(p) + [0, 0, off[l1]])
        ax.plot([a[0], b[0]], [a[1], b[1]], color=SUN, lw=0.9, ls=(0, (5, 4)), alpha=0.85, zorder=3)
    ys = []
    for key, title, sub in EXPLODE_LABELS:
        a = anchor(key)
        y = a[1]
        if ys and y < ys[-1] + 330:
            y = ys[-1] + 330
        ys.append(y)
        tx = W - 120
        ax.plot([a[0], tx - 30], [a[1], y], color=INK, lw=0.7, zorder=4)
        ax.add_patch(Circle(a, 11, fc=SUN, ec="none", zorder=5))
        ax.text(tx, y - 18, title, fontsize=12.5, color=INK, va="bottom", zorder=5)
        ax.text(tx, y + 14, sub, fontsize=9.5, color=MUTED, va="top", linespacing=1.5, zorder=5)
    fig.text(0.03, 1 - 0.36 / Hf, "爆炸轴测 · 分层构成", fontsize=18, color=INK, va="center")
    fig.text(0.03, 1 - 0.66 / Hf, "自下而上：基座与潮池 → 首层 → 主屋顶与低檐 → 观景厅 → 次屋顶；橙色虚线为层间对位",
             fontsize=11, color=MUTED, va="center")
    fig.savefig(path, facecolor=PAPER)
    plt.close(fig)


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


# ======================================================================= 支承距离检查
def support_map(path):
    """前三格：三片屋面任一点到最近竖向支承的距离（即该点的悬挑长度或半跨）；第四格：抗侧力体系。"""
    import structure as ST
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.lines import Line2D
    cmap = LinearSegmentedColormap.from_list("tide", ["#f1f4f5", "#9fbccb", "#3f7391", "#173a4f"])
    fig = plt.figure(figsize=(21, 7.4), dpi=200, facecolor=PAPER)
    vmax = 6.0
    titles = {P.ROOF_A: "A  主屋顶", P.ROOF_B: "B  次屋顶", P.ROOF_C: "C  低檐"}
    minx, miny, maxx, maxy = P.ROOF_A.poly.union(P.ROOF_C.poly).bounds
    frame = box(minx - 12, miny - 4, maxx + 10, maxy + 4)

    def panel(k):
        ax = fig.add_axes([0.012 + k * 0.247, 0.17, 0.235, 0.6])
        ax.set_facecolor(PAPER); ax.set_aspect("equal"); ax.axis("off")
        ax.set_xlim(frame.bounds[0], frame.bounds[2]); ax.set_ylim(frame.bounds[1], frame.bounds[3])
        return ax

    for k, roof in enumerate(P.ROOFS):
        ax = panel(k)
        xy, d = ST.support_distance(roof, step=0.2)
        # 栅格化为连续色块
        rminx, rminy, rmaxx, rmaxy = roof.poly.bounds
        gx = np.arange(rminx, rmaxx, 0.2); gy = np.arange(rminy, rmaxy, 0.2)
        grid = np.full((len(gy), len(gx)), np.nan)
        ix = np.clip(np.round((xy[:, 0] - rminx) / 0.2).astype(int), 0, len(gx) - 1)
        iy = np.clip(np.round((xy[:, 1] - rminy) / 0.2).astype(int), 0, len(gy) - 1)
        grid[iy, ix] = np.minimum(d, vmax)
        ax.imshow(np.ma.masked_invalid(grid), origin="lower", extent=(rminx - 0.1, gx[-1] + 0.1, rminy - 0.1, gy[-1] + 0.1),
                  cmap=cmap, vmin=0, vmax=vmax, interpolation="nearest", zorder=1)
        line_poly(ax, roof.poly, color=INK, lw=1.0, zorder=2)
        for kind, g in ST.supports(roof):
            if g.geom_type in ("LineString", "LinearRing"):
                x, y = g.xy
                ax.plot(x, y, color=SUN, lw=3.0 if g.geom_type == "LineString" else 2.2, solid_capstyle="round",
                        zorder=3)
            else:
                c = g.centroid
                if kind == "电梯筒":
                    ax.add_patch(Circle((c.x, c.y), P.CORE_R, fc=SUN, ec="none", zorder=3))
                elif kind == "水中石墩":
                    ax.add_patch(MPoly([(c.x - 0.45, c.y - 0.45), (c.x + 0.45, c.y - 0.45), (c.x + 0.45, c.y + 0.45),
                                        (c.x - 0.45, c.y + 0.45)], closed=True, fc=SUN, ec="none", zorder=3))
                elif kind == "Y 形斜柱":
                    ax.plot(c.x, c.y, marker="^", ms=7, color=SUN, zorder=3)
                else:
                    ax.plot(c.x, c.y, "o", ms=4.2, color=SUN, zorder=3)
        i = int(np.argmax(d))
        p = xy[i]
        ax.plot(*p, marker="x", ms=8, mew=1.6, color=INK, zorder=4)
        # 标注放在不压住屋面、也不出图框的一侧（文字约 9.5 × 2 m）
        cands = []
        for sx in ((1, -1) if p[0] > 0 else (-1, 1)):
            for dy in (-1.5, 1.5, -3.5, 3.5):
                tx, ty = p[0] + sx * 2.5, p[1] + dy
                rect = box(min(tx, tx + sx * 9.5), ty - 1.0, max(tx, tx + sx * 9.5), ty + 1.0)
                cost = rect.intersection(roof.poly).area + (0 if frame.contains(rect) else 100)
                cands.append((cost, tx, ty, sx))
        _, tx, ty, sx = min(cands, key=lambda c: c[0])
        ax.annotate(f"最远 {d.max():.1f} m", xy=p, xytext=(tx, ty), fontsize=13, color=INK,
                    ha="left" if sx > 0 else "right", va="center",
                    arrowprops=dict(arrowstyle="-", color=INK, lw=0.7, shrinkA=2), zorder=5)
        pos = ax.get_position()
        fig.text(pos.x0 + 0.02, 0.85, titles[roof], fontsize=17, color=INK)
        fig.text(pos.x0 + 0.02, 0.808, f"95% 的屋面距支承 ≤ {np.percentile(d, 95):.1f} m", fontsize=12.5, color=MUTED)
    # 第四格：抗侧力体系
    lat = ST.lateral_checks()
    ax = panel(3)
    for roof in P.ROOFS:
        fill_poly(ax, roof.poly, fc="#e9e4db", ec="#b3aba0", lw=0.7, zorder=1)
    grav = np.vstack([ST.E1_COLS_INNER, ST.E1_COLS_OUTER, ST.A_EAST_COLS, ST.B_COLS])
    ax.scatter(grav[:, 0], grav[:, 1], s=7, color="#9b948a", zorder=3)
    for w in list(ST.E1_WALLS) + list(ST.E3_WALLS) + [ST.MOON_LINE]:
        x, y = w.xy
        ax.plot(x, y, color=SUN, lw=3.0, solid_capstyle="round", zorder=4)
    fill_poly(ax, P.STAIR2, fc=SUN, ec="none", zorder=5)
    ax.add_patch(Circle(P.CORE_C, P.CORE_R, fc=SUN, ec="none", zorder=5))
    cf = np.vstack([ST.E2_COLS, ST.C_EXTRA_COLS, P.C_PIERS])
    ax.scatter(cf[:, 0], cf[:, 1], s=30, marker="s", color=SUN, zorder=5)
    for (x0, y0), (dx, dy) in (((frame.bounds[0] + 1.5, 2.0), (6.5, 0)), ((-2.0, frame.bounds[1] + 0.8), (0, 5.0))):
        ax.add_patch(FancyArrowPatch((x0, y0), (x0 + dx, y0 + dy), arrowstyle="simple,head_width=6,head_length=7",
                                     mutation_scale=2.2, color="#7fa3b8", lw=0, zorder=6))
    ax.text(frame.bounds[0] + 1.5, 3.6, "风", fontsize=13, color="#5d7f93", zorder=6)
    pos = ax.get_position()
    fig.text(pos.x0 + 0.02, 0.85, "抗侧力体系", fontsize=17, color=INK)
    fig.text(pos.x0 + 0.02, 0.808, f"风控制：设计风剪力 {lat['shafts']['Vw']:.0f} kN（地震 {lat['shafts']['VE']:.0f} kN）",
             fontsize=12.5, color=MUTED)
    hs = [Line2D([], [], ls="none", marker="o", ms=5, color=SUN), Line2D([], [], color=SUN, lw=3),
          Line2D([], [], ls="none", marker="o", ms=12, color=SUN), Line2D([], [], ls="none", marker="^", ms=8, color=SUN),
          Line2D([], [], ls="none", marker="s", ms=8, color=SUN), Line2D([], [], ls="none", marker="x", ms=8, mew=1.6, color=INK)]
    fig.legend(hs, ["柱", "承重墙 / 月洞墙 / 楼梯间", "电梯筒", "Y 形斜柱顶", "水中石墩 · 刚接柱", "离支承最远点"],
               loc="lower left", bbox_to_anchor=(0.012, 0.0), ncol=3, frameon=False, fontsize=12.5, labelcolor=INK,
               handletextpad=0.4, columnspacing=1.5)
    hs2 = [Line2D([], [], ls="none", marker="o", ms=4, color="#9b948a"), Line2D([], [], ls="none", marker="s", ms=10, color=SUN)]
    fig.legend(hs2, ["铰接柱：只承竖向荷载", "筒体 · 墙 · 刚接柱"], loc="lower left", bbox_to_anchor=(0.755, 0.0), ncol=1,
               frameon=False, fontsize=12.5, labelcolor=INK, handletextpad=0.4, columnspacing=1.2)
    cax = fig.add_axes([0.52, 0.07, 0.2, 0.024])
    sm = matplotlib.cm.ScalarMappable(cmap=cmap, norm=matplotlib.colors.Normalize(0, vmax))
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cb.set_ticks([0, 2, 4, 6]); cb.set_ticklabels(["0", "2", "4", "6 m"])
    cb.outline.set_visible(False); cb.ax.tick_params(colors=MUTED, labelsize=12)
    fig.text(0.012, 0.945, "支承距离检查 · 抗侧力体系", fontsize=22, color=INK)
    fig.text(0.012, 0.9, "前三格颜色为屋面任一点到最近竖向支承的距离，即该点的悬挑长度或半跨；第四格橙色为承担风与地震水平力的构件",
             fontsize=13, color=MUTED)
    fig.savefig(path, facecolor=PAPER)
    plt.close(fig)


# ======================================================================= 平面图
def _draw_envelope(ax, segments, wall_w=0.30, z=6):
    """围合体：实墙为涂黑带，玻璃为细双线，门为缺口 + 开启弧线。"""
    for kind, pts, inward in segments:
        if kind == "solid":
            fill_poly(ax, LineString(pts).buffer(wall_w / 2, cap_style="flat", join_style="mitre"), fc=POCHE,
                      ec="none", zorder=z)
        elif kind == "glass":
            for off in (-0.05, 0.05):
                q = pts + inward[None] * off
                ax.plot(q[:, 0], q[:, 1], color=GLASS, lw=0.7, zorder=z)
        else:
            a, b = pts[0], pts[-1]
            w = np.linalg.norm(b - a)
            for p0, p1 in ((a, b), (b, a)):              # 双扇门，向室内开启
                _door_swing(ax, p0, (p1 - p0) / w, inward, w / 2, z)


def _door_swing(ax, hinge, closed_dir, open_dir, leaf, z=7):
    a0 = math.degrees(math.atan2(closed_dir[1], closed_dir[0]))
    a1 = math.degrees(math.atan2(open_dir[1], open_dir[0]))
    if (a1 - a0) % 360 <= 180:
        t1, t2 = a0, a0 + (a1 - a0) % 360
    else:
        t1, t2 = a1, a1 + (a0 - a1) % 360
    ax.add_patch(matplotlib.patches.Arc(hinge, 2 * leaf, 2 * leaf, theta1=t1, theta2=t2, color=INK, lw=0.45, zorder=z))
    e = hinge + np.asarray(open_dir) * leaf
    ax.plot([hinge[0], e[0]], [hinge[1], e[1]], color=INK, lw=0.6, zorder=z)


def _egress_paths(ax, eg, k=3):
    import program as PG
    for x, y, d in PG.farthest_points(eg, k):
        path = PG.trace_path(eg, (x, y))
        ax.plot(path[:, 0], path[:, 1], color=SUN, lw=1.2, ls=(0, (3, 2)), zorder=9)
        if len(path) > 3:
            ax.add_patch(FancyArrowPatch(path[-3], path[-1], arrowstyle="-|>", mutation_scale=9, color=SUN, lw=1.2,
                                         zorder=9))
        ax.add_patch(Circle((x, y), 0.25, fc=SUN, ec="none", zorder=9))
        ax.text(x, y + 0.6, f"{d:.0f} m", ha="center", va="bottom", fontsize=9.5, color=SUN, zorder=10)


def _stair2_plan(ax, level):
    """疏散楼梯间：墙体涂黑，梯段踏步线，箭头表示上行方向。"""
    c, e_r, e_t = P.stair2_frame()
    hd, hl, t = (P.STAIR2_R_OUT - P.STAIR2_R_IN) / 2, P.STAIR2_HALF, P.STAIR2_WALL
    W = lambda u, v: c + e_r * u + e_t * v
    walls = P.STAIR2.difference(P.STAIR2.buffer(-t, join_style="mitre"))
    door = Polygon([W(0.05, hl - t - 0.01), W(1.15, hl - t - 0.01), W(1.15, hl + 0.01), W(0.05, hl + 0.01)])
    fill_poly(ax, walls.difference(door), fc=POCHE, ec="none", zorder=7)
    n = 15
    v_start = hl - t - 1.25
    for k in range(n + 1):
        vv = v_start - k * P.STAIR2_TREAD
        for u0, u1 in ((-hd + t, -0.05), (0.05, hd - t)):
            if level == "ground" and u0 > 0 and k < 9:
                continue
            a, b = W(u0, vv), W(u1, vv)
            ax.plot([a[0], b[0]], [a[1], b[1]], color=INK, lw=0.45, zorder=7)
    a, b = W(-0.05, v_start), W(-0.05, v_start - n * P.STAIR2_TREAD)
    ax.plot([a[0], b[0]], [a[1], b[1]], color=INK, lw=0.6, zorder=7)
    if level == "ground":
        p0, p1 = W(-0.65, v_start + 0.4), W(-0.65, v_start - 2.6)
        brk = [W(0.05, v_start - 1.9), W(0.65, v_start - 2.25), W(1.25, v_start - 2.6)]
        ax.plot([q[0] for q in brk], [q[1] for q in brk], color=INK, lw=0.6, zorder=7)
    else:
        p0, p1 = W(0.65, v_start - n * P.STAIR2_TREAD + 0.4), W(0.65, v_start + 0.4)
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=7, color=INK, lw=0.6, zorder=8))
    # 防火门：首层向展廊开启，平台层向楼梯间内开启（均为疏散方向）
    _door_swing(ax, W(0.05, hl), e_r, e_t if level == "ground" else -e_t, 1.1, 8)


def _room_label(ax, xy, name, area=None, size=10, color=INK, z=9):
    ax.text(xy[0], xy[1], name, ha="center", va="center", fontsize=size, color=color, zorder=z)
    if area is not None:
        ax.text(xy[0], xy[1] - 1.0, f"{area:.0f} m²", ha="center", va="center", fontsize=size - 2, color=MUTED, zorder=z)


def _north(ax, nx, ny, s=1.0):
    ax.add_patch(MPoly([(nx, ny + 2.6 * s), (nx - 0.9 * s, ny - 0.6 * s), (nx, ny + 0.2 * s), (nx + 0.9 * s, ny - 0.6 * s)],
                       closed=True, fc=INK, ec="none", zorder=9))
    ax.text(nx, ny + 3.0 * s, "N", ha="center", va="bottom", fontsize=11, color=INK)


def plan_ground(path):
    """首层平面 +0.85：展廊、门厅、卫生间与储藏、疏散楼梯间、电梯、潮池与坡道起点、疏散路线。"""
    import structure as ST
    import program as PG
    from facade import e1_solid, e1_doors, e2_doors
    fig = plt.figure(figsize=(12, 12.4), dpi=200, facecolor=PAPER)
    ax = fig.add_axes([0.02, 0.02, 0.96, 0.9])
    x0, x1, y0, y1 = -25, 25, -27, 24.5
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_aspect("equal"); ax.axis("off")
    ax.add_patch(MPoly([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], closed=True, fc=WATER, ec="none", zorder=0))
    fill_poly(ax, P.PLINTH_CUT, fc=STONE, ec="#8d877e", lw=0.8, zorder=1)
    for r_out, r_in, z in P.BASIN_STEPS:
        ring = Point(0, 0).buffer(r_out, 128).difference(Point(0, 0).buffer(r_in, 128)).difference(P.CHANNEL)
        fill_poly(ax, ring, fc=WATER_HT if z < P.HIGH_TIDE else "#e6dfd3", ec="#a59e93", lw=0.5, zorder=2)
    fill_poly(ax, Point(0, 0).buffer(P.BASIN_STEPS[-1][1], 128).union(P.CHANNEL.intersection(P.PLINTH_ALL.buffer(0.4))),
              fc=WATER_HT, ec="#a59e93", lw=0.5, zorder=2)
    import architecture as A                          # 临水栏杆：与模型同一套路线
    rails = A.edge_railing_paths()
    for pts in rails:
        ax.plot(pts[:, 0], pts[:, 1], color=INK, lw=1.0, zorder=4, solid_capstyle="butt")
    ax.add_patch(Circle(P.TREE_POS, P.TREE_ISLAND_R, fc="#c9cdb2", ec="#7f8466", lw=0.6, zorder=3))
    ax.add_patch(Circle(P.SCULPT_POS, 0.95, fc="white", ec=INK, lw=0.6, zorder=3))
    # 室内地面
    for g in (P.E1, P.E2):
        fill_poly(ax, g, fc="#fbf8f2", ec="none", zorder=3)
    fill_poly(ax, P.SERVICE, fc="#eee8dd", ec="none", zorder=3)
    # 设备间（基座内，虚线）
    line_poly(ax, P.PLANT_ROOM, color=MUTED, lw=0.8, ls=(0, (3, 2)), zorder=4)
    # 坡道（首层以上部分为虚线）
    th = np.radians(np.linspace(P.RAMP_START_DEG, P.RAMP_START_DEG + 120, 200))
    for r in (P.RAMP_R - P.RAMP_W / 2, P.RAMP_R + P.RAMP_W / 2):
        ax.plot(r * np.cos(th), r * np.sin(th), color=INK, lw=0.7, zorder=4)
        th2 = np.radians(np.linspace(P.RAMP_START_DEG + 120, P.RAMP_START_DEG + 360, 300))
        ax.plot(r * np.cos(th2), r * np.sin(th2), color=MUTED, lw=0.6, ls=(0, (3, 2)), zorder=4)
    a0 = math.radians(P.RAMP_START_DEG + 6); a1 = math.radians(P.RAMP_START_DEG + 40)
    ax.add_patch(FancyArrowPatch((P.RAMP_R * math.cos(a0), P.RAMP_R * math.sin(a0)),
                                 (P.RAMP_R * math.cos(a1), P.RAMP_R * math.sin(a1)), connectionstyle="arc3,rad=0.12",
                                 arrowstyle="-|>", mutation_scale=8, color=INK, lw=0.7, zorder=8))
    # 展墙、接待台
    for a_, b_, r in ((40, 64, 14.2), (126, 144, 14.0)):
        ang = np.radians(np.linspace(a_, b_, 40))
        ax.plot(r * np.cos(ang), r * np.sin(ang), color=MUTED, lw=2.2, solid_capstyle="butt", zorder=5)
    ang = np.radians(np.linspace(258, 282, 40))
    ax.plot(12.6 * np.cos(ang), 12.6 * np.sin(ang), color=MUTED, lw=4.0, solid_capstyle="butt", zorder=5)
    # 围合体
    _draw_envelope(ax, PG.bay_segments(P.E1, e1_solid, e1_doors))
    _draw_envelope(ax, PG.bay_segments(P.E2, None, e2_doors))
    # 卫生间隔墙
    d, r0, r1, (dr0, dr1) = PG.service_partition()
    for ra, rb in ((r0, dr0), (dr1, r1)):
        fill_poly(ax, LineString([d * ra, d * rb]).buffer(0.08, cap_style="flat"), fc=POCHE, ec="none", zorder=6)
    a = math.radians(P.WC_SPLIT_THETA)
    dd = np.array([math.cos(a), math.sin(a)])
    seg = LineString([dd * 5, dd * 30]).intersection(P.E1.buffer(-0.05))
    qa, qb = np.array(seg.coords[0]), np.array(seg.coords[-1])
    ax.plot([qa[0], qb[0]], [qa[1], qb[1]], color=POCHE, lw=1.6, zorder=6)
    # 竖向交通
    ax.add_patch(Circle(P.CORE_C, P.CORE_R, fc=POCHE, ec="none", zorder=7))
    ax.add_patch(Circle(P.CORE_C, P.CORE_R - 0.25, fc="white", ec="none", zorder=7))
    ax.plot([P.CORE_C[0] - 0.55, P.CORE_C[0] + 0.55], [P.CORE_C[1] - 0.55, P.CORE_C[1] + 0.55], color=INK, lw=0.5, zorder=8)
    ax.plot([P.CORE_C[0] - 0.55, P.CORE_C[0] + 0.55], [P.CORE_C[1] + 0.55, P.CORE_C[1] - 0.55], color=INK, lw=0.5, zorder=8)
    _stair2_plan(ax, "ground")
    ax.add_patch(Circle(P.STAIR_C, 1.6, fc="none", ec=INK, lw=0.6, zorder=6))
    for k in range(16):
        aa = k * 2 * math.pi / 16
        ax.plot([P.STAIR_C[0] + 0.35 * math.cos(aa), P.STAIR_C[0] + 1.6 * math.cos(aa)],
                [P.STAIR_C[1] + 0.35 * math.sin(aa), P.STAIR_C[1] + 1.6 * math.sin(aa)], color=INK, lw=0.35, zorder=6)
    # 柱、斜柱、石墩
    allc = np.vstack([ST.E1_COLS_INNER, ST.E1_COLS_OUTER, ST.E2_COLS, ST.A_EAST_COLS, ST.C_EXTRA_COLS])
    ax.scatter(allc[:, 0], allc[:, 1], s=10, color=INK, zorder=8)
    ax.add_patch(Circle(P.YCOL_BASE, 1.05, fc=STONE_BEYOND, ec=INK, lw=0.6, zorder=6))
    for q in P.C_PIERS:
        ax.add_patch(Circle(q, 0.62, fc=STONE_BEYOND, ec=INK, lw=0.6, zorder=6))
    # 屋顶投影（细虚线）
    for roof in P.ROOFS:
        line_poly(ax, roof.poly, color="#9b948a", lw=0.6, ls=(0, (6, 3)), zorder=5)
    # 疏散：对外门处的出门箭头 + 从最远点沿距离场下降的步行路线
    eg = PG.ground_egress()
    for kind, pts, inward in PG.bay_segments(P.E1, e1_solid, e1_doors) + PG.bay_segments(P.E2, None, e2_doors):
        if kind == "door":
            m = pts.mean(0)
            ax.add_patch(FancyArrowPatch(m + inward * 1.2, m - inward * 1.6, arrowstyle="-|>", mutation_scale=10,
                                         color=SUN, lw=1.5, zorder=9))
    _egress_paths(ax, eg)
    ax.add_patch(FancyArrowPatch((0.0, -17.6), (0.0, -24.8), arrowstyle="-|>", mutation_scale=10, color=SUN, lw=1.5,
                                 zorder=9))
    # 房间名称
    rooms = dict((n, a) for n, a, _ in PG.rooms())
    _room_label(ax, (6.3, 14.9), "展廊", rooms["展廊"], size=12)
    _room_label(ax, (5.6, -13.7), "门厅", rooms["门厅（含接待）"], size=11)
    c, e_r, e_t = P.stair2_frame()
    note_kw = dict(fontsize=10, color=INK, zorder=10)

    def note(xy, txt, xytext, ha="left"):
        ax.annotate(txt, xy=xy, xytext=xytext, ha=ha, va="center", arrowprops=dict(arrowstyle="-", color=INK, lw=0.6,
                                                                                    shrinkA=0, shrinkB=2), **note_kw)
    note(c + e_r * 1.2, "疏散楼梯间（封闭，直达屋面平台）", (-23.5, 21.0))
    note((P.CORE_C[0], P.CORE_C[1] + P.CORE_R), "电梯（无障碍）", (-1.5, 21.5), ha="center")
    wc_c = PG.WC.representative_point().coords[0]
    st_c = PG.STORE.representative_point().coords[0]
    note(wc_c, "卫生间（含无障碍）", (19.0, 13.5))
    note(st_c, "储藏 / 设备检修口", (19.5, 9.8))
    note(P.PLANT_ROOM.representative_point().coords[0], "设备间（基座内，虚线）", (19.5, 17.0))
    note(P.STAIR_C, "螺旋阶梯 → 观景圆台 +3.00", (-24.0, 5.5))
    ax.text(0.0, -11.7, "接待", ha="center", va="center", fontsize=9, color=MUTED, zorder=10)
    a_ = math.radians(P.RAMP_START_DEG + 20)
    note((P.RAMP_R * math.cos(a_), P.RAMP_R * math.sin(a_)), "1:12 坡道起点", (1.0, 3.6))
    note(P.YCOL_BASE, "Y 形斜柱石墩", (-23.5, -15.0))
    note(P.C_PIERS[1], "水中石墩", (21.0, 2.0))
    ax.text(2.4, -22.5, "栈桥 · 主入口", fontsize=9, color=INK, zorder=10)
    rp = min((q for r in rails for q in r), key=lambda q: np.hypot(q[0] - 14.0, q[1] + 11.0))
    note(tuple(rp), "临水玻璃栏板 h = 1.05 m", (15.0, -19.0))
    ax.text(-23.5, -21.0, "→ 疏散：展廊两处庭院门、门厅南门与庭院门，经环形步道至栈桥\n   虚线为距出口最远点的步行路线", fontsize=9,
            color=SUN, zorder=10)
    ax.text(-23.5, -23.3, f"任一点到最近对外门：步行 ≤ {eg['walk_max']:.0f} m，直线 ≤ {eg['straight_max']:.0f} m", fontsize=9,
            color=INK, zorder=10)
    _north(ax, 22.5, 20.0)
    scale_bar(ax, -23.5, -26.0, units=(0, 2, 5, 10))
    fig.text(0.03, 0.955, "首层平面 +0.85", fontsize=18, color=INK)
    fig.text(0.03, 0.932, "设计高潮位 ±0.00  ·  细虚线为上方屋顶投影  ·  坡道上层各圈为虚线", fontsize=10.5, color=MUTED)
    fig.savefig(path, facecolor=PAPER)
    plt.close(fig)


def plan_deck(path):
    """屋面平台平面 +6.30：观景厅、疏散楼梯间、电梯、月洞墙、坡道到达点与两条疏散路线。"""
    import structure as ST
    import program as PG
    from facade import e3_solid, e3_doors
    fig = plt.figure(figsize=(9.0, 6.45), dpi=260, facecolor=PAPER)
    ax = fig.add_axes([0.02, 0.02, 0.96, 0.86])
    x0, x1, y0, y1 = -19, 19, -3.5, 21.5
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_aspect("equal"); ax.axis("off")
    ax.add_patch(MPoly([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], closed=True, fc=WATER, ec="none", zorder=0))
    # 下方：主屋顶 A（不上人屋面）与庭院
    fill_poly(ax, P.ROOF_A.poly, fc="#e6e0d6", ec="#9b948a", lw=0.7, zorder=1)
    fill_poly(ax, P.ROOF_C.poly, fc="#ece7de", ec="#b3aba0", lw=0.5, zorder=1)
    fill_poly(ax, P.DECK_POLY, fc=STONE, ec=INK, lw=0.9, zorder=2)
    fill_poly(ax, P.E3, fc="#fbf8f2", ec="none", zorder=3)
    # 坡道到达点与上层坡道
    land = PG.ramp_landing()
    fill_poly(ax, land, fc=STONE, ec=INK, lw=0.7, zorder=3)
    th = np.radians(np.linspace(P.RAMP_END_DEG - 150, P.RAMP_END_DEG - 7.5, 200))
    for r in (P.RAMP_R - P.RAMP_W / 2, P.RAMP_R + P.RAMP_W / 2):
        ax.plot(r * np.cos(th), r * np.sin(th), color=INK, lw=0.7, zorder=3)
    a0 = math.radians(P.RAMP_END_DEG - 60); a1 = math.radians(P.RAMP_END_DEG - 12)
    ax.add_patch(FancyArrowPatch((P.RAMP_R * math.cos(a0), P.RAMP_R * math.sin(a0)),
                                 (P.RAMP_R * math.cos(a1), P.RAMP_R * math.sin(a1)), connectionstyle="arc3,rad=0.1",
                                 arrowstyle="-|>", mutation_scale=8, color=INK, lw=0.7, zorder=8))
    # 平台栏杆（沿边缘内收 0.25 m）
    edge = P.DECK_POLY.buffer(-0.25)
    line_poly(ax, edge, color=GLASS, lw=0.6, zorder=4)
    # 观景厅、楼梯间、电梯、月洞墙
    _draw_envelope(ax, PG.bay_segments(P.E3, e3_solid, e3_doors))
    _stair2_plan(ax, "deck")
    ax.add_patch(Circle(P.CORE_C, P.CORE_R, fc=POCHE, ec="none", zorder=7))
    ax.add_patch(Circle(P.CORE_C, P.CORE_R - 0.25, fc="white", ec="none", zorder=7))
    a = P.MOON_C - P.AXIS_N * P.MOON_HALF; b = P.MOON_C + P.AXIS_N * P.MOON_HALF
    for p0, p1 in ((a, P.MOON_C - P.AXIS_N * P.MOON_HOLE_R), (P.MOON_C + P.AXIS_N * P.MOON_HOLE_R, b)):
        fill_poly(ax, LineString([p0, p1]).buffer(0.17, cap_style="flat"), fc=POCHE, ec="none", zorder=7)
    ga, gb = P.MOON_C - P.AXIS_N * P.MOON_HOLE_R, P.MOON_C + P.AXIS_N * P.MOON_HOLE_R
    ax.plot([ga[0], gb[0]], [ga[1], gb[1]], color=GLASS, lw=0.8, zorder=7)
    # 平台家具
    am = math.radians(38)
    ax.add_patch(Circle((12.4 * math.cos(am), 12.4 * math.sin(am)), 1.3, fc="none", ec=MUTED, lw=0.6, ls=(0, (2, 2)),
                        zorder=5))
    # 次屋顶 B 投影
    line_poly(ax, P.ROOF_B.poly, color=INK, lw=0.7, ls=(0, (6, 3)), zorder=6)
    ax.scatter(ST.B_COLS[:, 0], ST.B_COLS[:, 1], s=10, color=INK, zorder=8)
    # 疏散：从最远点沿距离场下降的步行路线
    eg = PG.deck_egress()
    _egress_paths(ax, eg)
    rooms = dict((n, a) for n, a, _ in PG.rooms())

    def note(xy, txt, xytext, ha="left"):
        ax.annotate(txt, xy=xy, xytext=xytext, ha=ha, va="center", fontsize=10.5, color=INK, zorder=10,
                    arrowprops=dict(arrowstyle="-", color=INK, lw=0.6, shrinkA=0, shrinkB=2))
    _room_label(ax, (-0.2, 11.9), "观景厅", rooms["观景厅"], size=12.5)
    note((11.0, 8.0), f"屋面平台（室外）{rooms['屋面平台（室外）']:.0f} m²", (10.8, 2.6))
    note(PG.stair2_door(), "出口 2：疏散楼梯间", (-18.5, 18.5))
    note(land.centroid.coords[0], "出口 1：坡道到达点", (-6.0, 1.2), ha="right")
    note(P.CORE_C, "电梯", (6.5, 19.5))
    note(tuple(P.MOON_C), "月洞墙", (-18.5, 6.0))
    note((-15.0, 0.0), "主屋顶 A（不上人）", (-18.5, -1.5))
    note((14.8, 13.6), "次屋顶 B 投影", (14.0, 19.5))
    ax.text(18.5, -2.6, f"任一点到最近出口：步行 ≤ {eg['walk_max']:.0f} m，直线 ≤ {eg['straight_max']:.0f} m；"
            f"两出口相距 {eg['exit_gap']:.0f} m", fontsize=10, color=INK, ha="right", zorder=10)
    _north(ax, 17.0, 15.0, 0.8)
    scale_bar(ax, -18.5, -2.8, units=(0, 2, 5, 10), h=0.3, label_size=8)
    fig.text(0.03, 0.94, "屋面平台平面 +6.30", fontsize=16, color=INK)
    fig.text(0.03, 0.9, "观景厅位于次屋顶 B 之下  ·  两个出口：坡道与疏散楼梯间", fontsize=10.5, color=MUTED)
    fig.savefig(path, facecolor=PAPER)
    plt.close(fig)


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(out, exist_ok=True)
    site_plan(os.path.join(out, "plan_axis.png"))
    location_map(os.path.join(out, "location.png"))
    section_cross(os.path.join(out, "section_2.png"))
    detail_eave(os.path.join(out, "detail_eave.png"))
    detail_eave(os.path.join(out, "detail_eave_tall.png"), portrait=True)
    form_diagram(os.path.join(out, "form.png"))
    env_diagram(os.path.join(out, "environment.png"))
    section(os.path.join(out, "section.png"))
    tide_curve(os.path.join(out, "tide_curve.png"))
    support_map(os.path.join(out, "support_map.png"))
    plan_ground(os.path.join(out, "plan_ground.png"))
    plan_deck(os.path.join(out, "plan_deck.png"))
    if os.path.exists(os.path.join(out, "exploded_raw.png")):    # exploded.py 的线稿底图
        exploded_drawing(os.path.join(out, "exploded_raw.png"), os.path.join(out, "exploded.png"))
    if os.path.exists(os.path.join(out, "solar.json")):          # solar.py 的光线追踪结果
        solar_diagram(os.path.join(out, "solar.json"), os.path.join(out, "solar.png"))
    raw = os.path.join(out, "elevation_south_raw.png")          # elevation.py 的线稿底图（渲染较慢，单独生成）
    if os.path.exists(raw):
        elevation_drawing(raw, os.path.join(out, "elevation_south.png"))
    print("diagrams ->", out)
