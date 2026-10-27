"""立面规则（纯几何）：幕墙分格、实墙段、框景窗、门的位置。build.py 与 structure.py 共用。"""
import numpy as np
import shapely

from plan import ccw, resample_ring, STAIR2, WC_THETA


def ring_normals(ring):
    """逆时针环的外法向与切向。"""
    t = np.roll(ring, -1, 0) - np.roll(ring, 1, 0)
    t /= np.linalg.norm(t, axis=1, keepdims=True)
    return np.stack([t[:, 1], -t[:, 0]], 1), t


def envelope_bays(poly, bay=1.5, step=0.04):
    """沿围合体轮廓等宽分格。返回 dense（密采样环）、s（弧长）、L、sm（竖梃弧长位置）、mp（竖梃点）、nrm（外法向）。"""
    dense = ccw(resample_ring(poly.exterior.coords, step))
    seg = np.linalg.norm(np.roll(dense, -1, 0) - dense, axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    L = s[-1]
    n = max(6, int(round(L / bay)))
    sm = np.linspace(0, L, n, endpoint=False)
    closed = np.vstack([dense, dense[:1]])
    mp = np.stack([np.interp(sm, s, closed[:, 0]), np.interp(sm, s, closed[:, 1])], 1)
    nrm, _ = ring_normals(mp)
    return dense, s, L, sm, mp, nrm


def angle_deg(p):
    return np.degrees(np.arctan2(p[:, 1], p[:, 0])) % 360


def outer_side(mp, nrm):
    radial = mp / np.linalg.norm(mp, axis=1, keepdims=True)
    return (nrm * radial).sum(1)


def ranges_from_mask(mask):
    idx = np.where(mask)[0]
    if not len(idx):
        return []
    runs = np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)
    return [(int(r[0]), int(r[-1]) + 1) for r in runs]


def end_caps(which):
    """月牙端部（径向短边）做成实墙：玻璃与墙体在端部清晰交接。"""
    def f(mp, nrm):
        th = angle_deg(mp)
        radial = np.abs(outer_side(mp, nrm)) < 0.55
        out = []
        for lo, hi in which:
            out += ranges_from_mask(radial & (th > lo) & (th < hi))
        return out
    return f


_e3_ends = end_caps([(40, 66)])          # 二层观景厅东端弧墙：次屋顶的支承之一


def e3_solid(mp, nrm):
    """观景厅：东端弧墙 + 与疏散楼梯间相贴的一段（楼梯间墙体即观景厅北墙）。"""
    nxt = np.roll(mp, -1, 0)
    mid = (mp + nxt) / 2
    stair = shapely.distance(STAIR2.exterior, shapely.points(mid)) < 0.3
    return sorted(_e3_ends(mp, nrm) + ranges_from_mask(stair))


def e1_solid(mp, nrm):
    """一层展廊外侧整体为实墙（美术馆避免直射光），月牙两端为端墙；庭院一侧保持玻璃。"""
    th = angle_deg(mp)
    o = outer_side(mp, nrm)
    # 西端（螺旋楼梯所在的高厅）保持玻璃，让楼梯从外部可见
    # 东端尖角为卫生间与储藏：朝庭院一侧也改为实墙
    return ranges_from_mask(((o > 0.35) & (th < 148)) | ((np.abs(o) <= 0.35) & (th < 30)) | (th < WC_THETA))


def e1_windows(mp, nrm):
    """实墙上的四处框景窗，各宽两跨，朝向水面的不同方向。"""
    th = angle_deg(mp)
    o = outer_side(mp, nrm)
    out = []
    for target in (32, 66, 100, 132):
        cand = np.where(o > 0.6)[0]
        i = int(cand[np.argmin(np.abs(th[cand] - target))])
        out.append((i, i + 2))
    return out


def e1_doors(mp, nrm):
    th = angle_deg(mp)
    inner = outer_side(mp, nrm) < -0.8
    out = []
    for target in (40, 150):
        cand = np.where(inner)[0]
        if len(cand):
            out.append(int(cand[np.argmin(np.abs(th[cand] - target))]))
    return out


def e2_doors(mp, nrm):
    # 南侧凹入门廊的入口门 + 朝庭院的门
    out = []
    south = np.where((nrm[:, 1] < -0.85) & (np.abs(mp[:, 0]) < 2.2))[0]
    if len(south):
        out += [int(i) for i in south[np.argsort(np.abs(mp[south, 0]))[:1]]]
    inner = np.where(outer_side(mp, nrm) < -0.85)[0]
    th = angle_deg(mp)
    if len(inner):
        out.append(int(inner[np.argmin(np.abs(th[inner] - 270))]))
    return out


def e3_doors(mp, nrm):
    """观景厅三处门：南侧对坡道到达点，北侧通东平台，西端通月洞平台与疏散楼梯。"""
    th = angle_deg(mp)
    o = outer_side(mp, nrm)
    out = []
    for side, target in ((o < -0.8, 88.0), (o > 0.8, 64.0), (np.abs(o) < 0.6, 124.0)):
        cand = np.where(side)[0]
        if len(cand):
            out.append(int(cand[np.argmin(np.abs(th[cand] - target))]))
    return out
