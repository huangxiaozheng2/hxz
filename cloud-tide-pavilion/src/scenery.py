"""场地与配景：庭院树、磨砂玻璃雕塑、室内陈设、家具、岛屿、远山、水面。"""
import math
import numpy as np
import shapely
from shapely.geometry import Point, Polygon

import plan as P
from plan import ccw, resample_ring
from bl import collection, mesh_object, MeshBuf, tube, lathe, box
from architecture import triangulate, ring_normals, band


# =================================================================== 树
def _leaf_poly(c, d, side, size, w_ratio=0.32):
    """尖椭圆叶片：6 顶点平面多边形（无透明贴图，渲染干净）。"""
    d = d / np.linalg.norm(d)
    s = side - np.dot(side, d) * d
    s /= np.linalg.norm(s) + 1e-9
    L, W = size, size * w_ratio
    pts = [c, c + d * L * 0.25 + s * W * 0.5, c + d * L * 0.65 + s * W * 0.42, c + d * L,
           c + d * L * 0.65 - s * W * 0.42, c + d * L * 0.25 - s * W * 0.5]
    return np.array(pts), [[0, 1, 2, 3, 4, 5]]


def grow_tree(seed, base, height=7.0, trunk_r=0.22, depth=6, spread=1.0, leaf_size=0.11,
              leaves_per_twig=26, gnarl=0.25, up_bias=0.35, lean=(0.0, 0.0), flat_top=0.0,
              leaf_ratio=0.32, pine=False, cluster=0.12, trunk_frac=None):
    rng = np.random.default_rng(seed)
    branches = []   # (pts, radii)
    leaves = []

    def branch(p0, d0, length, r0, lvl):
        nseg = 7 if lvl < 2 else 4
        pts = [p0]; rads = [r0]
        d = d0 / np.linalg.norm(d0)
        seg = length / nseg
        r_end = r0 * (0.62 if lvl < depth - 1 else 0.35)
        for k in range(nseg):
            jitter = rng.normal(0, gnarl, 3)
            d = d + jitter * (0.6 if lvl == 0 else 1.0)
            d[2] += up_bias * (1 - flat_top * (lvl / depth)) * 0.4
            if pine and lvl >= 1:
                d[2] *= 0.35
            d /= np.linalg.norm(d)
            pts.append(pts[-1] + d * seg)
            rads.append(r0 + (r_end - r0) * (k + 1) / nseg)
        pts = np.array(pts)
        branches.append((pts, np.array(rads)))
        if lvl >= depth - 1 and pine:
            # 松：水平展开的扁平针叶团
            c0 = pts[-1]
            for _ in range(leaves_per_twig * 4):
                u = rng.normal(0, 1, 3) * [0.75, 0.75, 0.16] * spread
                c = c0 + u + [0, 0, 0.12]
                dd = np.array([rng.normal(), rng.normal(), rng.normal() * 0.3])
                leaves.append(_leaf_poly(c, dd, rng.normal(0, 1, 3), leaf_size * rng.uniform(0.7, 1.3), leaf_ratio))
            return
        if lvl >= depth - 1:
            # 叶簇
            for k in range(1, len(pts)):
                for _ in range(max(1, leaves_per_twig // (len(pts) - 1))):
                    c = pts[k] + rng.normal(0, cluster * spread * (1.6 if pine else 1.0), 3)
                    if pine:
                        c[2] = pts[k][2] + rng.normal(0, 0.05)
                    dd = rng.normal(0, 1, 3) + d * 0.5 + (np.array([0, 0, 0.6]) if not pine else 0)
                    side = rng.normal(0, 1, 3)
                    leaves.append(_leaf_poly(c, dd, side, leaf_size * rng.uniform(0.75, 1.25), leaf_ratio))
            return
        nchild = 2 if rng.random() < 0.55 else 3
        if lvl == 0:
            nchild = 3
        az0 = rng.uniform(0, 2 * np.pi)
        for c in range(nchild):
            az = az0 + 2 * np.pi * c / nchild + rng.normal(0, 0.35)
            tilt = rng.uniform(0.45, 0.85) * spread if lvl > 0 else rng.uniform(0.4, 0.7) * spread
            if pine:
                tilt = rng.uniform(1.1, 1.4)
            dd = d * math.cos(tilt) + np.array([math.cos(az), math.sin(az), 0]) * math.sin(tilt)
            start_k = len(pts) - 1 if rng.random() < 0.6 else max(2, len(pts) - 3)
            child_len = length * rng.uniform(0.62, 0.8)
            branch(pts[start_k], dd, child_len, rads[start_k] * 0.68, lvl + 1)

    d0 = np.array([lean[0], lean[1], 1.0])
    trunk_len = height * (trunk_frac or (0.38 if not pine else 0.55))
    branch(np.asarray(base, float), d0, trunk_len, trunk_r, 0)
    return branches, leaves


def grow_pine(seed, base, height=6.0, trunk_r=0.18, tiers=4, pad=0.9, needle=0.24, density=900, lean=(0, 0)):
    """层叠针叶团的松树（类黑松造型）：弯曲主干 + 多层水平枝 + 扁平针叶团。"""
    rng = np.random.default_rng(seed)
    branches, leaves = [], []
    n = 9
    pts = [np.asarray(base, float)]
    d = np.array([lean[0], lean[1], 1.0]); d /= np.linalg.norm(d)
    for k in range(n):
        d = d + rng.normal(0, 0.18, 3); d[2] = abs(d[2]) + 0.6; d /= np.linalg.norm(d)
        pts.append(pts[-1] + d * height / n)
    pts = np.array(pts)
    rads = np.linspace(trunk_r, trunk_r * 0.25, len(pts))
    branches.append((pts, rads))

    def pad_at(c, rad):
        for _ in range(int(density * (rad / pad) ** 2)):
            u = rng.normal(0, 1, 3) * [rad * 0.55, rad * 0.55, rad * 0.13]
            q = c + u
            dd = np.array([rng.normal(), rng.normal(), rng.normal() * 0.25])
            leaves.append(_leaf_poly(q, dd, rng.normal(0, 1, 3), needle * rng.uniform(0.7, 1.3), 0.14))
    az0 = rng.uniform(0, 2 * np.pi)
    for t in range(tiers):
        h = 0.42 + 0.5 * t / max(1, tiers - 1)
        k = int(h * (len(pts) - 1))
        for b in range(2 if t < tiers - 1 else 1):
            az = az0 + t * 2.1 + b * np.pi + rng.normal(0, 0.3)
            L = (1.0 - 0.55 * t / tiers) * height * 0.32 * rng.uniform(0.8, 1.15)
            bp = [pts[k]]
            dd = np.array([math.cos(az), math.sin(az), 0.25])
            for j in range(4):
                dd = dd + rng.normal(0, 0.15, 3); dd[2] = 0.15 + 0.1 * rng.normal(); dd /= np.linalg.norm(dd)
                bp.append(bp[-1] + dd * L / 4)
            bp = np.array(bp)
            branches.append((bp, np.linspace(rads[k] * 0.55, 0.025, len(bp))))
            pad_at(bp[-1] + [0, 0, 0.12], pad * (1.0 - 0.3 * t / tiers) * rng.uniform(0.85, 1.15))
    pad_at(pts[-1] + [0, 0, 0.1], pad * 0.75)
    return branches, leaves


def grow_willow(seed, base, height=9.0, trunk_r=0.3, limbs=5, strands=230, leaf=0.1, lean=(0.0, 0.0)):
    """垂柳：粗短主干 → 数根上扬外展的大枝与侧枝 → 自枝上垂下的柳条（细长叶沿柳条排列，柳条本身不建模）。"""
    rng = np.random.default_rng(seed)
    branches, leaves, anchors = [], [], []
    th = height * rng.uniform(0.28, 0.36)
    p = np.asarray(base, float)
    d = _unit3([lean[0], lean[1], 1.0])
    pts = [p]
    for k in range(5):
        d = _unit3(d + rng.normal(0, 0.12, 3) + [0, 0, 0.3])
        pts.append(pts[-1] + d * th / 5)
    pts = np.array(pts)
    branches.append((pts, np.linspace(trunk_r, trunk_r * 0.72, len(pts))))
    top = pts[-1]
    az0 = rng.uniform(0, 2 * np.pi)
    for i in range(limbs):
        az = az0 + 2 * np.pi * i / limbs + rng.normal(0, 0.3)
        d = _unit3([math.cos(az) * 0.55, math.sin(az) * 0.55, 0.85])
        L = (height - th) * rng.uniform(0.75, 1.0)
        q = [top + rng.normal(0, 0.05, 3)]
        for k in range(7):
            t = (k + 1) / 7
            d = _unit3(d + [math.cos(az) * 0.12 * t, math.sin(az) * 0.12 * t, -0.16 * t] + rng.normal(0, 0.06, 3))
            q.append(q[-1] + d * L / 7)
        q = np.array(q)
        branches.append((q, np.linspace(trunk_r * 0.55, 0.03, len(q))))
        anchors += [q[k] for k in range(2, len(q))]
        for j in range(3):                                         # 侧枝
            k0 = int(rng.integers(2, 6))
            az2 = az + rng.choice([-1, 1]) * rng.uniform(0.5, 1.1)
            d2 = _unit3([math.cos(az2) * 0.7, math.sin(az2) * 0.7, 0.45])
            q2 = [q[k0]]
            L2 = L * rng.uniform(0.35, 0.55)
            for k in range(4):
                d2 = _unit3(d2 + [0, 0, -0.12] + rng.normal(0, 0.06, 3))
                q2.append(q2[-1] + d2 * L2 / 4)
            q2 = np.array(q2)
            branches.append((q2, np.linspace(trunk_r * 0.22, 0.02, len(q2))))
            anchors += list(q2[1:])
    anchors = np.array(anchors)
    z_ground = float(base[2])
    for _ in range(strands):
        a = anchors[int(rng.integers(0, len(anchors)))] + rng.normal(0, 0.25, 3)
        drop = (a[2] - z_ground) * rng.uniform(0.55, 0.88)
        n = max(4, int(drop / 0.06))
        out = _unit3([a[0] - base[0], a[1] - base[1], 0.0]) * rng.uniform(0.02, 0.07)
        sway = rng.normal(0, 0.03, 2)
        for k in range(n):
            t = k / n
            c = a + np.array([out[0] * k + sway[0] * k * t, out[1] * k + sway[1] * k * t, -drop * t])
            dd = np.array([rng.normal(0, 0.35), rng.normal(0, 0.35), -1.0])
            leaves.append(_leaf_poly(c, dd, rng.normal(0, 1, 3), leaf * rng.uniform(0.75, 1.2), 0.17))
    return branches, leaves


def _unit3(v):
    v = np.asarray(v, float)
    return v / (np.linalg.norm(v) + 1e-12)


def grow_shrub(seed, base, radius=1.0, height=1.0, density=1300, leaf=0.12):
    """灌木团：在半个椭球壳内撒叶片，叶面大致朝外。"""
    rng = np.random.default_rng(seed)
    leaves = []
    c0 = np.asarray(base, float)
    for _ in range(int(density * radius * radius)):
        u = rng.normal(0, 1, 3)
        u[2] = abs(u[2])
        u /= np.linalg.norm(u)
        r = rng.uniform(0.62, 1.0)
        q = c0 + u * [radius * r, radius * r, height * r]
        dd = u + rng.normal(0, 0.6, 3)
        leaves.append(_leaf_poly(q, dd, rng.normal(0, 1, 3), leaf * rng.uniform(0.7, 1.3), 0.4))
    return leaves


def build_willow(name, coll, M, base, **kw):
    branches, leaves = grow_willow(base=base, **kw)
    wood = MeshBuf()
    for pts, rads in branches:
        v, f, uv = tube(pts, rads, 8 if rads[0] > 0.06 else 5, cap=True)
        wood.add(v, f, uv=uv)
    w = wood.build(f"{name}_Branches", coll, [M["bark"]], smooth=True)
    lb = MeshBuf()
    for v, f in leaves:
        lb.add(v, f)
    return w, lb.build(f"{name}_Foliage", coll, [M["leaf_willow"]], smooth=False)


def build_pine(name, coll, M, base, **kw):
    branches, leaves = grow_pine(base=base, **kw)
    wood = MeshBuf()
    for pts, rads in branches:
        v, f, uv = tube(pts, rads, 8, cap=True)
        wood.add(v, f, uv=uv)
    w = wood.build(f"{name}_Branches", coll, [M["bark"]], smooth=True)
    lb = MeshBuf()
    for v, f in leaves:
        lb.add(v, f)
    return w, lb.build(f"{name}_Foliage", coll, [M["leaf_pine"]], smooth=False)


def build_tree(name, coll, M, base, leaf_mat="leaf", **kw):
    branches, leaves = grow_tree(base=base, **kw)
    wood = MeshBuf()
    for pts, rads in branches:
        sides = 10 if rads[0] > 0.05 else 6
        v, f, uv = tube(pts, rads, sides, cap=True)
        wood.add(v, f, uv=uv)
    w = wood.build(f"{name}_Branches", coll, [M["bark"]], smooth=True)
    lb = MeshBuf()
    for v, f in leaves:
        lb.add(v, f)
    return w, lb.build(f"{name}_Foliage", coll, [M[leaf_mat]], smooth=False)


# =================================================================== 庭院
def build_courtyard(M, parent):
    """潮池中的树岛与磨砂玻璃雕塑：树岛高出首层 0.45 m，四周池水随潮涨落；
    雕塑立在池底石台上，高潮时下半段没入水中。"""
    coll = collection("Courtyard", parent)
    tx, ty = P.TREE_POS
    zb, top = P.BASIN_FLOOR - 0.05, P.FFL + 0.45
    R = P.TREE_ISLAND_R
    # 树岛：石材岛壁自池底升起，压顶出挑，压顶下暗藏灯带
    v, f = lathe([(R - 0.02, zb), (R, zb), (R, top - 0.12), (R + 0.07, top - 0.10), (R + 0.07, top),
                  (R - 0.14, top), (R - 0.14, top - 0.14)], 128, (tx, ty, 0))
    mesh_object("TreeIsland_Wall", v, f, coll, [M["stone_tidal"]], smooth=True)
    v, f = lathe([(R + 0.01, top - 0.125), (R + 0.04, top - 0.125), (R + 0.04, top - 0.105), (R + 0.01, top - 0.105)],
                 128, (tx, ty, 0))
    mesh_object("TreeIsland_UnderLipLED", v, f, coll, [M["led"]], smooth=True)
    soil_z = top - 0.12
    v, f = lathe([(0, soil_z), (R - 0.13, soil_z)], 64, (tx, ty, 0))
    mesh_object("TreeIsland_Soil", v, f, coll, [M["soil"]], smooth=False)
    rng = np.random.default_rng(3)
    gb = MeshBuf()
    for _ in range(2300):
        rr = (R - 0.18) * math.sqrt(rng.random()); a = rng.uniform(0, 2 * np.pi)
        c = np.array([tx + rr * math.cos(a), ty + rr * math.sin(a), soil_z + rng.uniform(0.0, 0.12)])
        vv, ff = _leaf_poly(c, rng.normal(0, 1, 3) + [0, 0, 1.2], rng.normal(0, 1, 3), rng.uniform(0.07, 0.13), 0.45)
        gb.add(vv, ff)
    gb.build("TreeIsland_GroundCover", coll, [M["groundcover"]], smooth=False)
    build_tree("Courtyard_Tree", coll, M, (tx, ty, soil_z), seed=11, height=7.8, trunk_r=0.30,
               depth=6, spread=1.3, leaf_size=0.15, leaves_per_twig=150, gnarl=0.30, up_bias=0.32,
               lean=(-0.12, 0.05), cluster=0.2, trunk_frac=0.30)
    # 磨砂玻璃雕塑：扭转的环带，喻潮汐回旋；立于池底石台
    sx, sy = P.SCULPT_POS
    z0 = -0.40
    v, f = lathe([(0, zb), (0.95, zb), (0.95, z0 - 0.04), (0.9, z0), (0, z0)], 64, (sx, sy, 0))
    mesh_object("Sculpture_Plinth", v, f, coll, [M["stone_tidal"]], smooth=True)
    nu, nv = 220, 24
    U = np.linspace(0, 2 * np.pi, nu, endpoint=False)
    Vv = np.linspace(0, 2 * np.pi, nv, endpoint=False)
    R = 1.0
    verts = []
    for u in U:
        c = np.array([R * math.cos(u) * 0.78, 0.0, R * math.sin(u)])
        # 环带切向与扭转截面
        tw = u / 2 + 0.6
        e1 = np.array([math.cos(u) * 0.78, 0, math.sin(u)]); e1 /= np.linalg.norm(e1)
        e2 = np.array([0, 1.0, 0])
        a1 = e1 * math.cos(tw) + e2 * math.sin(tw)
        a2 = -e1 * math.sin(tw) + e2 * math.cos(tw)
        for vv in Vv:
            p = c + a1 * 0.34 * math.cos(vv) + a2 * 0.085 * math.sin(vv)
            verts.append(p)
    verts = np.array(verts)
    rot = math.radians(35)
    Rz = np.array([[math.cos(rot), -math.sin(rot), 0], [math.sin(rot), math.cos(rot), 0], [0, 0, 1]])
    verts = verts @ Rz.T + [sx, sy, z0 + R + 0.05]
    faces = []
    for i in range(nu):
        j = (i + 1) % nu
        for k in range(nv):
            k2 = (k + 1) % nv
            faces.append([i * nv + k, j * nv + k, j * nv + k2, i * nv + k2])
    mesh_object("Sculpture_FrostedGlass_Tide", verts, faces, coll, [M["frosted"]], smooth=True)
    return coll


# =================================================================== 室内陈设
def bronze_form(kind, rng):
    if kind == 0:
        prof = [(0, 0), (0.16, 0), (0.18, 0.1), (0.1, 0.35), (0.2, 0.62), (0.08, 0.95), (0.13, 1.08), (0, 1.15)]
    elif kind == 1:
        prof = [(0, 0), (0.22, 0), (0.25, 0.22), (0.12, 0.5), (0.05, 0.55), (0.16, 0.75), (0.1, 0.92), (0, 0.95)]
    else:
        prof = [(0, 0), (0.12, 0), (0.28, 0.25), (0.3, 0.4), (0.2, 0.62), (0.05, 0.7), (0, 0.7)]
    s = rng.uniform(0.9, 1.25)
    return [(r * s, z * s) for r, z in prof]


KEEP_CLEAR = [(8.4, 7.5), (12.0, 3.2), (-6.6, 6.0), (-7.4, 7.6), (-12.0, 3.0), tuple(P.CORE_C)]
# 楼梯间与服务用房周边不放展品
NO_EXHIBIT = shapely.unary_union([P.STAIR2.buffer(1.1), P.SERVICE.buffer(1.0)])


def build_interior(M, parent, e1_info, e2_info, e3_info):
    coll = collection("Interior_Exhibits", parent)
    rng = np.random.default_rng(5)
    br, pl = MeshBuf(), MeshBuf()

    def pedestal(c, z0, h=0.95, w=0.5):
        v, f = box((c[0], c[1], z0 + h / 2), (w, w, h), rng.uniform(0, 3.14))
        pl.add(v, f)
        prof = bronze_form(int(rng.integers(0, 3)), rng)
        v, f = lathe([(r, z + z0 + h) for r, z in prof], 32, (c[0], c[1], 0))
        br.add(v, f)

    for (poly, info, z0, every) in ((P.E1, e1_info, P.FFL, 2), (P.E2, e2_info, P.FFL, 2)):
        mp, nrm = info[1], info[2]
        for i in range(1, len(mp), every):
            c = mp[i] - nrm[i] * 1.9 + (mp[(i + 1) % len(mp)] - mp[i]) * 0.5
            clear = all(Point(*c).distance(Point(*k)) > 3.2 for k in KEEP_CLEAR)
            if poly.buffer(-1.2).contains(Point(*c)) and rng.random() < 0.7 and clear and \
                    Point(*c).distance(Point(*P.STAIR_C)) > 3.4 and not NO_EXHIBIT.contains(Point(*c)):
                pedestal(c, z0)
    pl.build("Exhibit_Pedestals", coll, [M["wall"]], flat_shade=True)
    br.build("Exhibit_BronzeSculptures", coll, [M["bronze"]], smooth=True)

    # 一层展廊中部弧形展墙与画作
    walls, frames = MeshBuf(), MeshBuf()
    arts = [MeshBuf() for _ in range(6)]                                      # 六幅画，各用一张图
    art_spots = []
    sc_, se_r, se_t = P.stair2_frame()
    stair_face = sc_ - se_r * (P.STAIR2_R_OUT - P.STAIR2_R_IN) / 2           # 楼梯间朝庭院的墙面即展墙
    for k, (a0, a1, r) in enumerate(((40, 64, 14.2), (None, None, None), (196 - 70, 196 - 52, 14.0))):
        if a0 is not None:
            ang = np.radians(np.linspace(a0, a1, 40))
            pts = np.stack([r * np.cos(ang), r * np.sin(ang)], 1)
            if not all(P.E1.buffer(-0.6).contains(Point(*q)) for q in pts[::6]):
                continue
            nn = np.stack([np.cos(ang), np.sin(ang)], 1)
            v, f, uv = band(pts, P.FFL, P.FFL + 3.4, -0.12, 0.12, nn, closed=False)
            walls.add(v, f)
        # 画作挂于朝庭院一侧
        for q in (0.3, 0.7):
            if a0 is None:
                c = stair_face + se_t * (q - 0.5) * 4.0
                ndir = -se_r
            else:
                am = math.radians(a0 + (a1 - a0) * q)
                c = np.array([(r - 0.13) * math.cos(am), (r - 0.13) * math.sin(am)])
                ndir = -np.array([math.cos(am), math.sin(am)])
            w, h = 1.15, 1.45
            zc = P.FFL + 1.75
            right = np.array([-ndir[1], ndir[0]])         # 观者面对画面时的右手方向：画面不镜像，法线朝向观者
            corners = [c - right * w / 2, c + right * w / 2]
            V = [[*(corners[0] + ndir * 0.035), zc - h / 2], [*(corners[1] + ndir * 0.035), zc - h / 2],
                 [*(corners[1] + ndir * 0.035), zc + h / 2], [*(corners[0] + ndir * 0.035), zc + h / 2]]
            arts[k * 2 + int(q > 0.5)].add(V, [[0, 1, 2, 3]], uv=[[0, 0], [1, 0], [1, 1], [0, 1]])
            art_spots.append((np.r_[c + ndir * 0.04, zc], ndir))
            fr = np.array(V)
            for e in range(4):
                p0, p1 = fr[e], fr[(e + 1) % 4]
                vv, ff, _ = tube([p0, p1], 0.022, 4)
                frames.add(vv, ff)
    walls.build("Exhibit_DisplayWalls", coll, [M["wall"]], smooth=True)
    # 画作重点照明：自木梁下方射向画面的窄光束射灯
    import bpy
    from scene import aim
    lc = collection("Lights_GalleryAccent", parent)
    for k, (pc, ndir) in enumerate(art_spots):
        src = pc + np.r_[ndir * 2.2, 0]
        src[2] = float(P.ROOF_A.underside(np.array([src[0]]), np.array([src[1]]))[0]) - 0.15
        l = bpy.data.lights.new(f"Light_ArtAccent_{k:02d}", "SPOT")
        l.energy = 160; l.spot_size = math.radians(26); l.spot_blend = 0.35; l.color = (1.0, 0.86, 0.68)
        l.shadow_soft_size = 0.04
        o = bpy.data.objects.new(l.name, l)
        lc.objects.link(o)
        aim(o, tuple(src), tuple(pc))
    frames.build("Exhibit_ArtFrames", coll, [M["frame"]], smooth=False)
    for i, a in enumerate(arts):
        if a.n:                                                              # 某段展墙放不下时那两幅画为空
            a.build(f"Exhibit_Artwork_{i}", coll, [M["art"][i]], smooth=False)

    # 门厅接待台（弧形）
    ang = np.radians(np.linspace(258, 282, 40))
    r = 12.6
    pts = np.stack([r * np.cos(ang), r * np.sin(ang)], 1)
    nn = np.stack([np.cos(ang), np.sin(ang)], 1)
    desk = MeshBuf()
    v, f, uv = band(pts, P.FFL, P.FFL + 1.02, -0.3, 0.3, nn, closed=False)
    desk.add(v, f)
    desk.build("Lobby_ReceptionDesk", coll, [M["wall"]], smooth=True)
    return coll


def sofa(buf_f, buf_w, c, rot, length=2.2, depth=0.9):
    """圆润的低矮沙发：底座 + 坐垫 + 靠背（圆角盒）。"""
    def soft_box(cc, size, r=0.07):
        sx, sy, sz = size
        n = 7
        verts, faces = [], []
        u = np.linspace(-1, 1, n)
        for face_axis in range(3):
            for sgn in (-1, 1):
                base = len(verts)
                for i in range(n):
                    for j in range(n):
                        p = np.zeros(3)
                        ax = [a for a in range(3) if a != face_axis]
                        p[face_axis] = sgn; p[ax[0]] = u[i]; p[ax[1]] = u[j]
                        half = np.array([sx, sy, sz]) / 2
                        core = np.clip(p * half, -(half - r), half - r)
                        dvec = p * half - core
                        nd = np.linalg.norm(dvec)
                        q = core + (dvec / nd * r if nd > 1e-9 else 0)
                        verts.append(q)
                for i in range(n - 1):
                    for j in range(n - 1):
                        a, b, c2, d = base + i * n + j, base + (i + 1) * n + j, base + (i + 1) * n + j + 1, base + i * n + j + 1
                        faces.append([a, b, c2, d] if (sgn > 0) == (face_axis != 1) else [a, d, c2, b])
        V = np.array(verts)
        cr, sr = math.cos(rot), math.sin(rot)
        V = V @ np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]]).T + cc
        return V, faces
    cr, sr = math.cos(rot), math.sin(rot)
    def at(lx, ly, lz):
        return np.array([c[0] + lx * cr - ly * sr, c[1] + lx * sr + ly * cr, c[2] + lz])
    v, f = soft_box(at(0, 0, 0.12), (length, depth, 0.16), 0.03); buf_w.add(v, f)
    v, f = soft_box(at(0, 0.04, 0.32), (length - 0.06, depth - 0.1, 0.22), 0.09); buf_f.add(v, f)
    v, f = soft_box(at(0, -depth / 2 + 0.14, 0.62), (length - 0.06, 0.26, 0.48), 0.11); buf_f.add(v, f)


def build_deck_furniture(M, parent):
    coll = collection("Terrace_Furniture", parent)
    fab, wd = MeshBuf(), MeshBuf()
    # 平台东段休憩组团，面向庭院与水面
    a = math.radians(38)
    r = 12.4
    c0 = np.array([r * math.cos(a), r * math.sin(a), P.DECK])
    face = math.atan2(-c0[1], -c0[0]) - 0.35   # 面向庭院与西南方的落日
    rot = face + math.pi / 2 + math.pi
    sofa(fab, wd, c0 + [0.0, 0.0, 0], rot)
    for k, (dx, dang) in enumerate(((1.9, -1.2), (-1.9, 1.2))):
        off = np.array([math.cos(rot) * dx, math.sin(rot) * dx, 0]) + np.array([math.cos(face), math.sin(face), 0]) * 1.2
        sofa(fab, wd, c0 + off, rot + dang, length=1.0, depth=0.9)
    tc = c0 + np.array([math.cos(face), math.sin(face), 0]) * 1.55
    v, f = lathe([(0, 0.0), (0.42, 0.0), (0.42, 0.04), (0.2, 0.06), (0.2, 0.36), (0.55, 0.38), (0.55, 0.42), (0, 0.42)], 48,
                 tuple(tc))
    wd.add(v, f)
    fab.build("Terrace_Sofa_Cushions", coll, [M["fabric"]], smooth=True)
    wd.build("Terrace_Sofa_Bases_Table", coll, [M["wall"]], smooth=True)
    return coll


# =================================================================== 人物（尺度参照）
def build_people(M, parent):
    coll = collection("People_ScaleFigures", parent)
    rng = np.random.default_rng(17)
    spots = []                                   # (x, y, z, 朝向°, 步态)
    for x, y, fdeg in ((-9.2, -36.3, 40), (-3.6, -31.6, 52), (0.3, -24.0, 90), (0.9, -15.2, 92)):
        spots.append((x, y, P.FFL, fdeg, 1.0))
    for x, y, fdeg in ((7.7, -3.2, 160), (-3.1, -8.3, 70), (5.9, 5.9, 228), (-2.1, -4.5, 65)):
        z = P.FFL if np.hypot(x, y) > P.BASIN_R else P.BASIN_STEPS[0][2]
        spots.append((x, y, z, fdeg, 0.2))
    for phi in (-60.0, 205.0, 330.0):
        a = math.radians(phi)
        spots.append((P.RAMP_R * math.cos(a), P.RAMP_R * math.sin(a), float(P.ramp_z_of_deg(phi)), phi + 90, 0.9))
    deck = [(8.9, 9.4, 220), (3.15, 8.75, 250)]
    mx, my = P.MOON_C - P.AXIS * 2.2
    deck.append((mx, my, P.AXIS_DEG))
    e3x, e3y = P.MOON_C - P.AXIS * 4.6
    deck.append((e3x, e3y, P.AXIS_DEG))
    for x, y, fdeg in deck:
        spots.append((x, y, P.DECK, fdeg, 0.1))
    for x, y, fdeg in ((5.6, 10.2, 150), (-1.6, 11.4, 105), (-3.5, -12.2, 300), (2.6, -11.6, 120)):
        spots.append((x, y, P.FFL, fdeg, 0.15))
    # 岸边入口广场、滨水步道、栈桥起点
    for x, y, fdeg, wk in ((-57.4, -67.6, 42, 1.0), (-55.6, -66.4, 46, 0.9), (-50.5, -62.8, 40, 1.0),
                           (-63.4, -72.6, 48, 0.1), (-62.6, -73.4, 52, 0.1), (-68.5, -66.4, 30, 0.1),
                           (-77.5, -70.4, 200, 1.0), (-78.6, -71.0, 205, 0.9), (-44.0, -86.0, 120, 1.0)):
        spots.append((x, y, P.SHORE_TOP, fdeg, wk))
    import people as PP
    mats = {"top": M["fabric_tops"], "bottom": M["fabric_bottoms"], "shoe": M["shoes"], "hair": M["hairs"],
            "skin": M["skins"]}
    bufs = {(k, i): MeshBuf() for k in mats for i in range(len(mats[k]))}
    strands = {i: [] for i in range(len(M["hair_strands"]))}
    proxy = MeshBuf()
    import scene as SC                                            # 近景机位 12 m 以内的人用全精度
    near = [np.array(SC.CAMERAS[c][0][:2]) for c in ("moon", "terrace", "stair", "gallery", "interior", "arrival", "hero")]
    for k, (x, y, z, fdeg, walk) in enumerate(spots):
        kind = PP.KINDS[int(rng.choice(5, p=[0.27, 0.13, 0.22, 0.28, 0.10]))]
        h = rng.uniform(1.12, 1.32) if kind == "child" else (
            rng.uniform(1.58, 1.72) if kind.startswith("woman") else rng.uniform(1.68, 1.86))
        detail = 1.0 if min(np.linalg.norm(np.array([x, y]) - c) for c in near) < 12 else 0.7
        parts = PP.make_person(h, kind, walk, seed=k, detail=detail)
        a = math.radians(fdeg - 90)
        Rz = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])
        pick = {key: int(rng.integers(0, len(mats[key]))) for key in mats}
        if kind == "woman_dress":
            pick["bottom"] = int(rng.integers(0, len(mats["bottom"])))
        pick["hair"] = int(rng.choice(4, p=[0.55, 0.27, 0.12, 0.06] if kind.startswith("man") else [0.6, 0.28, 0.12, 0.0]))
        place = lambda V: np.asarray(V) @ Rz.T + [x, y, z]
        for key, src in (("top", "top"), ("bottom", "bottom"), ("shoe", "shoe"), ("shoe", "bag"), ("skin", "skin"),
                         ("hair", "hair_cap")):
            if src not in parts or not len(parts[src][0]):
                continue
            V, F = parts[src]
            bufs[(key, pick[key])].add_np(place(V), np.asarray(F))
        strands[pick["hair"]] += [place(st) for st in parts["strands"]]
        V, F = parts["hair_proxy"]
        proxy.add_np(place(V), np.asarray(F))
    PP.cleanup()
    for (key, i), b in bufs.items():
        if b.v:
            b.build(f"People_{key.capitalize()}_{i}", coll, [mats[key][i]], smooth=True)
    for i, st in strands.items():
        PP.add_hair_curves(f"People_HairStrands_{i}", coll, M["hair_strands"][i], st)
    proxy.build("People_HairProxy_Web", coll, [M["hairs"][0]], smooth=True)
    coll.objects["People_HairProxy_Web"].hide_render = True       # 只给网页模型用（发丝曲线无法导出 glTF）


def build_landscape(M, parent):
    coll = collection("Landscape", parent)
    # 水面
    s = 20000                    # 足够远：高视点时水面边缘不会在地平线下露出一条暗带
    mesh_object("Water_Surface", [[-s, -s, 0], [s, -s, 0], [s, s, 0], [-s, s, 0]], [[0, 1, 2, 3]], coll,
                [M["water"]], smooth=False)
    rng = np.random.default_rng(21)
    # 礁石小岛 + 松
    islets = [(-36, -5, 4.5), (-28, -32, 3.2), (32, -26, 4.0), (40, 2, 3.0), (-40, 34, 5.0), (8, 50, 3.6),
              (-16, 44, 3.2), (48, -42, 4.0), (-56, -40, 6.5)]
    rock, shrub = MeshBuf(), MeshBuf()
    tcoll = collection("Landscape_Trees", coll)
    for k, (x, y, r) in enumerate(islets):
        # 礁石：多面体碎石（平直着色），中心高、边缘低
        for j in range(int(r * 5)):
            ang = rng.uniform(0, 2 * np.pi); rad = r * math.sqrt(rng.random())
            q = np.array([x + rad * math.cos(ang), y + rad * math.sin(ang)])
            sz = rng.uniform(0.35, 1.0) * (1.25 - 0.6 * rad / r)
            hz = sz * rng.uniform(0.5, 0.9) * (1.3 - rad / r)
            sides = int(rng.integers(5, 8))
            v, f = lathe([(0, -1.3), (sz, -1.3), (sz * rng.uniform(0.8, 1.0), hz * 0.45),
                          (sz * rng.uniform(0.35, 0.6), hz * 0.9), (0, hz)], sides, (q[0], q[1], 0))
            v = v + rng.normal(0, 0.12 * sz, v.shape)
            rock.add(v, f)
        # 低矮灌丛
        for j in range(int(r * r * 60)):
            ang = rng.uniform(0, 2 * np.pi); rad = r * 0.8 * math.sqrt(rng.random())
            c = np.array([x + rad * math.cos(ang), y + rad * math.sin(ang), 0.2 + 0.5 * (1 - rad / r) + rng.uniform(0, 0.25)])
            vv, ff = _leaf_poly(c, rng.normal(0, 1, 3) + [0, 0, 0.8], rng.normal(0, 1, 3), rng.uniform(0.12, 0.2), 0.4)
            shrub.add(vv, ff)
        for t in range(1 if r < 4 else 2):
            p = np.array([x, y]) + rng.normal(0, r * 0.2, 2)
            build_pine(f"Islet{k}_Pine{t}", tcoll, M, (p[0], p[1], 0.35), seed=100 + k * 7 + t,
                       height=rng.uniform(3.8, 5.6), trunk_r=0.15, tiers=int(rng.integers(3, 5)), pad=0.95,
                       lean=tuple(rng.normal(0, 0.35, 2)))
    lib = build_shore(M, coll, tcoll, rock)
    rock.build("Islets_Rocks", coll, [M["rock"]], smooth=False)
    shrub.build("Islets_Shrubs", coll, [M["groundcover"]], smooth=False)
    build_town(M, coll, tcoll, lib)
    build_far_shore(M, coll)
    build_boats(M, coll)
    build_mountains(M, coll)
    return coll


def _fill(poly, z_of, name, coll, mat, area=6.0):
    """平面多边形（可带洞、可多块）→ 按 z_of(xy) 起伏的三角网。"""
    parts = [poly] if poly.geom_type == "Polygon" else [g for g in getattr(poly, "geoms", []) if g.geom_type == "Polygon"]
    V_all, F_all, off = [], [], 0
    for g in parts:
        if g.area < 0.5:
            continue
        rings = [ccw(resample_ring(g.exterior.coords, 0.8))]
        holes = []
        for h in g.interiors:
            hp = shapely.geometry.Polygon(h)
            if hp.area < 0.3:
                continue
            rings.append(resample_ring(h.coords, 0.8))
            holes.append(hp.representative_point().coords[0])
        V2, T = triangulate(rings, area, 20, holes=holes or None)
        V_all.append(np.c_[V2, z_of(V2)]); F_all.append(T + off); off += len(V2)
    if V_all:
        mesh_object(name, np.vstack(V_all), np.vstack(F_all), coll, [mat], smooth=True)


class _Variants:
    """配景树的变体库：每种变体只建一次（建在原点），其余位置用共享网格数据的实例，几何不重复存储。"""

    def __init__(self, coll):
        self.coll, self.cache, self.count, self.log = coll, {}, {}, []

    def place(self, key, make, name, loc, rot_z, scale):
        import bpy
        self.log.append(dict(kind=key[0], x=round(float(loc[0]), 2), y=round(float(loc[1]), 2), scale=round(float(scale), 3)))
        if key not in self.cache:
            objs = [o for o in make() if o is not None]
            self.cache[key] = objs
            first = True
        else:
            first = False
        out = []
        for o in self.cache[key]:
            if first:
                n = o
                n.name = f"{name}_{o.name.split('_')[-1]}"
            else:
                n = bpy.data.objects.new(f"{name}_{o.name.split('_')[-1]}", o.data)
                self.coll.objects.link(n)
            n.location = loc
            n.rotation_euler = (0.0, 0.0, rot_z)
            n.scale = (scale, scale, scale)
            out.append(n)
        return out


def build_shore(M, coll, tcoll, rock):
    """西南岸湖岸公园（设定场地）：石砌驳岸 + 压顶、6 m 滨水步道、栈桥入口广场、园路、草坡、树丛、坐凳与地灯。"""
    rng = np.random.default_rng(31)
    edge = shapely.geometry.LineString(P.SHORE_LINE).intersection(Point(0, 0).buffer(329, 128))
    edge = max(edge.geoms, key=lambda g: g.length) if edge.geom_type != "LineString" else edge

    def d_edge(V2):
        return shapely.distance(edge, shapely.points(V2))

    def ground(V2, sink=0.0):
        d = d_edge(V2)
        t = np.clip((d - 10) / 60, 0, 1)
        return P.SHORE_TOP - sink + 1.6 * t * t * (3 - 2 * t)
    paved = shapely.unary_union([P.PROMENADE, P.LANDING, P.PARK_PATH.intersection(P.SHORE.buffer(-1.0))])
    _fill(P.PARK.difference(paved.buffer(0.05)), lambda V: ground(V, 0.04), "Shore_Lawn", coll, M["lawn"], 12.0)
    _fill(paved, ground, "Shore_Paving", coll, M["paving"], 3.0)
    # 驳岸：石砌立面（随潮位干湿）+ 压顶
    pts = np.asarray(edge.coords)
    t = np.gradient(pts, axis=0); t /= np.linalg.norm(t, axis=1, keepdims=True)
    nrm = np.stack([t[:, 1], -t[:, 0]], 1)
    if shapely.contains_xy(P.SHORE, *(pts[len(pts) // 2] + nrm[len(pts) // 2] * 2.0)):
        nrm = -nrm                                    # 法向指向水面
    wall, cope = MeshBuf(), MeshBuf()
    v, f, uv = band(pts, -1.6, P.SHORE_TOP - 0.02, -0.6, 0.0, nrm, closed=False)
    wall.add(v, f)
    v, f, uv = band(pts, P.SHORE_TOP - 0.02, P.SHORE_TOP + 0.06, -0.45, 0.08, nrm, closed=False)
    cope.add(v, f)
    wall.build("Shore_Revetment", coll, [M["stone_tidal"]], smooth=True)
    cope.build("Shore_Coping", coll, [M["stone_edge"]], smooth=True)
    # 步道地灯（每 12 m）与坐凳（每 26 m，朝水）
    lamps, benches, glow = MeshBuf(), MeshBuf(), MeshBuf()
    L = edge.length
    for k, sdist in enumerate(np.arange(4.0, L, 12.0)):
        q = np.array(edge.interpolate(sdist).coords[0])
        i = int(np.argmin(np.linalg.norm(pts - q, axis=1)))
        if np.linalg.norm(q - P.LANDING_C) < 10 or np.linalg.norm(q) > 230:
            continue
        c = q - nrm[i] * 6.6
        v, f = lathe([(0, 0), (0.09, 0), (0.07, 0.04), (0.06, 0.82), (0.075, 0.86), (0, 0.9)], 16,
                     (c[0], c[1], float(ground(c[None])[0])))
        lamps.add(v, f)
        v, f = lathe([(0, 0.70), (0.062, 0.70), (0.062, 0.80), (0, 0.80)], 16, (c[0], c[1], float(ground(c[None])[0])))
        glow.add(v, f)
        if k % 2 == 1:
            b = q - nrm[i] * 5.6 + t[i] * 3.0
            ang = math.atan2(t[i][1], t[i][0])
            v, f = box((b[0], b[1], float(ground(b[None])[0]) + 0.22), (1.9, 0.5, 0.44), ang)
            benches.add(v, f)
    # 入口广场：环绕广场的坐凳 + 栈桥口的标识石
    for ang_deg in (200, 250, 300):
        a_ = math.radians(ang_deg)
        b = P.LANDING_C + np.array([math.cos(a_), math.sin(a_)]) * 8.6
        if P.SHORE.buffer(-1.5).contains(Point(*b)):
            v, f = box((b[0], b[1], P.SHORE_TOP + 0.22), (2.4, 0.55, 0.44), a_ + math.pi / 2)
            benches.add(v, f)
    sign = MeshBuf()                                  # 标识石：立在通往城市道路的园路口
    sp = np.array([-57.6, -79.0])
    v, f = box((sp[0], sp[1], float(ground(sp[None])[0]) + 1.3), (0.62, 0.16, 2.6), math.radians(80))
    sign.add(v, f)
    # 步道临水边的玻璃栏板（与栈桥、平台同一做法），栈桥接岸处留口
    from architecture import shore_railing_paths, balustrade
    rg, rm = MeshBuf(), MeshBuf()
    for pts_r in shore_railing_paths():
        balustrade(rg, rm, pts_r, P.SHORE_TOP, height=1.05, panel=1.5)
    rg.build("Shore_Balustrade_Glass", coll, [M["glass_rail"]], smooth=False)
    rm.build("Shore_Balustrade_Rail", coll, [M["metal"]], smooth=True)
    lamps.build("Shore_BollardLights", coll, [M["metal"]], smooth=True)
    glow.build("Shore_BollardLights_Glow", coll, [M["led_soft"]], smooth=False)
    benches.build("Shore_Benches", coll, [M["stone_edge"]], flat_shade=True)
    sign.build("Shore_EntranceSign", coll, [M["wall"]], flat_shade=True)
    # 种植：步道内侧一排垂柳（临水），其后是成片的阔叶大树与灌木，最里面是松；“到达”视角两侧留取景树
    placed = []
    cam = np.array([-72.5, -73.0])
    view = math.atan2(-8.0 - cam[1], -6.0 - cam[0])

    def in_view(p):                                   # 落在“到达”机位视野正中、又离机位不远的位置不种
        v = p - cam
        dist = np.linalg.norm(v)
        ang = abs((math.atan2(v[1], v[0]) - view + math.pi) % (2 * math.pi) - math.pi)
        return dist < 70 and ang < math.radians(24)

    park = P.PARK.buffer(-6.0)                        # 只在岸线与滨湖路之间种（离路 6 m 以上）

    def ok(p, gap, inset=9.0):
        return (P.SHORE.buffer(-inset).contains(Point(*p)) and park.contains(Point(*p))
                and not paved.buffer(2.2).contains(Point(*p))
                and all(np.linalg.norm(p - q) > gap for q in placed) and not in_view(p))
    for k, (p, hgt) in enumerate((((-76.5, -79.5), 8.6), ((-71.0, -83.0), 9.4))):
        p = np.array(p)
        placed.append(p)
        build_tree(f"Shore_FrameTree{k}", tcoll, M, (p[0], p[1], float(ground(p[None], 0.04)[0])), seed=480 + k,
                   height=hgt, trunk_r=0.26, depth=6, spread=1.3, leaf_size=0.15, leaves_per_twig=110, cluster=0.2,
                   trunk_frac=0.32)
    lib = _Variants(tcoll)
    # 垂柳：沿岸线内侧 9 m，间距约 10 m，栈桥口两侧各约 90 m
    s0 = edge.project(Point(*P.LANDING_C))
    willows = 0
    for sd in np.r_[np.arange(s0 - 14, s0 - 95, -10.5), np.arange(s0 + 14, s0 + 95, 10.5)]:
        if not 0 < sd < edge.length:
            continue
        q = np.array(edge.interpolate(sd).coords[0])
        i = int(np.argmin(np.linalg.norm(pts - q, axis=1)))
        p = q - nrm[i] * rng.uniform(8.6, 9.6) + t[i] * rng.normal(0, 1.0)
        if not ok(p, 7.5, inset=7.5):
            continue
        placed.append(p); willows += 1
        vi = willows % 3
        lib.place(("willow", vi), lambda vi=vi: build_willow(f"Shore_WillowV{vi}", tcoll, M, (0.0, 0.0, 0.0),
                                                             seed=700 + vi, height=9.4, trunk_r=0.31,
                                                             limbs=5 + vi % 2, strands=380, leaf=0.12),
                  f"Shore_Willow{willows}", (p[0], p[1], float(ground(p[None], 0.04)[0])),
                  math.atan2(nrm[i][1], nrm[i][0]) + rng.normal(0, 0.4), rng.uniform(0.88, 1.12))
    # 阔叶大树（樟树一类，树冠宽）：近处叶片细，远处用较大的叶片保持树冠饱满
    tries = 0
    groves = [P.LANDING_C + off for off in ([-26, -26], [10, -30], [-40, -6], [24, -52], [-8, -52], [-52, -30])]
    while len(placed) < willows + 2 + 30 and tries < 6000:
        tries += 1
        g = groves[int(rng.integers(0, len(groves)))]
        p = g + rng.normal(0, 1, 2) * [11, 9]
        if ok(p, 7.0):
            placed.append(p)
            near = np.linalg.norm(p - cam) < 40
            vi = int(rng.integers(0, 3 if near else 5))
            lib.place(("tree", near, vi),
                      lambda vi=vi, near=near: build_tree(f"Shore_TreeV{int(near)}{vi}", tcoll, M, (0.0, 0.0, 0.0),
                                                          seed=500 + vi + 10 * near, height=11.5, trunk_r=0.33, depth=6,
                                                          spread=1.42 + 0.06 * vi, leaf_size=0.17 if near else 0.29,
                                                          leaves_per_twig=120 if near else 75,
                                                          cluster=0.27 if near else 0.42, trunk_frac=0.3),
                      f"Shore_Tree{len(placed)}", (p[0], p[1], float(ground(p[None], 0.04)[0])),
                      rng.uniform(0, 2 * np.pi), rng.uniform(0.85, 1.18))
    pines = 0
    while pines < 10 and tries < 12000:
        tries += 1
        p = P.LANDING_C + rng.normal(0, 1, 2) * [45, 30] + [-10, -32]
        if ok(p, 7.5):
            placed.append(p); pines += 1
            vi = pines % 4
            lib.place(("pine", vi), lambda vi=vi: build_pine(f"Shore_PineV{vi}", tcoll, M, (0.0, 0.0, 0.0),
                                                             seed=300 + vi, height=7.5, trunk_r=0.2, tiers=4 + vi % 2,
                                                             pad=1.15, lean=(0.12 * (vi - 1.5), 0.1)),
                      f"Shore_Pine{pines}", (p[0], p[1], float(ground(p[None], 0.04)[0])),
                      rng.uniform(0, 2 * np.pi), rng.uniform(0.85, 1.2))
    # 灌木：树下与园路两侧成团种植（6 种变体）
    def shrub(vi):
        b = MeshBuf()
        for v, f in grow_shrub(900 + vi, (0.0, 0.0, -0.1), radius=1.2, height=0.8 + 0.06 * vi, density=850):
            b.add(v, f)
        return [b.build(f"Shore_ShrubV{vi}_Foliage", tcoll, [M["groundcover"]], smooth=False)]
    nsh = 0
    for p in list(placed[2:]) * 2:
        if nsh >= 46:
            break
        q = p + rng.normal(0, 1, 2) * 2.6
        if (not P.SHORE.buffer(-4.0).contains(Point(*q)) or not park.contains(Point(*q))
                or paved.buffer(1.4).contains(Point(*q)) or in_view(q)):
            continue
        vi = nsh % 6
        lib.place(("shrub", vi), lambda vi=vi: shrub(vi), f"Shore_Shrub{nsh}", (q[0], q[1], float(ground(q[None], 0.04)[0])),
                  rng.uniform(0, 2 * np.pi), rng.uniform(0.7, 1.3))
        nsh += 1
    import os
    if os.environ.get("PLANTING_JSON"):                 # 种植位置导出给区位图（不影响模型）
        import json
        log = lib.log + [dict(kind="tree", x=round(float(p[0]), 2), y=round(float(p[1]), 2), scale=1.0)
                         for p in placed[:2]]
        with open(os.environ["PLANTING_JSON"], "w") as f:
            json.dump(log, f)
    return lib


# =================================================================== 湖岸城区、对岸与船
def _quad_uv(a, b, z0, z1, u0=0.0):
    """竖直墙面四边形（a→b 水平，z0→z1），UV 以米计：u 沿墙，v 为高度。"""
    L = float(np.linalg.norm(np.asarray(b[:2]) - np.asarray(a[:2])))
    V = [(a[0], a[1], z0), (b[0], b[1], z0), (b[0], b[1], z1), (a[0], a[1], z1)]
    UV = [(u0, 0.0), (u0 + L, 0.0), (u0 + L, z1 - z0), (u0, z1 - z0)]
    return V, UV


def build_town(M, coll, tcoll, lib):
    """沿滨湖路的城区：路面与人行道、路灯、行道树、三排建筑（江南白墙灰瓦 / 现代低层 / 住宅 / 后排高层）。"""
    z0 = P.TOWN_GROUND_Z
    rng = np.random.default_rng(77)
    # 城区地面：滨湖路以南（含 330 m 以外）
    rc = np.asarray(P.ROAD_LINE.coords)
    south = Polygon(np.vstack([rc, [(rc[-1][0] + 400, -900), (rc[0][0] - 400, -900)]])).buffer(0)
    ground = south.union(P.SHORE.difference(P.PARK)).intersection(Point(0, 0).buffer(700, 64))
    _fill(ground, lambda V: np.full(len(V), z0), "Town_Ground", coll, M["paving"], 40.0)
    road = P.ROAD_LINE.buffer(6.0, cap_style="flat").intersection(Point(0, 0).buffer(700))
    _fill(road, lambda V: np.full(len(V), z0 + 0.03), "Town_Road", coll, M["asphalt"], 20.0)
    walk = P.ROAD_LINE.buffer(9.5, cap_style="flat").difference(P.ROAD_LINE.buffer(6.0, cap_style="flat"))
    _fill(walk.intersection(Point(0, 0).buffer(700)), lambda V: np.full(len(V), z0 + 0.12), "Town_Sidewalk", coll,
          M["stone_edge"], 10.0)
    # 路灯与行道树
    lamps = MeshBuf()
    L_ = P.ROAD_LINE.length
    for k, sd in enumerate(np.arange(6.0, L_, 12.0)):
        q = np.array(P.ROAD_LINE.interpolate(sd).coords[0])
        q2 = np.array(P.ROAD_LINE.interpolate(min(sd + 1.0, L_)).coords[0])
        tt = (q2 - q) / (np.linalg.norm(q2 - q) + 1e-9)
        nn = np.array([tt[1], -tt[0]])
        if P.PARK.contains(Point(*(q + nn * 12))):
            nn = -nn                                         # nn 指向城区一侧
        if np.hypot(*q) > 600:
            continue
        if k % 2 == 0:
            for side in (-1, 1):
                c = q + nn * side * 7.8
                v, f = lathe([(0, 0), (0.09, 0), (0.06, 0.3), (0.05, 6.2), (0.0, 6.2)], 12, (c[0], c[1], z0 + 0.12))
                lamps.add(v, f)
                v, f = box((c[0] - nn[0] * side * 0.5, c[1] - nn[1] * side * 0.5, z0 + 6.2), (1.1, 0.25, 0.12),
                           math.atan2(nn[1], nn[0]))
                lamps.add(v, f)
        c = q + nn * 8.0
        vi = k % 3
        lib.place(("street", vi), lambda vi=vi: build_tree(f"Town_StreetTreeV{vi}", tcoll, M, (0.0, 0.0, 0.0),
                                                           seed=860 + vi, height=7.5, trunk_r=0.22, depth=5,
                                                           spread=1.3, leaf_size=0.2, leaves_per_twig=70, cluster=0.3,
                                                           trunk_frac=0.36),
                  f"Town_StreetTree{k}", (c[0], c[1], z0 + 0.12), rng.uniform(0, 2 * np.pi), rng.uniform(0.85, 1.1))
    lamps.build("Town_StreetLamps", coll, [M["metal_dark"]], smooth=True)
    # 建筑
    walls = {k: MeshBuf() for k in M["town"]}
    variant = {"residential": ("residential", "residential_b", "residential_c"), "modern": ("modern", "modern_b")}
    roofs, flat_roofs, slabs = MeshBuf(), MeshBuf(), MeshBuf()
    for b in P.TOWN:
        kind = b["kind"]
        fh = P.TOWN_STYLES[kind]["floor"]
        H = b["floors"] * fh
        corners = np.asarray(b["poly"].exterior.coords)[:4]
        e1, e2 = corners[1] - corners[0], corners[2] - corners[1]
        if e1[0] * e2[1] - e1[1] * e2[0] < 0:                 # 保证逆时针，墙面法线朝外
            corners = corners[::-1]
        mk = variant.get(kind, (kind,))[b["seed"] % len(variant.get(kind, (kind,)))]   # 同类建筑换几种立面色
        u0 = 0.0
        for i in range(4):
            a, c = corners[i], corners[(i + 1) % 4]
            V, UV = _quad_uv(a, c, z0, z0 + H, u0)
            walls[mk].add(V, [[0, 1, 2, 3]], uv=UV)
            u0 += float(np.linalg.norm(c - a)) + 1.3
        t_, n_ = b["t"], b["n"]
        cc = np.asarray(b["c"])
        if kind == "jiangnan":                                # 双坡小青瓦屋面，出檐 0.6 m
            e, pitch = 0.6, math.radians(27)
            hw, hd = b["w"] / 2 + e, b["d"] / 2 + e
            zr = z0 + H + (b["d"] / 2) * math.tan(pitch)
            ze = z0 + H - e * math.tan(pitch)
            for sg in (-1, 1):
                p1 = cc - t_ * hw + n_ * sg * hd; p2 = cc + t_ * hw + n_ * sg * hd
                p3 = cc + t_ * hw; p4 = cc - t_ * hw
                V = [(p1[0], p1[1], ze), (p2[0], p2[1], ze), (p3[0], p3[1], zr), (p4[0], p4[1], zr)]
                sl = math.hypot(hd, zr - ze)
                roofs.add(V, [[0, 1, 2, 3]], uv=[(0, 0), (2 * hw, 0), (2 * hw, sl), (0, sl)])
            for sg in (-1, 1):                                # 山墙三角
                g1 = cc + t_ * sg * b["w"] / 2 - n_ * b["d"] / 2; g2 = cc + t_ * sg * b["w"] / 2 + n_ * b["d"] / 2
                g3 = cc + t_ * sg * b["w"] / 2
                walls[mk].add([(g1[0], g1[1], z0 + H), (g2[0], g2[1], z0 + H), (g3[0], g3[1], zr - 0.05)],
                              [[0, 1, 2]], uv=[(0, 0), (b["d"], 0), (b["d"] / 2, 0.01)])
            v, f, uv = tube(np.array([[*(cc - t_ * hw), zr + 0.08], [*(cc + t_ * hw), zr + 0.08]]), 0.16, 8)   # 屋脊
            roofs.add(v, f, uv=uv)
        else:                                                 # 平屋顶 + 女儿墙
            ring = np.vstack([corners, corners[:1]])
            flat_roofs.add([(x, y, z0 + H) for x, y in corners], [[0, 1, 2, 3]])
            for i in range(4):
                a, c = ring[i], ring[i + 1]
                V, UV = _quad_uv(a, c, z0 + H, z0 + H + 0.7)
                flat_roofs.add(V, [[0, 1, 2, 3]], uv=UV)
            if b["floors"] >= 5:                              # 屋顶楼梯间 / 设备房
                off_ = cc + t_ * b["w"] * 0.18 + n_ * b["d"] * 0.1
                v, f = box((off_[0], off_[1], z0 + H + 1.6), (min(7.0, b["w"] * 0.3), min(5.0, b["d"] * 0.35), 3.2),
                           math.atan2(t_[1], t_[0]))
                flat_roofs.add(v, f)
        if kind == "residential":                             # 朝湖一侧的阳台板
            face = cc - n_ * (b["d"] / 2 + 0.6)
            for fl in range(1, b["floors"]):
                v, f = box((face[0], face[1], z0 + fl * fh - 0.1), (b["w"] * 0.86, 1.2, 0.16),
                           math.atan2(t_[1], t_[0]))
                slabs.add(v, f)
    for k, buf in walls.items():
        buf.build(f"Town_Buildings_{k}", coll, [M["town"][k]], smooth=False)
    roofs.build("Town_Roofs_Tile", coll, [M["roof_tile"]], smooth=False)
    flat_roofs.build("Town_Roofs_Flat", coll, [M["stone_edge"]], smooth=False)
    slabs.build("Town_Balconies", coll, [M["wall"]], flat_shade=True)


def build_far_shore(M, coll):
    """对岸：一条带树林的岸线、天际线（随距离透出天色）与山顶七层塔。"""
    rng = np.random.default_rng(55)
    a0, a1 = P.FAR_SHORE_ARC
    th = np.radians(np.linspace(a0, a1, 400))
    R = P.FAR_SHORE_R + 60 * np.sin(3 * th) + 30 * np.cos(5 * th + 1.0)
    inner = np.stack([R * np.cos(th), R * np.sin(th)], 1)
    outer = np.stack([(R + 180) * np.cos(th), (R + 180) * np.sin(th)], 1)
    land = Polygon(np.vstack([inner, outer[::-1]])).buffer(0)
    _fill(land, lambda V: np.full(len(V), 0.9), "FarShore_Land", coll, M["far_land"], 400.0)
    # 临水树林带：起伏的低矮林冠
    verts = []
    for j, (dr, hz) in enumerate(((4, 0.9), (10, 9.0), (26, 13.0), (40, 6.0))):
        hh = hz * (0.7 + 0.3 * np.sin(th * 37 + j) ** 2) if j in (1, 2) else np.full(len(th), hz)
        verts.append(np.stack([(R + dr) * np.cos(th), (R + dr) * np.sin(th), hh], 1))
    V = np.vstack(verts); na = len(th)
    F = [[j * na + i, j * na + i + 1, (j + 1) * na + i + 1, (j + 1) * na + i] for j in range(3) for i in range(na - 1)]
    mesh_object("FarShore_TreeBand", V, F, coll, [M["far_land"]], smooth=True)
    # 天际线
    bufs = MeshBuf()
    ang = a0 + 2.0
    while ang < a1 - 2.0:
        a = math.radians(ang)
        r = P.FAR_SHORE_R + 60 * math.sin(3 * a) + 30 * math.cos(5 * a + 1.0) + rng.uniform(55, 160)
        if abs(ang - P.PAGODA_AT[0]) < 7:
            ang += 2.5
            continue
        w, d = rng.uniform(16, 40), rng.uniform(12, 26)
        h = float(rng.choice([rng.uniform(9, 22), rng.uniform(22, 40), rng.uniform(40, 65)], p=[0.6, 0.32, 0.08]))
        c = np.array([r * math.cos(a), r * math.sin(a)])
        tdir = np.array([-math.sin(a), math.cos(a)]); ndir = np.array([math.cos(a), math.sin(a)])
        cs = [c - tdir * w / 2 - ndir * d / 2, c + tdir * w / 2 - ndir * d / 2, c + tdir * w / 2 + ndir * d / 2,
              c - tdir * w / 2 + ndir * d / 2]
        u0 = 0.0
        for i in range(4):
            Vq, UV = _quad_uv(cs[i], cs[(i + 1) % 4], 0.9, 0.9 + h, u0)
            bufs.add(Vq, [[0, 1, 2, 3]], uv=UV)
            u0 += float(np.linalg.norm(cs[(i + 1) % 4] - cs[i]))
        bufs.add([(x, y, 0.9 + h) for x, y in cs], [[0, 1, 2, 3]], uv=[(0, 0)] * 4)
        ang += math.degrees((w + rng.uniform(8, 40)) / r)
    bufs.build("FarShore_Skyline", coll, [M["town_far"]], smooth=False)
    # 山与七层塔
    pa, pr = math.radians(P.PAGODA_AT[0]), P.PAGODA_AT[1]
    pc = np.array([pr * math.cos(pa), pr * math.sin(pa)])
    # 山：不规则的丘陵（按方位起伏的半径与高度），塔立在主峰
    na, nr = 72, 14
    phi = np.linspace(0, 2 * np.pi, na, endpoint=False)
    wob = 1 + 0.22 * np.sin(phi * 2 + 0.7) + 0.12 * np.sin(phi * 5 + 2.0) + 0.06 * rng.normal(size=na)
    rings = []
    for j in range(nr):
        u = j / (nr - 1)
        rr = 170 * (1 - u) * wob + 1e-3
        zz = 48.5 * (1 - (1 - u) ** 2.2) * (0.92 + 0.08 * np.sin(phi * 3 + j))
        rings.append(np.stack([pc[0] + rr * np.cos(phi), pc[1] + rr * np.sin(phi), zz], 1))
    V = np.vstack(rings)
    F = [[j * na + i, j * na + (i + 1) % na, (j + 1) * na + (i + 1) % na, (j + 1) * na + i]
         for j in range(nr - 1) for i in range(na)]
    mesh_object("FarShore_PagodaHill", V, F, coll, [M["far_land"]], smooth=True)
    pg = MeshBuf()
    z = 48.5
    for tier in range(7):
        r_ = 6.2 - 0.55 * tier
        hgt = 4.6 - 0.2 * tier
        v, f = lathe([(0, 0), (r_, 0), (r_, hgt), (0, hgt)], 8, (pc[0], pc[1], z))
        pg.add(v, f)
        v, f = lathe([(0, hgt - 0.3), (r_ + 2.1, hgt - 0.9), (r_ + 2.2, hgt - 0.6), (r_ * 0.7, hgt + 0.5), (0, hgt + 0.5)], 8,
                     (pc[0], pc[1], z))
        pg.add(v, f)
        z += hgt
    v, f = lathe([(0, 0), (1.0, 0), (0.25, 6.0), (0, 7.0)], 8, (pc[0], pc[1], z))
    pg.add(v, f)
    pg.build("FarShore_Pagoda", coll, [M["pagoda"]], smooth=False)


def build_boats(M, coll):
    """湖上的船：几艘帆船（远中景）与湖岸停泊的小木船。"""
    hulls, woods, sails, masts = MeshBuf(), MeshBuf(), MeshBuf(), MeshBuf()

    def hull(buf, c, heading, L, B, D):
        xs = np.linspace(-L / 2, L / 2, 14)
        rings = []
        for x in xs:
            s_ = max(0.05, 1 - (2 * x / L) ** 2) ** 0.5 * (1.0 if x < 0 else 1 - 0.35 * (2 * x / L) ** 2)
            ring = []
            for k in range(9):
                phi = math.pi * k / 8
                ring.append((x, -math.cos(phi) * B / 2 * s_, -math.sin(phi) * D * s_ ** 0.5 + 0.35))
            rings.append(ring)
        V = np.array([p for r_ in rings for p in r_])
        F = [[i * 9 + j, (i + 1) * 9 + j, (i + 1) * 9 + j + 1, i * 9 + j + 1] for i in range(len(xs) - 1) for j in range(8)]
        F += [[i * 9 + 0, i * 9 + 8, (i + 1) * 9 + 8, (i + 1) * 9 + 0] for i in range(len(xs) - 1)]   # 甲板
        ca, sa = math.cos(heading), math.sin(heading)
        Rz = np.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1]])
        buf.add(V @ Rz.T + [c[0], c[1], 0.0], F)
        return Rz
    for k, (deg, r, heading) in enumerate(((96, 330, 30), (58, 470, -20), (142, 560, 75), (75, 760, 10))):
        a = math.radians(deg)
        c = (r * math.cos(a), r * math.sin(a))
        Rz = hull(hulls, c, math.radians(heading), 6.5, 2.2, 0.8)
        mast_h = 8.5
        v, f = lathe([(0, 0), (0.07, 0), (0.05, mast_h), (0, mast_h)], 8, (c[0], c[1], 0.6))
        masts.add(v, f)
        main = np.array([[0.15, 0, 1.2], [2.7, 0, 1.3], [0.15, 0, mast_h - 0.3]])
        jib = np.array([[-0.1, 0, 1.1], [-3.0, 0, 1.0], [-0.1, 0, mast_h - 1.2]])
        for tri in (main, jib):
            tri = tri.copy(); tri[:, 1] += np.array([0.0, 0.35, 0.1]) * (1 if k % 2 else -1)
            sails.add(tri @ Rz.T + [c[0], c[1], 0.6], [[0, 1, 2]])
    for k, (x, y, hd) in enumerate(((-47.0, -60.5, 35), (-44.5, -63.0, 42), (-80.5, -63.0, 160))):
        hull(woods, (x, y), math.radians(hd), 4.2, 1.35, 0.5)
    hulls.build("Boats_Hulls", coll, [M["boat_hull"]], smooth=True)
    woods.build("Boats_Rowing", coll, [M["boat_wood"]], smooth=True)
    masts.build("Boats_Masts", coll, [M["metal"]], smooth=True)
    sails.build("Boats_Sails", coll, [M["sail"]], smooth=False)


def build_mountains(M, coll):
    rng = np.random.default_rng(9)
    for k, (a0, a1, dist, hmax, mat) in enumerate((
            # 冬至日落方位（平面角约 207°）两侧让开山体，夕阳落入开阔水面
            (140, 193, 1000, 75, "mountain"), (224, 330, 1200, 60, "mountain"), (20, 160, 1500, 110, "mountain_far"),
            (232, 300, 1900, 150, "mountain_far"), (330, 400, 1700, 95, "mountain_far"))):
        na, nr = 260, 12
        a = np.radians(np.linspace(a0, a1, na))
        prof = np.zeros(na)
        for octv in range(5):
            f = 2 ** octv * 1.1
            prof += rng.uniform(0.5, 1.0) / (2.2 ** octv) * np.sin(a * f * 3 + rng.uniform(0, 6)) ** 2
        prof = prof / prof.max()
        ends = np.minimum(1, np.minimum(np.arange(na), np.arange(na)[::-1]) / 30)
        h = hmax * (0.25 + 0.75 * prof) * ends
        verts = []
        for j in range(nr):
            u = j / (nr - 1)
            r = dist + u * 260
            zz = h * math.sin(math.pi * u) ** 0.7
            zz = zz - 5 * (u == 0)
            verts.append(np.stack([r * np.cos(a), r * np.sin(a), zz], 1))
        V = np.vstack(verts)
        F = [[j * na + i, j * na + i + 1, (j + 1) * na + i + 1, (j + 1) * na + i] for j in range(nr - 1) for i in range(na - 1)]
        mesh_object(f"Mountains_{k}", V, F, coll, [M[mat]], smooth=True)
