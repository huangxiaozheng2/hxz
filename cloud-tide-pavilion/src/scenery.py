"""场地与配景：庭院树、磨砂玻璃雕塑、室内陈设、家具、岛屿、远山、水面。"""
import math
import numpy as np
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


def build_pine(name, coll, M, base, **kw):
    branches, leaves = grow_pine(base=base, **kw)
    wood = MeshBuf()
    for pts, rads in branches:
        v, f, uv = tube(pts, rads, 8, cap=True)
        wood.add(v, f, uv=uv)
    wood.build(f"{name}_Branches", coll, [M["bark"]], smooth=True)
    lb = MeshBuf()
    for v, f in leaves:
        lb.add(v, f)
    lb.build(f"{name}_Foliage", coll, [M["leaf_pine"]], smooth=False)


def build_tree(name, coll, M, base, leaf_mat="leaf", **kw):
    branches, leaves = grow_tree(base=base, **kw)
    wood = MeshBuf()
    for pts, rads in branches:
        sides = 10 if rads[0] > 0.05 else 6
        v, f, uv = tube(pts, rads, sides, cap=True)
        wood.add(v, f, uv=uv)
    wood.build(f"{name}_Branches", coll, [M["bark"]], smooth=True)
    lb = MeshBuf()
    for v, f in leaves:
        lb.add(v, f)
    lb.build(f"{name}_Foliage", coll, [M[leaf_mat]], smooth=False)


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


KEEP_CLEAR = [(10.5, 4.25), (12.0, 3.2), (-6.6, 6.0), (-12.0, 3.0), tuple(P.CORE_C)]


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
                    Point(*c).distance(Point(*P.STAIR_C)) > 3.4:
                pedestal(c, z0)
    pl.build("Exhibit_Pedestals", coll, [M["wall"]], smooth=False)
    br.build("Exhibit_BronzeSculptures", coll, [M["bronze"]], smooth=True)

    # 一层展廊中部弧形展墙与画作
    walls, frames = MeshBuf(), MeshBuf()
    arts = [MeshBuf() for _ in range(4)]
    art_spots = []
    for k, (a0, a1, r) in enumerate(((40, 64, 14.2), (88, 118, 15.0), (196 - 70, 196 - 52, 14.0))):
        ang = np.radians(np.linspace(a0, a1, 40))
        pts = np.stack([r * np.cos(ang), r * np.sin(ang)], 1)
        if not all(P.E1.buffer(-0.6).contains(Point(*q)) for q in pts[::6]):
            continue
        nn = np.stack([np.cos(ang), np.sin(ang)], 1)
        v, f, uv = band(pts, P.FFL, P.FFL + 3.4, -0.12, 0.12, nn, closed=False)
        walls.add(v, f)
        # 画作挂于朝庭院一侧
        for q in (0.3, 0.7):
            am = math.radians(a0 + (a1 - a0) * q)
            c = np.array([(r - 0.13) * math.cos(am), (r - 0.13) * math.sin(am)])
            tdir = np.array([-math.sin(am), math.cos(am)])
            ndir = -np.array([math.cos(am), math.sin(am)])
            w, h = 1.15, 1.45
            zc = P.FFL + 1.75
            corners = [c - tdir * w / 2, c + tdir * w / 2]
            V = [[*(corners[0] + ndir * 0.035), zc - h / 2], [*(corners[1] + ndir * 0.035), zc - h / 2],
                 [*(corners[1] + ndir * 0.035), zc + h / 2], [*(corners[0] + ndir * 0.035), zc + h / 2]]
            arts[(k * 2 + int(q > 0.5)) % 4].add(V, [[0, 1, 2, 3]], uv=[[0, 0], [1, 0], [1, 1], [0, 1]])
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
def person_mesh(h=1.72, walk=0.0, seed=0):
    """简化人物：溜肩躯干、贴身手臂、收脚踝的腿、带颈部的头；局部坐标面朝 +y。"""
    rng = np.random.default_rng(seed)
    cloth, skin = MeshBuf(), MeshBuf()
    stride = 0.16 * h * walk
    hip = 0.53 * h
    for sx, sg in ((-0.052 * h, 1), (0.052 * h, -1)):
        knee = [sx * 0.95, sg * stride * 0.45, 0.285 * h]
        ankle = [sx * 0.9, sg * stride, 0.045 * h]
        pts = np.array([[sx, 0, hip], [sx, sg * stride * 0.2, 0.42 * h], knee, ankle])
        v, f, _ = tube(pts, [0.056 * h, 0.05 * h, 0.038 * h, 0.028 * h], 14)
        cloth.add(v, f)
        foot = np.array([[ankle[0], ankle[1] - 0.02 * h, 0.03 * h], [ankle[0], ankle[1] + 0.09 * h, 0.022 * h]])
        v, f, _ = tube(foot, [0.03 * h, 0.022 * h], 10)
        cloth.add(v, f)
    prof = [(0.0, hip - 0.03 * h), (0.098 * h, hip - 0.02 * h), (0.104 * h, hip + 0.04 * h), (0.09 * h, 0.62 * h),
            (0.102 * h, 0.70 * h), (0.1 * h, 0.745 * h), (0.086 * h, 0.78 * h), (0.058 * h, 0.81 * h),
            (0.034 * h, 0.828 * h), (0.025 * h, 0.84 * h), (0.0, 0.842 * h)]
    v, f = lathe(prof, 28)
    v[:, 1] *= 0.6
    cloth.add(v, f)
    for sx, sg in ((-1, -1), (1, 1)):
        sh = [sx * 0.082 * h, 0, 0.765 * h]
        el = [sx * 0.112 * h, sg * stride * 0.35 + 0.01 * h, 0.615 * h]
        hand = [sx * 0.104 * h, sg * stride * 0.65 + 0.025 * h, 0.47 * h]
        v, f, _ = tube(np.array([sh, el, hand]), [0.029 * h, 0.025 * h, 0.02 * h], 12)
        cloth.add(v, f)
    a = np.linspace(0, np.pi, 14)
    v, f = lathe([(0.058 * h * math.sin(t), 0.928 * h - 0.07 * h * math.cos(t)) for t in a], 24)
    v[:, 1] *= 0.92
    skin.add(v, f)
    v, f = lathe([(0.024 * h, 0.835 * h), (0.024 * h, 0.865 * h)], 16)
    skin.add(v, f)
    return cloth, skin


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
    for x, y, fdeg in ((8.4, 7.2, 40), (-3.5, 12.8, 105), (-3.5, -12.2, 300), (2.6, -11.6, 120)):
        spots.append((x, y, P.FFL, fdeg, 0.15))
    bufs = [MeshBuf() for _ in M["people"]]
    skinb = MeshBuf()
    for k, (x, y, z, fdeg, walk) in enumerate(spots):
        h = rng.uniform(1.6, 1.84)
        cloth, skin = person_mesh(h, walk, seed=k)
        a = math.radians(fdeg - 90)
        Rz = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])
        ci = int(rng.integers(0, len(bufs)))
        for src, dst in ((cloth, bufs[ci]), (skin, skinb)):
            V = np.vstack(src.v) @ Rz.T + [x, y, z]
            dst.add(V, src.f)
    for i, b in enumerate(bufs):
        b.build(f"People_Clothing_{i}", coll, [M["people"][i]], smooth=True)
    skinb.build("People_Skin", coll, [M["skin"]], smooth=True)
    return coll


# =================================================================== 景观
def build_landscape(M, parent):
    coll = collection("Landscape", parent)
    # 水面
    s = 2600
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
    # 西南岸线（栈桥起点）
    shore = Polygon([(-140, -150), (-20, -150), (-38, -90), (-52, -70), (-64, -62), (-90, -55), (-140, -50)]).buffer(6)
    sr = ccw(resample_ring(shore.exterior.coords, 1.0))
    V2, T = triangulate([sr], 20.0, 20)
    z = 0.5 + 0.4 * np.clip((-V2[:, 1] - 60) / 40, 0, 1) * 3
    mesh_object("Shore_Ground", np.c_[V2, z], T, coll, [M["grass"]], smooth=True)
    nn, _ = ring_normals(sr)
    v, f, uv = band(sr, -1.4, 0.55, -0.4, 0.3, nn)
    rock.add(v, f)
    for t in range(9):
        p = np.array([-60 - rng.uniform(0, 50), -66 - rng.uniform(0, 40)])
        if shore.buffer(-3).contains(Point(*p)):
            build_pine(f"Shore_Pine{t}", tcoll, M, (p[0], p[1], 0.6), seed=300 + t,
                       height=rng.uniform(5.5, 8.0), trunk_r=0.2, tiers=int(rng.integers(4, 6)), pad=1.15,
                       lean=tuple(rng.normal(0, 0.25, 2)))
    rock.build("Islets_Rocks", coll, [M["rock"]], smooth=False)
    shrub.build("Islets_Shrubs", coll, [M["groundcover"]], smooth=False)
    build_mountains(M, coll)
    return coll


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
