"""建筑主体构件：屋顶与天窗、胶合木梁、幕墙与框景窗、柱、基座与潮池、平台、坡道、电梯筒、翼尖支承、螺旋楼梯。"""
import math
import numpy as np
import shapely
import triangle
from shapely.geometry import Polygon, Point

import plan as P
from plan import ccw, resample_ring
from bl import collection, mesh_object, MeshBuf, sweep, tube, lathe, oriented_box, frames_along

UP = np.array([0, 0, 1.0])


# =================================================================== 工具
def triangulate(rings, max_area=0.12, quality=28, holes=None):
    """rings: 闭合环列表（第一个为外轮廓）。返回 (verts2d, tris)。前 len(rings[0]) 个顶点即外轮廓。
    holes: 洞内一点的列表（洞的边界须已包含在 rings 中）。"""
    V, S, off = [], [], 0
    for r in rings:
        n = len(r)
        V.append(r)
        S.append(np.stack([np.arange(n) + off, (np.arange(n) + 1) % n + off], 1))
        off += n
    V = np.vstack(V); S = np.vstack(S)
    data = {"vertices": V, "segments": S}
    if holes:
        data["holes"] = np.asarray(holes, float)
    out = triangle.triangulate(data, f"pYq{quality}a{max_area}")
    return out["vertices"], out["triangles"]


def offset_rings(poly, ds, step):
    rings = []
    for d in ds:
        g = poly.buffer(-d, 64)
        parts = [g] if g.geom_type == "Polygon" else list(getattr(g, "geoms", []))
        for p in parts:
            if p.area > 1.5:
                rings.append(ccw(resample_ring(p.exterior.coords, step)))
    return rings


def ring_normals(ring):
    """逆时针环的外法向。"""
    t = np.roll(ring, -1, 0) - np.roll(ring, 1, 0)
    t /= np.linalg.norm(t, axis=1, keepdims=True)
    return np.stack([t[:, 1], -t[:, 0]], 1), t


def band(pts2, zb, zt, o0, o1, nrm2, closed=True, cap=True):
    """沿平面折线的竖向带：每点 (法向偏移 o0..o1, 高度 zb..zt)，可逐点变化。"""
    N = len(pts2)
    zb = np.broadcast_to(zb, (N,)); zt = np.broadcast_to(zt, (N,))
    o0 = np.broadcast_to(o0, (N,)); o1 = np.broadcast_to(o1, (N,))
    V = []
    for i in range(N):
        p = pts2[i]; n = nrm2[i]
        for (o, z) in ((o0[i], zb[i]), (o1[i], zb[i]), (o1[i], zt[i]), (o0[i], zt[i])):
            V.append([p[0] + n[0] * o, p[1] + n[1] * o, z])
    F = []
    segs = N if closed else N - 1
    for i in range(segs):
        j = (i + 1) % N
        for k in range(4):
            k2 = (k + 1) % 4
            F.append([i * 4 + k, j * 4 + k, j * 4 + k2, i * 4 + k2])
    if cap and not closed:
        F.append([3, 2, 1, 0]); F.append([(N - 1) * 4 + k for k in range(4)])
    s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(pts2, axis=0), axis=1))])
    uv = np.stack([np.repeat(s, 4), np.tile([0, .25, .5, .75], N)], 1)
    return np.array(V), F, uv


# =================================================================== 屋顶
def build_roof(roof, M, parent, skylights=()):
    coll = collection(roof.name, parent)
    poly = roof.poly
    outer = ccw(resample_ring(poly.exterior.coords, 0.16))
    rings = [outer] + offset_rings(poly, [P.REVEAL, 0.45, 0.75], 0.2) + offset_rings(poly, [1.6, 3.2], 0.45)
    sky_rings = [ccw(resample_ring(h.exterior.coords, 0.12)) for h in skylights]
    holes = [h.representative_point().coords[0] for h in skylights]
    V2, T = triangulate(rings + sky_rings, max_area=0.10, holes=holes)
    nb = len(outer)
    d = roof.dist_edge(V2[:, 0], V2[:, 1])
    d[:nb] = 0.0
    top, bot = roof.surfaces(V2[:, 0], V2[:, 1], d)
    N = len(V2)

    # --- 屋面（顶面 + 檐下底面）
    verts = np.vstack([np.c_[V2, top], np.c_[V2, bot]])
    dc = d[T].mean(1)
    faces_top = T
    faces_bot = T[:, ::-1] + N
    mat_top = np.where((dc > 0.45) & (dc < 0.75), 1, 0)          # 排水槽为金属格栅
    mat_bot = np.where(dc < P.REVEAL, 3, 2)                       # 阴影缝 / 底板
    # --- 楔形边缘的圆润金属包边
    nrm, _ = ring_normals(outer)
    K = 12
    phi = np.linspace(0, np.pi, K)
    zt, zb = top[:nb], bot[:nb]
    rho = (zt - zb) / 2; zc = (zt + zb) / 2
    rim = np.zeros((nb, K, 3))
    rim[:, :, 0] = outer[:, None, 0] + nrm[:, None, 0] * rho[:, None] * np.sin(phi)[None] * 0.85
    rim[:, :, 1] = outer[:, None, 1] + nrm[:, None, 1] * rho[:, None] * np.sin(phi)[None] * 0.85
    rim[:, :, 2] = zc[:, None] + rho[:, None] * np.cos(phi)[None]
    rim_off = 2 * N
    rim_faces = []
    for i in range(nb):
        j = (i + 1) % nb
        for k in range(K - 1):
            rim_faces.append([rim_off + j * K + k, rim_off + i * K + k, rim_off + i * K + k + 1, rim_off + j * K + k + 1])
    all_v = np.vstack([verts, rim.reshape(-1, 3)])
    all_f = faces_top.tolist() + faces_bot.tolist() + rim_faces
    all_m = np.concatenate([mat_top, mat_bot, np.full(len(rim_faces), 4)])
    mats = [M["roof"], M["metal_dark"], M["soffit"], M["gap"], M["metal"]]
    # 分为 屋面壳体 与 金属包边 两个对象，便于替换材质
    nt = len(faces_top) + len(faces_bot)
    mesh_object(f"{roof.name}_Shell", all_v[:2 * N], all_f[:nt], coll, mats, all_m[:nt])
    trim_v = rim.reshape(-1, 3)
    trim_f = [[i - rim_off for i in f] for f in rim_faces]
    mesh_object(f"{roof.name}_EdgeTrim", trim_v, trim_f, coll, [M["metal"]])

    if skylights:
        build_skylights(roof, skylights, M, coll)
    build_ribs(roof, M, coll, skylights=skylights)
    return coll


def build_skylights(roof, holes, M, coll):
    """屋面光缝：洞口四周金属内衬贯通屋面厚度，顶部嵌超白玻璃，玻璃随屋面曲率。"""
    lining, glass = MeshBuf(), MeshBuf()
    for h in holes:
        ring = ccw(resample_ring(h.exterior.coords, 0.1))
        d = roof.dist_edge(ring[:, 0], ring[:, 1])
        top, bot = roof.surfaces(ring[:, 0], ring[:, 1], d)
        under = bot - roof.rib_depth(d)
        nrm, _ = ring_normals(ring)
        v, f, uv = band(ring, under - 0.01, top + 0.02, 0.0, 0.025, nrm)
        lining.add(v, f)
        V2, T = triangulate([ring], 0.05, 25)
        gt, _ = roof.surfaces(V2[:, 0], V2[:, 1])
        glass.add(np.c_[V2, gt - 0.05], T.tolist())
    lining.build(f"{roof.name}_Skylight_Lining", coll, [M["metal"]], smooth=False)
    glass.build(f"{roof.name}_Skylight_Glass", coll, [M["glass"]], smooth=False)


def build_ribs(roof, M, coll, spacing=0.24, r_ref=14.0, skylights=()):
    """檐下胶合木径向梁：以庭院中心为极点排布，沿屋面底部曲率连续；截面高度由檐口 100 mm
    向支座方向加深到 340 mm（悬挑弯矩越大截面越高），端部统一收于阴影缝内线。"""
    inset = roof.poly.buffer(-(P.REVEAL + 0.04), 64)
    for h in skylights:
        inset = inset.difference(h.buffer(0.06, 16))
    shapely.prepare(inset)
    dth = spacing / r_ref
    ths = np.arange(math.radians(roof.ts) + dth / 2, math.radians(roof.te), dth)
    rs = np.arange(3.0, 34.0, 0.08)
    TH, RR = np.meshgrid(ths, rs, indexing="ij")
    inside = shapely.contains_xy(inset, RR * np.cos(TH), RR * np.sin(TH))
    runs = []
    for i, th in enumerate(ths):
        row = inside[i]
        if not row.any():
            continue
        dif = np.diff(np.concatenate([[0], row.astype(int), [0]]))
        starts = np.where(dif == 1)[0]; ends = np.where(dif == -1)[0] - 1
        for a, b in zip(starts, ends):
            runs.append((th, rs[a], rs[b], a > 0, True))
    runs = np.array([(r[0], r[1], r[2]) for r in runs])
    # 端点二分细化，确保收口线整齐
    def refine(th, r_in, r_out):
        lo, hi = r_out.copy(), r_in.copy()   # lo 在外（不含），hi 在内（含）
        for _ in range(12):
            mid = (lo + hi) / 2
            ok = shapely.contains_xy(inset, mid * np.cos(th), mid * np.sin(th))
            hi = np.where(ok, mid, hi); lo = np.where(ok, lo, mid)
        return hi
    th = runs[:, 0]
    r0 = refine(th, runs[:, 1], runs[:, 1] - 0.08)
    r1 = refine(th, runs[:, 2], runs[:, 2] + 0.08)
    keep = (r1 - r0) > 0.35
    th, r0, r1 = th[keep], r0[keep], r1[keep]

    buf_v, buf_f, buf_uv = [], [], []
    off = 0
    allp = []
    meta = []
    for k in range(len(th)):
        m = max(2, int(math.ceil((r1[k] - r0[k]) / 0.3))) + 1
        r = np.linspace(r0[k], r1[k], m)
        allp.append(np.stack([r * math.cos(th[k]), r * math.sin(th[k])], 1))
        meta.append((th[k], r))
    P2 = np.vstack(allp)
    d_all = roof.dist_edge(P2[:, 0], P2[:, 1])
    _, zb_all = roof.surfaces(P2[:, 0], P2[:, 1], d_all)
    dep_all = roof.rib_depth(d_all)
    idx = 0
    rng = np.random.default_rng(7)
    for (t, r) in meta:
        m = len(r)
        zb = zb_all[idx: idx + m]; dep = dep_all[idx: idx + m]; idx += m
        e = np.array([-math.sin(t), math.cos(t)])
        c = np.stack([r * math.cos(t), r * math.sin(t)], 1)
        w = np.clip(0.045 + 0.16 * r * dth, 0.05, 0.11)
        z_up = zb + 0.03; z_dn = zb - dep
        corners = [(-0.5, z_dn), (0.5, z_dn), (0.5, z_up), (-0.5, z_up)]
        V = np.zeros((m, 4, 3))
        for q, (s, z) in enumerate(corners):
            V[:, q, 0] = c[:, 0] + e[0] * s * w
            V[:, q, 1] = c[:, 1] + e[1] * s * w
            V[:, q, 2] = z
        F = []
        for i in range(m - 1):
            for q in range(4):
                q2 = (q + 1) % 4
                F.append([off + i * 4 + q, off + i * 4 + q2, off + (i + 1) * 4 + q2, off + (i + 1) * 4 + q])
        F.append([off + 3, off + 2, off + 1, off + 0])
        F.append([off + (m - 1) * 4 + q for q in range(4)])
        u0 = rng.random() * 10
        uv = np.stack([np.repeat(r + u0, 4), np.tile([0, .3, .6, .9], m) + rng.random()], 1)
        buf_v.append(V.reshape(-1, 3)); buf_f.extend(F); buf_uv.append(uv)
        off += 4 * m
    mesh_object(f"{roof.name}_SoffitRibs", np.vstack(buf_v), buf_f, coll, [M["wood"]], smooth=False,
                uv=np.vstack(buf_uv))


# =================================================================== 幕墙
def envelope_ring(poly, step=0.04):
    return ccw(resample_ring(poly.exterior.coords, step))


def build_envelope(name, poly, roof, floor_z, M, parent, bay=1.5, solid=(), doors=(), column_every=3,
                   column_skip=(), moon=None, windows=(), column_side="outer"):
    """幕墙：等宽分段，竖梃与室内柱位对应；顶部金属收口带封至木梁底。
    solid: [(i0,i1)] 以竖梃编号表示的实墙段；windows: 实墙中的框景窗 [(i0,i1)]；doors: [bay 编号]。"""
    coll = collection(name, parent)
    dense = envelope_ring(poly)
    seg = np.linalg.norm(np.roll(dense, -1, 0) - dense, axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    L = s[-1]
    n = max(6, int(round(L / bay)))
    sm = np.linspace(0, L, n, endpoint=False)
    closed = np.vstack([dense, dense[:1]])
    mp = np.stack([np.interp(sm, s, closed[:, 0]), np.interp(sm, s, closed[:, 1])], 1)
    nrm, tan = ring_normals(mp)
    z_head = roof.underside(mp[:, 0], mp[:, 1]) - 0.04
    if callable(solid):
        solid = solid(mp, nrm)
    if callable(windows):
        windows = windows(mp, nrm)
    if callable(doors):
        doors = doors(mp, nrm)
    if callable(moon):
        moon = moon(mp, nrm)
    if moon:
        solid = list(solid) + [moon]

    def is_solid(i):
        return any(a <= i < b for a, b in solid)

    glass, metal, wall, led = MeshBuf(), MeshBuf(), MeshBuf(), MeshBuf()
    # 竖梃
    for i in range(n):
        if is_solid(i) and is_solid(i - 1):
            continue
        door_edge = (i in doors) or ((i - 1) % n in doors)
        w = 0.075 if door_edge else 0.05
        c = np.r_[mp[i] - nrm[i] * 0.03, 0]
        v, f = oriented_box(c + [0, 0, floor_z + 0.02], c + [0, 0, z_head[i] + 0.02], w, 0.13,
                            np.r_[nrm[i], 0])
        metal.add(v, f)
    # 玻璃（单片，薄壁）与门
    for i in range(n):
        j = (i + 1) % n
        if is_solid(i):
            continue
        a, b = mp[i], mp[j]
        zb = floor_z + 0.06
        q = [[a[0], a[1], zb], [b[0], b[1], zb], [b[0], b[1], z_head[j]], [a[0], a[1], z_head[i]]]
        glass.add(q, [[0, 1, 2, 3]])
        if i in doors:
            nn = np.r_[(nrm[i] + nrm[j]) / 2, 0]
            tt = np.r_[b - a, 0]; width = np.linalg.norm(tt); tt /= width
            mid = np.r_[(a + b) / 2, 0]
            # 门楣横档
            h = floor_z + 2.75
            v, f = oriented_box(mid - tt * width / 2 + [0, 0, h], mid + tt * width / 2 + [0, 0, h], 0.07, 0.12, nn)
            metal.add(v, f)
            # 中缝竖框 + 两侧拉手
            v, f = oriented_box(mid + [0, 0, floor_z], mid + [0, 0, h], 0.05, 0.10, nn)
            metal.add(v, f)
            for side in (-1, 1):
                for face in (-1, 1):
                    p0 = mid + tt * side * 0.16 + nn * face * 0.07
                    v, f, _ = tube([p0 + [0, 0, floor_z + 0.65], p0 + [0, 0, floor_z + 1.85]], 0.016, 10)
                    metal.add(v, f)
                    for zz in (floor_z + 0.75, floor_z + 1.75):
                        v, f, _ = tube([p0 + [0, 0, zz], p0 - nn * face * 0.055 + [0, 0, zz]], 0.008, 8)
                        metal.add(v, f)
    # 底槛 / 顶部收口带（连续）
    dn, _ = ring_normals(dense)
    _, dso = roof.surfaces(dense[:, 0], dense[:, 1])
    dhead = roof.underside(dense[:, 0], dense[:, 1]) - 0.04
    v, f, uv = band(dense, floor_z - 0.01, floor_z + 0.07, -0.09, 0.05, dn)
    metal.add(v, f)
    v, f, uv = band(dense, dhead, dso + 0.04, -0.10, 0.07, dn)
    metal.add(v, f)
    # 室内侧暗藏灯带：向上洗亮木肋
    v, f, uv = band(dense, dhead - 0.02, dhead, -0.16, -0.10, dn)
    led.add(v, f)
    # 实墙段（其中的框景窗：墙体只留窗下与窗上部分，外挑深窗框）
    def span_sel(a, b):
        sa, sb = sm[a % n], (sm[b] if b < n else L)
        return (s[:-1] >= sa - 1e-6) & (s[:-1] <= sb + 1e-6)

    sill_z, head_z = floor_z + 0.55, floor_z + 3.15
    frames = MeshBuf()
    for a, b in solid:
        if moon and (a, b) == tuple(moon):
            continue
        cuts = sorted(w for w in windows if a <= w[0] and w[1] <= b)
        edges = [a]
        for wa, wb in cuts:
            edges += [wa, wb]
        edges.append(b)
        for k in range(0, len(edges) - 1):
            pa, pb = edges[k], edges[k + 1]
            if pb <= pa:
                continue
            sel = span_sel(pa, pb)
            pts = dense[sel]; nn = dn[sel]; zz = dso[sel]
            if len(pts) < 2:
                continue
            if k % 2 == 0:          # 整片实墙
                v, f, uv = band(pts, floor_z - 0.01, zz + 0.03, -0.22, 0.12, nn, closed=False)
                wall.add(v, f, uv=uv)
            else:                   # 框景窗
                v, f, uv = band(pts, floor_z - 0.01, sill_z, -0.22, 0.12, nn, closed=False)
                wall.add(v, f, uv=uv)
                v, f, uv = band(pts, head_z, zz + 0.03, -0.22, 0.12, nn, closed=False)
                wall.add(v, f, uv=uv)
                for i in range(len(pts) - 1):
                    glass.add([[*pts[i], sill_z], [*pts[i + 1], sill_z], [*pts[i + 1], head_z], [*pts[i], head_z]],
                              [[0, 1, 2, 3]])
                # 外挑深窗框：上下框沿墙曲线，两侧竖框
                for z0, z1 in ((sill_z - 0.07, sill_z), (head_z, head_z + 0.07)):
                    v, f, uv = band(pts, z0, z1, 0.0, 0.62, nn, closed=False)
                    frames.add(v, f)
                for end in (0, -1):
                    p = pts[end]; q = nn[end]
                    tdir = pts[min(end + 1, len(pts) - 1) if end == 0 else end - 1] - p
                    tdir = tdir / (np.linalg.norm(tdir) + 1e-9)
                    o = p + q * 0.31 + tdir * 0.035
                    v, f = oriented_box(np.r_[o, sill_z - 0.07], np.r_[o, head_z + 0.07], 0.62, 0.07,
                                        np.r_[tdir, 0])
                    frames.add(v, f)
    if frames.v:
        frames.build(f"{name}_WindowFrames", coll, [M["bronze_satin"]], smooth=False)
    if moon:
        a, b = moon
        sa, sb = sm[a], (sm[b] if b < n else L)
        sel = (s[:-1] >= sa - 1e-6) & (s[:-1] <= sb + 1e-6)
        moon_wall(f"{name}_MoonWindowWall", dense[sel], dn[sel], floor_z, dso[sel] + 0.03, M, coll)
    glass.build(f"{name}_Glass", coll, [M["glass"]], smooth=False)
    metal.build(f"{name}_Mullions_Frames", coll, [M["metal"]], smooth=False)
    if wall.v:
        wall.build(f"{name}_SolidWall", coll, [M["wall"]], smooth=True)
    led.build(f"{name}_CoveLED", coll, [M["led_soft"]], smooth=False)
    # 柱：与竖梃对应，退入室内
    cols = MeshBuf()
    col_pts = []
    radial_out = (nrm * (mp / np.linalg.norm(mp, axis=1, keepdims=True))).sum(1)
    for i in (range(0, n, column_every) if column_every else []):
        if is_solid(i) or i in column_skip:
            continue
        if column_side == "outer" and radial_out[i] < 0.35:
            continue
        if column_side == "inner" and radial_out[i] > -0.35:
            continue
        c = mp[i] - nrm[i] * 0.85
        if not poly.buffer(-0.4).contains(Point(*c)):
            continue
        col_pts.append(c)
    if col_pts:
        cp = np.array(col_pts)
        cund = roof.underside(cp[:, 0], cp[:, 1])
        for c, zu in zip(cp, cund):
            add_column(cols, c, floor_z, zu)
        cols.build(f"{name}_Columns", coll, [M["metal"]], smooth=True)
    return coll, mp, nrm, z_head


def moon_wall(name, pts, nrm, z0, ztop, M, coll, thick=0.34, hole_r=1.1):
    """带圆形框景开口（月洞）的弧形实墙：开口内衬金属环，嵌超白玻璃。"""
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    u = np.concatenate([[0], np.cumsum(seg)])
    U = u[-1]
    hc = (U / 2, z0 + 0.5 + hole_r)
    outer2 = np.vstack([np.stack([u, np.full(len(u), z0)], 1),
                        np.stack([u[::-1], ztop[::-1]], 1)])
    a = np.linspace(0, 2 * np.pi, 96, endpoint=False)
    hole = np.stack([hc[0] + hole_r * np.cos(a), hc[1] + hole_r * np.sin(a)], 1)
    V = np.vstack([outer2, hole]); no = len(outer2)
    S = np.vstack([np.stack([np.arange(no), (np.arange(no) + 1) % no], 1),
                   np.stack([np.arange(96) + no, (np.arange(96) + 1) % 96 + no], 1)])
    out = triangle.triangulate({"vertices": V, "segments": S, "holes": [hc]}, "pYq25a0.05")
    V2, T = out["vertices"], out["triangles"]

    def place(uv, off):
        x = np.interp(uv[:, 0], u, pts[:, 0]); y = np.interp(uv[:, 0], u, pts[:, 1])
        nx = np.interp(uv[:, 0], u, nrm[:, 0]); ny = np.interp(uv[:, 0], u, nrm[:, 1])
        return np.stack([x + nx * off, y + ny * off, uv[:, 1]], 1)
    o_out, o_in = 0.12, 0.12 - thick
    F = place(V2, o_out); B = place(V2, o_in)
    N = len(V2)
    faces = [list(t) for t in T] + [list(t[::-1] + N) for t in T]
    verts = [F, B]
    # 外周与洞口侧面
    for loop in (np.arange(no), np.arange(no, no + 96)[::-1]):
        base = sum(len(v) for v in verts)
        lf, lb = F[loop], B[loop]
        verts += [lf, lb]
        m = len(loop)
        for i in range(m):
            j = (i + 1) % m
            faces.append([base + j, base + i, base + m + i, base + m + j])
    mesh_object(name, np.vstack(verts), faces, coll, [M["wall"]], smooth=False)
    # 洞口金属内衬与玻璃
    ring = MeshBuf()
    c2 = place(np.array([hc]), 0)[0]
    rim_pts = place(hole, (o_out + o_in) / 2)
    rim_n = place(hole, o_out) - place(hole, o_in)
    rim_n /= np.linalg.norm(rim_n, axis=1, keepdims=True)
    radial = rim_pts - c2; radial /= np.linalg.norm(radial, axis=1, keepdims=True)
    Vr = []
    for k in range(96):
        for (dr, dt) in ((-0.0, -thick / 2 - 0.015), (-0.0, thick / 2 + 0.015), (-0.035, thick / 2 + 0.015), (-0.035, -thick / 2 - 0.015)):
            Vr.append(rim_pts[k] + radial[k] * dr + rim_n[k] * dt)
    Fr = []
    for k in range(96):
        j = (k + 1) % 96
        for q in range(4):
            q2 = (q + 1) % 4
            Fr.append([k * 4 + q, k * 4 + q2, j * 4 + q2, j * 4 + q])
    ring.add(Vr, Fr)
    ring.build(f"{name}_MetalReveal", coll, [M["metal"]], smooth=True)
    gp = rim_pts - radial * 0.03
    mesh_object(f"{name}_Glass", np.vstack([gp, c2[None]]), [[k, (k + 1) % 96, 96] for k in range(96)],
                coll, [M["glass"]], smooth=False)


def add_column(buf, c, z0, z_under, r=0.095):
    """细长柱：柱脚节点（底板+套环）、柱身、外张柱头与顶板；柱头顶板承于木梁底。"""
    top = z_under - 0.01
    prof = [
        (0.0, z0), (0.24, z0), (0.24, z0 + 0.025), (0.205, z0 + 0.03),       # 底板
        (0.13, z0 + 0.035), (0.13, z0 + 0.16), (r + 0.004, z0 + 0.175),     # 套环
        (r, z0 + 0.19), (r, top - 0.62),                                      # 柱身
    ]
    # 柱头：凹曲线外张
    for k in range(1, 9):
        u = k / 8
        prof.append((r + (0.21 - r) * u ** 2.2, top - 0.62 + 0.6 * u))
    prof += [(0.25, top - 0.02), (0.25, top + 0.12), (0.0, top + 0.12)]
    v, f = lathe(prof, 28, (c[0], c[1], 0))
    buf.add(v, f)


# =================================================================== 基座 / 平台
def build_plinth(poly, z, M, parent, name="Plinth", lip=0.30, recess=0.6, depth=-1.7, led=True, mat="stone",
                 cut=None):
    """浮于水面的石材基座：挑出的檐口 + 内凹深色基座 + 檐下灯带。
    cut：潮池与水口区域；沿其边缘改为直落池底的石材池壁。"""
    coll = collection(name, parent)
    outer = ccw(resample_ring(poly.exterior.coords, 0.22))
    V2, T = triangulate([outer], max_area=4.0, quality=20)
    mesh_object(f"{name}_Top", np.c_[V2, np.full(len(V2), z)], T, coll, [M[mat]], smooth=False)
    nrm, _ = ring_normals(outer)
    if cut is not None:
        wallmask = shapely.distance(cut, shapely.points(outer)) < 0.06
    else:
        wallmask = np.zeros(len(outer), bool)
    edge, base, lb, pool = MeshBuf(), MeshBuf(), MeshBuf(), MeshBuf()
    for mask, kind in ((~wallmask, "edge"), (wallmask, "wall")):
        idx = np.where(mask)[0]
        if not len(idx):
            continue
        runs = np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)
        if mask[0] and mask[-1] and len(runs) > 1:
            runs = [np.concatenate([runs[-1], runs[0]])] + runs[1:-1]
        closed = len(runs) == 1 and len(runs[0]) == len(outer)
        for rr in runs:
            if len(rr) < 2:
                continue
            if not closed:          # 与相邻段搭接一个点，避免缝隙
                rr = np.concatenate([[(rr[0] - 1) % len(outer)], rr, [(rr[-1] + 1) % len(outer)]])
            pts, nn = outer[rr], nrm[rr]
            if kind == "edge":
                v, f, uv = band(pts, z - lip, z, -0.01, 0.0, nn, closed=closed)
                edge.add(v, f)
                v, f, uv = band(pts, z - lip - 0.01, z - lip, -recess, 0.0, nn, closed=closed)
                edge.add(v, f)
                v, f, uv = band(pts, depth, z - lip, -recess - 0.4, -recess, nn, closed=closed)
                base.add(v, f)
                if led:
                    v, f, uv = band(pts, z - lip - 0.02, z - lip - 0.012, -0.16, -0.08, nn, closed=closed)
                    lb.add(v, f)
            else:
                v, f, uv = band(pts, depth, z, -0.35, 0.0, nn, closed=closed)
                pool.add(v, f)
    edge.build(f"{name}_EdgeFascia", coll, [M["stone_edge"]], smooth=False)
    base.build(f"{name}_RecessedBase", coll, [M["plinth"]], smooth=False)
    if led:
        lb.build(f"{name}_UnderLipLED", coll, [M["led"]], smooth=False)
    if pool.v:
        pool.build(f"{name}_TidalPoolWalls", coll, [M["stone_tidal"]], smooth=False)
    return coll


def prism(buf, poly, z0, z1, step=0.2, area=0.6):
    """平面多边形拉伸成实体（顶面 + 外周侧壁）。"""
    parts = [poly] if poly.geom_type == "Polygon" else list(poly.geoms)
    for p in parts:
        if p.area < 0.05:
            continue
        ring = ccw(resample_ring(p.exterior.coords, step))
        inner = [ccw(resample_ring(i.coords, step))[::-1] for i in p.interiors]
        holes = [Polygon(i.coords).representative_point().coords[0] for i in p.interiors]
        V2, T = triangulate([ring] + inner, area, 20, holes=holes or None)
        buf.add(np.c_[V2, np.full(len(V2), z1)], T.tolist())
        for rg in [ring] + inner:
            nn, _ = ring_normals(rg)
            v, f, uv = band(rg, z0, z1, -0.01, 0.0, nn)
            buf.add(v, f)


def build_tidal_basin(M, parent):
    """阶梯式潮池：四级台阶由首层标高降至池底，经水口与外部水面相通；
    高潮位时池水漫至第三级，落潮时池底与潮痕全部露出。"""
    coll = collection("Courtyard_TidalBasin", parent)
    buf = MeshBuf()
    ch = P.CHANNEL
    for r_out, r_in, z in P.BASIN_STEPS:
        ring = Point(0, 0).buffer(r_out, 128).difference(Point(0, 0).buffer(r_in, 128)).difference(ch)
        prism(buf, ring, P.BASIN_FLOOR - 0.05, z)
    floor = Point(0, 0).buffer(P.BASIN_STEPS[-1][1] + 0.02, 128).union(ch.intersection(P.PLINTH_ALL.buffer(0.4, 32)))
    floor = floor.union(Point(0, 0).buffer(P.BASIN_R, 128).intersection(ch))
    prism(buf, floor, -1.7, P.BASIN_FLOOR)
    buf.build("TidalBasin_Steps", coll, [M["stone_tidal"]], smooth=False)
    # 横跨水口的两座步行桥（首层标高，侧面金属收边 + 玻璃栏板）
    br = MeshBuf()
    g, m = MeshBuf(), MeshBuf()
    for r_mid, width in ((8.3, 1.8), (11.6, 1.5)):
        c = P.AXIS * r_mid
        a, n_ = P.AXIS * width / 2, P.AXIS_N * (P.CHANNEL_W / 2 + 0.01)
        quad = Polygon([c - a - n_, c + a - n_, c + a + n_, c - a + n_])
        prism(br, quad, P.FFL - 0.22, P.FFL - 0.002)
        for sgn in (-1, 1):
            p0 = c + a * sgn * 0.98 - P.AXIS_N * (P.CHANNEL_W / 2)
            p1 = c + a * sgn * 0.98 + P.AXIS_N * (P.CHANNEL_W / 2)
            balustrade(g, m, np.stack([p0, p1]), P.FFL, height=1.05, panel=1.4)
    br.build("Channel_Footbridges", coll, [M["stone"]], smooth=False)
    # 水口两侧栏板（在桥处断开）
    for sgn in (-1, 1):
        off = P.AXIS_N * sgn * (P.CHANNEL_W / 2 + 0.25)
        segs = [(P.BASIN_R + 0.2, 8.3 - 0.9 - 0.05), (8.3 + 0.9 + 0.05, 11.6 - 0.75 - 0.05)]
        for r0, r1 in segs:
            pts = np.stack([P.AXIS * r0 + off, P.AXIS * r1 + off])
            if shapely.contains_xy(P.PLINTH_CUT, *pts.mean(0)):
                balustrade(g, m, pts, P.FFL, height=1.05, panel=1.4)
    g.build("Channel_Balustrade_Glass", coll, [M["glass_rail"]], smooth=False)
    m.build("Channel_Balustrade_Rail", coll, [M["metal"]], smooth=True)
    return coll


def balustrade(buf_glass, buf_metal, pts, z_base, height=1.08, panel=1.5, closed=False, nrm=None,
               zfun=None, zs=None):
    """玻璃栏板：分段玻璃 + 顶部圆管扶手 + 底部压条。pts 为平面折线；
    zs 为每个折线点的底标高（用于坡道、楼梯等斜向栏板）。"""
    pts = np.asarray(pts, float)
    if closed:
        pts = np.vstack([pts, pts[:1]])
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    L = s[-1]
    if L < 0.3:
        return
    if zs is not None:
        zs = np.asarray(zs, float)
        zfun = None
    n = max(1, int(round(L / panel)))
    sm = np.linspace(0, L, max(int(L / 0.12), 4))
    path = np.stack([np.interp(sm, s, pts[:, 0]), np.interp(sm, s, pts[:, 1])], 1)

    def zat(ss, q):
        if zs is not None:
            return np.interp(ss, s, zs)
        return np.full(len(q), z_base) if zfun is None else zfun(q)
    zb = zat(sm, path)
    _, nn = frames_along(np.c_[path, zb])
    nn2 = nn[:, :2]
    # 底部压条
    v, f, uv = band(path, zb - 0.01, zb + 0.09, -0.035, 0.035, nn2, closed=False)
    buf_metal.add(v, f)
    # 扶手
    rail = np.c_[path, zb + height]
    v, f, uv = tube(rail, 0.024, 12)
    buf_metal.add(v, f)
    # 玻璃分段（板间 12 mm 缝）
    edges = np.linspace(0, L, n + 1)
    for k in range(n):
        a, b = edges[k] + 0.006, edges[k + 1] - 0.006
        ss = np.linspace(a, b, max(2, int((b - a) / 0.2) + 1))
        q = np.stack([np.interp(ss, s, pts[:, 0]), np.interp(ss, s, pts[:, 1])], 1)
        qz = zat(ss, q)
        V = []; F = []
        for i in range(len(q)):
            V.append([q[i, 0], q[i, 1], qz[i] + 0.08]); V.append([q[i, 0], q[i, 1], qz[i] + height - 0.035])
        for i in range(len(q) - 1):
            F.append([2 * i, 2 * i + 2, 2 * i + 3, 2 * i + 1])
        buf_glass.add(V, F)
        # 玻璃夹 / 扶手支撑
    for k in range(n + 1):
        x = np.interp(edges[k], s, pts[:, 0]); y = np.interp(edges[k], s, pts[:, 1])
        zz = float(zat(np.array([edges[k]]), np.array([[x, y]]))[0])
        v, f, _ = tube([[x, y, zz + height - 0.07], [x, y, zz + height - 0.02]], 0.012, 8)
        buf_metal.add(v, f)


def build_moon_gate(M, parent):
    """月洞墙：位于屋面平台西端，墙面正对冬至日落方位；圆洞对准二层观景厅，
    冬至傍晚夕阳穿洞入室。墙两端在顶部以凹曲线外张、贴合次屋顶底面——与柱头同一曲线语言，
    也是次屋顶的承重墙之一。"""
    coll = collection("Terrace_MoonGate", parent)
    c, ax, nrm = P.MOON_C, P.AXIS_N, P.AXIS          # 墙沿 AXIS_N 延伸，法向朝日落
    half, z0 = P.MOON_HALF, P.DECK - 0.05
    fw, fh = 0.9, 1.6                                 # 顶部外张宽度 / 高度

    def ztop(u):
        q = c[None] + ax[None] * np.atleast_1d(u)[:, None]
        _, bot = P.ROOF_B.surfaces(q[:, 0], q[:, 1])
        return bot + 0.03
    # 立面轮廓 (u, z)：下段竖直，上段凹曲线外张至屋面
    us = np.linspace(-(half + fw), half + fw, 70)
    zt = ztop(us)
    prof = [(-half, z0), (half, z0)]
    for k in range(1, 13):
        t = k / 12
        u = half + fw * t ** 2.4
        prof.append((u, float(ztop(np.array([u]))[0]) - fh * (1 - t)))
    prof += [(u, z) for u, z in zip(us[::-1], zt[::-1])][1:-1]
    for k in range(12, 0, -1):
        t = k / 12
        u = -(half + fw * t ** 2.4)
        prof.append((u, float(ztop(np.array([u]))[0]) - fh * (1 - t)))
    outer2 = np.array(prof)
    hr = P.MOON_HOLE_R
    hc = (0.0, P.DECK + 0.5 + hr)
    a_ = np.linspace(0, 2 * np.pi, 120, endpoint=False)
    hole = np.stack([hc[0] + hr * np.cos(a_), hc[1] + hr * np.sin(a_)], 1)
    V = np.vstack([outer2, hole]); no = len(outer2)
    S = np.vstack([np.stack([np.arange(no), (np.arange(no) + 1) % no], 1),
                   np.stack([np.arange(120) + no, (np.arange(120) + 1) % 120 + no], 1)])
    out = triangle.triangulate({"vertices": V, "segments": S, "holes": [hc]}, "pYq25a0.04")
    V2, T = out["vertices"], out["triangles"]
    th = 0.42

    def place(uv, off):
        q = c[None] + ax[None] * uv[:, 0:1] + nrm[None] * off
        return np.c_[q, uv[:, 1]]
    Fv, Bv = place(V2, th / 2), place(V2, -th / 2)
    N = len(V2)
    faces = [list(t) for t in T] + [list(t[::-1] + N) for t in T]
    verts = [Fv, Bv]
    for loop in (np.arange(no), np.arange(no, no + 120)[::-1]):
        base = sum(len(v) for v in verts)
        verts += [Fv[loop], Bv[loop]]
        m = len(loop)
        for i in range(m):
            j = (i + 1) % m
            faces.append([base + j, base + i, base + m + i, base + m + j])
    mesh_object("MoonGate_Wall", np.vstack(verts), faces, coll, [M["wall"]], smooth=False)
    # 洞口拉丝金属内衬 + 超白玻璃
    ring = MeshBuf()
    cen = place(np.array([hc]), 0)[0]
    Vr, Fr = [], []
    for k in range(120):
        pth = place(hole[k:k + 1], 0)[0]
        rad = (pth - cen) / np.linalg.norm(pth - cen)
        for dr, dt in ((0.0, -th / 2 - 0.015), (0.0, th / 2 + 0.015), (-0.04, th / 2 + 0.015), (-0.04, -th / 2 - 0.015)):
            Vr.append(pth + rad * dr + np.r_[nrm, 0] * dt)
    for k in range(120):
        j = (k + 1) % 120
        for q in range(4):
            q2 = (q + 1) % 4
            Fr.append([k * 4 + q, k * 4 + q2, j * 4 + q2, j * 4 + q])
    ring.add(Vr, Fr)
    ring.build("MoonGate_MetalReveal", coll, [M["metal"]], smooth=True)
    gp = place(hole, 0.0)
    gp = cen + (gp - cen) * 0.97
    mesh_object("MoonGate_Glass", np.vstack([gp, cen[None]]), [[k, (k + 1) % 120, 120] for k in range(120)],
                coll, [M["glass"]], smooth=False)
    return coll


def build_deck(M, parent):
    coll = collection("Roof_Terrace_Deck", parent)
    poly = P.DECK_POLY
    outer = ccw(resample_ring(poly.exterior.coords, 0.2))
    V2, T = triangulate([outer], max_area=1.5, quality=20)
    mesh_object("Deck_Paving", np.c_[V2, np.full(len(V2), P.DECK)], T, coll, [M["stone"]], smooth=False)
    nrm, _ = ring_normals(outer)
    top_a, _ = P.ROOF_A.surfaces(outer[:, 0], outer[:, 1])
    ed = MeshBuf()
    v, f, uv = band(outer, top_a - 0.05, P.DECK, -0.02, 0.0, nrm)
    ed.add(v, f)
    ed.build("Deck_EdgeFascia", coll, [M["metal"]], smooth=False)
    # 栏杆：沿平台边缘，留出坡道落点与二层观景厅
    g, m = MeshBuf(), MeshBuf()
    th = np.degrees(np.arctan2(outer[:, 1], outer[:, 0])) % 360
    r = np.hypot(outer[:, 0], outer[:, 1])
    landing = (np.abs(th - (P.RAMP_END_DEG % 360)) < 8.5) & (r < 9.6)
    e3 = P.E3.buffer(0.35)
    in_e3 = shapely.contains_xy(e3, outer[:, 0], outer[:, 1])
    ok = ~(landing | in_e3)
    # 切成连续段
    idx = np.where(ok)[0]
    if len(idx):
        runs = np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)
        if ok[0] and ok[-1] and len(runs) > 1:
            runs = [np.concatenate([runs[-1], runs[0]])] + runs[1:-1]
        for rr in runs:
            if len(rr) < 3:
                continue
            pts = outer[rr] - nrm[rr] * 0.25
            balustrade(g, m, pts, P.DECK)
    g.build("Deck_Balustrade_Glass", coll, [M["glass_rail"]], smooth=False)
    m.build("Deck_Balustrade_Rail", coll, [M["metal"]], smooth=True)
    return coll


# =================================================================== 坡道
def build_ramp(M, parent):
    """绕潮池上升的无障碍坡道：坡度 1:12，每升约 0.68 m 设 1.5 m 休息平台，共约 1.96 圈；
    上下两圈净高约 2.5 m。外侧一圈细柱以短悬臂托住各圈坡道。"""
    coll = collection("Courtyard_Ramp", parent)
    R, W = P.RAMP_R, P.RAMP_W
    phi = np.arange(P.RAMP_START_DEG, P.RAMP_END_DEG + 0.01, 0.5)
    z = P.ramp_z_of_deg(phi)
    rad = np.radians(phi)
    c = np.stack([R * np.cos(rad), R * np.sin(rad), z], 1)
    out = np.stack([np.cos(rad), np.sin(rad), np.zeros_like(rad)], 1)   # 径向外
    hw = W / 2
    prof = [(-hw, 0.0), (-hw, -0.07), (-hw + 0.25, -0.17), (-0.25, -0.26), (0.25, -0.26),
            (hw - 0.25, -0.17), (hw, -0.07), (hw, 0.0)]
    prof = prof[::-1]
    v, f, uv = sweep(c, prof, nrm=out)
    slab = MeshBuf(); slab.add(v, f)
    # 端部平台：落于屋面平台
    ang = np.radians(np.linspace(P.RAMP_END_DEG - 7.5, P.RAMP_END_DEG + 6.5, 30))
    land = Polygon(np.vstack([
        np.stack([(R - hw) * np.cos(ang), (R - hw) * np.sin(ang)], 1),
        np.stack([9.2 * np.cos(ang[::-1]), 9.2 * np.sin(ang[::-1])], 1)]))
    lr = ccw(resample_ring(land.exterior.coords, 0.25))
    V2, T = triangulate([lr], 0.5, 20)
    mesh_object("Ramp_Landing_Top", np.c_[V2, np.full(len(V2), P.DECK)], T, coll, [M["stone"]], smooth=False)
    ln, _ = ring_normals(lr)
    v, f, uv = band(lr, P.DECK - 0.26, P.DECK, -0.02, 0.0, ln)
    slab.add(v, f)
    slab.build("Ramp_Slab", coll, [M["wall"]], smooth=True)
    # 坡面铺装
    top = [c + out * side + [0, 0, 0.008] for side in (-hw + 0.02, hw - 0.02)]
    V = np.vstack(top); n = len(c)
    F = [[i, i + 1, n + i + 1, n + i] for i in range(n - 1)]
    mesh_object("Ramp_Paving", V, F, coll, [M["stone"]], smooth=False)
    # 栏板 + 坡道边缘暗藏灯带
    g, m, led = MeshBuf(), MeshBuf(), MeshBuf()
    for side in (-hw + 0.06, hw - 0.06):
        pts = c[:, :2] + out[:, :2] * side
        balustrade(g, m, pts[:-14], 0, height=1.05, zs=z[:-14])
        sgn = np.sign(side)
        e = c + out * (sgn * (hw + 0.004)) + [0, 0, -0.045]
        v, f, uv = tube(e, 0.012, 6, cap=True)
        led.add(v, f)
    g.build("Ramp_Balustrade_Glass", coll, [M["glass_rail"]], smooth=False)
    m.build("Ramp_Balustrade_Rail", coll, [M["metal"]], smooth=True)
    led.build("Ramp_EdgeLED", coll, [M["led"]], smooth=True)
    # 外侧细柱列：每 30° 一根，以短悬臂托住经过的每一圈坡道
    sp = MeshBuf()
    rc = R + hw + 0.16
    for tdeg in np.arange(0, 360, 30) + 15:
        dax = (tdeg - P.AXIS_DEG + 180) % 360 - 180
        if abs(dax) < 16:          # 让开水口
            continue
        passes = [ph for ph in (tdeg - 720, tdeg - 360, tdeg, tdeg + 360) if P.RAMP_START_DEG + 4 < ph < P.RAMP_END_DEG - 4]
        if not passes:
            continue
        zs_ = [float(P.ramp_z_of_deg(ph)) - 0.26 for ph in passes]
        a = math.radians(tdeg)
        dirv = np.array([math.cos(a), math.sin(a)])
        p = dirv * rc
        ztop = max(zs_) - 0.02
        v, f = lathe([(0, P.FFL), (0.16, P.FFL), (0.16, P.FFL + 0.02), (0.065, P.FFL + 0.04),
                      (0.065, ztop), (0, ztop)], 20, (p[0], p[1], 0))
        sp.add(v, f)
        for zz in zs_:
            q0 = np.r_[p, zz - 0.05]; q1 = np.r_[dirv * (R + 0.2), zz - 0.05]
            v, f = oriented_box(q0, q1, 0.09, 0.14, (0, 0, 1))
            sp.add(v, f)
    sp.build("Ramp_Colonnade", coll, [M["metal"]], smooth=True)
    return coll


def build_core(M, parent):
    """电梯筒：贯通首层、屋面平台与二层观景厅，直抵次屋顶——竖向交通与次屋顶的主要支承。"""
    coll = collection("Structure_LiftCore", parent)
    cx, cy = P.CORE_C
    zt = float(P.ROOF_B.underside(np.array([cx]), np.array([cy]))[0]) + 0.05
    v, f = lathe([(0, P.FFL - 0.01), (P.CORE_R, P.FFL - 0.01), (P.CORE_R, zt), (0, zt)], 72, (cx, cy, 0))
    mesh_object("LiftCore_Shaft", v, f, coll, [M["wall"]], smooth=True)
    # 两层电梯门：朝庭院一侧的古铜门板
    face = math.atan2(-cy, -cx)
    doors = MeshBuf()
    for z0 in (P.FFL, P.DECK):
        for k in range(2):
            a0 = face + (k - 0.5) * 0.46 - 0.23
            ang = np.linspace(a0, a0 + 0.44, 10)
            pts = np.stack([cx + (P.CORE_R + 0.012) * np.cos(ang), cy + (P.CORE_R + 0.012) * np.sin(ang)], 1)
            nn = np.stack([np.cos(ang), np.sin(ang)], 1)
            v, f, uv = band(pts, z0, z0 + 2.35, 0.0, 0.01, nn, closed=False)
            doors.add(v, f)
    doors.build("LiftCore_Doors", coll, [M["bronze_satin"]], smooth=True)
    return coll


def build_supports(M, parent):
    """西翼尖下的 Y 形斜柱（立于水中石墩），与 C 东翼尖下的水中石墩。"""
    coll = collection("Structure_WingSupports", parent)
    stone, steel = MeshBuf(), MeshBuf()
    bx, by = P.YCOL_BASE
    v, f = lathe([(0, -1.7), (1.05, -1.7), (1.05, P.FFL - 0.35), (0.95, P.FFL - 0.2), (0.85, P.FFL), (0, P.FFL)], 64, (bx, by, 0))
    stone.add(v, f)
    node = np.array([bx, by, P.FFL + 0.55])
    v, f = lathe([(0, P.FFL), (0.42, P.FFL), (0.42, P.FFL + 0.06), (0.26, P.FFL + 0.12), (0.22, P.FFL + 0.6), (0, P.FFL + 0.6)],
                 40, (bx, by, 0))
    steel.add(v, f)
    for r, a in P.YCOL_TOPS:
        tx, ty = r * math.cos(math.radians(a)), r * math.sin(math.radians(a))
        tz = float(P.ROOF_A.underside(np.array([tx]), np.array([ty]))[0])
        top = np.array([tx, ty, tz])
        k = np.linspace(0, 1, 24)
        path = node[None] * (1 - k[:, None]) + top[None] * k[:, None]
        rad = 0.22 - 0.07 * k + 0.17 * np.clip((k - 0.86) / 0.14, 0, 1) ** 2.2
        v, f, uv = tube(path, rad, 20)
        steel.add(v, f)
        v, f = lathe([(0, tz - 0.02), (0.44, tz - 0.02), (0.44, tz + 0.1), (0, tz + 0.1)], 32, (tx, ty, 0))
        steel.add(v, f)
    # C 东翼尖的水中石墩与古铜支座
    px, py = P.C_PIER
    pz = float(P.ROOF_C.underside(np.array([px]), np.array([py]))[0])
    v, f = lathe([(0, -1.7), (0.62, -1.7), (0.62, pz - 0.45), (0.5, pz - 0.3), (0, pz - 0.3)], 48, (px, py, 0))
    stone.add(v, f)
    v, f = lathe([(0, pz - 0.3), (0.34, pz - 0.3), (0.26, pz - 0.12), (0.4, pz - 0.05), (0.4, pz + 0.05), (0, pz + 0.05)],
                 40, (px, py, 0))
    steel.add(v, f)
    stone.build("WingSupports_StonePiers", coll, [M["stone_tidal"]], smooth=True)
    steel.build("WingSupports_SteelStruts", coll, [M["metal"]], smooth=True)
    return coll


# =================================================================== 螺旋楼梯 + 圆形平台
def build_spiral_stair(M, parent):
    coll = collection("Hall_SpiralStair", parent)
    cx, cy = P.STAIR_C
    under = float(P.ROOF_A.underside(np.array([cx]), np.array([cy]))[0])
    pz = min(P.STAIR_PLATFORM_Z + P.FFL, under - 2.6)
    R0, R1 = 1.55, 2.55
    # 平台
    plat = MeshBuf()
    v, f = lathe([(0, pz - 0.16), (1.38, pz - 0.16), (1.5, pz - 0.10), (1.5, pz), (0, pz)], 64, (cx, cy, 0))
    plat.add(v, f)
    plat.build("Stair_Platform_Disc", coll, [M["wall"]], smooth=True)
    mesh_disc = lathe([(0, pz + 0.004), (1.44, pz + 0.004)], 64, (cx, cy, 0))
    mesh_object("Stair_Platform_Floor", mesh_disc[0], mesh_disc[1], coll, [M["oak"]], smooth=False,
                uv=np.c_[mesh_disc[0][:, 0], mesh_disc[0][:, 1]])
    # 吊杆
    rods = MeshBuf()
    for k in range(4):
        a = math.radians(45 + 90 * k)
        x, y = cx + 1.3 * math.cos(a), cy + 1.3 * math.sin(a)
        sz = P.ROOF_A.underside(np.array([x]), np.array([y]))
        v, f, _ = tube([[x, y, pz], [x, y, float(sz[0])]], 0.011, 8)
        rods.add(v, f)
    rods.build("Stair_HangerRods", coll, [M["metal"]], smooth=True)
    # 踏步 + 外侧螺旋梁
    rise = pz - P.FFL
    nsteps = int(math.ceil(rise / 0.165))
    h = rise / nsteps
    a_start = math.radians(110)
    sweep_ang = math.radians(290)
    treads = MeshBuf()
    for k in range(nsteps):
        a0 = a_start + sweep_ang * k / nsteps
        a1 = a_start + sweep_ang * (k + 1) / nsteps + 0.012
        zt = P.FFL + h * (k + 1)
        angs = np.linspace(a0, a1, 6)
        inner = np.stack([cx + R0 * np.cos(angs), cy + R0 * np.sin(angs)], 1)
        outer = np.stack([cx + R1 * np.cos(angs), cy + R1 * np.sin(angs)], 1)
        ring = np.vstack([outer, inner[::-1]])
        n = len(ring)
        V = np.vstack([np.c_[ring, np.full(n, zt - 0.05)], np.c_[ring, np.full(n, zt)]])
        F = [list(range(n))[::-1], list(range(n, 2 * n))]
        for i in range(n):
            j = (i + 1) % n
            F.append([i, j, n + j, n + i])
        treads.add(V, F)
    treads.build("Stair_Treads_Oak", coll, [M["oak"]], smooth=False)
    angs = np.linspace(a_start, a_start + sweep_ang, 220)
    zz = P.FFL + (angs - a_start) / sweep_ang * rise
    stringer = MeshBuf()
    for R, w in ((R1 + 0.02, 0.03), (R0 - 0.02, 0.03)):
        path = np.stack([cx + R * np.cos(angs), cy + R * np.sin(angs), zz], 1)
        out = np.stack([np.cos(angs), np.sin(angs), np.zeros_like(angs)], 1)
        v, f, uv = sweep(path, [(w / 2, 0.02), (-w / 2, 0.02), (-w / 2, -0.32), (w / 2, -0.32)][::-1], nrm=out)
        stringer.add(v, f)
    stringer.build("Stair_Stringers", coll, [M["wall"]], smooth=True)
    g, m = MeshBuf(), MeshBuf()
    pts = np.stack([cx + (R1 - 0.03) * np.cos(angs), cy + (R1 - 0.03) * np.sin(angs)], 1)
    balustrade(g, m, pts[3:-3], 0, height=1.0, panel=1.1, zs=zz[3:-3])
    # 平台栏板（留出楼梯到达口）
    a_end = a_start + sweep_ang
    pa = np.linspace(a_end + 0.55, a_end + 2 * np.pi - 0.15, 120)
    pts = np.stack([cx + 1.44 * np.cos(pa), cy + 1.44 * np.sin(pa)], 1)
    balustrade(g, m, pts, pz, height=1.05, panel=1.0)
    g.build("Stair_Balustrade_Glass", coll, [M["glass_rail"]], smooth=False)
    m.build("Stair_Balustrade_Rail", coll, [M["metal"]], smooth=True)
    return coll
