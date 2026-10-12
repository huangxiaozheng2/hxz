"""建筑主体构件：屋顶、木肋、幕墙、柱、基座、平台、坡道、螺旋楼梯、栈桥。"""
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
def triangulate(rings, max_area=0.12, quality=28):
    """rings: 闭合环列表（第一个为外轮廓）。返回 (verts2d, tris)。前 len(rings[0]) 个顶点即外轮廓。"""
    V, S, off = [], [], 0
    for r in rings:
        n = len(r)
        V.append(r)
        S.append(np.stack([np.arange(n) + off, (np.arange(n) + 1) % n + off], 1))
        off += n
    V = np.vstack(V); S = np.vstack(S)
    out = triangle.triangulate({"vertices": V, "segments": S}, f"pYq{quality}a{max_area}")
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
def build_roof(roof, M, parent):
    coll = collection(roof.name, parent)
    poly = roof.poly
    outer = ccw(resample_ring(poly.exterior.coords, 0.16))
    rings = [outer] + offset_rings(poly, [P.REVEAL, 0.45, 0.75], 0.2) + offset_rings(poly, [1.6, 3.2], 0.45)
    V2, T = triangulate(rings, max_area=0.10)
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

    build_ribs(roof, M, coll)
    return coll


def build_ribs(roof, M, coll, spacing=0.24, r_ref=14.0):
    """檐下浅木色细肋：以庭院中心为极点径向排布，沿屋面底部曲率连续，端部统一收于阴影缝内线。"""
    inset = roof.poly.buffer(-(P.REVEAL + 0.04), 64)
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
    _, zb_all = roof.surfaces(P2[:, 0], P2[:, 1])
    idx = 0
    rng = np.random.default_rng(7)
    for (t, r) in meta:
        m = len(r)
        zb = zb_all[idx: idx + m]; idx += m
        e = np.array([-math.sin(t), math.cos(t)])
        c = np.stack([r * math.cos(t), r * math.sin(t)], 1)
        w = np.clip(0.045 + 0.16 * r * dth, 0.05, 0.11)
        z_up = zb + 0.03; z_dn = zb - P.RIB_DEPTH
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
                   column_skip=(), moon=None):
    """玻璃幕墙：等宽分段，竖梃与室内柱位对应；顶部金属收口带封至屋面底板。
    solid: [(i0,i1)] 以竖梃编号表示的实墙段；doors: [bay 编号]。"""
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
    _, soff = roof.surfaces(mp[:, 0], mp[:, 1])
    z_head = soff - P.RIB_DEPTH - 0.04
    if callable(solid):
        solid = solid(mp, nrm)
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
    dhead = dso - P.RIB_DEPTH - 0.04
    v, f, uv = band(dense, floor_z - 0.01, floor_z + 0.07, -0.09, 0.05, dn)
    metal.add(v, f)
    v, f, uv = band(dense, dhead, dso + 0.04, -0.10, 0.07, dn)
    metal.add(v, f)
    # 室内侧暗藏灯带：向上洗亮木肋
    v, f, uv = band(dense, dhead - 0.02, dhead, -0.16, -0.10, dn)
    led.add(v, f)
    # 实墙段
    for a, b in solid:
        if moon and (a, b) == tuple(moon):
            continue
        sa, sb = sm[a % n], sm[b % n] if b < n else L
        sel = (s[:-1] >= sa - 1e-6) & (s[:-1] <= sb + 1e-6)
        pts = dense[sel]; nn = dn[sel]; zz = dso[sel]
        if len(pts) < 2:
            continue
        v, f, uv = band(pts, floor_z - 0.01, zz + 0.03, -0.22, 0.12, nn, closed=False)
        wall.add(v, f, uv=uv)
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
        if radial_out[i] < 0.35:        # 柱列只沿外侧幕墙布置，庭院侧保持通透
            continue
        c = mp[i] - nrm[i] * 0.85
        if not poly.buffer(-0.4).contains(Point(*c)):
            continue
        col_pts.append(c)
    if col_pts:
        cp = np.array(col_pts)
        _, csoff = roof.surfaces(cp[:, 0], cp[:, 1])
        for c, zs in zip(cp, csoff):
            add_column(cols, c, floor_z, zs)
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


def add_column(buf, c, z0, z_soffit, r=0.095):
    """细长柱：柱脚节点（底板+套环）、柱身、外张柱头与顶板。"""
    top = z_soffit - P.RIB_DEPTH - 0.01
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
def build_plinth(poly, z, M, parent, name="Plinth", lip=0.30, recess=0.6, depth=-1.2, led=True, mat="stone"):
    coll = collection(name, parent)
    outer = ccw(resample_ring(poly.exterior.coords, 0.25))
    V2, T = triangulate([outer], max_area=4.0, quality=20)
    mesh_object(f"{name}_Top", np.c_[V2, np.full(len(V2), z)], T, coll, [M[mat]], smooth=False)
    nrm, _ = ring_normals(outer)
    edge = MeshBuf()
    v, f, uv = band(outer, z - lip, z, -0.01, 0.0, nrm)
    edge.add(v, f)
    # 檐口底面与凹入的阴影基座
    v, f, uv = band(outer, z - lip - 0.01, z - lip, -recess, 0.0, nrm)
    edge.add(v, f)
    edge.build(f"{name}_EdgeFascia", coll, [M["stone_edge"]], smooth=False)
    base = MeshBuf()
    v, f, uv = band(outer, depth, z - lip, -recess - 0.4, -recess, nrm)
    base.add(v, f)
    base.build(f"{name}_RecessedBase", coll, [M["plinth"]], smooth=False)
    if led:
        lb = MeshBuf()
        v, f, uv = band(outer, z - lip - 0.02, z - lip - 0.012, -0.16, -0.08, nrm)
        lb.add(v, f)
        lb.build(f"{name}_UnderLipLED", coll, [M["led"]], smooth=False)
    return coll


def balustrade(buf_glass, buf_metal, pts, z_base, height=1.08, panel=1.5, closed=False, nrm=None,
               zfun=None):
    """玻璃栏板：分段玻璃 + 顶部圆管扶手 + 底部压条。pts 为平面折线。"""
    pts = np.asarray(pts, float)
    if closed:
        pts = np.vstack([pts, pts[:1]])
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    L = s[-1]
    if L < 0.3:
        return
    n = max(1, int(round(L / panel)))
    sm = np.linspace(0, L, max(int(L / 0.12), 4))
    path = np.stack([np.interp(sm, s, pts[:, 0]), np.interp(sm, s, pts[:, 1])], 1)
    zb = np.full(len(path), z_base) if zfun is None else zfun(path)
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
        qz = np.full(len(q), z_base) if zfun is None else zfun(q)
        V = []; F = []
        for i in range(len(q)):
            V.append([q[i, 0], q[i, 1], qz[i] + 0.08]); V.append([q[i, 0], q[i, 1], qz[i] + height - 0.035])
        for i in range(len(q) - 1):
            F.append([2 * i, 2 * i + 2, 2 * i + 3, 2 * i + 1])
        buf_glass.add(V, F)
        # 玻璃夹 / 扶手支撑
    for k in range(n + 1):
        x = np.interp(edges[k], s, pts[:, 0]); y = np.interp(edges[k], s, pts[:, 1])
        zz = z_base if zfun is None else float(zfun(np.array([[x, y]]))[0])
        v, f, _ = tube([[x, y, zz + height - 0.07], [x, y, zz + height - 0.02]], 0.012, 8)
        buf_metal.add(v, f)


def build_moon_gate(M, parent, theta=137.5, r0=10.0, r1=14.2):
    """屋面平台西端的框景墙：径向布置的直墙，圆形开口正对二层观景厅，
    远处水面与天空穿过开口进入室内；墙顶贴合次屋顶底面，形成承托关系。"""
    coll = collection("Terrace_MoonGate", parent)
    a = math.radians(theta)
    rr = np.linspace(r0, r1, 60)
    pts = np.stack([rr * math.cos(a), rr * math.sin(a)], 1)
    nrm = np.repeat([[-math.sin(a), math.cos(a)]], len(rr), 0)
    _, sb = P.ROOF_B.surfaces(pts[:, 0], pts[:, 1])
    inside = shapely.contains_xy(P.ROOF_B.poly.buffer(-0.3), pts[:, 0], pts[:, 1])
    top = np.where(inside, sb + 0.03, P.DECK + 3.6)
    moon_wall("MoonGate_Wall", pts, nrm, P.DECK - 0.05, top, M, coll, thick=0.38, hole_r=1.42)
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
    landing = (np.abs(th - (P.RAMP_T1 % 360)) < 8.5) & (r < 9.6)
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
def ramp_z(theta_deg):
    t0, t1 = P.RAMP_T0, P.RAMP_T1
    u = np.clip((np.asarray(theta_deg) - t0 - 5) / (t1 - t0 - 13), 0, 1)
    return P.FFL + (P.DECK - P.FFL) * u


def ramp_theta(q):
    th = np.degrees(np.arctan2(q[:, 1], q[:, 0])) % 360
    return np.where(th < P.RAMP_T0 - 2, th + 360, th)


def build_ramp(M, parent):
    coll = collection("Courtyard_Ramp", parent)
    R, W = P.RAMP_R, P.RAMP_W
    th = np.linspace(P.RAMP_T0, P.RAMP_T1, 520)
    z = ramp_z(th)
    rad = np.radians(th)
    c = np.stack([R * np.cos(rad), R * np.sin(rad), z], 1)
    out = np.stack([np.cos(rad), np.sin(rad), np.zeros_like(rad)], 1)   # 径向外
    hw = W / 2
    prof = [(-hw, 0.0), (-hw, -0.07), (-hw + 0.25, -0.17), (-0.25, -0.26), (0.25, -0.26),
            (hw - 0.25, -0.17), (hw, -0.07), (hw, 0.0)]
    prof = prof[::-1]
    v, f, uv = sweep(c, prof, nrm=out)
    slab = MeshBuf(); slab.add(v, f)
    # 端部平台：落于屋面平台
    a = math.radians(P.RAMP_T1)
    ang = np.radians(np.linspace(P.RAMP_T1 - 7.5, P.RAMP_T1 + 6.5, 30))
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
    # 坡面铺装（顶面略高）
    top = []
    for side in (-hw + 0.02, hw - 0.02):
        top.append(c + out * side + [0, 0, 0.008])
    V = np.vstack([top[0], top[1]]); n = len(c)
    F = [[i, i + 1, n + i + 1, n + i] for i in range(n - 1)]
    mesh_object("Ramp_Paving", V, F, coll, [M["stone"]], smooth=False)
    # 栏板 + 灯带
    g, m, led = MeshBuf(), MeshBuf(), MeshBuf()
    for side in (-hw + 0.06, hw - 0.06):
        pts = c[:, :2] + out[:, :2] * side
        zf = lambda q: ramp_z(ramp_theta(q))
        balustrade(g, m, pts[:-12], 0, height=1.05, zfun=zf)
        # 坡道边缘暗藏灯带
        sgn = np.sign(side)
        e = c + out * (sgn * (hw + 0.004)) + [0, 0, -0.045]
        v, f, uv = tube(e, 0.012, 6, cap=True)
        led.add(v, f)
    g.build("Ramp_Balustrade_Glass", coll, [M["glass_rail"]], smooth=False)
    m.build("Ramp_Balustrade_Rail", coll, [M["metal"]], smooth=True)
    led.build("Ramp_EdgeLED", coll, [M["led"]], smooth=True)
    # 细柱支撑
    sp = MeshBuf()
    for tdeg in np.arange(P.RAMP_T0 + 40, P.RAMP_T1 - 10, 38):
        a = math.radians(tdeg)
        zz = float(ramp_z(tdeg)) - 0.26
        if zz < 1.0:
            continue
        p = np.array([R * math.cos(a), R * math.sin(a)])
        v, f = lathe([(0, P.FFL), (0.14, P.FFL), (0.14, P.FFL + 0.02), (0.06, P.FFL + 0.03),
                      (0.06, zz - 0.05), (0.11, zz), (0, zz)], 20, (p[0], p[1], 0))
        sp.add(v, f)
    sp.build("Ramp_Supports", coll, [M["metal"]], smooth=True)
    return coll


# =================================================================== 螺旋楼梯 + 圆形平台
def build_spiral_stair(M, parent):
    coll = collection("Hall_SpiralStair", parent)
    cx, cy = P.STAIR_C
    _, soff = P.ROOF_A.surfaces(np.array([cx]), np.array([cy]))
    soff = float(soff[0])
    pz = min(P.STAIR_PLATFORM_Z, soff - P.RIB_DEPTH - 2.7)
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
        _, sz = P.ROOF_A.surfaces(np.array([x]), np.array([y]))
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
    zf = lambda q: P.FFL + ((np.arctan2(q[:, 1] - cy, q[:, 0] - cx) - a_start) % (2 * np.pi)) / sweep_ang * rise
    balustrade(g, m, pts[3:-3], 0, height=1.0, panel=1.1, zfun=zf)
    # 平台栏板（留出楼梯到达口）
    a_end = a_start + sweep_ang
    pa = np.linspace(a_end + 0.55, a_end + 2 * np.pi - 0.15, 120)
    pts = np.stack([cx + 1.44 * np.cos(pa), cy + 1.44 * np.sin(pa)], 1)
    balustrade(g, m, pts, pz, height=1.05, panel=1.0)
    g.build("Stair_Balustrade_Glass", coll, [M["glass_rail"]], smooth=False)
    m.build("Stair_Balustrade_Rail", coll, [M["metal"]], smooth=True)
    return coll
