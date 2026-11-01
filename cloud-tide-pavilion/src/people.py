"""人物（近景可用的配景人）。

- 身体：头、颈、躯干、四肢、手（掌 + 拇指 + 四指两节）的体块并集，体素重建后平滑，关节处没有接缝；
  脸部有眉弓、鼻、颧、唇、下颌与耳，眼窝与口线由顶点内推得到。
- 衣物：上衣 / 外套、长裤 / 短裤 / 裙是体块外扩后的外壳，竖向拉伸的噪声位移做出垂坠褶皱；鞋由跟、掌、趾与鞋口组成。
- 头发：发丝曲线（Cycles 毛发 BSDF），按发型（短发、齐耳、披肩、马尾、发髻）从头皮梳出、受重力下垂并避开头、颈与肩背；
  头皮另有一层同色发壳，避免发丝之间透出皮肤；网页导出用的简化发型网格不参与渲染。
局部坐标：面朝 +y，脚底 z = 0，长度单位 m。
"""
import math

import bmesh
import bpy
import numpy as np

from bl import lathe

KINDS = ("man_shirt", "man_coat", "woman_dress", "woman_trousers", "child")


# ------------------------------------------------------------------ 几何工具
_ICO = {}


def _ico(sub):
    if sub not in _ICO:
        bm = bmesh.new()
        bmesh.ops.create_icosphere(bm, subdivisions=sub, radius=1.0)
        V = np.array([v.co[:] for v in bm.verts])
        F = np.array([[v.index for v in f.verts] for f in bm.faces])
        bm.free()
        _ICO[sub] = (V, F)
    return _ICO[sub]


def _unit(v):
    v = np.asarray(v, float)
    return v / (np.linalg.norm(v) + 1e-12)


def _rx(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _ry(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _rot_to(d):
    """把 z 轴转到方向 d。"""
    d = _unit(d)
    z = np.array([0.0, 0.0, 1.0])
    v = np.cross(z, d)
    c = float(np.dot(z, d))
    if np.linalg.norm(v) < 1e-9:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx / (1 + c)


def _loft(levels, n=40):
    """超椭圆截面放样（两端封口）：levels 自下而上为 (z, a, b, xc, yc, e)，a/b 为 x/y 半轴，e 为超椭圆指数。"""
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    c, s = np.cos(th), np.sin(th)
    rings = []
    for z, a, b, xc, yc, e in levels:
        x = xc + a * np.sign(c) * np.abs(c) ** (2 / e)
        y = yc + b * np.sign(s) * np.abs(s) ** (2 / e)
        rings.append(np.stack([x, y, np.full(n, z)], 1))
    m = len(rings)
    V = np.vstack(rings + [np.array([[levels[0][3], levels[0][4], levels[0][0]],
                                     [levels[-1][3], levels[-1][4], levels[-1][0]]])])
    F = []
    for i in range(m - 1):
        for j in range(n):
            a0, a1 = i * n + j, i * n + (j + 1) % n
            b0, b1 = a0 + n, a1 + n
            F += [[a0, a1, b1], [a0, b1, b0]]
    bot, top = m * n, m * n + 1
    for j in range(n):
        F.append([bot, (j + 1) % n, j])
        F.append([top, (m - 1) * n + j, (m - 1) * n + (j + 1) % n])
    return V, np.array(F)


class Prims:
    """体块清单：('ell', c, r, R) 或 ('cap', a, b, ra, rb)，可整体外扩后生成网格，也可当作发丝的碰撞体。"""

    def __init__(self):
        self.items = []

    def ell(self, c, r, R=None, sub=3):
        self.items.append(("ell", np.asarray(c, float), np.asarray(r, float),
                           np.eye(3) if R is None else np.asarray(R, float), sub))

    def cap(self, a, b, ra, rb, sub=2):
        self.items.append(("cap", np.asarray(a, float), np.asarray(b, float), float(ra), float(rb), sub))

    def extend(self, other, grow=0.0, keep=None):
        for it in other.items:
            if keep is not None and not keep(it):
                continue
            if it[0] == "ell":
                self.items.append(("ell", it[1], it[2] + grow, it[3], it[4]))
            else:
                self.items.append(("cap", it[1], it[2], it[3] + grow, it[4] + grow, it[5]))

    def meshes(self):
        out = []
        for it in self.items:
            if it[0] == "ell":
                V, F = _ico(it[4])
                out.append(((V * it[2]) @ it[3].T + it[1], F))
            else:
                _, a, b, ra, rb, sub = it
                V, F = _ico(sub)
                L = float(np.linalg.norm(b - a))
                W = V.copy()
                top = V[:, 2] >= 0
                W[top] *= rb
                W[~top] *= ra
                W[top, 2] += L
                out.append((W @ _rot_to(b - a).T + a, F))
        return out

    def push_out(self, p, margin):
        """把点推到所有体块之外（外扩 margin）。"""
        for it in self.items:
            if it[0] == "ell":
                c, r, R = it[1], it[2] + margin, it[3]
                q = (R.T @ (p - c)) / r
                n = np.linalg.norm(q)
                if n < 1.0:
                    p = c + R @ (q / max(n, 1e-6) * r)
            else:
                a, b, ra, rb = it[1], it[2], it[3] + margin, it[4] + margin
                ab = b - a
                t = float(np.clip(np.dot(p - a, ab) / (np.dot(ab, ab) + 1e-12), 0, 1))
                c = a + ab * t
                r = ra + (rb - ra) * t
                d = p - c
                n = np.linalg.norm(d)
                if n < r:
                    p = c + (d / max(n, 1e-6)) * r
        return p


# ------------------------------------------------------------------ 体素重建
_FOLD = {}


def _fold_space(seed):
    """竖向拉伸的褶皱噪声：Displace 修改器以一个竖向放大的空物体为纹理坐标。"""
    if "tex" not in _FOLD:
        t = bpy.data.textures.new("People_Folds", "CLOUDS")
        t.noise_scale = 0.05
        t.noise_depth = 2
        e = bpy.data.objects.new("People_FoldSpace", None)
        e.scale = (1.0, 1.0, 3.2)
        bpy.context.scene.collection.objects.link(e)
        _FOLD.update(tex=t, empty=e)
    rng = np.random.default_rng(seed)
    _FOLD["empty"].location = tuple(rng.uniform(-50, 50, 3))
    return _FOLD["tex"], _FOLD["empty"]


def _solidify(prims, voxel, smooth_iter=4, folds=0.0, decimate=1.0, seed=0):
    """体块并集 → 体素重建 → 平滑 →（褶皱位移）→（减面）→ 三角面。返回 (V, F) numpy。"""
    parts = prims.meshes()
    if not parts:
        return np.zeros((0, 3)), np.zeros((0, 3), int)
    Vs, Fs, n = [], [], 0
    for V, F in parts:
        Vs.append(V); Fs.append(F + n); n += len(V)
    V = np.vstack(Vs); F = np.vstack(Fs)
    me = bpy.data.meshes.new("_person_tmp")
    me.from_pydata(V.tolist(), [], F.tolist())
    ob = bpy.data.objects.new("_person_tmp", me)
    bpy.context.scene.collection.objects.link(ob)
    m = ob.modifiers.new("remesh", "REMESH")
    m.mode = "VOXEL"; m.voxel_size = voxel; m.adaptivity = 0.0
    s = ob.modifiers.new("smooth", "SMOOTH")
    s.factor = 0.6; s.iterations = smooth_iter
    if folds > 0:
        tex, emp = _fold_space(seed)
        bpy.context.view_layer.update()
        d = ob.modifiers.new("folds", "DISPLACE")
        d.texture = tex; d.texture_coords = "OBJECT"; d.texture_coords_object = emp
        d.strength = folds; d.mid_level = 0.5
        s2 = ob.modifiers.new("smooth2", "SMOOTH")
        s2.factor = 0.5; s2.iterations = 2
    if decimate < 1.0:
        dm = ob.modifiers.new("decimate", "DECIMATE")
        dm.ratio = decimate
    ob.modifiers.new("tri", "TRIANGULATE")
    dg = bpy.context.evaluated_depsgraph_get()
    me2 = bpy.data.meshes.new_from_object(ob.evaluated_get(dg))
    V2 = np.empty(len(me2.vertices) * 3, np.float32)
    me2.vertices.foreach_get("co", V2)
    F2 = np.empty(len(me2.loops), np.int32)
    me2.loops.foreach_get("vertex_index", F2)
    bpy.data.objects.remove(ob)
    bpy.data.meshes.remove(me)
    bpy.data.meshes.remove(me2)
    return V2.reshape(-1, 3).astype(float), F2.reshape(-1, 3)


def _dent(V, c, radius, depth, axis):
    """以 c 为中心、半径 radius 的范围内沿 -axis 内推（余弦衰减），用于眼窝与口线。"""
    d = np.linalg.norm((V - c) / radius, axis=1)
    k = np.clip(1 - d, 0, 1)
    k = 0.5 - 0.5 * np.cos(np.pi * k)
    return V - np.outer(k * depth, axis)


# ------------------------------------------------------------------ 骨架与体块
def _ik(H, A, L1, L2, fwd):
    d = A - H
    dist = float(np.linalg.norm(d))
    u = d / dist
    if dist >= L1 + L2 - 1e-6:
        return H + u * L1
    a = (L1 ** 2 - L2 ** 2 + dist ** 2) / (2 * dist)
    hgt = math.sqrt(max(L1 ** 2 - a ** 2, 0.0))
    f = np.asarray(fwd, float)
    f = _unit(f - np.dot(f, u) * u)
    return H + u * a + f * hgt


def make_person(h=1.72, kind="man_shirt", walk=0.0, seed=0, detail=1.0):
    """返回各层网格（局部坐标）、发丝与发型代理网格。detail < 1 时放大体素以降低面数（远处人物）。"""
    rng = np.random.default_rng(seed)
    female = kind.startswith("woman")
    child = kind == "child"
    stand = walk < 0.3
    sw = (0.91 if female else 1.0) * (0.92 if child else 1.0) * rng.uniform(0.96, 1.04)   # 肩宽
    hw = (1.07 if female else 0.97) * rng.uniform(0.97, 1.03)                             # 髋宽
    bulk = rng.uniform(0.95, 1.08) * (0.94 if female else 1.0)                           # 胖瘦
    hk = 1.15 if child else 1.0                                                          # 头部比例
    zc = dict(hc=0.928, neck=0.81, sh=0.806, chest=0.722, waist=0.628, pel=0.532, hip=0.512, knee=0.285, ank=0.042)
    if child:
        zc.update(hc=0.905, neck=0.79, sh=0.783, chest=0.70, waist=0.61, pel=0.505, hip=0.486, knee=0.27, ank=0.045)
    Z = {k: v * h for k, v in zc.items()}
    s = 1 if rng.random() < 0.5 else -1          # 站姿：重心腿；行走：前腿
    pel = np.array([0.0, 0.0, Z["pel"]])
    if stand:
        pel[0] += s * 0.012 * h
    else:
        pel[2] -= 0.012 * h * walk
    sx = pel[0] * 0.45                            # 上身随骨盆略移
    # 躯干：超椭圆截面放样（身体本身被衣物遮住，只用于衣物外形与发丝碰撞）
    wf, cf, bf, bust = (0.9, 0.93, 1.05, 0.006) if female else (1.0, 1.0, 1.0, 0.0)
    hwf = hw * (1.07 if female else 1.0)

    def torso_levels(grow=0.0, hem=None, top_z=None):
        g = grow / h
        L = [(Z["hip"] - 0.047 * h, 0.086 * hwf, 0.060, pel[0], -0.004, 2.2),
             (Z["pel"], 0.088 * hwf, 0.062, pel[0], -0.004, 2.2),
             (Z["waist"] - 0.028 * h, 0.080 * wf, 0.056, (pel[0] + sx) / 2, 0.004, 2.1),
             (Z["waist"] + 0.022 * h, 0.080 * wf, 0.057, sx, 0.006, 2.2),
             (Z["chest"] - 0.012 * h, 0.090 * cf * sw, 0.064 * bf, sx, 0.006 + bust, 2.4),
             (Z["chest"] + 0.038 * h, 0.094 * sw, 0.062, sx, 0.002, 2.4),
             (Z["sh"] - 0.011 * h, 0.096 * sw, 0.054, sx, -0.004, 2.4),
             (Z["sh"] + 0.012 * h, 0.07 * sw, 0.044, sx, -0.008, 2.2),
             (Z["sh"] + 0.026 * h, 0.045, 0.038, sx, -0.006, 2.0)]
        out_ = []
        for z, a, b, xc, yc, e in L:
            if hem is not None and z < hem - 1e-6:
                continue
            if top_z is not None and z > top_z + 1e-6:
                continue
            out_.append((z, (a * bulk + g) * h, (b * bulk + g) * h, xc, yc * h, e))
        if hem is not None and out_[0][0] > hem + 1e-6:
            z0, a0, b0, xc0, yc0, e0 = out_[0]
            out_.insert(0, (hem, a0, b0, xc0, yc0, e0))
        return out_
    S = {side: np.array([sx + side * 0.094 * h * sw, -0.006 * h, Z["sh"] - 0.006 * h]) for side in (-1, 1)}
    coll_body = Prims()                                     # 发丝碰撞：肩背与肩头
    coll_body.ell([sx, -0.012 * h, Z["sh"] - 0.03 * h], np.array([0.098 * sw, 0.058, 0.06]) * h * bulk)
    coll_body.ell([sx, 0.004 * h, Z["chest"]], np.array([0.09 * sw, 0.064, 0.085]) * h * bulk)
    for side in (-1, 1):
        coll_body.ell(S[side], np.array([0.036, 0.04, 0.04]) * h * bulk)
    # 腿
    Lt, Ls = Z["hip"] - Z["knee"], Z["knee"] - Z["ank"]
    hips = {side: pel + np.array([side * 0.05 * hw * h, 0.0, Z["hip"] - Z["pel"]]) for side in (-1, 1)}
    feet, pitch = {}, {}
    for side in (-1, 1):
        if stand:
            if side == s:
                A = [hips[side][0] * 0.92, -0.006 * h, Z["ank"]]
                pitch[side] = 0.0
            else:
                A = [side * 0.082 * h, 0.04 * h, Z["ank"] + 0.003 * h]
                pitch[side] = 0.05
        else:
            fwd = side == s
            A = [side * 0.052 * h, (0.17 if fwd else -0.15) * h * walk, Z["ank"] + (0.006 if fwd else 0.034 * walk) * h]
            pitch[side] = 0.22 * walk if fwd else -0.45 * walk
        feet[side] = np.array(A, float)
    knees = {side: _ik(hips[side], feet[side], Lt, Ls, [0, 1, 0]) for side in (-1, 1)}
    legs = Prims()
    for side in (-1, 1):
        H, K, A = hips[side], knees[side], feet[side]
        legs.cap(H, K, 0.058 * h * bulk * (1.05 if female else 1.0), 0.036 * h)
        legs.cap(K, A, 0.033 * h, 0.0195 * h)
        sh_dir = _unit(A - K)
        legs.ell(K + (A - K) * 0.3 - np.array([0, 0.006 * h, 0]), np.array([0.027, 0.029, 0.075]) * h,
                 _rot_to(sh_dir))
        legs.ell(K + np.array([0, 0.01 * h, 0]), np.array([0.018, 0.015, 0.02]) * h)
    # 手臂与手
    La, Lf = 0.186 * h * (0.95 if child else 1.0), 0.146 * h * (0.95 if child else 1.0)
    upper, fore, hands = Prims(), Prims(), Prims()
    wrists, elbows = {}, {}
    for side in (-1, 1):
        swing = 0.0 if stand else (-1.0 if side == s else 1.0) * 0.3 * walk
        abd = math.radians(rng.uniform(6, 11))
        d_up = _rx(swing) @ _ry(-side * abd) @ np.array([0, 0, -1.0])
        E = S[side] + d_up * La
        flex = math.radians(rng.uniform(12, 22) + (28 * walk if swing > 0 else 6 * walk))
        d_fo = _rx(swing + flex) @ _ry(-side * abd * 0.5) @ np.array([0, 0, -1.0])
        W = E + d_fo * Lf
        wrists[side], elbows[side] = W, E
        upper.cap(S[side], E, 0.033 * h * bulk, 0.026 * h * bulk)
        upper.ell(S[side] + (E - S[side]) * 0.45, np.array([0.032, 0.034, 0.06]) * h * bulk, _rot_to(E - S[side]))
        fore.cap(E, W, 0.026 * h, 0.0172 * h)
        fore.ell(E + (W - E) * 0.28, np.array([0.029, 0.027, 0.05]) * h, _rot_to(W - E))
        # 手：掌心朝向身体，拇指在前
        u = _unit(_rx(math.radians(8)) @ d_fo)
        w = _unit(np.array([-side, 0.0, 0.0]) - np.dot([-side, 0.0, 0.0], u) * u)
        v = _unit(np.cross(w, u))
        if v[1] < 0:
            v = -v
        Rp = np.stack([v, w, u], 1)
        hands.ell(W + u * 0.028 * h, np.array([0.021, 0.0095, 0.03]) * h, Rp, sub=2)
        for off, ln in ((0.0135, 0.95), (0.0045, 1.05), (-0.0045, 1.0), (-0.0135, 0.82)):
            K = W + u * 0.054 * h + v * off * h
            b1 = rng.uniform(0.25, 0.45)
            d1 = _unit(u * math.cos(b1) + w * math.sin(b1))
            Mk = K + d1 * 0.023 * h * ln
            b2 = b1 + rng.uniform(0.35, 0.55)
            d2 = _unit(u * math.cos(b2) + w * math.sin(b2))
            T = Mk + d2 * 0.018 * h * ln
            hands.cap(K, Mk, 0.0047 * h, 0.0041 * h, sub=1)
            hands.cap(Mk, T, 0.0041 * h, 0.0034 * h, sub=1)
        Tb = W + u * 0.016 * h + v * 0.015 * h + w * 0.005 * h
        Tm = Tb + _unit(u * 0.55 + v * 0.45 + w * 0.45) * 0.021 * h
        Tt = Tm + _unit(u * 0.75 + v * 0.1 + w * 0.6) * 0.016 * h
        hands.cap(Tb, Tm, 0.0064 * h, 0.0051 * h, sub=1)
        hands.cap(Tm, Tt, 0.0051 * h, 0.004 * h, sub=1)
    # 头颈
    yaw = math.radians(rng.normal(0, 9))
    pitch_h = math.radians(rng.uniform(-6, 4))
    Rh = _rz(yaw) @ _rx(pitch_h)
    s_h = h * hk
    Hc = np.array([sx * 0.6, 0.008 * h, Z["hc"]])

    def hp(x, y, z):
        return Hc + Rh @ (np.array([x, y, z]) * s_h)
    head = Prims()
    cran_c, cran_r = hp(0, -0.006, 0.014), np.array([0.046, 0.054, 0.054]) * s_h
    head.ell(cran_c, cran_r, Rh)
    for c, r in (((0, 0.022, -0.02), (0.039, 0.037, 0.046)), ((0, 0.03, -0.05), (0.027, 0.022, 0.017)),
                 ((0, 0.047, 0.012), (0.031, 0.011, 0.008)), ((0, 0.056, -0.008), (0.0065, 0.009, 0.016)),
                 ((0, 0.062, -0.022), (0.0085, 0.0085, 0.0075)), ((0, 0.052, -0.038), (0.015, 0.008, 0.0065)),
                 ((0.0445, -0.006, -0.008), (0.005, 0.0095, 0.0175)), ((-0.0445, -0.006, -0.008), (0.005, 0.0095, 0.0175))):
        head.ell(hp(*c), np.array(r) * s_h, Rh, sub=2 if r[0] < 0.02 else 3)
    neck_top = hp(0, -0.008, -0.04)
    head.cap([sx * 0.7, -0.006 * h, Z["neck"]], neck_top, 0.031 * h, 0.029 * h)
    # ---------------------------------------------------------------- 衣物
    top_kind = {"man_shirt": "shirt", "man_coat": "coat", "woman_dress": "dress", "woman_trousers": "shirt",
                "child": "shirt"}[kind]
    short_sleeve = top_kind == "dress" or (top_kind == "shirt" and rng.random() < 0.45)
    bottom_kind = {"man_shirt": "trousers", "man_coat": "trousers", "woman_dress": "skirt",
                   "woman_trousers": "trousers", "child": "shorts" if rng.random() < 0.5 else "trousers"}[kind]
    grow = {"shirt": 0.007, "coat": 0.016, "dress": 0.005}[top_kind] * h
    top = Prims()
    if top_kind == "coat":
        lv = torso_levels(grow)
        z_h, a_h, b_h, xc_h, yc_h, e_h = lv[0]
        lv = [(0.355 * h, a_h * 1.16, b_h * 1.2, xc_h, yc_h, e_h), (0.41 * h, a_h * 1.1, b_h * 1.14, xc_h, yc_h, e_h)] + lv
    elif top_kind == "dress":
        lv = torso_levels(grow, hem=Z["waist"] - 0.01 * h)
    else:
        lv = torso_levels(grow)
    top.items.append(("mesh",) + _loft(lv))
    if female:
        for side in (-1, 1):
            top.ell([sx + side * 0.031 * h, (0.062 * bulk + grow / h) * h, Z["chest"] - 0.008 * h],
                    np.array([0.03, 0.013, 0.025]) * h)
    for side in (-1, 1):
        E, W = elbows[side], wrists[side]
        if short_sleeve:
            top.cap(S[side] - [side * 0.012 * h, 0, 0.012 * h], S[side] + (E - S[side]) * 0.45,
                    0.033 * h * bulk + grow, 0.029 * h * bulk + grow)
        else:
            top.cap(S[side] - [side * 0.012 * h, 0, 0.012 * h], E, 0.031 * h * bulk + grow * 0.7,
                    0.025 * h * bulk + grow * 0.7)
            top.cap(E, E + (W - E) * 0.9, 0.025 * h + grow * 0.7, 0.0195 * h + grow * 0.7)
    bottom = Prims()
    if bottom_kind in ("trousers", "shorts"):
        g = 0.006 * h
        lv = [(Z["hip"] - 0.05 * h, 0.084 * hwf * h * bulk + g, 0.058 * h * bulk + g, pel[0], -0.004 * h, 2.2),
              (Z["hip"] - 0.03 * h, 0.082 * hwf * h * bulk, 0.056 * h * bulk, pel[0], -0.004 * h, 2.2),
              (Z["pel"], 0.078 * hwf * h * bulk, 0.054 * h * bulk, pel[0], -0.004 * h, 2.2),
              (Z["waist"] - 0.01 * h, 0.066 * wf * h * bulk, 0.046 * h * bulk, (pel[0] + sx) / 2, 0.004 * h, 2.1)]
        bottom.items.append(("mesh",) + _loft(lv))
        for side in (-1, 1):
            H, K, A = hips[side], knees[side], feet[side]
            H0 = H + np.array([-side * 0.012 * h, 0.0, -0.035 * h])      # 大腿顶端收在上衣以内
            if bottom_kind == "trousers":
                bottom.cap(H0, K, 0.05 * h * bulk, 0.042 * h)
                bottom.cap(K, A + [0, 0, 0.012 * h], 0.04 * h, 0.031 * h)
            else:
                bottom.cap(H0, H + (K - H) * 0.62, 0.052 * h * bulk, 0.05 * h)
    else:
        hem = 0.33 * h if rng.random() < 0.6 else 0.25 * h
        v, f = lathe([(0, Z["waist"] + 0.01 * h), (0.078 * h, Z["waist"] + 0.01 * h), (0.092 * h, Z["hip"] + 0.02 * h),
                      (0.11 * h, (Z["hip"] + hem) / 2), (0.128 * h, hem + 0.006 * h), (0.126 * h, hem), (0, hem)], 36)
        v[:, 1] *= 0.8
        v += [pel[0], 0, 0]
        bottom.items.append(("mesh", v, np.asarray(f)))
    # 露出的皮肤：脸与手（轻平滑，保留五官与手指）、四肢（重平滑，去掉分段感）
    skin, limbs = Prims(), Prims()
    skin.extend(head)
    skin.extend(hands)
    if short_sleeve:
        limbs.extend(fore)
        for side in (-1, 1):
            E = elbows[side]
            limbs.cap(S[side] + (E - S[side]) * 0.35, E, 0.03 * h * bulk, 0.026 * h)
    else:
        for side in (-1, 1):
            W = wrists[side]
            skin.cap(W - _unit(W - elbows[side]) * 0.03 * h, W, 0.019 * h, 0.0172 * h)
    if bottom_kind in ("skirt", "shorts"):
        for side in (-1, 1):
            H, K, A = hips[side], knees[side], feet[side]
            limbs.cap(H + (K - H) * (0.45 if bottom_kind == "shorts" else 0.3), K, 0.05 * h, 0.036 * h)
        limbs.extend(legs, 0, keep=lambda it: it[0] == "cap" and it[1][2] < 0.4 * h or it[0] == "ell")
    # 鞋
    shoe = Prims()
    for side in (-1, 1):
        A = feet[side]
        Rf = _rz(-side * math.radians(rng.uniform(4, 10))) @ _rx(pitch[side])
        for c, r in (((0, -0.022, -0.02), (0.026, 0.034, 0.022)), ((0, 0.04, -0.023), (0.03, 0.05, 0.019)),
                     ((0, 0.09, -0.028), (0.027, 0.03, 0.014)), ((0, 0.0, 0.0), (0.024, 0.027, 0.02))):
            shoe.ell(A + Rf @ (np.array(c) * h), np.array(r) * h, Rf)
    # ---------------------------------------------------------------- 网格
    vx = 0.0045 * h / 1.72 / detail
    out = {}
    V, F = _solidify(skin, vx, 3, decimate=0.36)
    if limbs.items:
        V2, F2 = _solidify(limbs, vx * 1.2, 9, decimate=0.2)
        F = np.vstack([F, F2 + len(V)]); V = np.vstack([V, V2])
    for side in (-1, 1):                                          # 眼窝、口线
        V = _dent(V, hp(side * 0.0175, 0.05, -0.002), 0.012 * s_h, 0.0028 * s_h, Rh @ [0, 1, 0])
    V = _dent(V, hp(0, 0.058, -0.036), 0.007 * s_h, 0.0012 * s_h, Rh @ [0, 1, 0])
    out["skin"] = (V, F)
    cloth_vx = 0.0075 * h / 1.72 / detail
    out["top"] = _solidify(_meshify(top), cloth_vx, 4, folds=0.006 * h / 1.72, decimate=0.22, seed=seed)
    out["bottom"] = _solidify(_meshify(bottom), cloth_vx, 4,
                              folds=(0.012 if bottom_kind == "skirt" else 0.007) * h / 1.72, decimate=0.22,
                              seed=seed + 7)
    out["shoe"] = _solidify(shoe, 0.006 * h / 1.72 / detail, 4, decimate=0.25)
    # 包：部分成年人斜挎包
    if not child and rng.random() < 0.35:
        side = 1 if rng.random() < 0.5 else -1
        bag = Prims()
        bc = pel + np.array([side * 0.115 * h, 0.01 * h, -0.02 * h])
        bag.ell(bc, np.array([0.022, 0.07, 0.055]) * h, _rz(0.0))
        sh_o = S[-side] + np.array([0, 0, 0.03 * h])
        mid = np.array([sx, 0.07 * h * bulk, Z["chest"]])
        bag.cap(sh_o, mid, 0.004 * h, 0.004 * h, sub=1)
        bag.cap(mid, bc + [0, 0, 0.05 * h], 0.004 * h, 0.004 * h, sub=1)
        out["bag"] = _solidify(bag, 0.006 * h / 1.72 / detail, 2, decimate=0.3)
    # ---------------------------------------------------------------- 头发
    style = _hair_style(kind, rng)
    colliders = Prims()
    colliders.extend(head, 0.0, keep=lambda it: it[0] == "cap")
    colliders.extend(coll_body, grow + 0.004 * h)
    strands, cap_mesh, proxy = _hair(style, cran_c, cran_r, Rh, Hc, s_h, colliders, rng, detail)
    out["strands"] = strands
    out["hair_cap"] = cap_mesh
    out["hair_proxy"] = proxy
    out["style"] = style
    return out


def _tris(F):
    F = np.asarray(F)
    if F.shape[1] == 3:
        return F
    return np.vstack([F[:, [0, 1, 2]], F[:, [0, 2, 3]]])


def _meshify(prims):
    """Prims 中混入的 ('mesh', V, F)（如裙、外套下摆的旋转体）直接并入。"""
    p = Prims()
    p.items = [it for it in prims.items if it[0] != "mesh"]
    extra = [(it[1], _tris(it[2])) for it in prims.items if it[0] == "mesh"]
    base = p.meshes
    p.meshes = lambda: base() + extra
    return p


# ------------------------------------------------------------------ 头发
def _hair_style(kind, rng):
    if kind.startswith("man"):
        return "buzz" if rng.random() < 0.15 else "short"
    if kind == "child":
        return ["short", "bob", "ponytail"][int(rng.integers(0, 3))]
    return ["long", "long", "bob", "ponytail", "bun"][int(rng.integers(0, 5))]


def _scalp_mask(q, style):
    """q：单位球方向（头部局部，y 前 z 上）。返回是否在发际线以内。"""
    front = (q[:, 1] > 0.18) & (q[:, 2] < 0.42 + 0.3 * np.clip(q[:, 1] - 0.18, 0, 1))
    ears = (np.abs(q[:, 0]) > 0.62) & (q[:, 2] < 0.08) & (q[:, 1] > -0.42)
    nape = q[:, 2] < (-0.5 if style in ("long", "bob") else -0.42)
    return ~(front | ears | nape)


def _hair(style, cc, cr, Rh, Hc, s_h, colliders, rng, detail):
    """发丝：先在头皮上撒导向发丝并按发型生长，再在每根导向发丝周围生成子发丝（向发梢聚拢）。"""
    n_guide = int({"buzz": 260, "short": 300, "bob": 320, "long": 340, "ponytail": 300, "bun": 280}[style] * detail)
    n_child = {"buzz": 6, "short": 8, "bob": 10, "long": 11, "ponytail": 10, "bun": 9}[style]
    if detail < 1:
        n_child = max(4, int(n_child * detail))
    # 头皮采样（单位球方向 → 椭球）
    q = rng.normal(0, 1, (n_guide * 6, 3))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    q = q[_scalp_mask(q, style)][:n_guide]

    def on_scalp(qq, lift=0.0):
        return cc + (np.asarray(qq) * (cr + lift)) @ Rh.T

    def normal(qq):
        n_ = (np.asarray(qq) / cr) @ Rh.T
        return n_ / np.linalg.norm(n_, axis=-1, keepdims=True)

    def reproject(p, lift):
        lq = Rh.T @ (p - cc) / (cr + lift)
        return cc + Rh @ (lq / np.linalg.norm(lq) * (cr + lift))

    up = np.array([0, 0, 1.0])
    back = Rh @ np.array([0, -1.0, 0])
    whorl_q = _unit([0.08, -0.5, 0.86])
    whorl = on_scalp(whorl_q)
    tie = Hc + Rh @ (np.array([0, -0.06, 0.004]) * s_h)
    bun_c = Hc + Rh @ (np.array([0, -0.052, 0.036]) * s_h)
    guides = []
    L_of = {"buzz": (0.008, 0.014), "short": (0.035, 0.065), "bob": (0.13, 0.17), "long": (0.27, 0.38),
            "ponytail": (0.0, 0.0), "bun": (0.0, 0.0)}
    for qi in q:
        p0 = on_scalp(qi, 0.0015)
        n0 = normal(qi)
        pts = [p0]
        if style in ("buzz", "short"):
            L = rng.uniform(*L_of[style]); nseg = 6
            p = p0.copy()
            for k in range(nseg):
                t = (k + 1) / nseg
                flow = p - whorl
                flow = _unit(flow - np.dot(flow, n0) * n0) if np.linalg.norm(flow) > 1e-4 else back
                if Rh.T.dot(p - cc)[1] > 0.3 * cr[1]:              # 前额上方：向前略偏一侧
                    flow = _unit(flow + Rh @ np.array([0.35, 0.6, 0.0]))
                p = p + _unit(flow - up * 0.15) * (L / nseg)
                p = reproject(p, 0.0015 + (0.004 if style == "buzz" else 0.009) * t)
                n0 = _unit(p - cc)
                pts.append(p)
        elif style in ("long", "bob"):
            L = rng.uniform(*L_of[style]); nseg = 12 if style == "long" else 9
            p = p0.copy()
            lq0 = Rh.T.dot(p - cc) / cr
            side = np.sign(lq0[0]) or 1.0
            fr = float(np.clip((lq0[1] + 0.25) / 0.7, 0, 1))         # 前部发丝向两侧分开，后部直接下垂
            d = _unit(Rh @ np.array([side * (0.12 + 0.5 * fr), -0.3 - 0.2 * (1 - fr), -0.25 - 0.4 * (1 - fr)]))
            for k in range(nseg):
                t = (k + 1) / nseg
                lq = Rh.T @ (p - cc)
                above = lq[2] > -0.15 * cr[2]
                g = 0.15 if above else 0.55 + 0.6 * t
                d = _unit(d + np.array([0, 0, -g]) + Rh @ np.array([side * 0.06 * fr, -0.06, 0]))
                p = p + d * (L / nseg)
                if above:
                    p = reproject(p, 0.002 + 0.01 * t)
                p = colliders.push_out(p, 0.004)
                lq = Rh.T @ (p - cc) / cr
                if np.linalg.norm(lq) < 1.04:
                    p = cc + Rh @ (lq / np.linalg.norm(lq) * 1.04 * cr)
                pts.append(p)
            if style == "bob":                                   # 发梢内扣
                pts[-1] = pts[-1] + _unit(Hc - pts[-1]) * 0.012
        else:                                                    # 马尾 / 发髻：向扎点梳拢
            target = tie if style == "ponytail" else bun_c
            p = p0.copy()
            for k in range(10):
                to = target - p
                if np.linalg.norm(to) < 0.01:
                    break
                n_ = _unit(p - cc)
                step = _unit(to - np.dot(to, n_) * n_ * 0.6)
                p = p + step * min(0.02, np.linalg.norm(to))
                if np.linalg.norm(target - p) > 0.02:
                    p = reproject(p, 0.0025)
                pts.append(p)
        if len(pts) >= 3:
            guides.append(np.array(pts))
    if style == "ponytail":                                      # 马尾：自扎点下垂
        for _ in range(int(140 * max(detail, 0.5))):
            p = tie + rng.normal(0, 0.006, 3)
            d = _unit(back * 0.8 - up * 0.6 + rng.normal(0, 0.15, 3))
            L = rng.uniform(0.17, 0.26); nseg = 12
            pts = [p.copy()]
            for k in range(nseg):
                t = (k + 1) / nseg
                d = _unit(d + np.array([0, 0, -0.35 - 0.6 * t]))
                p = p + d * (L / nseg)
                p = colliders.push_out(p, 0.004)
                pts.append(p.copy())
            guides.append(np.array(pts))
    # 子发丝
    strands = []
    for g in guides:
        n_ = len(g)
        t = np.linspace(0, 1, n_)[:, None]
        for _ in range(n_child):
            off = rng.normal(0, 0.0055, 3)
            clump = rng.uniform(0.35, 0.75)
            frizz = rng.normal(0, 0.0012, (n_, 3)) * t
            c = g + off * (1 - clump * t) + frizz
            c[0] = g[0] + off * 0.5
            strands.append(c)
        strands.append(g)
    # 发壳：头皮范围内的椭球面（略低于发根），与发丝同色
    V, F = _ico(4)
    keep = _scalp_mask(V, style)
    face_keep = keep[F].all(axis=1)
    Vc = cc + (V * (cr + 0.0008)) @ Rh.T
    cap_mesh = (Vc, F[face_keep])
    # 网页用的简化发型（不渲染）
    proxy = Prims()
    proxy.ell(cc, cr * 1.07, Rh, sub=2)
    if style == "long":
        proxy.ell(Hc + Rh @ (np.array([0, -0.03, -0.09]) * s_h), np.array([0.058, 0.035, 0.11]) * s_h, Rh, sub=2)
    elif style == "bob":
        proxy.ell(Hc + Rh @ (np.array([0, -0.012, -0.03]) * s_h), np.array([0.055, 0.058, 0.06]) * s_h, Rh, sub=2)
    elif style == "ponytail":
        proxy.cap(tie, tie + back * 0.06 - up * 0.17, 0.018, 0.01, sub=1)
    elif style == "bun":
        proxy.ell(bun_c, np.array([0.03, 0.028, 0.027]) * s_h, Rh, sub=2)
    if style == "bun":                                           # 发髻本体（渲染，与发壳同材质）
        bun = Prims(); bun.ell(bun_c + back * 0.008, np.array([0.03, 0.026, 0.026]) * s_h, Rh, sub=3)
        Vb, Fb = bun.meshes()[0]
        cap_mesh = (np.vstack([cap_mesh[0], Vb]), np.vstack([cap_mesh[1], Fb + len(cap_mesh[0])]))
    pm = proxy.meshes()
    Vp = np.vstack([m[0] for m in pm])
    Fp = np.vstack([m[1] + sum(len(x[0]) for x in pm[:i]) for i, m in enumerate(pm)])
    return strands, cap_mesh, (Vp, Fp)


# ------------------------------------------------------------------ 场景
def add_hair_curves(name, coll, mat, strands, r_root=0.00032, r_tip=0.00007):
    if not strands:
        return None
    hc = bpy.data.hair_curves.new(name)
    sizes = [len(s) for s in strands]
    hc.add_curves(sizes)
    pos = np.concatenate(strands).astype(np.float32)
    hc.position_data.foreach_set("vector", pos.ravel())
    rad = np.concatenate([np.linspace(r_root, r_tip, n) for n in sizes]).astype(np.float32)
    ra = hc.attributes.new("radius", "FLOAT", "POINT")
    ra.data.foreach_set("value", rad)
    hc.materials.append(mat)
    ob = bpy.data.objects.new(name, hc)
    coll.objects.link(ob)
    return ob


def cleanup():
    if "empty" in _FOLD:
        bpy.data.objects.remove(_FOLD["empty"])
        _FOLD.clear()
