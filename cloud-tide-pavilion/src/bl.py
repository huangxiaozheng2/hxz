"""Blender 低层工具：numpy → 网格、扫掠、集合管理。"""
import math
import numpy as np
import bpy


# ------------------------------------------------------------------ 集合
def collection(name, parent=None):
    c = bpy.data.collections.get(name)
    if c is None:
        c = bpy.data.collections.new(name)
        (parent or bpy.context.scene.collection).children.link(c)
    return c


# ------------------------------------------------------------------ 网格
def mesh_object(name, verts, faces, coll, mats=None, mat_idx=None, smooth=True, uv=None,
                loop_uv=None, location=(0, 0, 0)):
    """verts: (N,3)；faces: 等长整数数组 (F,k) 或 list[list]；uv: (N,2) 逐顶点。"""
    verts = np.asarray(verts, np.float32).reshape(-1, 3)
    if isinstance(faces, np.ndarray) and faces.ndim == 2:
        sizes = np.full(len(faces), faces.shape[1], np.int32)
        flat = faces.reshape(-1).astype(np.int32)
    else:
        sizes = np.array([len(f) for f in faces], np.int32)
        flat = np.fromiter((i for f in faces for i in f), np.int32, int(sizes.sum()))
    me = bpy.data.meshes.new(name)
    me.vertices.add(len(verts))
    me.vertices.foreach_set("co", verts.ravel())
    me.loops.add(len(flat))
    me.loops.foreach_set("vertex_index", flat)
    me.polygons.add(len(sizes))
    starts = np.concatenate([[0], np.cumsum(sizes)[:-1]]).astype(np.int32)
    me.polygons.foreach_set("loop_start", starts)
    if mat_idx is not None:
        me.polygons.foreach_set("material_index", np.asarray(mat_idx, np.int32))
    if uv is not None or loop_uv is not None:
        lay = me.uv_layers.new(name="UVMap")
        if loop_uv is None:
            loop_uv = np.asarray(uv, np.float32)[flat]
        lay.data.foreach_set("uv", np.asarray(loop_uv, np.float32).ravel())
    me.update(calc_edges=True)
    me.validate(clean_customdata=False)
    if smooth:
        me.polygons.foreach_set("use_smooth", np.ones(len(me.polygons), bool))
    ob = bpy.data.objects.new(name, me)
    ob.location = location
    coll.objects.link(ob)
    for m in (mats or []):
        me.materials.append(m)
    return ob


class MeshBuf:
    """把许多小构件合并为一个网格的缓冲区。"""

    def __init__(self):
        self.v = []; self.f = []; self.m = []; self.uv = []; self.n = 0

    def add(self, verts, faces, mat=0, uv=None):
        verts = np.asarray(verts, float).reshape(-1, 3)
        self.v.append(verts)
        if uv is None:
            uv = np.zeros((len(verts), 2))
        self.uv.append(np.asarray(uv, float))
        for f in faces:
            self.f.append([i + self.n for i in f])
            self.m.append(mat)
        self.n += len(verts)

    def add_np(self, verts, faces, mat=0, uv=None):
        verts = np.asarray(verts, float).reshape(-1, 3)
        faces = np.asarray(faces) + self.n
        self.v.append(verts)
        self.uv.append(np.zeros((len(verts), 2)) if uv is None else np.asarray(uv, float))
        self.f.extend(faces.tolist())
        self.m.extend([mat] * len(faces))
        self.n += len(verts)

    def build(self, name, coll, mats, smooth=True):
        if not self.v:
            return None
        return mesh_object(name, np.vstack(self.v), self.f, coll, mats, self.m, smooth,
                           uv=np.vstack(self.uv))


# ------------------------------------------------------------------ 扫掠
def frames_along(path, closed=False, up=(0, 0, 1)):
    """沿 3D 折线的水平法向（左侧）与切向。"""
    p = np.asarray(path, float)
    if closed:
        tan = np.roll(p, -1, 0) - np.roll(p, 1, 0)
    else:
        tan = np.gradient(p, axis=0)
    th = tan.copy(); th[:, 2] = 0
    th /= np.linalg.norm(th, axis=1, keepdims=True) + 1e-12
    nrm = np.stack([-th[:, 1], th[:, 0], np.zeros(len(p))], 1)  # 左手侧水平法向
    tan /= np.linalg.norm(tan, axis=1, keepdims=True) + 1e-12
    return tan, nrm


def sweep(path, profile, closed=False, cap=True, nrm=None):
    """沿 3D 路径扫掠 2D 截面。profile: (K,2) 为 (水平法向偏移, 竖向偏移)，逆时针闭合。
    返回 verts, faces, uv。"""
    p = np.asarray(path, float)
    prof = np.asarray(profile, float)
    if nrm is None:
        _, nrm = frames_along(p, closed)
    N, K = len(p), len(prof)
    V = p[:, None, :] + nrm[:, None, :] * prof[None, :, 0:1] + np.array([0, 0, 1.0]) * prof[None, :, 1:2]
    V = V.reshape(-1, 3)
    faces = []
    segs = N if closed else N - 1
    for i in range(segs):
        j = (i + 1) % N
        for k in range(K):
            k2 = (k + 1) % K
            faces.append([i * K + k, i * K + k2, j * K + k2, j * K + k])
    if cap and not closed:
        faces.append(list(range(K))[::-1])
        faces.append([(N - 1) * K + k for k in range(K)])
    s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))])
    uv = np.stack([np.repeat(s, K), np.tile(np.linspace(0, 1, K), N)], 1)
    return V, faces, uv


def tube(path, radius, sides=10, closed=False, cap=True):
    """圆管（半径可逐点变化）。"""
    p = np.asarray(path, float)
    N = len(p)
    r = np.broadcast_to(np.asarray(radius, float), (N,))
    if closed:
        tan = np.roll(p, -1, 0) - np.roll(p, 1, 0)
    else:
        tan = np.gradient(p, axis=0) if N > 2 else np.repeat((p[1] - p[0])[None], N, 0)
    tan /= np.linalg.norm(tan, axis=1, keepdims=True) + 1e-12
    # 平行传输标架
    ref = np.array([0, 0, 1.0]) if abs(tan[0, 2]) < 0.9 else np.array([1.0, 0, 0])
    n0 = np.cross(tan[0], ref); n0 /= np.linalg.norm(n0)
    ns = [n0]
    for i in range(1, N):
        n = ns[-1] - np.dot(ns[-1], tan[i]) * tan[i]
        n /= np.linalg.norm(n) + 1e-12
        ns.append(n)
    ns = np.array(ns); bs = np.cross(tan, ns)
    a = np.linspace(0, 2 * np.pi, sides, endpoint=False)
    V = p[:, None, :] + r[:, None, None] * (np.cos(a)[None, :, None] * ns[:, None, :] + np.sin(a)[None, :, None] * bs[:, None, :])
    V = V.reshape(-1, 3)
    faces = []
    segs = N if closed else N - 1
    for i in range(segs):
        j = (i + 1) % N
        for k in range(sides):
            k2 = (k + 1) % sides
            faces.append([i * sides + k, i * sides + k2, j * sides + k2, j * sides + k])
    if cap and not closed:
        faces.append(list(range(sides))[::-1])
        faces.append([(N - 1) * sides + k for k in range(sides)])
    s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))])
    uv = np.stack([np.tile(np.linspace(0, 1, sides), N), np.repeat(s, sides)], 1)
    return V, faces, uv


def lathe(profile, sides=48, center=(0, 0, 0)):
    """绕 z 轴旋转 (r,z) 轮廓。"""
    prof = np.asarray(profile, float)
    a = np.linspace(0, 2 * np.pi, sides, endpoint=False)
    K = len(prof)
    V = np.zeros((sides, K, 3))
    V[:, :, 0] = np.cos(a)[:, None] * prof[None, :, 0]
    V[:, :, 1] = np.sin(a)[:, None] * prof[None, :, 0]
    V[:, :, 2] = prof[None, :, 1]
    V = V.reshape(-1, 3) + np.asarray(center)
    faces = []
    for i in range(sides):
        j = (i + 1) % sides
        for k in range(K - 1):
            faces.append([i * K + k, j * K + k, j * K + k + 1, i * K + k + 1])
    return V, faces


def box(c, size, rot_z=0.0):
    sx, sy, sz = np.asarray(size) / 2
    v = np.array([[-sx, -sy, -sz], [sx, -sy, -sz], [sx, sy, -sz], [-sx, sy, -sz],
                  [-sx, -sy, sz], [sx, -sy, sz], [sx, sy, sz], [-sx, sy, sz]])
    cz, sz_ = math.cos(rot_z), math.sin(rot_z)
    R = np.array([[cz, -sz_, 0], [sz_, cz, 0], [0, 0, 1]])
    v = v @ R.T + np.asarray(c)
    f = [[0, 3, 2, 1], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]]
    return v, f


def oriented_box(p0, p1, w, d, nrm):
    """以 p0→p1 为轴（竖直或任意），w 沿 nrm 的垂直方向，d 沿 nrm。"""
    p0 = np.asarray(p0, float); p1 = np.asarray(p1, float)
    ax = p1 - p0; ax /= np.linalg.norm(ax)
    n = np.asarray(nrm, float); n = n - np.dot(n, ax) * ax; n /= np.linalg.norm(n)
    t = np.cross(ax, n)
    hw, hd = w / 2, d / 2
    corners = [(-hw, -hd), (hw, -hd), (hw, hd), (-hw, hd)]
    V = [p + t * a + n * b for p in (p0, p1) for a, b in corners]
    f = [[0, 3, 2, 1], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]]
    return np.array(V), f
