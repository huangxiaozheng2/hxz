"""AR 模型：1:100 桌面模型。圆形底座（水面 + 胡桃木边框，直径约 56 cm），栈桥在水面边缘截断并补上剖切面。
供手机 AR 查看：iPhone 用 USDZ（AR Quick Look），安卓用 GLB（Scene Viewer / WebXR），网页用 model-viewer。
只保留建筑、庭院、室内陈设与底座范围内的人物；去掉 LED 灯带、发丝曲线与湖岸配景。
程序化材质换成同色的基础 PBR；玻璃改为半透明（AR 查看器不做折射）。
屋顶、屋面平台、观景厅等上部构件改用名称以 "Up_" 开头的材质副本，网页把这些材质设为全透明即可“掀开屋顶”看首层。
单位米（已按 1:100 缩小），原点在底座底面中心；glTF 的 +Z（模型正面）朝南，即栈桥一侧。
用法：python export_ar.py scene.blend 输出前缀    → 前缀.glb、前缀.usdz、前缀.json（网页标注点的模型坐标）
"""
import glob
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bpy  # noqa: E402
import bmesh  # noqa: E402
import numpy as np  # noqa: E402
import shapely  # noqa: E402
from mathutils import Matrix, Vector  # noqa: E402
from scipy.sparse import coo_matrix  # noqa: E402
from scipy.sparse.csgraph import connected_components  # noqa: E402
from shapely.geometry import MultiLineString, Point, box  # noqa: E402
from shapely.ops import polygonize, unary_union  # noqa: E402

import plan as P  # noqa: E402
from export_glb import decimated, thin_foliage  # noqa: E402

SCALE = 0.01                      # 1:100
R = 27.0                          # 水面半径（实际尺度，m）
RIM_W, RIM_UP, BASE_T = 1.0, 0.25, 1.8     # 木边框宽、高出水面、水面以下的底座厚度（包住 -1.7 m 的池底与墩墙）
KEEP = {"Architecture", "Courtyard", "Interior_Exhibits", "Terrace_Furniture", "People_ScaleFigures"}
DROP = ("LED", "Glow", "HairStrands")
UPPER = ("Roof_", "Deck_", "Envelope_E3_", "MoonGate_", "Terrace_Sofa_")
SPLIT_LOOSE = ("Supports_Columns", "Supports_RingBeams_Steel")   # 按构件底标高分上下
CORE_CUT = ("LiftCore_", "EgressStair_Walls", "EgressStair_FireDoors")   # 在屋面平台标高处切开
PEOPLE_RATIO = 0.08
DECIMATE = (("SoffitRibs", 0.25), ("PrimaryBeams", 0.4), ("Balustrade_Rail", 0.25), ("Mullions_Frames", 0.6),
            ("Ramp_Slab", 0.5), ("Columns", 0.5), ("EdgeTrim", 0.6), ("_Shell", 0.6), ("Sculpture_FrostedGlass", 0.4),
            ("TidalBasin_Steps", 0.6), ("Courtyard_Tree_Branches", 0.5))     # 1:100 下看不出的细部：按构件减面
FONT = "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"
NAMEPLATE = "云上潮汐馆 1:100"
HOTSPOTS = {                      # 网页标注点（实际尺度的设计坐标 x, y, z）
    "bridge": (0.3, -23.5, 1.0), "lobby": (1.0, -12.6, 2.6), "pool": (2.2, -1.2, 0.35),
    "ramp": (P.RAMP_R * math.cos(math.radians(300)), P.RAMP_R * math.sin(math.radians(300)),
             float(P.ramp_z_of_deg(300)) + 0.6),
    "gallery": (5.0, 14.6, 2.8), "ystrut": (-11.6, -10.4, 3.2),
    "deck": (8.6, 7.4, 7.0), "moon": (-9.0, 7.9, 8.2), "belvedere": (0.0, 11.0, 8.6),
}
UPPER_SPOTS = {"deck", "moon", "belvedere"}


def lin(c):
    """sRGB 0–1 → 线性。"""
    return tuple(((v + 0.055) / 1.055) ** 2.4 if v > 0.04045 else v / 12.92 for v in c)


def in_keep(o):
    if o.type != "MESH" or any(k in o.name for k in DROP):
        return False
    names = {c.name for c in o.users_collection}
    for c in bpy.data.collections:
        if any(o.name in ch.objects for ch in c.children_recursive):
            names.add(c.name)
    return bool(names & KEEP)


def bake(o, dg):
    """修改器与物体变换烘进网格，物体变换归零，此后所有坐标都是世界坐标。"""
    me = bpy.data.meshes.new_from_object(o.evaluated_get(dg))
    mw = o.matrix_world.copy()
    me.transform(mw)
    if mw.determinant() < 0:
        me.flip_normals()
    o.modifiers.clear()
    o.data = me
    o.parent = None
    o.matrix_world = Matrix.Identity(4)


def arrays(me):
    co = np.empty(len(me.vertices) * 3)
    me.vertices.foreach_get("co", co)
    ed = np.empty(len(me.edges) * 2, np.int64)
    me.edges.foreach_get("vertices", ed)
    ls = np.empty(len(me.polygons), np.int64)
    me.polygons.foreach_get("loop_start", ls)
    lv = np.empty(len(me.loops), np.int64)
    me.loops.foreach_get("vertex_index", lv)
    return co.reshape(-1, 3), ed.reshape(-1, 2), ls, lv


def components(me):
    """每个面所属的松散部件编号，以及各部件的顶点包围盒（xmin, ymin, zmin, xmax, ymax, zmax）。"""
    co, ed, ls, lv = arrays(me)
    n = len(co)
    if n == 0:
        return np.zeros(0, int), np.zeros((0, 6))
    g = coo_matrix((np.ones(len(ed)), (ed[:, 0], ed[:, 1])), shape=(n, n))
    k, lab = connected_components(g, directed=False)
    face_lab = lab[lv[ls]] if len(ls) else np.zeros(0, int)
    lo = np.full((k, 3), np.inf)
    hi = np.full((k, 3), -np.inf)
    np.minimum.at(lo, lab, co)
    np.maximum.at(hi, lab, co)
    return face_lab, np.hstack([lo, hi])


def delete_faces(me, mask):
    if not mask.any():
        return
    bm = bmesh.new()
    bm.from_mesh(me)
    bm.faces.ensure_lookup_table()
    bmesh.ops.delete(bm, geom=[bm.faces[i] for i in np.nonzero(mask)[0]], context="FACES")
    loose = [v for v in bm.verts if not v.link_faces]
    bmesh.ops.delete(bm, geom=loose, context="VERTS")
    bm.to_mesh(me)
    bm.free()


def face_centers(me):
    c = np.empty(len(me.polygons) * 3)
    me.polygons.foreach_get("center", c)
    return c.reshape(-1, 3)


def cut_plane():
    """栈桥中线与水面圆的交点处、与圆相切的竖直切面（外法线 n、到圆心距离 R）。"""
    inter = P.BRIDGE.intersection(Point(0, 0).buffer(R, 512).exterior)
    c = inter.centroid
    ang = math.atan2(c.y, c.x)
    return Vector((math.cos(ang), math.sin(ang), 0.0)), ang


def crop(o, n):
    """切掉底座以外的部分：人物按整个人（松散部件）取舍；其余构件用切面切开（栈桥、栏杆），远处的零碎直接删。"""
    me = o.data
    if o.name.startswith("People_"):
        lab, bb = components(me)
        cx, cy = (bb[:, 0] + bb[:, 3]) / 2, (bb[:, 1] + bb[:, 4]) / 2
        out = (np.hypot(cx, cy) > R - 0.5) | (cx * n.x + cy * n.y > R - 0.6)
        delete_faces(me, out[lab])
        return int(out.sum())
    co = arrays(me)[0]
    if len(co) == 0 or (np.hypot(co[:, 0], co[:, 1]).max() <= R and (co[:, :2] @ (n.x, n.y)).max() <= R):
        return 0
    bm = bmesh.new()
    bm.from_mesh(me)
    bmesh.ops.bisect_plane(bm, geom=bm.verts[:] + bm.edges[:] + bm.faces[:], plane_co=n * R, plane_no=n,
                           clear_outer=True)
    far = [f for f in bm.faces if math.hypot(*f.calc_center_median().xy) > R + 1.5]
    bmesh.ops.delete(bm, geom=far, context="FACES")
    bmesh.ops.delete(bm, geom=[v for v in bm.verts if not v.link_faces], context="VERTS")
    bm.to_mesh(me)
    bm.free()
    return 1


def section_cap(objs, n):
    """栈桥剖切面：收集落在切面上的边，在（横向 u, 高度 z）平面里拼成截面，只取水面以上部分。
    下部两道墩墙在切面上是闭合的轮廓，直接成面；桥面板只有面层和薄封边（不闭合），按其切点的包络矩形算实心。"""
    t = Vector((-n.y, n.x, 0.0))
    shapes = []
    for o in objs:
        co, ed, _, _ = arrays(o.data)
        if len(co) == 0:
            continue
        on = np.abs(co[:, :2] @ (n.x, n.y) - R) < 2e-3
        if on.sum() < 2:
            continue
        uz = np.stack([co[:, :2] @ (t.x, t.y), co[:, 2]], 1)
        m = on[ed[:, 0]] & on[ed[:, 1]]
        segs = [(tuple(uz[a]), tuple(uz[b])) for a, b in ed[m] if np.linalg.norm(uz[a] - uz[b]) > 1e-5]
        polys = list(polygonize(unary_union(MultiLineString(segs)))) if segs else []
        if sum(p.area for p in polys) > 0.3:
            shapes.extend(polys)
        else:
            a = uz[on]
            b = box(a[:, 0].min(), a[:, 1].min(), a[:, 0].max(), a[:, 1].max())
            if b.area > 1e-3:
                shapes.append(b)
    shape = unary_union(shapes).buffer(0.002, join_style="mitre").buffer(-0.002, join_style="mitre")
    shape = shape.intersection(box(-50, P.HIGH_TIDE - 0.02, 50, 50))
    parts = [shape] if shape.geom_type == "Polygon" else [g for g in shape.geoms if g.geom_type == "Polygon"]
    from architecture import triangulate
    verts, faces = [], []
    for g in parts:
        ring = list(g.exterior.coords)[:-1]
        if not shapely.Polygon(ring).exterior.is_ccw:
            ring = ring[::-1]
        V2, T = triangulate([np.array(ring)], max_area=50.0)
        base = len(verts)
        verts.extend([tuple(n * (R + 0.005) + t * u + Vector((0, 0, z))) for u, z in V2])
        faces.extend([(base + a, base + b, base + c) for a, b, c in T])
    me = bpy.data.meshes.new("AR_BridgeSection")
    me.from_pydata(verts, [], faces)
    me.validate()
    for pg in me.polygons:                         # 法线朝外（沿 n）
        if Vector(pg.normal).dot(n) < 0:
            pg.flip()
    return me, sum(g.area for g in parts)


def up_copy(m):
    u = bpy.data.materials.get("Up_" + m.name)
    if u is None:
        u = m.copy()
        u.name = "Up_" + m.name
    return u


def assign_upper(o, face_mask):
    """把选中的面换成 Up_ 材质副本（同一网格里追加材质槽）。"""
    me = o.data
    if not face_mask.any():
        return 0
    mats = list(me.materials)
    if not mats:
        return 0
    off = len(mats)
    for m in mats:
        me.materials.append(up_copy(m) if m else None)
    mi = np.empty(len(me.polygons), np.int64)
    me.polygons.foreach_get("material_index", mi)
    mi[face_mask] += off
    me.polygons.foreach_set("material_index", mi)
    return int(face_mask.sum())


def split_upper(o):
    name = o.name
    me = o.data
    if name.startswith(UPPER):
        for i, m in enumerate(me.materials):
            if m:
                me.materials[i] = up_copy(m)
        return len(me.polygons)
    if name.startswith(SPLIT_LOOSE):
        lab, bb = components(me)
        return assign_upper(o, (bb[:, 2] > P.DECK - 0.4)[lab])
    if name.startswith(CORE_CUT):
        bm = bmesh.new()
        bm.from_mesh(me)
        bmesh.ops.bisect_plane(bm, geom=bm.verts[:] + bm.edges[:] + bm.faces[:], plane_co=(0, 0, P.DECK + 0.02),
                               plane_no=(0, 0, 1))
        bm.to_mesh(me)
        bm.free()
        return assign_upper(o, face_centers(me)[:, 2] > P.DECK + 0.02)
    if name.startswith("People_"):
        lab, bb = components(me)
        cx, cy = (bb[:, 0] + bb[:, 3]) / 2, (bb[:, 1] + bb[:, 4]) / 2
        on_ramp = np.abs(np.hypot(cx, cy) - P.RAMP_R) < P.RAMP_W / 2 + 0.4
        return assign_upper(o, ((bb[:, 2] > P.DECK - 0.5) & ~on_ramp)[lab])
    return 0


def simplify_materials():
    """程序化材质 → 同色基础 PBR；玻璃半透明；不发光。Up_ 副本的粗糙度略作区分，避免压缩时被当作重复材质合并。
    返回 {材质名: 不透明度}，供 USD 补写（Blender 的 USD 导出不写 Alpha）。"""
    alphas = {}
    for m in bpy.data.materials:
        if not m.use_nodes:
            continue
        nt = m.node_tree
        pb = next((x for x in nt.nodes if x.bl_idname == "ShaderNodeBsdfPrincipled"), None)
        out = next((x for x in nt.nodes if x.bl_idname == "ShaderNodeOutputMaterial"), None)
        if out is None:
            continue
        new = nt.nodes.new("ShaderNodeBsdfPrincipled")
        new.inputs["Base Color"].default_value = m.diffuse_color
        rough, metal, trans = 0.5, 0.0, 0.0
        if pb:
            rough = pb.inputs["Roughness"].default_value
            metal = pb.inputs["Metallic"].default_value
            trans = pb.inputs["Transmission Weight"].default_value
        low = m.name.lower()
        if trans > 0.5 or "glass" in low:
            frosted = "frost" in low
            new.inputs["Base Color"].default_value = (*lin((0.86, 0.92, 0.95) if frosted else (0.72, 0.82, 0.86)), 1)
            new.inputs["Alpha"].default_value = alphas[m.name] = 0.6 if frosted else 0.28
            rough, metal = (0.45 if frosted else 0.05), 0.0
        new.inputs["Roughness"].default_value = min(1.0, rough + (0.004 if m.name.startswith("Up_") else 0.0))
        new.inputs["Metallic"].default_value = metal
        for link in list(out.inputs[0].links):
            nt.links.remove(link)
        nt.links.new(new.outputs[0], out.inputs[0])
    return alphas


def pxr_modules():
    try:
        from pxr import Usd, UsdShade, UsdUtils
    except ImportError:                                   # pip 版 bpy：pxr 在 bpy 自带的 site-packages 里
        sys.path += glob.glob(os.path.join(bpy.utils.resource_path("LOCAL"), "python", "lib", "python3*",
                                           "site-packages"))
        from pxr import Usd, UsdShade, UsdUtils
    return Usd, UsdShade, UsdUtils


def export_usdz(path, alphas):
    """先导出 .usdc，补写玻璃的 opacity，再按 ARKit 规则打包成 .usdz。"""
    tmp = path[:-5] + ".usdc"
    bpy.ops.wm.usd_export(filepath=tmp, selected_objects_only=True, export_materials=True,
                          generate_preview_surface=True, export_uvmaps=False, export_mesh_colors=False,
                          export_normals=True, use_instancing=False, convert_orientation=True,
                          export_global_forward_selection="NEGATIVE_Z", export_global_up_selection="Y",
                          convert_world_material=False, root_prim_path="/CloudTidePavilion")
    Usd, UsdShade, UsdUtils = pxr_modules()
    st = Usd.Stage.Open(tmp)
    patched = 0
    for prim in st.Traverse():
        if not prim.IsA(UsdShade.Shader):
            continue
        sh = UsdShade.Shader(prim)
        a = alphas.get(prim.GetParent().GetName())
        if sh.GetIdAttr().Get() == "UsdPreviewSurface" and a is not None:
            sh.GetInput("opacity").Set(float(a))
            patched += 1
    st.GetRootLayer().Save()
    del st
    if os.path.exists(path):
        os.remove(path)
    ok = UsdUtils.CreateNewARKitUsdzPackage(tmp, path)
    os.remove(tmp)
    print("usdz", path, "opacity patched", patched, "ok", ok)


def strip_upper(o):
    """删掉 Up_ 材质的面（屋顶、屋面平台、观景厅等），得到“掀开屋顶”的版本。"""
    me = o.data
    up = np.array([bool(m and m.name.startswith("Up_")) for m in me.materials])
    mi = np.empty(len(me.polygons), np.int64)
    me.polygons.foreach_get("material_index", mi)
    delete_faces(me, up[mi])


def flat_material(name, srgb, rough, metal=0.0):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    m.diffuse_color = (*lin(srgb), 1)
    b = m.node_tree.nodes.get("Principled BSDF")
    b.inputs["Base Color"].default_value = (*lin(srgb), 1)
    b.inputs["Roughness"].default_value = rough
    b.inputs["Metallic"].default_value = metal
    return m


def ring_mesh(name, r0, r1, z0, z1, seg=256, top=True, bottom=False, inner=True, outer=True):
    """圆盘或圆环（r0 = 0 时为圆盘），带内外侧壁。"""
    bm = bmesh.new()
    angs = np.linspace(0, 2 * math.pi, seg, endpoint=False)

    def loop(r, z):
        return [bm.verts.new((r * math.cos(a), r * math.sin(a), z)) for a in angs]
    if r0 == 0:
        if top:
            bm.faces.new(loop(r1, z1))
        if bottom:
            bm.faces.new(loop(r1, z0)[::-1])
        if outer:
            a, b = loop(r1, z0), loop(r1, z1)
            for i in range(seg):
                j = (i + 1) % seg
                bm.faces.new((a[i], a[j], b[j], b[i]))
    else:
        it, ot, ib, ob = loop(r0, z1), loop(r1, z1), loop(r0, z0), loop(r1, z0)
        for i in range(seg):
            j = (i + 1) % seg
            if top:
                bm.faces.new((it[i], ot[i], ot[j], it[j]))
            if outer:
                bm.faces.new((ob[i], ob[j], ot[j], ot[i]))
            if inner:
                bm.faces.new((ib[j], ib[i], it[i], it[j]))
            if bottom:
                bm.faces.new((ib[i], ib[j], ob[j], ob[i]))
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    return me


def glyph_meshes(text, size, depth):
    """逐字生成文字网格（每个字单独一份，便于沿圆弧排布）；字体加载失败时返回空列表。"""
    try:
        font = bpy.data.fonts.load(FONT)
    except Exception as e:  # noqa: BLE001
        print("font failed", e)
        return []
    out = []
    dg = bpy.context.evaluated_depsgraph_get()
    for ch in text:
        if ch == " ":
            out.append((None, size * 0.45))
            continue
        cu = bpy.data.curves.new("AR_Glyph", "FONT")
        cu.body = ch
        cu.font = font
        cu.size = size
        cu.extrude = depth / 2
        cu.resolution_u = 3
        cu.align_x = "CENTER"
        ob = bpy.data.objects.new("AR_Glyph", cu)
        bpy.context.scene.collection.objects.link(ob)
        bpy.context.view_layer.update()
        dg = bpy.context.evaluated_depsgraph_get()
        me = bpy.data.meshes.new_from_object(ob.evaluated_get(dg))
        w = ob.dimensions.x
        bpy.data.objects.remove(ob)
        me.transform(Matrix.Translation((0, 0, depth / 2)))
        out.append((me, max(w, size * 0.35) + size * 0.12))
    return out


def nameplate(mat, centre_deg, n_arrow_deg):
    """边框顶面上的铭牌（项目名 + 比例）和指北针，黄铜色浅浮雕。"""
    z = P.HIGH_TIDE + RIM_UP
    rr = R + RIM_W / 2
    objs = []
    glyphs = glyph_meshes(NAMEPLATE, 0.62, 0.05)
    total = sum(w for _, w in glyphs)
    a = math.radians(centre_deg) - (total / rr) / 2           # 文字自西向东排（从南边看是从左到右）
    for me, w in glyphs:
        amid = a + (w / 2) / rr
        if me is not None:
            # 字的“上方”朝向圆心：先绕 z 转到切线方向，再放到圆弧上
            rot = Matrix.Rotation(amid + math.pi / 2, 4, "Z")
            pos = Matrix.Translation((rr * math.cos(amid), rr * math.sin(amid), z))
            me.transform(pos @ rot @ Matrix.Translation((0, -0.62 * 0.36, 0)))
            ob = bpy.data.objects.new("AR_Nameplate", me)
            me.materials.append(mat)
            objs.append(ob)
        a += w / rr
    # 指北针：指向 +Y 的细长三角（棱柱，立在边框顶面上）+ “N”
    am = math.radians(n_arrow_deg)
    c = Vector((rr * math.cos(am), rr * math.sin(am), z))
    outline = [Vector(v) for v in ((0, 0.42, 0), (-0.17, -0.3, 0), (0, -0.16, 0), (0.17, -0.3, 0))]   # 逆时针
    bm = bmesh.new()
    top = [bm.verts.new(c + p + Vector((0, 0, 0.05))) for p in outline]
    bot = [bm.verts.new(c + p) for p in outline]
    bm.faces.new(top)
    for i in range(4):
        j = (i + 1) % 4
        bm.faces.new((bot[i], bot[j], top[j], top[i]))
    me = bpy.data.meshes.new("AR_NorthArrow")
    bm.to_mesh(me)
    bm.free()
    me.materials.append(mat)
    objs.append(bpy.data.objects.new("AR_NorthArrow", me))
    nm = glyph_meshes("N", 0.36, 0.05)
    if nm and nm[0][0] is not None:
        me = nm[0][0]
        me.transform(Matrix.Translation(c + Vector((0.0, 0.5, z - c.z))))
        me.materials.append(mat)
        objs.append(bpy.data.objects.new("AR_NorthN", me))
    return objs


def show_all(lc):
    lc.exclude = False
    lc.hide_viewport = False
    lc.collection.hide_viewport = False
    lc.collection.hide_select = False
    for ch in lc.children:
        show_all(ch)


def main(blend, prefix):
    bpy.ops.wm.open_mainfile(filepath=blend)
    sc = bpy.context.scene
    show_all(bpy.context.view_layer.layer_collection)
    keep = [o for o in bpy.data.objects if in_keep(o)]
    for o in list(bpy.data.objects):
        if o not in keep:
            bpy.data.objects.remove(o)
    for o in keep:
        if o.data.users > 1:
            o.data = o.data.copy()
    dg = bpy.context.evaluated_depsgraph_get()
    for o in keep:
        bake(o, dg)
    n, ang = cut_plane()
    print(f"cut plane at {math.degrees(ang):.1f} deg")
    cut_people = 0
    for o in keep:
        k = crop(o, n)
        if o.name.startswith("People_"):
            cut_people += k
    print("people parts removed", cut_people)
    # 减面：庭院树叶抽稀；人物减到一成以下；屋檐下木肋、栏杆、竖梃等细部按比例减面
    for k, o in enumerate(keep):
        if not len(o.data.polygons):
            continue
        if o.name == "Courtyard_Tree_Foliage":
            thin_foliage(o, k, keep=0.3, scale=1.6)
        elif o.name == "TreeIsland_GroundCover":
            thin_foliage(o, k, keep=0.5, scale=1.3)
        elif o.name.startswith("People_"):
            o.data = decimated(o, PEOPLE_RATIO)
        else:
            r = next((r for key, r in DECIMATE if key in o.name), None)
            if r:
                o.data = decimated(o, r)
        o.data.validate()
    keep = [o for o in keep if len(o.data.polygons)]
    for o in list(bpy.data.objects):
        if o.type == "MESH" and o not in keep:
            bpy.data.objects.remove(o)
    # 剖切面
    plinth = [o for o in keep if o.name.startswith(("Plinth_And_Bridge", "Plinth_Edge"))]
    sec_me, sec_area = section_cap([o for o in plinth if o.name.startswith("Plinth_And_Bridge")], n)
    print(f"section cap {sec_area:.2f} m2, {len(sec_me.polygons)} tris")
    # 上部构件换 Up_ 材质
    up_faces = sum(split_upper(o) for o in keep)
    print("upper faces", up_faces)
    alphas = simplify_materials()
    # 底座：水面圆盘 + 胡桃木边框（内壁、顶面、外壁、底面）+ 剖切面 + 铭牌与指北针
    water = flat_material("AR_Water", (0.22, 0.41, 0.46), 0.06)
    wood = flat_material("AR_BaseWalnut", (0.33, 0.23, 0.165), 0.5)
    brass = flat_material("AR_Brass", (0.80, 0.66, 0.40), 0.32, 0.85)
    cut = flat_material("AR_Section", (0.24, 0.23, 0.22), 0.8)
    base = []
    me = ring_mesh("AR_WaterSurface", 0, R + 0.02, P.HIGH_TIDE, P.HIGH_TIDE, top=True, outer=False)
    me.materials.append(water)
    base.append(bpy.data.objects.new("AR_WaterSurface", me))
    me = ring_mesh("AR_BaseRim", R, R + RIM_W, P.HIGH_TIDE - BASE_T, P.HIGH_TIDE + RIM_UP, bottom=False)
    me.materials.append(wood)
    base.append(bpy.data.objects.new("AR_BaseRim", me))
    me = ring_mesh("AR_BaseBottom", 0, R + RIM_W, P.HIGH_TIDE - BASE_T, P.HIGH_TIDE - BASE_T, top=False,
                   bottom=True, outer=False)
    me.materials.append(wood)
    base.append(bpy.data.objects.new("AR_BaseBottom", me))
    sec_me.materials.append(cut)
    base.append(bpy.data.objects.new("AR_BridgeSection", sec_me))
    bridge_deg = math.degrees(ang)
    base += nameplate(brass, bridge_deg + 33.0, bridge_deg + 58.0)
    for ob in base:
        sc.collection.objects.link(ob)
    # 缩放到 1:100，原点移到底座底面
    M = Matrix.Translation((0, 0, BASE_T * SCALE)) @ Matrix.Scale(SCALE, 4)
    allobj = keep + base
    counts = {}
    for o in allobj:
        o.data.transform(M)
        o.hide_set(False)
        o.hide_viewport = False
        o.hide_render = False
        o.hide_select = False
        lt = np.empty(len(o.data.polygons), np.int64)
        o.data.polygons.foreach_get("loop_total", lt)
        counts[o.name] = int((lt - 2).sum())
    tris = sum(counts.values())
    print(f"objects {len(allobj)}, triangles {tris}")
    for k, v in sorted(counts.items(), key=lambda kv: -kv[1])[:30]:
        print(f"  {v:7d}  {k}")
    # 合成两个网格：建筑与人物（含上部构件）、底座
    for group, name in ((keep, "CloudTidePavilion"), (base, "ModelBase")):
        bpy.ops.object.select_all(action="DESELECT")
        for o in group:
            o.select_set(True)
        bpy.context.view_layer.objects.active = group[0]
        bpy.ops.object.join()
        j = bpy.context.view_layer.objects.active
        j.name = j.data.name = name
        j.data.validate()
    for o in bpy.data.objects:
        o.select_set(o.type == "MESH")
    bpy.ops.export_scene.gltf(filepath=prefix + ".glb", export_format="GLB", use_selection=True,
                              export_apply=True, export_texcoords=False, export_normals=True,
                              export_tangents=False, export_vertex_color="NONE", export_attributes=False,
                              export_cameras=False, export_lights=False, export_yup=True)
    export_usdz(prefix + ".usdz", alphas)
    strip_upper(bpy.data.objects["CloudTidePavilion"])               # iPhone 上“掀开屋顶”用的另一份 USDZ
    export_usdz(prefix + "_open.usdz", alphas)
    spots = {}
    for k, (x, y, z) in HOTSPOTS.items():                 # glTF 坐标（Y 向上，+Z 朝南）
        spots[k] = [round(x * SCALE, 4), round((z + BASE_T) * SCALE, 4), round(-y * SCALE, 4)]
    info = dict(scale=SCALE, radius=round((R + RIM_W) * SCALE, 3), height_base=round((BASE_T + RIM_UP) * SCALE, 4),
                hotspots=spots, upper_spots=sorted(UPPER_SPOTS), triangles=tris)
    with open(prefix + ".json", "w") as f:
        json.dump(info, f, ensure_ascii=False, indent=1)
    print("exported", prefix, info)


if __name__ == "__main__":
    main(sys.argv[-2], sys.argv[-1])
