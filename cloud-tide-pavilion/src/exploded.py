"""爆炸轴测：把建筑分层抬开（基座 → 首层 → A/C 屋顶与屋面平台 → 观景厅 → B 屋顶），白模 + Freestyle 线稿，
透明底；同时写出各层标注锚点与层间对位线端点的图像坐标（JSON），由 diagrams.exploded_drawing() 加标注。
用法：python exploded.py scene.blend 输出.png [--samples=32] [--scale=1.0] [--lines=0]
"""
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bpy  # noqa: E402
from bpy_extras.object_utils import world_to_camera_view  # noqa: E402
from mathutils import Vector  # noqa: E402

import plan as P  # noqa: E402
from elevation import flat, classify  # noqa: E402

HIDE = ("Mountains_", "Islet", "Shore_", "Water", "People_", "Exhibit_", "Lobby_", "Terrace_Sofa", "Courtyard_Tree",
        "TreeIsland_GroundCover", "Sculpture_")
HIDE_SUB = ("LED", "Glow")
# 层：(键, 抬升高度 m, 名称前缀)
LAYERS = [
    ("base", 0.0, ("Plinth_And_Bridge", "Plinth_Edge", "TidalBasin", "Channel_", "Supports_StonePiers", "TreeIsland_")),
    ("ground", 6.5, ("Envelope_E1", "Envelope_E2", "Service_", "Ramp_", "Stair_", "LiftCore_", "EgressStair_",
                     "Supports_YStrut")),
    ("roofs", 16.0, ("Roof_A_Main", "Roof_C_Low", "Deck_")),
    ("upper", 23.5, ("Envelope_E3", "MoonGate_")),
    ("roofB", 31.0, ("Roof_B_Upper",)),
]
SPLIT = ("Supports_Columns", "Supports_RingBeams_Steel")      # 按构件高度分到首层或观景厅层
LABEL = {
    "roofB": "B  次屋顶\n胶合木主梁 · 木格栅 · 金属包边",
    "upper": "观景厅 · 月洞墙\n二层柱立在主屋顶结构层上",
    "roofs": "A  主屋顶 · C  低檐 · 屋面平台\n屋面光缝 8 段",
    "ground": "首层：展廊 · 门厅 · 坡道\n电梯筒 · 疏散楼梯间 · Y 形斜柱",
    "base": "基座 · 潮池 · 水口\n栈桥 · 水中石墩",
}


def layer_of(name):
    for key, off, prefixes in LAYERS:
        if name.startswith(prefixes):
            return key, off
    return None, 0.0


def split_loose(o):
    """把一个物体按松散部件拆开，返回新物体列表。"""
    bpy.ops.object.select_all(action="DESELECT")
    o.select_set(True)
    bpy.context.view_layer.objects.active = o
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.separate(type="LOOSE")
    bpy.ops.object.mode_set(mode="OBJECT")
    return [x for x in bpy.context.selected_objects]


def main(blend, out_png, samples=32, scale=1.0, lines=True):
    bpy.ops.wm.open_mainfile(filepath=blend)
    sc = bpy.context.scene
    mats = {k: flat(f"Exp_{k}", rgb, a) for k, (rgb, a) in {
        "wall": ((0.93, 0.915, 0.89), 1.0), "wood": ((0.86, 0.76, 0.62), 1.0), "metal": ((0.74, 0.74, 0.74), 1.0),
        "glass": ((0.80, 0.87, 0.90), 0.32), "water": ((0.72, 0.82, 0.88), 1.0), "people": ((0.35, 0.35, 0.36), 1.0),
        "leaf": ((0.70, 0.76, 0.62), 1.0), "bark": ((0.45, 0.40, 0.35), 1.0), "stone": ((0.86, 0.85, 0.83), 1.0)}.items()}
    for key, m in mats.items():                     # 白模加一点自发光：基座里的缝隙与内腔不再发黑
        if key == "glass":
            continue
        t = m.node_tree
        b = next(n for n in t.nodes if n.bl_idname == "ShaderNodeBsdfPrincipled")
        out = next(n for n in t.nodes if n.bl_idname == "ShaderNodeOutputMaterial")
        em = t.nodes.new("ShaderNodeEmission")
        em.inputs["Color"].default_value = b.inputs["Base Color"].default_value
        em.inputs["Strength"].default_value = 0.13
        add = t.nodes.new("ShaderNodeAddShader")
        t.links.new(b.outputs[0], add.inputs[0]); t.links.new(em.outputs[0], add.inputs[1])
        t.links.new(add.outputs[0], out.inputs[0])
    for o in list(bpy.data.objects):
        if o.type == "LIGHT":
            o.hide_render = True
        elif o.type == "CURVES":
            o.hide_render = True
    # 柱与环梁：拆开后按高度归层
    for name in SPLIT:
        o = bpy.data.objects.get(name)
        if o is None:
            continue
        for part in split_loose(o):
            zmin = min((part.matrix_world @ Vector(c)).z for c in part.bound_box)
            part.name = f"{name}_{'upper' if zmin > P.DECK - 0.4 else 'ground'}"
    anchors = {}
    for o in bpy.data.objects:
        if o.type != "MESH":
            continue
        if o.name.startswith(HIDE) or any(k in o.name for k in HIDE_SUB):
            o.hide_render = True
            continue
        if o.name.startswith(SPLIT):
            key = "upper" if "_upper" in o.name else "ground"
            off = dict((k, v) for k, v, _ in LAYERS)[key]
        else:
            key, off = layer_of(o.name)
        if key is None:
            o.hide_render = True
            continue
        o.location.z += off
        if o.name.startswith(("Plinth_And_Bridge", "TidalBasin")):     # 池壁法线朝向不一致时白模里会发黑：统一外向并用平直着色
            import bmesh
            bm = bmesh.new(); bm.from_mesh(o.data)
            bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
            bm.to_mesh(o.data); bm.free()
            for pg in o.data.polygons:
                pg.use_smooth = False
        o.data.materials.clear()
        o.data.materials.append(classify(o.name, mats))
        for pg in o.data.polygons:
            pg.material_index = 0
    # 潮池与水口里补一片局部水面（全局水面已隐藏，否则池中露出基座内部的黑洞）
    from architecture import triangulate
    from plan import ccw, resample_ring
    wpoly = P.WATER_CUT.intersection(P.PLINTH_ALL.buffer(0.6, 32))
    wparts = [wpoly] if wpoly.geom_type == "Polygon" else list(wpoly.geoms)
    for k, g in enumerate(wparts):
        ring = ccw(resample_ring(g.exterior.coords, 0.4))
        V2, T = triangulate([ring], max_area=0.6)
        me = bpy.data.meshes.new(f"Exp_PoolWater_{k}")
        me.from_pydata([(x, y, P.HIGH_TIDE) for x, y in V2], [], [tuple(t) for t in T])
        ob = bpy.data.objects.new(f"Exp_PoolWater_{k}", me)
        sc.collection.objects.link(ob)
        me.materials.append(mats["water"])
    w = sc.world
    nt = w.node_tree
    nt.nodes.clear()
    bg = nt.nodes.new("ShaderNodeBackground")
    bg.inputs["Color"].default_value = (1, 1, 1, 1)
    bg.inputs["Strength"].default_value = 0.45
    o_ = nt.nodes.new("ShaderNodeOutputWorld")
    nt.links.new(bg.outputs[0], o_.inputs[0])
    sun = bpy.data.lights.new("Exp_Sun", "SUN"); sun.energy = 0.95; sun.angle = math.radians(8)
    so = bpy.data.objects.new("Exp_Sun", sun); sc.collection.objects.link(so)
    so.rotation_euler = (math.radians(42), 0, math.radians(30))
    # 轴测相机：自东南方向、俯角 32°
    phi, el = math.radians(-50), math.radians(32)
    target = Vector((1.5, 0.0, 16.5))
    d = Vector((math.cos(el) * math.cos(phi), math.cos(el) * math.sin(phi), math.sin(el)))
    cam = bpy.data.cameras.new("Exp"); cam.type = "ORTHO"; cam.ortho_scale = 70.0
    cam.clip_start = 1.0; cam.clip_end = 600
    co = bpy.data.objects.new("Cam_exp", cam); sc.collection.objects.link(co)
    co.location = target + d * 200
    co.rotation_euler = (-d).to_track_quat("-Z", "Y").to_euler()
    sc.camera = co
    sc.render.resolution_x, sc.render.resolution_y = int(2400 * scale), int(3000 * scale)
    sc.cycles.samples = samples
    sc.cycles.use_denoising = True
    sc.view_settings.view_transform = "Standard"
    sc.view_settings.look = "None"
    sc.view_settings.exposure = 0.0
    sc.render.film_transparent = True
    sc.render.image_settings.color_mode = "RGBA"
    sc.render.use_freestyle = lines
    sc.render.line_thickness_mode = "ABSOLUTE"
    sc.render.line_thickness = 1.5 * scale
    vl = sc.view_layers[0]
    vl.use_freestyle = lines
    fs = vl.freestyle_settings
    fs.crease_angle = math.radians(135)
    ls = fs.linesets[0] if fs.linesets else fs.linesets.new("Lines")
    ls.select_by_visibility = True
    ls.select_silhouette = True; ls.select_border = True; ls.select_crease = True
    ls.linestyle.color = (0.17, 0.165, 0.155)
    ls.linestyle.thickness = 1.5 * scale
    bpy.context.view_layer.update()
    # 标注锚点：各层的代表点；对位线：同一平面位置在相邻两层的高度
    W, H = sc.render.resolution_x, sc.render.resolution_y

    def px(p):
        v = world_to_camera_view(sc, co, Vector(p))
        return [round(v.x * W, 1), round((1 - v.y) * H, 1)]
    offs = dict((k, v) for k, v, _ in LAYERS)
    reps = {"base": (-14.0, -4.0, 0.9), "ground": (-15.6, 5.0, 3.0), "roofs": (-17.0, -8.0, 7.5),
            "upper": (-9.5, 8.5, P.DECK + 2.0), "roofB": (-11.0, -2.5, 11.5)}
    anchors = {k: px((x, y, z + offs[k])) for k, (x, y, z) in reps.items()}
    pairs = [("base", "ground", (14.0, -5.0, P.FFL), (14.0, -5.0, P.FFL)),
             ("ground", "roofs", (-10.5, -13.4, 5.2), (-10.5, -13.4, 5.2)),
             ("roofs", "upper", (-9.0, 7.9, P.DECK), (-9.0, 7.9, P.DECK)),
             ("upper", "roofB", (4.0, 12.0, 9.6), (4.0, 12.0, 9.6))]
    guides = [[px((a[0], a[1], a[2] + offs[l0])), px((b[0], b[1], b[2] + offs[l1]))] for l0, l1, a, b in pairs]
    sc.render.filepath = out_png
    bpy.ops.render.render(write_still=True)
    with open(out_png[:-4] + ".json", "w") as f:
        json.dump(dict(size=[W, H], anchors=anchors, labels=LABEL, guides=guides), f, ensure_ascii=False)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    opts = dict(a[2:].split("=") for a in sys.argv[1:] if a.startswith("--") and "=" in a)
    main(args[0], os.path.abspath(args[1]), int(opts.get("samples", 32)), float(opts.get("scale", 1.0)),
         opts.get("lines", "1") != "0")
