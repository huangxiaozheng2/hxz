"""结构体系轴测图：屋面半透明，承重构件以橙色突出，并自动标注。
用法：python struct_axo.py scene.blend 输出.png [--samples=64] [--labels-only]
"""
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bpy  # noqa: E402
import numpy as np  # noqa: E402
from mathutils import Vector  # noqa: E402
from bpy_extras.object_utils import world_to_camera_view  # noqa: E402

import plan as P  # noqa: E402
import structure as ST  # noqa: E402

HIDE = ("People_", "Water_Surface", "Mountains_", "Islet", "Shore_", "Exhibit_", "Lobby_", "Terrace_Sofa",
        "Courtyard_Tree", "TreeIsland_GroundCover", "_SoffitRibs", "LED", "_Glass", "Skylight_Glass", "Planter",
        "Stair_Balustrade", "Ramp_Balustrade", "Deck_Balustrade", "Channel_Balustrade", "MoonGate_Glass",
        "Sculpture_FrostedGlass", "_Mullions_Frames", "_WindowFrames", "LiftCore_Doors")
ACCENT = ("_Columns", "Supports_", "LiftCore_Shaft", "MoonGate_Wall", "_SolidWall", "Ramp_Colonnade",
          "_PrimaryBeams", "Stair_HangerRods")
GHOST = ("_Shell", "_EdgeTrim", "Skylight_Lining")


def flat_mat(name, rgb, alpha=1.0):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    d = nt.nodes.new("ShaderNodeBsdfPrincipled")
    d.inputs["Base Color"].default_value = (*rgb, 1)
    d.inputs["Roughness"].default_value = 0.9
    d.inputs["Specular IOR Level"].default_value = 0.1
    if alpha < 1.0:
        tr = nt.nodes.new("ShaderNodeBsdfTransparent")
        mix = nt.nodes.new("ShaderNodeMixShader")
        mix.inputs[0].default_value = 1 - alpha
        nt.links.new(d.outputs[0], mix.inputs[1]); nt.links.new(tr.outputs[0], mix.inputs[2])
        nt.links.new(mix.outputs[0], out.inputs[0])
    else:
        nt.links.new(d.outputs[0], out.inputs[0])
    return m


def main(blend, out_png, samples=64, render=True):
    bpy.ops.wm.open_mainfile(filepath=blend)
    sc = bpy.context.scene
    acc = flat_mat("Axo_Accent", (0.78, 0.36, 0.14))
    ghost = flat_mat("Axo_Ghost", (0.97, 0.96, 0.94), alpha=0.16)
    grey = flat_mat("Axo_Grey", (0.80, 0.79, 0.77))
    for o in bpy.data.objects:
        if o.type == "LIGHT":
            o.hide_render = True
            continue
        if o.type != "MESH":
            continue
        n = o.name
        if any(h in n for h in HIDE):
            o.hide_render = True
            continue
        mat = acc if any(a in n for a in ACCENT) else ghost if any(g in n for g in GHOST) else grey
        o.data.materials.clear()
        o.data.materials.append(mat)
        for p in o.data.polygons:
            p.material_index = 0
    # 灯光：均匀环境光 + 一盏太阳光
    w = sc.world
    nt = w.node_tree
    nt.nodes.clear()
    bg = nt.nodes.new("ShaderNodeBackground"); bg.inputs["Color"].default_value = (0.9, 0.9, 0.9, 1)
    bg.inputs["Strength"].default_value = 0.9
    o_ = nt.nodes.new("ShaderNodeOutputWorld"); nt.links.new(bg.outputs[0], o_.inputs[0])
    sun = bpy.data.lights.new("Axo_Sun", "SUN"); sun.energy = 2.6; sun.angle = math.radians(6)
    so = bpy.data.objects.new("Axo_Sun", sun); sc.collection.objects.link(so)
    so.rotation_euler = (math.radians(42), 0, math.radians(-38))
    # 正交相机：南偏东上方，能同时看到 Y 形斜柱两肢、电梯筒和水中石墩
    cam = bpy.data.cameras.new("Axo"); cam.type = "ORTHO"; cam.ortho_scale = 46
    co = bpy.data.objects.new("Cam_axo", cam); sc.collection.objects.link(co)
    tgt = Vector((0.0, -0.4, 4.2))
    co.location = tgt + Vector((10, -60, 34)).normalized() * 120
    co.rotation_euler = (tgt - co.location).to_track_quat("-Z", "Y").to_euler()
    cam.clip_end = 500
    sc.camera = co
    sc.render.resolution_x, sc.render.resolution_y = 2400, 1800
    sc.render.film_transparent = True
    sc.render.image_settings.color_mode = "RGBA"
    sc.cycles.samples = samples
    sc.view_settings.look = "None"
    sc.view_settings.exposure = 0.0
    sc.render.filepath = out_png
    bpy.context.view_layer.update()
    if render:
        bpy.ops.render.render(write_still=True)

    # 标注锚点 → 画面坐标
    def under(roof, xy):
        return float(roof.underside(np.array([xy[0]]), np.array([xy[1]]))[0])

    def top(roof, xy):
        t, _ = roof.surfaces(np.array([xy[0]]), np.array([xy[1]]))
        return float(t[0])
    c1 = ST.E1_COLS_INNER[2]
    b1 = ST.B_COLS[len(ST.B_COLS) // 2]
    yt = [P.YCOL_TOPS[0][0] * math.cos(math.radians(P.YCOL_TOPS[0][1])),
          P.YCOL_TOPS[0][0] * math.sin(math.radians(P.YCOL_TOPS[0][1]))]
    ramp_a = math.radians(285)
    ramp_p = [(P.RAMP_R + P.RAMP_W / 2 + 0.16) * math.cos(ramp_a), (P.RAMP_R + P.RAMP_W / 2 + 0.16) * math.sin(ramp_a)]
    anchors = {
        "主屋顶 A（半透明显示）": (-14.0, 12.0, top(P.ROOF_A, (-14.0, 12.0))),
        "次屋顶 B": (0.0, 12.5, top(P.ROOF_B, (0.0, 12.5))),
        "低檐 C": (0.0, -16.5, top(P.ROOF_C, (0.0, -16.5))),
        "胶合木主梁 约 1.5 m 一根": (13.5, -4.5, under(P.ROOF_A, (13.5, -4.5)) + 0.3) if P.ROOF_A.poly.contains(
            __import__("shapely").Point(13.5, -4.5)) else (8.0, 13.0, under(P.ROOF_A, (8.0, 13.0)) + 0.2),
        "钢环梁 + 柱 φ194": (c1[0], c1[1], under(P.ROOF_A, c1) - 1.2),
        "二层柱（与一层柱对齐）": (b1[0], b1[1], P.DECK + 1.8),
        "电梯筒（贯通至 B）": (P.CORE_C[0], P.CORE_C[1], P.DECK + 1.6),
        "月洞墙（承重）": (P.MOON_C[0], P.MOON_C[1], P.DECK + 3.0),
        "Y 形斜柱": ((P.YCOL_BASE[0] + yt[0]) / 2, (P.YCOL_BASE[1] + yt[1]) / 2, 5.0),
        "水中石墩 + 钢柱": (P.C_PIERS[1][0], P.C_PIERS[1][1], 1.2),
        "坡道钢柱与悬臂": (ramp_p[0], ramp_p[1], 2.4),
    }
    res = {}
    for k, (x, y, z) in anchors.items():
        v = world_to_camera_view(sc, co, Vector((x, y, z)))
        res[k] = (v.x * sc.render.resolution_x, (1 - v.y) * sc.render.resolution_y)
    with open(out_png.replace(".png", "_labels.json"), "w") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    opts = dict(a[2:].split("=") for a in sys.argv[1:] if a.startswith("--") and "=" in a)
    main(args[0], os.path.abspath(args[1]), int(opts.get("samples", 64)), "--labels-only" not in sys.argv)
