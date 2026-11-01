"""立面图：正交相机 + Freestyle 线稿 + 浅色材质，输出透明底的立面渲染与图幅信息（JSON）。
标高、水位与比例尺由 diagrams.elevation_drawing() 叠加。
近裁剪面放在台基南缘外 1.9 m：栈桥在此被剖切（黑色剖面），其后的建筑完整可见。
水面不进入相机，但仍反射天光照亮檐下。
用法：python elevation.py scene.blend 输出.png [--side=south|west] [--samples=48] [--scale=1.0] [--lines=0]
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bpy  # noqa: E402
from mathutils import Vector  # noqa: E402


HIDE = ("Mountains_", "Islet", "Shore_", "Exhibit_", "Lobby_", "Terrace_Sofa", "LED", "Lights_", "Courtyard_Tree",
        "TreeIsland_GroundCover", "Town_", "FarShore_", "Boats_")
SIDES = {
    # 视线方向（相机看向的方向）、相机位置、正交宽度
    "south": (Vector((0, 1, 0)), Vector((0.0, -19.6, 5.5)), 62.0),
    "west": (Vector((1, 0, 0)), Vector((-36.0, 0.0, 5.5)), 62.0),
}


def flat(name, rgb, alpha=1.0, rough=0.85):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    b = nt.nodes.new("ShaderNodeBsdfPrincipled")
    b.inputs["Base Color"].default_value = (*rgb, 1)
    b.inputs["Roughness"].default_value = rough
    b.inputs["Specular IOR Level"].default_value = 0.2
    if alpha < 1:
        tr = nt.nodes.new("ShaderNodeBsdfTransparent")
        mix = nt.nodes.new("ShaderNodeMixShader")
        mix.inputs[0].default_value = 1 - alpha
        nt.links.new(b.outputs[0], mix.inputs[1]); nt.links.new(tr.outputs[0], mix.inputs[2])
        nt.links.new(mix.outputs[0], out.inputs[0])
    else:
        nt.links.new(b.outputs[0], out.inputs[0])
    return m


def classify(name, mats):
    n = name
    if "Glass" in n:
        return mats["glass"]
    if any(k in n for k in ("SoffitRibs", "PrimaryBeams", "Stair_Treads", "Platform_Floor")):
        return mats["wood"]
    if any(k in n for k in ("Mullions", "Columns", "Supports_", "Rail", "EdgeTrim", "WindowFrames", "Colonnade",
                            "FireDoors", "Reveal", "Bollard")):
        return mats["metal"]
    if n.startswith("Water"):
        return mats["water"]
    if n.startswith("People_"):
        return mats["people"]
    if any(k in n for k in ("Foliage", "Shrub")):
        return mats["leaf"]
    if any(k in n for k in ("Branches",)):
        return mats["bark"]
    if any(k in n for k in ("Plinth", "TidalBasin", "TidalPool", "StonePiers", "Deck_Paving", "Footbridges")):
        return mats["stone"]
    return mats["wall"]


def main(blend, out_png, side="south", samples=48, scale=1.0, lines=True):
    bpy.ops.wm.open_mainfile(filepath=blend)
    sc = bpy.context.scene
    mats = {k: flat(f"Elev_{k}", rgb, a) for k, (rgb, a) in {
        "wall": ((0.93, 0.915, 0.89), 1.0), "wood": ((0.86, 0.76, 0.62), 1.0), "metal": ((0.74, 0.74, 0.74), 1.0),
        "glass": ((0.80, 0.87, 0.90), 0.38), "water": ((0.84, 0.89, 0.91), 1.0), "people": ((0.35, 0.35, 0.36), 1.0),
        "leaf": ((0.70, 0.76, 0.62), 1.0), "bark": ((0.45, 0.40, 0.35), 1.0), "stone": ((0.86, 0.85, 0.83), 1.0)}.items()}
    nolines = bpy.data.collections.new("Elev_NoLines")
    sc.collection.children.link(nolines)
    for o in bpy.data.objects:
        if o.type == "LIGHT":
            o.hide_render = True
            continue
        if o.type != "MESH":
            continue
        if o.name.startswith(HIDE):
            o.hide_render = True
            continue
        if o.name.startswith("Water"):
            o.visible_camera = False
            nolines.objects.link(o)                     # 水面也不画线（否则水平面边缘会成一条通长线）
        m = classify(o.name, mats)
        o.data.materials.clear()
        o.data.materials.append(m)
        for pgon in o.data.polygons:
            pgon.material_index = 0
    w = sc.world
    nt = w.node_tree
    nt.nodes.clear()
    bg = nt.nodes.new("ShaderNodeBackground"); bg.inputs["Color"].default_value = (1, 1, 1, 1)
    bg.inputs["Strength"].default_value = 1.1
    o_ = nt.nodes.new("ShaderNodeOutputWorld"); nt.links.new(bg.outputs[0], o_.inputs[0])
    sun = bpy.data.lights.new("Elev_Sun", "SUN"); sun.energy = 1.3; sun.angle = math.radians(8)
    so = bpy.data.objects.new("Elev_Sun", sun); sc.collection.objects.link(so)
    so.rotation_euler = (math.radians(50), 0, math.radians(-35 if side == "south" else -120))
    look, loc, width = SIDES[side]
    cam = bpy.data.cameras.new("Elev"); cam.type = "ORTHO"; cam.ortho_scale = width
    cam.clip_start = 0.1; cam.clip_end = 400
    co = bpy.data.objects.new("Cam_elev", cam); sc.collection.objects.link(co)
    co.location = loc
    co.rotation_euler = look.to_track_quat("-Z", "Y").to_euler()
    sc.camera = co
    sc.render.resolution_x, sc.render.resolution_y = int(3600 * scale), int(1260 * scale)
    sc.cycles.samples = samples
    sc.cycles.use_denoising = True
    sc.view_settings.view_transform = "Standard"
    sc.view_settings.look = "None"
    sc.view_settings.exposure = 0.0
    sc.render.film_transparent = True
    sc.render.image_settings.color_mode = "RGBA"
    # Freestyle 线稿
    sc.render.use_freestyle = lines
    sc.render.line_thickness_mode = "ABSOLUTE"
    sc.render.line_thickness = 1.6 * scale
    vl = sc.view_layers[0]
    vl.use_freestyle = lines
    fs = vl.freestyle_settings
    fs.crease_angle = math.radians(135)
    ls = fs.linesets[0] if fs.linesets else fs.linesets.new("Lines")
    ls.select_by_visibility = True
    ls.select_silhouette = True; ls.select_border = True; ls.select_crease = True
    ls.select_by_collection = True; ls.collection = nolines; ls.collection_negation = "EXCLUSIVE"
    ls.linestyle.color = (0.17, 0.165, 0.155)
    ls.linestyle.thickness = 1.6 * scale
    sc.render.filepath = out_png
    bpy.ops.render.render(write_still=True)
    # 图幅：图像左右对应的水平坐标（南立面为世界 x，西立面为世界 -y）与上下对应的标高
    px_per_m = sc.render.resolution_x / width
    hh = sc.render.resolution_y / px_per_m / 2
    u0 = loc.x if side == "south" else -loc.y
    import json
    with open(out_png.replace(".png", ".json"), "w") as f:
        json.dump(dict(side=side, px_per_m=px_per_m, extent=[u0 - width / 2, u0 + width / 2, loc.z - hh, loc.z + hh],
                       cut=loc.y + cam.clip_start if side == "south" else loc.x + cam.clip_start), f)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    opts = dict(a[2:].split("=") for a in sys.argv[1:] if a.startswith("--") and "=" in a)
    main(args[0], os.path.abspath(args[1]), opts.get("side", "south"), int(opts.get("samples", 48)),
         float(opts.get("scale", 1.0)), opts.get("lines", "1") != "0")
