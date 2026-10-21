"""天空、光照、渲染设置与相机机位。"""
import math
import numpy as np
import bpy

import plan as P
from bl import collection

SUN_AZ = P.SOLSTICE_SUNSET_AZ   # 冬至日落方位（北纬 30°）
SUN_EL = 4.0                    # 落日前约 20 分钟


def setup_world(sun_az=SUN_AZ, sun_el=SUN_EL, strength=0.55, clouds=True):
    sc = bpy.context.scene
    w = bpy.data.worlds.get("World_TideSky") or bpy.data.worlds.new("World_TideSky")
    sc.world = w
    w.use_nodes = True
    nt = w.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputWorld"); out.location = (900, 0)
    bg = nt.nodes.new("ShaderNodeBackground"); bg.location = (650, 0)
    bg.inputs["Strength"].default_value = strength
    sky = nt.nodes.new("ShaderNodeTexSky"); sky.location = (-200, 200)
    sky.sky_type = "MULTIPLE_SCATTERING"
    sky.sun_elevation = math.radians(sun_el)
    sky.sun_rotation = math.radians(sun_az)
    sky.sun_size = math.radians(1.6)
    sky.sun_intensity = 0.9
    sky.altitude = 20.0
    sky.air_density = 1.0
    sky.aerosol_density = 0.6
    sky.ozone_density = 2.0
    col = sky.outputs[0]
    if clouds:
        # 高空薄云：将视线方向投影到云层平面，噪声拉伸成丝缕
        tc = nt.nodes.new("ShaderNodeTexCoord"); tc.location = (-1400, -300)
        sep = nt.nodes.new("ShaderNodeSeparateXYZ"); sep.location = (-1200, -300)
        nt.links.new(tc.outputs["Generated"], sep.inputs[0])

        def m(op, a, b=None, loc=(0, 0)):
            n = nt.nodes.new("ShaderNodeMath"); n.operation = op; n.location = loc
            for i, x in enumerate((a, b)):
                if x is None:
                    continue
                if isinstance(x, (int, float)):
                    n.inputs[i].default_value = x
                else:
                    nt.links.new(x, n.inputs[i])
            return n.outputs[0]
        zz = m("ADD", sep.outputs[2], 0.06)
        u = m("DIVIDE", sep.outputs[0], zz); v = m("DIVIDE", sep.outputs[1], zz)
        comb = nt.nodes.new("ShaderNodeCombineXYZ"); comb.location = (-800, -300)
        nt.links.new(m("MULTIPLY", u, 0.55), comb.inputs[0])
        nt.links.new(m("MULTIPLY", v, 2.2), comb.inputs[1])
        nz = nt.nodes.new("ShaderNodeTexNoise"); nz.location = (-600, -300)
        nz.inputs["Scale"].default_value = 1.15
        nz.inputs["Detail"].default_value = 10.0
        nz.inputs["Roughness"].default_value = 0.62
        nz.inputs["Distortion"].default_value = 0.25
        nt.links.new(comb.outputs[0], nz.inputs["Vector"])
        mr = nt.nodes.new("ShaderNodeMapRange"); mr.location = (-400, -300)
        mr.inputs["From Min"].default_value = 0.52
        mr.inputs["From Max"].default_value = 0.78
        nt.links.new(nz.outputs["Fac"], mr.inputs["Value"])
        hz = nt.nodes.new("ShaderNodeMapRange"); hz.location = (-400, -550)
        hz.inputs["From Min"].default_value = 0.015
        hz.inputs["From Max"].default_value = 0.16
        nt.links.new(sep.outputs[2], hz.inputs["Value"])
        hz2 = nt.nodes.new("ShaderNodeMapRange"); hz2.location = (-400, -750)
        hz2.inputs["From Min"].default_value = 0.65
        hz2.inputs["From Max"].default_value = 0.3
        nt.links.new(sep.outputs[2], hz2.inputs["Value"])
        mask = m("MULTIPLY", m("MULTIPLY", mr.outputs[0], hz.outputs[0]), hz2.outputs[0])
        mask = m("MULTIPLY", mask, 0.75)
        warm = nt.nodes.new("ShaderNodeMix"); warm.data_type = "RGBA"; warm.blend_type = "MULTIPLY"
        warm.location = (200, -100)
        warm.inputs["Factor"].default_value = 1.0
        nt.links.new(col, warm.inputs["A"])
        warm.inputs["B"].default_value = (1.9, 1.55, 1.38, 1)
        mix = nt.nodes.new("ShaderNodeMix"); mix.data_type = "RGBA"; mix.location = (420, 0)
        nt.links.new(mask, mix.inputs["Factor"])
        nt.links.new(col, mix.inputs["A"])
        nt.links.new(warm.outputs["Result"], mix.inputs["B"])
        col = mix.outputs["Result"]
    nt.links.new(col, bg.inputs["Color"])
    nt.links.new(bg.outputs[0], out.inputs[0])
    return w


def setup_lights(parent):
    coll = collection("Lights", parent)

    def area(name, loc, size, power, color=(1.0, 0.78, 0.55), rot=(0, 0, 0), shape="DISK"):
        l = bpy.data.lights.new(name, "AREA")
        l.energy = power; l.size = size; l.color = color; l.shape = shape
        o = bpy.data.objects.new(name, l)
        o.location = loc; o.rotation_euler = rot
        o.visible_camera = False
        o.visible_glossy = False
        o.visible_transmission = False      # 透过玻璃也看不到灯具本身
        coll.objects.link(o)
        return o

    def spot(name, loc, rot, power, angle=40, color=(1.0, 0.8, 0.6), size=0.05):
        l = bpy.data.lights.new(name, "SPOT")
        l.energy = power; l.spot_size = math.radians(angle); l.color = color; l.shadow_soft_size = size
        l.spot_blend = 0.6
        o = bpy.data.objects.new(name, l)
        o.location = loc; o.rotation_euler = rot
        coll.objects.link(o)
        return o

    # 展厅：吊顶下方的柔和面光（暖白 3000K），按网格放于围合体内
    import shapely
    for poly, roof, z0, pw, nm in ((P.E1, P.ROOF_A, P.FFL, 520, "E1"), (P.E2, P.ROOF_C, P.FFL, 380, "E2"),
                                    (P.E3, P.ROOF_B, P.DECK, 380, "E3")):
        minx, miny, maxx, maxy = poly.bounds
        xs = np.arange(minx + 1.5, maxx, 4.2); ys = np.arange(miny + 1.5, maxy, 4.2)
        X, Y = np.meshgrid(xs, ys)
        inside = shapely.contains_xy(poly.buffer(-1.4), X, Y)
        pts = np.stack([X[inside], Y[inside]], 1)
        und = roof.underside(pts[:, 0], pts[:, 1])
        for k, (p, zu) in enumerate(zip(pts, und)):
            area(f"Light_{nm}_Ceiling_{k:02d}", (p[0], p[1], zu - 0.3), 1.6, pw)
    # 树岛上照灯
    tx, ty = P.TREE_POS
    top = P.FFL + 0.45
    for k in range(3):
        a = 2 * math.pi * k / 3 + 0.4
        x, y = tx + 1.15 * math.cos(a), ty + 1.15 * math.sin(a)
        o = spot(f"Light_Tree_Uplight_{k}", (x, y, top), (0, 0, 0), 420, angle=55)
        aim(o, (x, y, top), (tx, ty, top + 5.0))
    # 雕塑侧光（装在第一级台阶上）
    sx, sy = P.SCULPT_POS
    a = math.atan2(sy, sx)
    src = (4.9 * math.cos(a), 4.9 * math.sin(a), P.BASIN_STEPS[0][2] + 0.12)
    o = spot("Light_Sculpture_Accent", src, (0, 0, 0), 220, angle=40)
    aim(o, src, (sx, sy, 0.7))
    return coll


def setup_render(samples=128, res=(1600, 900), denoise=True):
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    cy = sc.cycles
    cy.device = "CPU"
    cy.samples = samples
    cy.use_adaptive_sampling = True
    cy.adaptive_threshold = 0.02
    cy.use_denoising = denoise
    try:
        cy.denoiser = "OPENIMAGEDENOISE"
    except Exception:
        pass
    cy.max_bounces = 10
    cy.diffuse_bounces = 3
    cy.glossy_bounces = 4
    cy.transmission_bounces = 10
    cy.transparent_max_bounces = 8
    cy.caustics_reflective = False
    cy.caustics_refractive = False
    cy.sample_clamp_indirect = 6.0
    cy.light_sampling_threshold = 0.01
    sc.render.resolution_x, sc.render.resolution_y = res
    sc.render.resolution_percentage = 100
    sc.view_settings.view_transform = "AgX"
    try:
        sc.view_settings.look = "AgX - Medium High Contrast"
    except Exception:
        pass
    sc.view_settings.exposure = 0.0
    sc.render.film_transparent = False
    sc.render.threads_mode = "AUTO"
    sc.render.image_settings.file_format = "PNG"
    sc.render.image_settings.color_mode = "RGB"


def _axis(r, z):
    p = P.AXIS * r
    return (float(p[0]), float(p[1]), z)


def _ramp_cam(phi):
    a = math.radians(phi)
    r = P.RAMP_R - 0.35
    return (r * math.cos(a), r * math.sin(a), float(P.ramp_z_of_deg(phi)) + 1.6)


_mc = P.MOON_C
CAMERAS = {
    # 名称: (位置, 目标点, 焦距mm, 曝光, 潮位)
    "hero":      ((14.0, -44.0, 2.6), (-4.0, 0.0, 5.2), 24, 0.0, "high"),
    "aerial":    ((30.0, -38.0, 46.0), (-2.0, -1.0, 1.0), 30, 0.0, "high"),
    "axis":      (_axis(-5.0, P.BASIN_STEPS[0][2] + 1.6), _axis(40.0, 2.6), 22, 0.2, "high"),
    "axis_out":  (_axis(46.0, 2.2), (0.0, 0.0, 4.2), 30, 0.0, "high"),
    "court_ht":  ((3.5, -12.5, 17.5), (-0.8, 0.4, -0.3), 26, 0.2, "high"),
    "court_lt":  ((3.5, -12.5, 17.5), (-0.8, 0.4, -0.3), 26, 0.2, "low"),
    "interior":  ((10.5, 4.25, P.FFL + 1.55), (3.9, 10.8, P.FFL + 2.0), 18, 0.6, "high"),
    "moon":      ((float((_mc - P.AXIS * 8.6)[0]), float((_mc - P.AXIS * 8.6)[1]), P.DECK + 1.6),
                  (float(_mc[0]), float(_mc[1]), P.DECK + 1.95), 26, 0.3, "high"),
    "terrace":   ((10.6, 6.2, P.DECK + 1.6), (-9.0, 7.9, P.DECK + 1.6), 20, 0.2, "high"),
    "stair":     ((-6.6, 6.0, P.FFL + 1.5), (-12.6, 2.4, P.FFL + 2.0), 18, 0.5, "high"),
    "detail":    ((11.6, -19.6, P.FFL + 1.1), (8.2, -14.4, 3.6), 42, 0.2, "high"),
    "structure": ((-10.3, -9.2, P.FFL + 1.6), (-13.0, -8.2, 7.2), 24, 0.3, "high"),
    "dusk":      ((26.0, -40.0, 2.6), (-3.0, 0.0, 5.4), 24, 1.6, "high"),
}


def add_cameras(parent):
    coll = collection("Cameras", parent)
    out = {}
    for name, (loc, tgt, lens, ev, tide) in CAMERAS.items():
        cam = bpy.data.cameras.new(f"Cam_{name}")
        cam.lens = lens
        cam.sensor_width = 36
        cam.clip_start = 0.05
        cam.clip_end = 6000
        o = bpy.data.objects.new(f"Cam_{name}", cam)
        coll.objects.link(o)
        aim(o, loc, tgt)
        o["exposure"] = ev
        o["tide"] = tide
        out[name] = o
    return out


def aim(o, loc, tgt):
    from mathutils import Vector
    o.location = loc
    d = Vector(tgt) - Vector(loc)
    o.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()
