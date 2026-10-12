"""材质库：所有材质集中定义，命名统一以 M_ 开头，便于后续替换。"""
import bpy


class NT:
    """节点树小助手。"""

    def __init__(self, mat):
        mat.use_nodes = True
        self.mat = mat
        self.t = mat.node_tree
        self.t.nodes.clear()
        self.out = self.n("ShaderNodeOutputMaterial", (900, 0))

    def n(self, kind, loc=(0, 0), **props):
        node = self.t.nodes.new(kind)
        node.location = loc
        for k, v in props.items():
            if k in node.inputs:
                node.inputs[k].default_value = v
            else:
                setattr(node, k, v)
        return node

    def link(self, a, b):
        self.t.links.new(a, b)

    def bsdf(self, **kw):
        b = self.n("ShaderNodeBsdfPrincipled", (500, 0))
        for k, v in kw.items():
            b.inputs[k].default_value = v
        return b

    def math(self, op, a, b=None, loc=(0, 0)):
        m = self.n("ShaderNodeMath", loc, operation=op)
        for i, x in enumerate((a, b)):
            if x is None:
                continue
            if isinstance(x, (int, float)):
                m.inputs[i].default_value = x
            else:
                self.link(x, m.inputs[i])
        return m.outputs[0]


def _new(name):
    m = bpy.data.materials.get(name)
    if m:
        bpy.data.materials.remove(m)
    return bpy.data.materials.new(name)


def ivory_mineral(name="M_IvoryMineral", color=(0.83, 0.80, 0.745, 1), rough=0.52, scale=1.0):
    """象牙白，带极轻矿物颗粒与色差。"""
    m = _new(name); t = NT(m)
    tc = t.n("ShaderNodeTexCoord", (-900, 0))
    fine = t.n("ShaderNodeTexNoise", (-650, -200), **{"Scale": 160.0 * scale, "Detail": 6.0, "Roughness": 0.65})
    t.link(tc.outputs["Object"], fine.inputs["Vector"])
    big = t.n("ShaderNodeTexNoise", (-650, 200), **{"Scale": 0.35, "Detail": 3.0})
    t.link(tc.outputs["Object"], big.inputs["Vector"])
    ramp = t.n("ShaderNodeValToRGB", (-400, 200))
    ramp.color_ramp.elements[0].color = tuple(c * 0.975 for c in color[:3]) + (1,)
    ramp.color_ramp.elements[1].color = tuple(min(1, c * 1.02) for c in color[:3]) + (1,)
    t.link(big.outputs["Fac"], ramp.inputs["Fac"])
    speck = t.n("ShaderNodeTexVoronoi", (-650, -450), **{"Scale": 900.0 * scale})
    t.link(tc.outputs["Object"], speck.inputs["Vector"])
    bump = t.n("ShaderNodeBump", (200, -300), **{"Strength": 0.06, "Distance": 0.002})
    mixh = t.math("ADD", fine.outputs["Fac"], t.math("MULTIPLY", speck.outputs["Distance"], 0.25))
    t.link(mixh, bump.inputs["Height"])
    b = t.bsdf(**{"Roughness": rough, "Specular IOR Level": 0.45})
    t.link(ramp.outputs["Color"], b.inputs["Base Color"])
    t.link(bump.outputs["Normal"], b.inputs["Normal"])
    rr = t.math("ADD", t.math("MULTIPLY", fine.outputs["Fac"], 0.12), rough - 0.06)
    t.link(rr, b.inputs["Roughness"])
    t.link(b.outputs[0], t.out.inputs[0])
    m.diffuse_color = color
    return m


def brushed_metal(name="M_BrushedMetal", color=(0.78, 0.76, 0.72, 1), rough=0.34):
    """柔和拉丝金属（香槟银）。"""
    m = _new(name); t = NT(m)
    tc = t.n("ShaderNodeTexCoord", (-900, 0))
    mp = t.n("ShaderNodeMapping", (-700, 0))
    mp.inputs["Scale"].default_value = (1.0, 1.0, 260.0)
    t.link(tc.outputs["Object"], mp.inputs["Vector"])
    nz = t.n("ShaderNodeTexNoise", (-500, 0), **{"Scale": 6.0, "Detail": 8.0})
    t.link(mp.outputs[0], nz.inputs["Vector"])
    b = t.bsdf(**{"Base Color": color, "Metallic": 1.0, "Anisotropic": 0.55})
    rr = t.math("ADD", t.math("MULTIPLY", nz.outputs["Fac"], 0.10), rough - 0.05)
    t.link(rr, b.inputs["Roughness"])
    bump = t.n("ShaderNodeBump", (200, -300), **{"Strength": 0.04, "Distance": 0.0005})
    t.link(nz.outputs["Fac"], bump.inputs["Height"])
    t.link(bump.outputs["Normal"], b.inputs["Normal"])
    t.link(b.outputs[0], t.out.inputs[0])
    m.diffuse_color = color
    return m


def glass(name="M_Glass", tint=(0.93, 0.965, 0.96, 1), rough=0.0, thin=True):
    m = _new(name); t = NT(m)
    b = t.bsdf(**{"Base Color": tint, "Transmission Weight": 1.0, "Roughness": rough, "IOR": 1.52,
                  "Thin Wall": thin})
    t.link(b.outputs[0], t.out.inputs[0])
    m.diffuse_color = (0.6, 0.7, 0.72, 0.3)
    return m


def frosted_glass(name="M_FrostedGlass"):
    m = _new(name); t = NT(m)
    b = t.bsdf(**{"Base Color": (0.95, 0.97, 0.98, 1), "Transmission Weight": 1.0, "Roughness": 0.42,
                  "IOR": 1.48, "Subsurface Weight": 0.0})
    vol = t.n("ShaderNodeVolumeAbsorption", (500, -300), **{"Color": (0.9, 0.95, 0.96, 1), "Density": 0.6})
    t.link(b.outputs[0], t.out.inputs[0])
    t.link(vol.outputs[0], t.out.inputs[1])
    return m


def wood_rib(name="M_WoodRib"):
    """浅木色细肋：沿长度方向的纹理 + 每根肋轻微色差。"""
    m = _new(name); t = NT(m)
    uvn = t.n("ShaderNodeUVMap", (-1000, 0))
    mp = t.n("ShaderNodeMapping", (-800, 0))
    mp.inputs["Scale"].default_value = (0.6, 40.0, 1.0)
    t.link(uvn.outputs[0], mp.inputs["Vector"])
    wv = t.n("ShaderNodeTexWave", (-600, 0), **{"Scale": 2.0, "Distortion": 6.0, "Detail": 4.0,
                                               "Detail Scale": 1.5})
    wv.bands_direction = "Y"
    t.link(mp.outputs[0], wv.inputs["Vector"])
    ramp = t.n("ShaderNodeValToRGB", (-300, 0))
    ramp.color_ramp.elements[0].color = (0.66, 0.52, 0.38, 1)
    ramp.color_ramp.elements[1].color = (0.82, 0.69, 0.53, 1)
    t.link(wv.outputs["Fac"], ramp.inputs["Fac"])
    hsv = t.n("ShaderNodeHueSaturation", (-50, 0))
    t.link(ramp.outputs[0], hsv.inputs["Color"])
    # Random per island → 明度 ±4%
    gi = t.n("ShaderNodeNewGeometry", (-400, 400))
    v = t.math("ADD", t.math("MULTIPLY", gi.outputs["Random Per Island"], 0.08), 0.96)
    t.link(v, hsv.inputs["Value"])
    b = t.bsdf(**{"Roughness": 0.5, "Specular IOR Level": 0.35})
    t.link(hsv.outputs[0], b.inputs["Base Color"])
    t.link(b.outputs[0], t.out.inputs[0])
    m.diffuse_color = (0.75, 0.58, 0.40, 1)
    return m


def flat(name, color, rough=0.6, metallic=0.0, spec=0.5):
    m = _new(name); t = NT(m)
    b = t.bsdf(**{"Base Color": color, "Roughness": rough, "Metallic": metallic, "Specular IOR Level": spec})
    t.link(b.outputs[0], t.out.inputs[0])
    m.diffuse_color = color
    return m


def emission(name, color=(1.0, 0.78, 0.52, 1), strength=8.0):
    m = _new(name); t = NT(m)
    e = t.n("ShaderNodeEmission", (500, 0), **{"Color": color, "Strength": strength})
    t.link(e.outputs[0], t.out.inputs[0])
    m.diffuse_color = color
    return m


def stone_floor(name="M_StoneFloor", base=(0.80, 0.775, 0.725, 1), ring=1.2, seg=1.2, polish=0.28):
    """低对比石材；铺装缝为以庭院中心为极点的同心环 + 错缝径向缝，与建筑轴线一致。"""
    m = _new(name); t = NT(m)
    tc = t.n("ShaderNodeTexCoord", (-1400, 0))
    sep = t.n("ShaderNodeSeparateXYZ", (-1200, 0))
    t.link(tc.outputs["Object"], sep.inputs[0])
    x, y = sep.outputs[0], sep.outputs[1]
    r = t.math("SQRT", t.math("ADD", t.math("MULTIPLY", x, x), t.math("MULTIPLY", y, y)))
    th = t.math("ARCTAN2", y, x)
    # 环向缝
    ri = t.math("FLOOR", t.math("DIVIDE", r, ring))
    fr = t.math("FRACT", t.math("DIVIDE", r, ring))
    dr = t.math("MULTIPLY", t.math("MINIMUM", fr, t.math("SUBTRACT", 1.0, fr)), ring)
    # 径向缝：弧长坐标，逐环错缝
    s = t.math("ADD", t.math("DIVIDE", t.math("MULTIPLY", r, th), seg), t.math("MULTIPLY", ri, 0.5))
    fs = t.math("FRACT", s)
    ds = t.math("MULTIPLY", t.math("MINIMUM", fs, t.math("SUBTRACT", 1.0, fs)), seg)
    dmin = t.math("MINIMUM", dr, ds)
    joint = t.n("ShaderNodeMapRange", (-200, -300), **{"From Min": 0.002, "From Max": 0.0055,
                                                      "To Min": 1.0, "To Max": 0.0})
    t.link(dmin, joint.inputs["Value"])
    # 低对比纹理
    nz = t.n("ShaderNodeTexNoise", (-700, 300), **{"Scale": 1.4, "Detail": 9.0, "Roughness": 0.62,
                                                  "Distortion": 0.35})
    t.link(tc.outputs["Object"], nz.inputs["Vector"])
    ramp = t.n("ShaderNodeValToRGB", (-450, 300))
    ramp.color_ramp.elements[0].position = 0.35
    ramp.color_ramp.elements[1].position = 0.70
    ramp.color_ramp.elements[0].color = tuple(c * 0.955 for c in base[:3]) + (1,)
    ramp.color_ramp.elements[1].color = tuple(min(1, c * 1.025) for c in base[:3]) + (1,)
    t.link(nz.outputs["Fac"], ramp.inputs["Fac"])
    # 每块板细微色差
    slab = t.n("ShaderNodeTexWhiteNoise", (-700, 600))
    slab.noise_dimensions = "2D"
    comb = t.n("ShaderNodeCombineXYZ", (-900, 600))
    t.link(ri, comb.inputs[0]); t.link(t.math("FLOOR", s), comb.inputs[1])
    t.link(comb.outputs[0], slab.inputs["Vector"])
    hsv = t.n("ShaderNodeHueSaturation", (-150, 300))
    t.link(ramp.outputs[0], hsv.inputs["Color"])
    t.link(t.math("ADD", t.math("MULTIPLY", slab.outputs["Value"], 0.05), 0.975), hsv.inputs["Value"])
    mixc = t.n("ShaderNodeMix", (150, 200), data_type="RGBA")
    t.link(joint.outputs[0], mixc.inputs["Factor"])
    t.link(hsv.outputs[0], mixc.inputs["A"])
    mixc.inputs["B"].default_value = tuple(c * 0.62 for c in base[:3]) + (1,)
    b = t.bsdf(**{"Specular IOR Level": 0.5})
    t.link(mixc.outputs["Result"], b.inputs["Base Color"])
    rough = t.math("ADD", t.math("MULTIPLY", joint.outputs[0], 0.5), polish)
    t.link(t.math("ADD", rough, t.math("MULTIPLY", nz.outputs["Fac"], 0.08)), b.inputs["Roughness"])
    bump = t.n("ShaderNodeBump", (250, -300), **{"Strength": 0.25, "Distance": 0.003, "invert": True})
    t.link(joint.outputs[0], bump.inputs["Height"])
    t.link(bump.outputs[0], b.inputs["Normal"])
    t.link(b.outputs[0], t.out.inputs[0])
    m.diffuse_color = base
    return m


def water(name="M_Water"):
    """深色水面：清晰反射 + 轻微波纹。"""
    m = _new(name); t = NT(m)
    tc = t.n("ShaderNodeTexCoord", (-1000, 0))
    mp = t.n("ShaderNodeMapping", (-800, 0))
    mp.inputs["Scale"].default_value = (1.0, 2.2, 1.0)
    t.link(tc.outputs["Object"], mp.inputs["Vector"])
    n1 = t.n("ShaderNodeTexNoise", (-550, 100), **{"Scale": 0.22, "Detail": 4.0, "Roughness": 0.55})
    t.link(mp.outputs[0], n1.inputs["Vector"])
    n2 = t.n("ShaderNodeTexNoise", (-550, -150), **{"Scale": 1.6, "Detail": 3.0, "Roughness": 0.5})
    t.link(mp.outputs[0], n2.inputs["Vector"])
    h = t.math("ADD", n1.outputs["Fac"], t.math("MULTIPLY", n2.outputs["Fac"], 0.35))
    bump = t.n("ShaderNodeBump", (200, -300), **{"Strength": 0.16, "Distance": 0.08})
    t.link(h, bump.inputs["Height"])
    b = t.bsdf(**{"Base Color": (0.012, 0.035, 0.05, 1), "Roughness": 0.015, "IOR": 1.333,
                  "Specular IOR Level": 0.5})
    t.link(bump.outputs[0], b.inputs["Normal"])
    t.link(b.outputs[0], t.out.inputs[0])
    m.diffuse_color = (0.05, 0.12, 0.18, 1)
    return m


def leaf(name="M_Leaf", c1=(0.16, 0.22, 0.10, 1), c2=(0.32, 0.36, 0.20, 1)):
    m = _new(name); t = NT(m)
    gi = t.n("ShaderNodeNewGeometry", (-700, 200))
    ramp = t.n("ShaderNodeValToRGB", (-450, 200))
    ramp.color_ramp.elements[0].color = c1
    ramp.color_ramp.elements[1].color = c2
    t.link(gi.outputs["Random Per Island"], ramp.inputs["Fac"])
    b = t.bsdf(**{"Roughness": 0.55, "Specular IOR Level": 0.35})
    t.link(ramp.outputs[0], b.inputs["Base Color"])
    tr = t.n("ShaderNodeBsdfTranslucent", (500, -250))
    t.link(ramp.outputs[0], tr.inputs["Color"])
    mix = t.n("ShaderNodeMixShader", (720, 0), **{"Fac": 0.25})
    t.link(b.outputs[0], mix.inputs[1]); t.link(tr.outputs[0], mix.inputs[2])
    t.link(mix.outputs[0], t.out.inputs[0])
    m.diffuse_color = c2
    return m


def bark(name="M_Bark", color=(0.16, 0.135, 0.11, 1)):
    m = _new(name); t = NT(m)
    tc = t.n("ShaderNodeTexCoord", (-800, 0))
    nz = t.n("ShaderNodeTexNoise", (-550, 0), **{"Scale": 18.0, "Detail": 8.0, "Distortion": 1.2})
    t.link(tc.outputs["Object"], nz.inputs["Vector"])
    bump = t.n("ShaderNodeBump", (200, -300), **{"Strength": 0.4, "Distance": 0.01})
    t.link(nz.outputs["Fac"], bump.inputs["Height"])
    ramp = t.n("ShaderNodeValToRGB", (-300, 200))
    ramp.color_ramp.elements[0].color = tuple(c * 0.7 for c in color[:3]) + (1,)
    ramp.color_ramp.elements[1].color = tuple(c * 1.3 for c in color[:3]) + (1,)
    t.link(nz.outputs["Fac"], ramp.inputs["Fac"])
    b = t.bsdf(**{"Roughness": 0.85})
    t.link(ramp.outputs[0], b.inputs["Base Color"])
    t.link(bump.outputs[0], b.inputs["Normal"])
    t.link(b.outputs[0], t.out.inputs[0])
    return m


def distance_haze(name, color, haze=(0.78, 0.72, 0.70, 1), near=250.0, far=1600.0, rough=0.9):
    """远景材质：随视距混入雾色（薄雾只出现在远景）。"""
    m = _new(name); t = NT(m)
    cam = t.n("ShaderNodeCameraData", (-800, 0))
    mr = t.n("ShaderNodeMapRange", (-550, 0), **{"From Min": near, "From Max": far})
    t.link(cam.outputs["View Distance"], mr.inputs["Value"])
    tc = t.n("ShaderNodeTexCoord", (-800, 300))
    nz = t.n("ShaderNodeTexNoise", (-550, 300), **{"Scale": 0.02, "Detail": 6.0})
    t.link(tc.outputs["Object"], nz.inputs["Vector"])
    ramp = t.n("ShaderNodeValToRGB", (-300, 300))
    ramp.color_ramp.elements[0].color = tuple(c * 0.75 for c in color[:3]) + (1,)
    ramp.color_ramp.elements[1].color = tuple(c * 1.15 for c in color[:3]) + (1,)
    t.link(nz.outputs["Fac"], ramp.inputs["Fac"])
    b = t.bsdf(**{"Roughness": rough})
    t.link(ramp.outputs[0], b.inputs["Base Color"])
    # 空气透视：远处逐渐透出天空，使雾色始终与天色一致
    e = t.n("ShaderNodeBsdfTransparent", (500, -200))
    mix = t.n("ShaderNodeMixShader", (720, 0))
    pw = t.math("MULTIPLY", t.math("POWER", mr.outputs[0], 0.8), 0.86)
    t.link(pw, mix.inputs[0])
    t.link(b.outputs[0], mix.inputs[1]); t.link(e.outputs[0], mix.inputs[2])
    t.link(mix.outputs[0], t.out.inputs[0])
    return m


def painting(name, seed=0.0):
    m = _new(name); t = NT(m)
    uvn = t.n("ShaderNodeTexCoord", (-900, 0))
    nz = t.n("ShaderNodeTexNoise", (-650, 0), **{"Scale": 2.2, "Detail": 3.0, "Distortion": 2.5})
    nz.noise_dimensions = "4D"
    nz.inputs["W"].default_value = seed
    t.link(uvn.outputs["Generated"], nz.inputs["Vector"])
    ramp = t.n("ShaderNodeValToRGB", (-400, 0))
    ramp.color_ramp.elements[0].color = (0.08, 0.075, 0.07, 1)
    ramp.color_ramp.elements[1].color = (0.82, 0.78, 0.70, 1)
    ramp.color_ramp.elements[0].position = 0.42
    ramp.color_ramp.elements[1].position = 0.6
    t.link(nz.outputs["Fac"], ramp.inputs["Fac"])
    b = t.bsdf(**{"Roughness": 0.8})
    t.link(ramp.outputs[0], b.inputs["Base Color"])
    t.link(b.outputs[0], t.out.inputs[0])
    return m


def build_library():
    L = {}
    L["roof"] = ivory_mineral("M_RoofSkin_Ivory", (0.86, 0.835, 0.79, 1), rough=0.48)
    L["wall"] = ivory_mineral("M_Wall_IvoryMineral", (0.83, 0.80, 0.74, 1), rough=0.6)
    L["metal"] = brushed_metal("M_Metal_BrushedChampagne")
    L["metal_dark"] = brushed_metal("M_Metal_Gutter", (0.40, 0.39, 0.37, 1), rough=0.35)
    L["glass"] = glass("M_Glass_Clear")
    L["glass_rail"] = glass("M_Glass_Balustrade", (0.95, 0.975, 0.97, 1))
    L["frosted"] = frosted_glass("M_Glass_FrostedSculpture")
    L["wood"] = wood_rib("M_Wood_SoffitRib")
    L["soffit"] = flat("M_Soffit_Backing", (0.30, 0.235, 0.17, 1), rough=0.85, spec=0.2)
    L["gap"] = flat("M_ShadowGap", (0.035, 0.034, 0.033, 1), rough=0.9, spec=0.1)
    L["stone"] = stone_floor("M_Stone_Floor")
    L["stone_edge"] = ivory_mineral("M_Stone_Edge", (0.74, 0.715, 0.67, 1), rough=0.55, scale=0.6)
    L["water"] = water()
    L["led"] = emission("M_LED_Warm", (1.0, 0.74, 0.46, 1), 9.0)
    L["led_soft"] = emission("M_LED_Soft", (1.0, 0.78, 0.55, 1), 6.0)
    L["bronze"] = flat("M_Bronze", (0.38, 0.25, 0.15, 1), rough=0.32, metallic=1.0)
    L["leaf"] = leaf()
    L["leaf_pine"] = leaf("M_Leaf_Pine", (0.06, 0.12, 0.07, 1), (0.14, 0.21, 0.10, 1))
    L["bark"] = bark()
    L["soil"] = flat("M_Planter_Soil", (0.10, 0.085, 0.07, 1), rough=0.95)
    L["groundcover"] = leaf("M_GroundCover", (0.12, 0.20, 0.08, 1), (0.25, 0.33, 0.13, 1))
    L["rock"] = ivory_mineral("M_Rock", (0.15, 0.14, 0.125, 1), rough=0.85, scale=0.2)
    L["grass"] = flat("M_Island_Grass", (0.16, 0.20, 0.09, 1), rough=0.95)
    L["mountain"] = distance_haze("M_Mountain_Haze", (0.20, 0.24, 0.27, 1), (0.80, 0.70, 0.66, 1))
    L["mountain_far"] = distance_haze("M_Mountain_Far", (0.32, 0.34, 0.38, 1), (0.82, 0.72, 0.68, 1),
                                      near=150, far=1300)
    L["fabric"] = flat("M_Fabric_Cream", (0.86, 0.83, 0.77, 1), rough=0.95, spec=0.2)
    L["oak"] = wood_rib("M_Wood_Furniture")
    L["plinth"] = flat("M_Plinth_Shadow", (0.13, 0.13, 0.13, 1), rough=0.8)
    L["art"] = [painting(f"M_Art_{i}", seed=i * 3.7) for i in range(4)]
    L["frame"] = flat("M_ArtFrame", (0.10, 0.09, 0.08, 1), rough=0.4)
    return L
