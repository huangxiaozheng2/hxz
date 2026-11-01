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
    m.diffuse_color = color
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


def fabric(name, color, sheen=0.35, rough=0.88):
    """织物：高粗糙度 + 柔和光泽边缘 + 细密织纹凹凸（约 1.5 mm）与轻微色差。"""
    m = flat(name, color, rough=rough, spec=0.3)
    t = m.node_tree
    b = next(n for n in t.nodes if n.bl_idname == "ShaderNodeBsdfPrincipled")
    if "Sheen Weight" in b.inputs:
        b.inputs["Sheen Weight"].default_value = sheen
        b.inputs["Sheen Roughness"].default_value = 0.5
    tc = t.nodes.new("ShaderNodeTexCoord")
    weave = t.nodes.new("ShaderNodeTexWave")
    weave.wave_type = "BANDS"; weave.bands_direction = "DIAGONAL"
    weave.inputs["Scale"].default_value = 650.0
    weave.inputs["Distortion"].default_value = 2.0
    t.links.new(tc.outputs["Object"], weave.inputs["Vector"])
    nz = t.nodes.new("ShaderNodeTexNoise")
    nz.inputs["Scale"].default_value = 6.0
    t.links.new(tc.outputs["Object"], nz.inputs["Vector"])
    bump = t.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.12
    bump.inputs["Distance"].default_value = 0.0006
    t.links.new(weave.outputs["Fac"], bump.inputs["Height"])
    t.links.new(bump.outputs["Normal"], b.inputs["Normal"])
    mix = t.nodes.new("ShaderNodeMix"); mix.data_type = "RGBA"
    mix.inputs["Factor"].default_value = 0.12
    mix.inputs["A"].default_value = color
    mix.inputs["B"].default_value = tuple(c * 0.8 for c in color[:3]) + (1,)
    t.links.new(nz.outputs["Fac"], mix.inputs["Factor"])
    t.links.new(mix.outputs["Result"], b.inputs["Base Color"])
    return m


def skin(name, color):
    """皮肤：次表面散射（红色通道散得更远）+ 细微毛孔凹凸。"""
    m = flat(name, color, rough=0.5, spec=0.45)
    t = m.node_tree
    b = next(n for n in t.nodes if n.bl_idname == "ShaderNodeBsdfPrincipled")
    b.inputs["Subsurface Weight"].default_value = 0.25
    b.inputs["Subsurface Radius"].default_value = (1.0, 0.38, 0.22)
    b.inputs["Subsurface Scale"].default_value = 0.012
    tc = t.nodes.new("ShaderNodeTexCoord")
    nz = t.nodes.new("ShaderNodeTexNoise")
    nz.inputs["Scale"].default_value = 900.0
    t.links.new(tc.outputs["Object"], nz.inputs["Vector"])
    bump = t.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.05
    bump.inputs["Distance"].default_value = 0.0002
    t.links.new(nz.outputs["Fac"], bump.inputs["Height"])
    t.links.new(bump.outputs["Normal"], b.inputs["Normal"])
    return m


def hair_strands(name, melanin=0.9, redness=0.25, color=None):
    """发丝：Chiang 毛发模型；color 给定时直接着色（花白发）。"""
    m = _new(name); t = NT(m)
    hb = t.n("ShaderNodeBsdfHairPrincipled", (500, 0))
    hb.model = "CHIANG"
    if color is not None:
        hb.parametrization = "COLOR"
        hb.inputs["Color"].default_value = color
    else:
        hb.parametrization = "MELANIN"
        hb.inputs["Melanin"].default_value = melanin
        hb.inputs["Melanin Redness"].default_value = redness
    for k, v in (("Roughness", 0.32), ("Radial Roughness", 0.38), ("Random Color", 0.08), ("Random Roughness", 0.2)):
        if k in hb.inputs:
            hb.inputs[k].default_value = v
    t.link(hb.outputs[0], t.out.inputs[0])
    m.diffuse_color = color or (0.02, 0.018, 0.015, 1)
    return m


def lawn(name="M_Lawn"):
    """草坪：两种绿色的大尺度斑驳 + 细碎明暗与微凹凸。"""
    m = _new(name); t = NT(m)
    tc = t.n("ShaderNodeTexCoord", (-900, 0))
    big = t.n("ShaderNodeTexNoise", (-650, 200), **{"Scale": 0.08, "Detail": 4.0, "Roughness": 0.55})
    t.link(tc.outputs["Object"], big.inputs["Vector"])
    fine = t.n("ShaderNodeTexNoise", (-650, -150), **{"Scale": 9.0, "Detail": 8.0, "Roughness": 0.7})
    t.link(tc.outputs["Object"], fine.inputs["Vector"])
    ramp = t.n("ShaderNodeValToRGB", (-400, 200))
    ramp.color_ramp.elements[0].color = (0.12, 0.17, 0.06, 1)
    ramp.color_ramp.elements[1].color = (0.33, 0.35, 0.13, 1)
    mixv = t.math("ADD", t.math("MULTIPLY", big.outputs["Fac"], 0.75), t.math("MULTIPLY", fine.outputs["Fac"], 0.25))
    t.link(mixv, ramp.inputs["Fac"])
    bump = t.n("ShaderNodeBump", (200, -300), **{"Strength": 0.25, "Distance": 0.02})
    t.link(fine.outputs["Fac"], bump.inputs["Height"])
    b = t.bsdf(**{"Roughness": 0.92, "Specular IOR Level": 0.25})
    t.link(ramp.outputs["Color"], b.inputs["Base Color"])
    t.link(bump.outputs["Normal"], b.inputs["Normal"])
    t.link(b.outputs[0], t.out.inputs[0])
    m.diffuse_color = (0.17, 0.21, 0.08, 1)
    return m


def tidal_stone(name="M_Stone_Tidal", color=(0.70, 0.675, 0.63, 1), tide=0.0):
    """潮池石材：设计高潮位以下颜色加深、表面湿润，近池底泛出极淡的青褐色——落潮后留下潮痕。"""
    m = _new(name); t = NT(m)
    geo = t.n("ShaderNodeNewGeometry", (-1200, 200))
    sep = t.n("ShaderNodeSeparateXYZ", (-1000, 200))
    t.link(geo.outputs["Position"], sep.inputs[0])
    wet = t.n("ShaderNodeMapRange", (-800, 200), **{"From Min": tide + 0.06, "From Max": tide - 0.04,
                                                   "To Min": 0.0, "To Max": 1.0})
    t.link(sep.outputs[2], wet.inputs["Value"])
    deep = t.n("ShaderNodeMapRange", (-800, -50), **{"From Min": tide - 0.25, "From Max": tide - 0.75,
                                                    "To Min": 0.0, "To Max": 1.0})
    t.link(sep.outputs[2], deep.inputs["Value"])
    tc = t.n("ShaderNodeTexCoord", (-1200, -300))
    nz = t.n("ShaderNodeTexNoise", (-900, -300), **{"Scale": 2.5, "Detail": 8.0, "Roughness": 0.6})
    t.link(tc.outputs["Object"], nz.inputs["Vector"])
    ramp = t.n("ShaderNodeValToRGB", (-650, -300))
    ramp.color_ramp.elements[0].color = tuple(c * 0.94 for c in color[:3]) + (1,)
    ramp.color_ramp.elements[1].color = tuple(min(1, c * 1.04) for c in color[:3]) + (1,)
    t.link(nz.outputs["Fac"], ramp.inputs["Fac"])
    dark = t.n("ShaderNodeMix", (-350, 0), data_type="RGBA")
    t.link(wet.outputs[0], dark.inputs["Factor"])
    t.link(ramp.outputs[0], dark.inputs["A"])
    dark.inputs["B"].default_value = (0.36, 0.34, 0.30, 1)
    algae = t.n("ShaderNodeMix", (-120, 0), data_type="RGBA")
    t.link(t.math("MULTIPLY", deep.outputs[0], 0.55), algae.inputs["Factor"])
    t.link(dark.outputs["Result"], algae.inputs["A"])
    algae.inputs["B"].default_value = (0.20, 0.24, 0.17, 1)
    b = t.bsdf(**{"Specular IOR Level": 0.5})
    t.link(algae.outputs["Result"], b.inputs["Base Color"])
    t.link(t.math("SUBTRACT", 0.6, t.math("MULTIPLY", wet.outputs[0], 0.38)), b.inputs["Roughness"])
    bump = t.n("ShaderNodeBump", (250, -300), **{"Strength": 0.15, "Distance": 0.004})
    t.link(nz.outputs["Fac"], bump.inputs["Height"])
    t.link(bump.outputs[0], b.inputs["Normal"])
    t.link(b.outputs[0], t.out.inputs[0])
    m.diffuse_color = color
    return m


def facade(name, wall, glass=(0.05, 0.06, 0.07, 1), bay=3.2, floor=3.4, win=1.4, sill=0.9, head=2.6, lit=0.18,
           frame=None, glass_rough=0.12, haze=None):
    """建筑立面：按 UV（u 沿立面、v 为高度，单位 m）生成窗格；一部分窗亮灯（节点 WindowLight 控制亮度，黄昏调高）。
    haze=(near, far) 时随视距透出天色（对岸远景）。"""
    m = _new(name); t = NT(m)
    uv = t.n("ShaderNodeUVMap", (-1400, 0))
    sep = t.n("ShaderNodeSeparateXYZ", (-1200, 0))
    t.link(uv.outputs[0], sep.inputs[0])
    u, v = sep.outputs[0], sep.outputs[1]
    cu = t.math("FLOOR", t.math("DIVIDE", u, bay))
    cv = t.math("FLOOR", t.math("DIVIDE", v, floor))
    fu = t.math("FRACT", t.math("DIVIDE", u, bay))
    fv = t.math("FRACT", t.math("DIVIDE", v, floor))
    a = (1 - win / bay) / 2
    inx = t.math("MULTIPLY", t.math("GREATER_THAN", fu, a), t.math("LESS_THAN", fu, 1 - a))
    inz = t.math("MULTIPLY", t.math("GREATER_THAN", fv, sill / floor), t.math("LESS_THAN", fv, head / floor))
    mask = t.math("MULTIPLY", inx, inz)
    cmb = t.n("ShaderNodeCombineXYZ", (-600, -300))
    t.link(cu, cmb.inputs[0]); t.link(cv, cmb.inputs[1])
    cmb.inputs[2].default_value = (sum(map(ord, name)) % 997) / 7.0          # 确定性的种子（不用 hash()）
    wn = t.n("ShaderNodeTexWhiteNoise", (-400, -300))
    wn.noise_dimensions = "3D"
    t.link(cmb.outputs[0], wn.inputs["Vector"])
    litm = t.math("MULTIPLY", t.math("GREATER_THAN", wn.outputs["Value"], 1 - lit), mask)
    # 墙面细微色差
    tc = t.n("ShaderNodeTexCoord", (-1400, 400))
    nz = t.n("ShaderNodeTexNoise", (-1200, 400), **{"Scale": 0.35, "Detail": 3.0})
    t.link(tc.outputs["Object"], nz.inputs["Vector"])
    wmix = t.n("ShaderNodeMix", (-800, 400)); wmix.data_type = "RGBA"
    t.link(nz.outputs["Fac"], wmix.inputs["Factor"])
    wmix.inputs["A"].default_value = tuple(c * 0.92 for c in wall[:3]) + (1,)
    wmix.inputs["B"].default_value = tuple(min(1.0, c * 1.05) for c in wall[:3]) + (1,)
    col = t.n("ShaderNodeMix", (-300, 200)); col.data_type = "RGBA"
    t.link(mask, col.inputs["Factor"])
    t.link(wmix.outputs["Result"], col.inputs["A"])
    col.inputs["B"].default_value = glass
    rough = t.n("ShaderNodeMapRange", (-300, 0), **{"To Min": 0.82, "To Max": glass_rough})
    t.link(mask, rough.inputs["Value"])
    b = t.bsdf(**{"Specular IOR Level": 0.5})
    t.link(col.outputs["Result"], b.inputs["Base Color"])
    t.link(rough.outputs[0], b.inputs["Roughness"])
    wl = t.n("ShaderNodeValue", (-300, -400)); wl.name = "WindowLight"; wl.label = "WindowLight"
    wl.outputs[0].default_value = 1.2
    b.inputs["Emission Color"].default_value = (1.0, 0.72, 0.42, 1)
    t.link(t.math("MULTIPLY", litm, wl.outputs[0]), b.inputs["Emission Strength"])
    if haze is None:
        t.link(b.outputs[0], t.out.inputs[0])
    else:
        cam = t.n("ShaderNodeCameraData", (-300, -700))
        mr = t.n("ShaderNodeMapRange", (0, -700), **{"From Min": haze[0], "From Max": haze[1]})
        t.link(cam.outputs["View Distance"], mr.inputs["Value"])
        tr = t.n("ShaderNodeBsdfTransparent", (500, -300))
        mx = t.n("ShaderNodeMixShader", (720, 0))
        t.link(t.math("MULTIPLY", t.math("POWER", mr.outputs[0], 0.75), 0.9), mx.inputs[0])
        t.link(b.outputs[0], mx.inputs[1]); t.link(tr.outputs[0], mx.inputs[2])
        t.link(mx.outputs[0], t.out.inputs[0])
    m.diffuse_color = wall
    return m


def roof_tile(name="M_Roof_Tile", color=(0.10, 0.105, 0.11, 1)):
    """小青瓦：沿坡向的瓦垄（UV 的 u 为沿屋脊方向）+ 轻微凹凸。"""
    m = _new(name); t = NT(m)
    uv = t.n("ShaderNodeUVMap", (-900, 0))
    wv = t.n("ShaderNodeTexWave", (-650, 0), **{"Scale": 1.0, "Distortion": 0.4, "Detail": 1.0})
    wv.wave_type = "BANDS"; wv.bands_direction = "X"
    mp = t.n("ShaderNodeMapping", (-800, 0))
    mp.inputs["Scale"].default_value = (4.2, 1.0, 1.0)
    t.link(uv.outputs[0], mp.inputs["Vector"]); t.link(mp.outputs[0], wv.inputs["Vector"])
    ramp = t.n("ShaderNodeValToRGB", (-400, 0))
    ramp.color_ramp.elements[0].color = tuple(c * 0.7 for c in color[:3]) + (1,)
    ramp.color_ramp.elements[1].color = tuple(c * 1.35 for c in color[:3]) + (1,)
    t.link(wv.outputs["Fac"], ramp.inputs["Fac"])
    b = t.bsdf(**{"Roughness": 0.62, "Specular IOR Level": 0.45})
    t.link(ramp.outputs[0], b.inputs["Base Color"])
    bump = t.n("ShaderNodeBump", (250, -250), **{"Strength": 0.35, "Distance": 0.02})
    t.link(wv.outputs["Fac"], bump.inputs["Height"]); t.link(bump.outputs[0], b.inputs["Normal"])
    t.link(b.outputs[0], t.out.inputs[0])
    m.diffuse_color = color
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
    L["leaf_willow"] = leaf("M_Leaf_Willow", (0.20, 0.28, 0.09, 1), (0.40, 0.46, 0.17, 1))
    L["bark"] = bark()
    L["soil"] = flat("M_Planter_Soil", (0.10, 0.085, 0.07, 1), rough=0.95)
    L["groundcover"] = leaf("M_GroundCover", (0.12, 0.20, 0.08, 1), (0.25, 0.33, 0.13, 1))
    L["rock"] = ivory_mineral("M_Rock", (0.15, 0.14, 0.125, 1), rough=0.85, scale=0.2)
    L["grass"] = flat("M_Island_Grass", (0.16, 0.20, 0.09, 1), rough=0.95)
    L["lawn"] = lawn()
    L["paving"] = stone_floor("M_Paving_Granite", (0.37, 0.345, 0.31, 1), ring=0.9, seg=0.9, polish=0.86)
    L["mountain"] = distance_haze("M_Mountain_Haze", (0.20, 0.24, 0.27, 1), (0.80, 0.70, 0.66, 1))
    L["mountain_far"] = distance_haze("M_Mountain_Far", (0.32, 0.34, 0.38, 1), (0.82, 0.72, 0.68, 1),
                                      near=150, far=1300)
    L["fabric"] = flat("M_Fabric_Cream", (0.86, 0.83, 0.77, 1), rough=0.95, spec=0.2)
    L["oak"] = wood_rib("M_Wood_Furniture")
    L["plinth"] = flat("M_Plinth_Shadow", (0.13, 0.13, 0.13, 1), rough=0.8)
    L["stone_tidal"] = tidal_stone()
    L["bronze_satin"] = flat("M_Bronze_Satin", (0.42, 0.33, 0.24, 1), rough=0.38, metallic=1.0)
    L["fabric_tops"] = [fabric(f"M_People_Top_{k}", c) for k, c in enumerate((
        (0.80, 0.78, 0.74, 1), (0.06, 0.08, 0.14, 1), (0.60, 0.50, 0.36, 1), (0.03, 0.03, 0.035, 1),
        (0.23, 0.24, 0.14, 1), (0.44, 0.17, 0.08, 1), (0.42, 0.52, 0.63, 1), (0.36, 0.36, 0.37, 1)))]
    L["fabric_bottoms"] = [fabric(f"M_People_Bottom_{k}", c) for k, c in enumerate((
        (0.06, 0.08, 0.12, 1), (0.03, 0.03, 0.033, 1), (0.46, 0.40, 0.28, 1), (0.23, 0.23, 0.24, 1),
        (0.72, 0.68, 0.59, 1), (0.30, 0.12, 0.10, 1)))]
    L["shoes"] = [flat(f"M_People_Shoe_{k}", c, rough=0.55, spec=0.4) for k, c in enumerate((
        (0.03, 0.028, 0.026, 1), (0.82, 0.80, 0.77, 1), (0.17, 0.10, 0.055, 1)))]
    L["hairs"] = [flat(f"M_People_Hair_{k}", c, rough=0.5, spec=0.45) for k, c in enumerate((
        (0.018, 0.016, 0.014, 1), (0.06, 0.04, 0.028, 1), (0.15, 0.095, 0.055, 1), (0.42, 0.41, 0.39, 1)))]
    L["hair_strands"] = [hair_strands("M_People_HairStrands_0", 0.95, 0.15),
                         hair_strands("M_People_HairStrands_1", 0.85, 0.3),
                         hair_strands("M_People_HairStrands_2", 0.66, 0.45),
                         hair_strands("M_People_HairStrands_3", color=(0.36, 0.35, 0.34, 1))]
    L["skins"] = [skin(f"M_People_Skin_{k}", c) for k, c in enumerate((
        (0.74, 0.56, 0.45, 1), (0.62, 0.46, 0.36, 1), (0.52, 0.36, 0.26, 1), (0.36, 0.24, 0.17, 1)))]
    L["art"] = [painting(f"M_Art_{i}", seed=i * 3.7) for i in range(4)]
    # 湖岸城区与对岸
    L["town"] = {
        "jiangnan": facade("M_Facade_Jiangnan", (0.86, 0.845, 0.81, 1), glass=(0.07, 0.055, 0.045, 1), bay=3.0, floor=3.4,
                           win=1.3, sill=1.0, head=2.6, lit=0.22),
        "modern": facade("M_Facade_Modern", (0.66, 0.62, 0.56, 1), bay=3.0, floor=4.0, win=2.6, sill=0.3, head=3.5,
                         lit=0.25),
        "modern_b": facade("M_Facade_Modern_Timber", (0.55, 0.40, 0.27, 1), bay=2.4, floor=4.0, win=1.9, sill=0.3,
                           head=3.5, lit=0.25),
        "residential": facade("M_Facade_Residential", (0.80, 0.765, 0.71, 1), bay=3.6, floor=3.0, win=1.8, sill=0.9,
                              head=2.4, lit=0.2),
        "residential_b": facade("M_Facade_Residential_Terracotta", (0.66, 0.50, 0.40, 1), bay=3.4, floor=3.0, win=1.6,
                                sill=0.9, head=2.4, lit=0.2),
        "residential_c": facade("M_Facade_Residential_Grey", (0.58, 0.59, 0.58, 1), bay=3.9, floor=3.0, win=2.2,
                                sill=0.6, head=2.5, lit=0.2),
        "tower": facade("M_Facade_Tower", (0.30, 0.33, 0.35, 1), glass=(0.08, 0.1, 0.12, 1), bay=1.5, floor=3.4,
                        win=1.38, sill=0.8, head=3.2, lit=0.16, glass_rough=0.06),
    }
    L["town_far"] = facade("M_Facade_Far", (0.62, 0.6, 0.58, 1), bay=4.0, floor=3.2, win=2.4, sill=0.8, head=2.6,
                           lit=0.12, haze=(150.0, 1400.0))
    L["roof_tile"] = roof_tile()
    L["asphalt"] = ivory_mineral("M_Asphalt", (0.10, 0.10, 0.105, 1), rough=0.88, scale=0.6)
    L["far_land"] = distance_haze("M_FarShore_Land", (0.18, 0.21, 0.13, 1), near=250, far=1700)
    L["pagoda"] = distance_haze("M_Pagoda", (0.55, 0.42, 0.30, 1), near=250, far=1800, rough=0.7)
    L["boat_hull"] = flat("M_Boat_Hull", (0.82, 0.80, 0.76, 1), rough=0.35, spec=0.5)
    L["boat_wood"] = wood_rib("M_Boat_Wood")
    L["sail"] = flat("M_Sail", (0.92, 0.90, 0.85, 1), rough=0.75, spec=0.2)
    L["frame"] = flat("M_ArtFrame", (0.10, 0.09, 0.08, 1), rough=0.4)
    return L
