"""导出 GLB：建筑主体、庭院、室内陈设、人物与湖岸配景（不含水面与远山），便于在网页/其他软件中查看。
程序化材质无法完整导出到 glTF，导出前替换为同色的基础 PBR 参数。
配景树叶在网页里过重：只保留约一成叶片并放大，枝干减面，树冠轮廓与渲染图一致。
用法：python export_glb.py scene.blend out.glb
"""
import sys

import bpy
import bmesh  # noqa: E402  （须在 bpy 之后导入）
import numpy as np

KEEP = {"Architecture", "Courtyard", "Interior_Exhibits", "Terrace_Furniture", "People_ScaleFigures", "Landscape"}
SKIP = ("Water_Surface", "Mountains_")            # 水面由网页自己绘制，远山不导出
LEAF_KEEP, LEAF_SCALE, BRANCH_RATIO = 0.12, 2.4, 0.3


def thin_foliage(o, seed, keep=LEAF_KEEP, scale=LEAF_SCALE):
    """配景树：随机保留一部分叶片，每片绕自身中心放大，树冠疏密与轮廓基本不变。"""
    rng = np.random.default_rng(seed)
    bm = bmesh.new()
    bm.from_mesh(o.data)
    drop = []
    for f in bm.faces:
        if rng.random() < keep:
            c = f.calc_center_median()
            for v in f.verts:
                v.co = c + (v.co - c) * scale
        else:
            drop.append(f)
    bmesh.ops.delete(bm, geom=drop, context="FACES")
    loose = [v for v in bm.verts if not v.link_faces]
    bmesh.ops.delete(bm, geom=loose, context="VERTS")
    bm.to_mesh(o.data)
    bm.free()


def in_keep(o):
    names = {c.name for c in o.users_collection}
    for c in bpy.data.collections:
        for ch in c.children_recursive:
            if o.name in ch.objects:
                names.add(c.name)
    return bool(names & KEEP) and o.type == "MESH" and not o.name.startswith(SKIP)


def main(src, dst):
    bpy.ops.wm.open_mainfile(filepath=src)
    for k, o in enumerate(bpy.data.objects):
        o.select_set(False)
        o.hide_set(False)
        if not in_keep(o):
            continue
        o.select_set(True)
        landscape = o.name.startswith(("Shore_", "Islet"))
        if landscape and o.name.endswith("_Foliage"):
            thin_foliage(o, k)
        elif o.name == "Islets_Shrubs":
            thin_foliage(o, k, keep=0.25, scale=1.8)
        elif o.name == "Courtyard_Tree_Foliage":          # 庭院主景树：只轻度抽稀
            thin_foliage(o, k, keep=0.45, scale=1.4)
        elif landscape and o.name.endswith("_Branches"):
            mod = o.modifiers.new("web_decimate", "DECIMATE")
            mod.ratio = BRANCH_RATIO
    # 简化材质：取 Principled 的基础色 / 金属度 / 粗糙度
    for m in bpy.data.materials:
        if not m.use_nodes:
            continue
        nt = m.node_tree
        pb = next((n for n in nt.nodes if n.bl_idname == "ShaderNodeBsdfPrincipled"), None)
        em = next((n for n in nt.nodes if n.bl_idname == "ShaderNodeEmission"), None)
        out = next(n for n in nt.nodes if n.bl_idname == "ShaderNodeOutputMaterial")
        new = nt.nodes.new("ShaderNodeBsdfPrincipled")
        new.inputs["Base Color"].default_value = m.diffuse_color
        if pb:
            for k in ("Metallic", "Roughness", "Transmission Weight", "IOR"):
                new.inputs[k].default_value = pb.inputs[k].default_value
        if em:
            new.inputs["Emission Color"].default_value = em.inputs["Color"].default_value
            new.inputs["Emission Strength"].default_value = 2.0
        for l in list(out.inputs[0].links):
            nt.links.remove(l)
        nt.links.new(new.outputs[0], out.inputs[0])
    bpy.ops.export_scene.gltf(filepath=dst, export_format="GLB", use_selection=True, export_apply=True,
                              export_cameras=False, export_lights=False, export_yup=True)
    print("exported", dst)


if __name__ == "__main__":
    main(sys.argv[-2], sys.argv[-1])
