"""导出 GLB（建筑主体 + 庭院，不含远景配景），便于在网页/其他软件中查看。
程序化材质无法完整导出到 glTF，导出前替换为同色的基础 PBR 参数。
用法：python export_glb.py scene.blend out.glb
"""
import sys
import bpy


def main(src, dst):
    bpy.ops.wm.open_mainfile(filepath=src)
    keep = {"Architecture", "Site", "Courtyard", "Interior_Exhibits", "Terrace_Furniture"}
    for o in bpy.data.objects:
        names = {c.name for c in o.users_collection}
        parents = set()
        for c in bpy.data.collections:
            for ch in c.children_recursive:
                if o.name in ch.objects:
                    parents.add(c.name)
        inside = bool((names | parents) & keep) and o.type == "MESH"
        o.select_set(False)
        o.hide_set(False)
        if inside:
            o.select_set(True)
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
