"""渲染指定机位。用法：python render.py scene.blend outdir view1,view2 [--samples N] [--scale S]"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bpy  # noqa: E402
import plan as P  # noqa: E402

RES = {
    "hero": (2400, 1000), "hero2": (2400, 1000), "aerial": (1600, 1000), "rear": (1600, 1000), "court": (1600, 1000),
    "interior": (1600, 1000), "terrace": (1600, 1000), "detail": (1600, 1000), "moon": (1600, 1000), "stair": (1600, 1000), "dusk": (2400, 1000), "entry": (1600, 1000),
    "axis": (1600, 1000), "axis_out": (1600, 1000), "court_ht": (1600, 1000), "court_lt": (1600, 1000), "structure": (1600, 1000),
}


def set_time(sc, dusk=False):
    """黄昏版本：太阳没入地平线，室内灯光成为主光。"""
    import math
    sky = next(n for n in sc.world.node_tree.nodes if n.bl_idname == "ShaderNodeTexSky")
    bgn = next(n for n in sc.world.node_tree.nodes if n.bl_idname == "ShaderNodeBackground")
    if "day_el" not in sc:
        sc["day_el"] = sky.sun_elevation
        sc["day_bg"] = bgn.inputs["Strength"].default_value
    sky.sun_elevation = math.radians(-3.0) if dusk else sc["day_el"]
    bgn.inputs["Strength"].default_value = sc["day_bg"] * (1.0 if not dusk else 0.9)
    k = 2.6 if dusk else 1.0
    for l in bpy.data.lights:
        if "base_energy" not in l:
            l["base_energy"] = l.energy
        l.energy = l["base_energy"] * k
    for m in bpy.data.materials:
        if m.name.startswith("M_LED"):
            e = m.node_tree.nodes.get("Emission")
            if e:
                if "base_strength" not in m:
                    m["base_strength"] = e.inputs["Strength"].default_value
                e.inputs["Strength"].default_value = m["base_strength"] * (2.0 if dusk else 1.0)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    opts = dict(a[2:].split("=") for a in sys.argv[1:] if a.startswith("--") and "=" in a)
    blend, outdir, views = args[0], args[1], args[2].split(",")
    samples = int(opts.get("samples", 128))
    scale = float(opts.get("scale", 1.0))
    bpy.ops.wm.open_mainfile(filepath=blend)
    sc = bpy.context.scene
    sc.cycles.samples = samples
    os.makedirs(outdir, exist_ok=True)
    for v in views:
        cam = bpy.data.objects[f"Cam_{v}"]
        sc.camera = cam
        sc.view_settings.exposure = float(cam.get("exposure", 0.0)) + float(opts.get("ev", -0.2))
        w, h = RES.get(v, (1600, 1000))
        sc.render.resolution_x, sc.render.resolution_y = int(w * scale), int(h * scale)
        sc.render.filepath = os.path.join(outdir, f"{v}.png")
        set_time(sc, dusk=(v == "dusk"))
        water = bpy.data.objects.get("Water_Surface")
        if water:
            water.location.z = P.LOW_TIDE if cam.get("tide") == "low" else P.HIGH_TIDE
        t = time.time()
        bpy.ops.render.render(write_still=True)
        print(f"rendered {v} in {time.time() - t:.1f}s", flush=True)


if __name__ == "__main__":
    main()
