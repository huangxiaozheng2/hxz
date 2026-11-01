"""日照分析（北纬 30°，真太阳时）：在模型上做光线追踪，算冬至与夏至两天首层地面、潮池水面每一点的直射日照时数，
以及潮池与门厅地面被晒到的面积比例随时间的变化。玻璃视为透光（不遮挡），人物与湖岸配景不参与；庭院的树参与遮挡。
用法：python solar.py scene.blend 输出.json [--step=0.5] [--minutes=10]
"""
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bpy  # noqa: E402
import numpy as np  # noqa: E402
import shapely  # noqa: E402
from mathutils import Vector  # noqa: E402
from shapely.geometry import Point  # noqa: E402

import plan as P  # noqa: E402
from plan import sun_vector, day_length  # noqa: E402

DAYS = {"winter": -23.44, "equinox": 0.0, "summer": 23.44}
SKIP = ("Glass", "People_", "Shore_", "Islet", "Mountains_", "Water", "LED", "Glow", "Lights_", "Boats_", "Sails")
# 城区与对岸参与遮挡（低角度的冬季午后阳光可能被西南方的城区挡住）


def main(blend, out_json, step=0.5, minutes=10):
    bpy.ops.wm.open_mainfile(filepath=blend)
    for o in bpy.data.objects:
        if o.type != "MESH" or o.name.startswith(SKIP) or any(k in o.name for k in SKIP):
            o.hide_viewport = True
    dg = bpy.context.evaluated_depsgraph_get()
    sc = bpy.context.scene
    # 采样点：首层地面（基座顶 +0.85，含室内）与潮池水面（设计高潮位 ±0.00）
    region = P.PLINTH_CUT.intersection(Point(0, 0).buffer(21.0, 128))
    xs = np.arange(-21, 21 + 1e-6, step)
    X, Y = np.meshgrid(xs, xs)
    X, Y = X.ravel(), Y.ravel()
    on_plinth = shapely.contains_xy(region, X, Y)
    in_pool = shapely.contains_xy(P.BASIN, X, Y)
    keep = on_plinth | in_pool
    X, Y, in_pool = X[keep], Y[keep], in_pool[keep]
    r = np.hypot(X, Y)                                   # 潮池里高出水面的台阶取台阶面
    zp = np.full(len(X), P.HIGH_TIDE + 0.02)
    for r_out, r_in, z in P.BASIN_STEPS:
        ring = (r <= r_out) & (r > r_in)
        zp[ring] = np.maximum(zp[ring], z + 0.02)
    Z = np.where(in_pool, zp, P.FFL + 0.02)
    lobby = shapely.contains_xy(P.E2, X, Y)
    gallery = shapely.contains_xy(P.E1, X, Y)
    pts = np.stack([X, Y, Z], 1)
    out = dict(step=step, x=X.round(2).tolist(), y=Y.round(2).tolist(), pool=in_pool.tolist(), lobby=lobby.tolist(),
               gallery=gallery.tolist(), days={})
    for name, dec in DAYS.items():
        t0, t1 = day_length(P.LATITUDE, dec)
        times = np.arange(math.ceil(t0 * 60 / minutes) * minutes / 60, t1, minutes / 60)
        hours = np.zeros(len(pts))
        series = []
        for t in times:
            v, alt, az = sun_vector(P.LATITUDE, dec, t)
            if alt <= 0.5:
                series.append([round(t, 3), round(alt, 2), round(az, 1), 0.0, 0.0, 0.0])
                continue
            d = Vector(v)
            lit = np.zeros(len(pts), bool)
            for i, p in enumerate(pts):
                hit, *_ = sc.ray_cast(dg, Vector(p) + d * 0.01, d)
                lit[i] = not hit
            hours += lit * (minutes / 60)
            series.append([round(t, 3), round(alt, 2), round(az, 1), round(float(lit[in_pool].mean()), 3),
                           round(float(lit[lobby].mean()), 3), round(float(lit[gallery].mean()), 3)])
            print(name, f"{t:5.2f}", f"alt {alt:5.1f}", "pool", round(float(lit[in_pool].mean()), 2), flush=True)
        out["days"][name] = dict(dec=dec, sunrise=round(t0, 3), sunset=round(t1, 3), hours=hours.round(2).tolist(),
                                 series=series)
    with open(out_json, "w") as f:
        json.dump(out, f)
    print("saved", out_json, len(pts), "points")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    opts = dict(a[2:].split("=") for a in sys.argv[1:] if a.startswith("--") and "=" in a)
    main(args[0], os.path.abspath(args[1]), float(opts.get("step", 0.5)), int(opts.get("minutes", 10)))
