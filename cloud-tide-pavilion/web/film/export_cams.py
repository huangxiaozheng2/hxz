"""导出 Blender 场景里各机位的相机（场景坐标 x 东、y 上、z 南，与网页 GLB 一致）→ web/film/cams.json。
导览影片里三维镜头经过这些机位时与对应的渲染图同一视角，两者叠化。相机或模型调整后重新运行：
    ../venv/bin/python web/film/export_cams.py model/cloud_tide_pavilion.blend
"""
import json
import math
import os
import sys

import bpy

bpy.ops.wm.open_mainfile(filepath=sys.argv[-1])
cams = {}
for o in sorted(bpy.data.objects, key=lambda o: o.name):
    if o.type != "CAMERA":
        continue
    c, m = o.data, o.matrix_world
    fwd = -(m.to_3x3().col[2])                                     # 相机看向自身 -Z
    gl = lambda v: [round(v[0], 4), round(v[2], 4), round(-v[1], 4)]   # Blender (x, y, z) → glTF (x, z, -y)
    # 横幅渲染图的传感器宽度对应画面宽度：水平视角
    cams[o.name] = dict(pos=gl(m.translation), dir=gl(fwd), hfov=round(2 * math.degrees(math.atan(c.sensor_width / 2 / c.lens)), 3))
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cams.json")
with open(out, "w") as f:
    json.dump(cams, f, ensure_ascii=False, indent=1)
print("cameras →", out, len(cams))
