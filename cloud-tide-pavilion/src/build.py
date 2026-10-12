"""总装：生成完整的“云上潮汐馆”场景并保存为 .blend。

用法：python build.py [输出.blend] [--lite]
"""
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bpy  # noqa: E402

import plan as P  # noqa: E402
import materials  # noqa: E402
import architecture as A  # noqa: E402
import scenery as S  # noqa: E402
import scene as SC  # noqa: E402
from bl import collection  # noqa: E402


def angle_deg(p):
    return np.degrees(np.arctan2(p[:, 1], p[:, 0])) % 360


def outer_side(mp, nrm):
    radial = mp / np.linalg.norm(mp, axis=1, keepdims=True)
    return (nrm * radial).sum(1)


def ranges_from_mask(mask):
    idx = np.where(mask)[0]
    if not len(idx):
        return []
    runs = np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)
    return [(int(r[0]), int(r[-1]) + 1) for r in runs]


def end_caps(which):
    """月牙端部（径向短边）做成实墙：玻璃与墙体在端部清晰交接。"""
    def f(mp, nrm):
        th = angle_deg(mp)
        radial = np.abs(outer_side(mp, nrm)) < 0.55
        out = []
        for lo, hi in which:
            out += ranges_from_mask(radial & (th > lo) & (th < hi))
        return out
    return f


e1_solid = end_caps([(0, 30)])
e2_solid = end_caps([(220, 245), (305, 330)])


def e1_doors(mp, nrm):
    th = angle_deg(mp)
    inner = outer_side(mp, nrm) < -0.8
    out = []
    for target in (40, 150):
        cand = np.where(inner)[0]
        if len(cand):
            out.append(int(cand[np.argmin(np.abs(th[cand] - target))]))
    return out


def e2_doors(mp, nrm):
    # 南侧凹入门廊的入口门 + 朝庭院的门
    out = []
    south = np.where((nrm[:, 1] < -0.85) & (np.abs(mp[:, 0]) < 2.2))[0]
    if len(south):
        out += [int(i) for i in south[np.argsort(np.abs(mp[south, 0]))[:1]]]
    inner = np.where(outer_side(mp, nrm) < -0.85)[0]
    th = angle_deg(mp)
    if len(inner):
        out.append(int(inner[np.argmin(np.abs(th[inner] - 270))]))
    return out


def main(out_path, lite=False):
    t0 = time.time()
    # 清空默认场景
    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o)
    for c in list(bpy.data.collections):
        bpy.data.collections.remove(c)
    root = collection("CloudTidePavilion")
    arch = collection("Architecture", root)
    roofs = collection("Roofs", arch)
    env = collection("Envelopes", arch)
    base = collection("Plinth_Terraces", arch)
    circ = collection("Circulation", arch)
    M = materials.build_library()
    print("materials", time.time() - t0)

    for roof in P.ROOFS:
        A.build_roof(roof, M, roofs)
        print("roof", roof.name, time.time() - t0)

    e1 = A.build_envelope("Envelope_E1_Gallery", P.E1, P.ROOF_A, P.FFL, M, env, solid=e1_solid, doors=e1_doors)
    e2 = A.build_envelope("Envelope_E2_Lobby", P.E2, P.ROOF_C, P.FFL, M, env, solid=e2_solid, doors=e2_doors)
    e3 = A.build_envelope("Envelope_E3_Belvedere", P.E3, P.ROOF_B, P.DECK, M, env, column_every=None)
    print("envelopes", time.time() - t0)

    A.build_plinth(P.PLINTH_ALL, P.FFL, M, base, name="Plinth_And_Bridge")
    A.build_deck(M, base)
    A.build_moon_gate(M, base)
    A.build_ramp(M, circ)
    A.build_spiral_stair(M, circ)
    print("base/circ", time.time() - t0)

    site = collection("Site", root)
    S.build_courtyard(M, site)
    S.build_interior(M, site, e1, e2, e3)
    S.build_deck_furniture(M, site)
    if not lite:
        S.build_landscape(M, site)
    else:
        from bl import mesh_object
        s = 2600
        mesh_object("Water_Surface", [[-s, -s, 0], [s, -s, 0], [s, s, 0], [-s, s, 0]], [[0, 1, 2, 3]],
                    collection("Landscape", site), [M["water"]], smooth=False)
    print("site", time.time() - t0)

    SC.setup_world()
    SC.setup_lights(root)
    SC.add_cameras(root)
    SC.setup_render()
    bpy.context.scene.camera = bpy.data.objects["Cam_hero"]
    bpy.context.scene.unit_settings.system = "METRIC"
    bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(out_path), compress=True)
    print("saved", out_path, time.time() - t0)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    main(args[0] if args else "cloud_tide_pavilion.blend", lite="--lite" in sys.argv)
