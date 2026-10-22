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
import structure as ST  # noqa: E402
from facade import e1_solid, e1_windows, e1_doors, e2_doors, e3_solid  # noqa: E402


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
        A.build_roof(roof, M, roofs, skylights=P.SKYLIGHTS if roof is P.ROOF_A else ())
        print("roof", roof.name, time.time() - t0)

    e1 = A.build_envelope("Envelope_E1_Gallery", P.E1, P.ROOF_A, P.FFL, M, env, solid=e1_solid, doors=e1_doors,
                          windows=e1_windows,
                          columns=np.vstack([ST.E1_COLS_INNER, ST.E1_COLS_OUTER]))
    e2 = A.build_envelope("Envelope_E2_Lobby", P.E2, P.ROOF_C, P.FFL, M, env, doors=e2_doors, columns=ST.E2_COLS)
    e3 = A.build_envelope("Envelope_E3_Belvedere", P.E3, P.ROOF_B, P.DECK, M, env, solid=e3_solid)
    print("envelopes", time.time() - t0)

    struct = collection("Structure", arch)
    A.build_plinth(P.PLINTH_CUT, P.FFL, M, base, name="Plinth_And_Bridge", cut=P.WATER_CUT)
    A.build_tidal_basin(M, base)
    A.build_deck(M, base)
    A.build_moon_gate(M, base)
    A.build_core(M, struct)
    A.build_supports(M, struct)
    A.build_ramp(M, circ)
    A.build_spiral_stair(M, circ)
    print("base/circ", time.time() - t0)

    site = collection("Site", root)
    S.build_courtyard(M, site)
    S.build_interior(M, site, e1, e2, e3)
    S.build_deck_furniture(M, site)
    S.build_people(M, site)
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
