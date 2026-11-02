"""网页构建：把仓库里的页面源文件、渲染图与模型组装成可直接发布的静态页面。

    ../venv/bin/python web/build.py            # 展示网站与 AR 网页都建（在 cloud-tide-pavilion/ 下运行）
    ../venv/bin/python web/build.py site       # 只建展示网站 → web/dist/site
    ../venv/bin/python web/build.py ar         # 只建 AR 网页 → web/dist/ar
    ../venv/bin/python web/build.py film       # 导览影片的逐帧渲染页 → web/dist/film（再用 film/render.mjs 出帧）

先在 web/ 里运行一次 npm install（gltf-transform 等工具，版本与网页引用的 CDN 版本一致）。

· 展示网站：页面引用的图片，凡与渲染图同名的直接复制；分析图裁去白边、结构轴测加标注、材料局部从渲染图裁出，
  都由 src/boards.py 的同一套函数生成；网页模型由 model/cloud_tide_pavilion.glb 经 site_model.mjs 简化、
  meshopt 压缩得到（约 7 MB / 63 万面，原样压缩是 10 MB / 127 万面）。
· AR 网页：模板里的导航数据、二维码、文件大小与面数都在这里填入；二维码与下载链接指向最后一次修改 AR 文件的
  那次提交（经 jsDelivr 分发），所以 AR 文件改动后要先提交、推送，再构建发布。
· 两个页面都以 base64 JSON 承载模型（发布平台只收常见网页文件类型）。
"""
import argparse
import base64
import io
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                                   # cloud-tide-pavilion/
REPO = os.path.dirname(ROOT)
RENDERS = os.path.join(ROOT, "renders")
MODEL = os.path.join(ROOT, "model")
DIST = os.path.join(HERE, "dist")
GLTF_TRANSFORM = os.path.join(HERE, "node_modules", ".bin", "gltf-transform")
sys.path.insert(0, os.path.join(ROOT, "src"))

# 展示网站模型：先由 site_model.mjs 去掉不用的 UV、逐个构件按 1 cm 绝对误差简化（人物、金属柱与斜撑不动），
# 再压缩；保留构件名（网页按构件剖切），不合并网格
SITE_ERR_M = 0.01
SITE_GLTF = ["--compress", "meshopt", "--simplify", "false", "--join", "false", "--palette", "false", "--texture-compress", "false"]
# AR 网页模型：同样的简化；不做实例化（网页按材质名分层）
AR_GLTF = ["--compress", "meshopt", "--texture-compress", "false", "--palette", "false", "--instance", "false",
           "--simplify", "true", "--simplify-ratio", "0", "--simplify-error", "0.0003", "--simplify-lock-border", "true"]
# 展示网站上由渲染图加工出来的图片
TRIMMED = {"detail_eave", "detail_eave_tall", "elevation_south", "environment", "exploded", "form", "location",
           "plan_deck", "plan_ground", "section_2", "solar", "support_map"}
STATIC = {"waternormals.jpg"}                                  # 直接放在 web/site/ 里的素材


def mb(nbytes):
    return f"{nbytes / 1e6:.1f}"


def optimize(src, dst, args):
    if not os.path.exists(GLTF_TRANSFORM):
        sys.exit("找不到 gltf-transform：先在 web/ 里运行 npm install")
    subprocess.run([GLTF_TRANSFORM, "optimize", src, dst, *args], check=True, stdout=subprocess.DEVNULL)


def glb_triangles(path):
    """GLB 的三角面数：每个网格算一次（与 gltf-transform 的统计口径一致）。"""
    data = open(path, "rb").read()
    n = struct.unpack_from("<I", data, 12)[0]
    j = json.loads(data[20:20 + n])
    acc = j["accessors"]
    tris = 0
    for m in j.get("meshes", []):
        for p in m["primitives"]:
            tris += (acc[p["indices"]]["count"] if "indices" in p else acc[p["attributes"]["POSITION"]]["count"]) // 3
    return tris


def write_model_json(glb, out):
    """模型以 base64 写进 JSON：{"format", "encoding", "data"}。"""
    data = open(glb, "rb").read()
    with open(out, "w") as f:
        json.dump({"format": "glb+meshopt", "encoding": "base64", "data": base64.b64encode(data).decode()}, f)
    return len(data)


def fresh(path):
    shutil.rmtree(path, ignore_errors=True)
    os.makedirs(path)
    return path


def build_site():
    import boards
    from PIL import Image
    out = fresh(os.path.join(DIST, "site"))
    img, mdl = fresh(os.path.join(out, "img")), fresh(os.path.join(out, "model"))
    html = open(os.path.join(HERE, "site", "index.html"), encoding="utf-8").read()
    shutil.copy(os.path.join(HERE, "site", "index.html"), out)
    names = sorted(set(re.findall(r"""img/([\w.-]+\.(?:jpg|png|webp))""", html)))
    for name in names:
        stem, dst = os.path.splitext(name)[0], os.path.join(img, name)
        if name in STATIC:
            shutil.copy(os.path.join(HERE, "site", name), dst)
        elif stem in TRIMMED:
            boards.trimmed(os.path.join(RENDERS, stem + ".png")).save(dst, optimize=True)
        elif name == "struct_axo.jpg":
            boards.axo_with_labels(os.path.join(RENDERS, "struct_axo.png"), 2400, 1560, label_w=460, fs=34,
                                   margin=36).save(dst, quality=88)
        elif re.fullmatch(r"mat_\d\.jpg", name):
            _, _, _, src, box = boards.MATERIAL_CARDS[int(name[4])]
            boards.shot(RENDERS, src).crop(box).resize((500, 390), Image.LANCZOS).save(dst, quality=88)
        elif os.path.exists(os.path.join(RENDERS, name)):
            shutil.copy(os.path.join(RENDERS, name), dst)
        else:
            sys.exit(f"页面引用了 img/{name}，但 renders/ 里没有，也不知道怎么生成")
    size, tris = site_model_json(os.path.join(mdl, "pavilion.json"))
    js = os.path.getsize(os.path.join(mdl, "pavilion.json"))
    print(f"site → {os.path.relpath(out, ROOT)}：{len(names)} 张图，网页模型 {mb(size)} MB（下载 {mb(js)} MB）/ {tris / 1e4:.1f} 万面")


def site_model_json(out):
    """展示网站（与导览影片）用的网页模型：简化、压缩后写成 base64 JSON；返回 GLB 字节数与面数。"""
    with tempfile.TemporaryDirectory() as tmp:
        pre, glb = os.path.join(tmp, "pre.glb"), os.path.join(tmp, "web.glb")
        subprocess.run(["node", os.path.join(HERE, "site_model.mjs"), os.path.join(MODEL, "cloud_tide_pavilion.glb"), pre,
                        str(SITE_ERR_M)], check=True, stdout=subprocess.DEVNULL)
        optimize(pre, glb, SITE_GLTF)
        return write_model_json(glb, out), glb_triangles(glb)


def copy_fonts(dst):
    """思源宋体 400/600、思源黑体 400/500（fontsource 的 woff2 分片，按 unicode-range 按需加载）。"""
    fresh(os.path.join(dst, "files"))
    for pkg, tag, weights in (("noto-serif-sc", "serif", (400, 600)), ("noto-sans-sc", "sans", (400, 500))):
        src = os.path.join(HERE, "node_modules", "@fontsource", pkg)
        if not os.path.isdir(src):
            sys.exit(f"找不到 @fontsource/{pkg}：先在 web/ 里运行 npm install")
        for w in weights:
            css = open(os.path.join(src, f"{w}.css"), encoding="utf-8").read()
            css = re.sub(r",\s*url\(\./files/[^)]+\.woff\) format\('woff'\)", "", css)     # 只留 woff2
            for name in re.findall(r"url\(\./files/([^)]+\.woff2)\)", css):
                shutil.copy(os.path.join(src, "files", name), os.path.join(dst, "files", name))
            open(os.path.join(dst, f"{tag}-{w}.css"), "w", encoding="utf-8").write(css)


def build_film():
    out = fresh(os.path.join(DIST, "film"))
    img, mdl = fresh(os.path.join(out, "img")), fresh(os.path.join(out, "model"))
    html = open(os.path.join(HERE, "film", "index.html"), encoding="utf-8").read()
    shutil.copy(os.path.join(HERE, "film", "index.html"), out)
    shutil.copy(os.path.join(HERE, "film", "cams.json"), out)
    shutil.copy(os.path.join(HERE, "site", "waternormals.jpg"), img)
    names = sorted(set(re.findall(r"img: '(\w+)'", html)))
    for n in names:
        shutil.copy(os.path.join(RENDERS, n + ".jpg"), img)
    site_json = os.path.join(DIST, "site", "model", "pavilion.json")       # 展示网站已建好且比模型新就直接用
    if os.path.exists(site_json) and os.path.getmtime(site_json) > os.path.getmtime(os.path.join(MODEL, "cloud_tide_pavilion.glb")):
        shutil.copy(site_json, mdl)
    else:
        site_model_json(os.path.join(mdl, "pavilion.json"))
    copy_fonts(os.path.join(out, "fonts"))
    print(f"film → {os.path.relpath(out, ROOT)}：{len(names)} 张渲染图（{'、'.join(names)}）")


def ar_commit():
    """最后一次修改 AR 文件的提交：二维码与下载链接经 jsDelivr 指向这次提交里的文件。"""
    files = [os.path.join("cloud-tide-pavilion", "model", f) for f in
             ("cloud_tide_pavilion_ar.usdz", "cloud_tide_pavilion_ar_open.usdz", "cloud_tide_pavilion_ar.glb")]
    git = lambda *a: subprocess.run(["git", "-C", REPO, *a], capture_output=True, text=True).stdout.strip()
    if git("status", "--porcelain", "--", *files):
        print("注意：AR 文件有未提交的改动，链接仍指向上一次提交的版本")
    sha = git("log", "-1", "--format=%H", "--", *files)
    if not sha:
        sys.exit("找不到修改 AR 文件的提交")
    if not git("branch", "-r", "--contains", sha):
        print(f"注意：提交 {sha[:7]} 还没推送到远端，二维码与下载链接暂时打不开")
    return sha


def build_ar(sha=None):
    import segno
    out = fresh(os.path.join(DIST, "ar"))
    img, mdl = fresh(os.path.join(out, "img")), fresh(os.path.join(out, "model"))
    sha = sha or ar_commit()
    nav = json.load(open(os.path.join(MODEL, "cloud_tide_pavilion_ar.json"), encoding="utf-8"))
    nav.pop("triangles", None)
    url = f"https://cdn.jsdelivr.net/gh/huangxiaozheng2/hxz@{sha}/cloud-tide-pavilion/model/cloud_tide_pavilion_ar.usdz"
    q = segno.make(url, error="m", micro=False)
    buf = io.BytesIO()
    q.save(buf, kind="svg", xmldecl=False, svgns=True, nl=False, scale=1, border=2, dark="#1d2023", light=None)
    svg = buf.getvalue().decode()
    d = re.search(r' d="([^"]+)"', svg).group(1)
    w = re.search(r'width="(\d+)"', svg).group(1)
    qr = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {w}" shape-rendering="crispEdges" role="img" '
          f'aria-label="二维码：在 iPhone 上打开 USDZ 模型"><path d="{d}"/></svg>')
    with tempfile.TemporaryDirectory() as tmp:
        glb = os.path.join(tmp, "web_ar.glb")
        optimize(os.path.join(MODEL, "cloud_tide_pavilion_ar.glb"), glb, AR_GLTF)
        size, tris = write_model_json(glb, os.path.join(mdl, "pavilion_ar.json")), glb_triangles(glb)
    size_of = lambda f: os.path.getsize(os.path.join(MODEL, f))
    fill = {
        "/*NAV*/null": json.dumps(nav, ensure_ascii=False, separators=(",", ":")),
        "<!--QR-->": qr,
        "__SHA__": sha,
        "__AR_WEB_MB__": mb(size),
        "__AR_WEB_WAN__": f"{tris / 1e4:.1f}",
        "__USDZ_MB__": mb(size_of("cloud_tide_pavilion_ar.usdz")),
        "__USDZ_OPEN_MB__": mb(size_of("cloud_tide_pavilion_ar_open.usdz")),
        "__AR_GLB_MB__": mb(size_of("cloud_tide_pavilion_ar.glb")),
    }
    html = open(os.path.join(HERE, "ar", "index.html"), encoding="utf-8").read()
    for k, v in fill.items():
        html = html.replace(k, v)
    left = re.findall(r"/\*NAV\*/|<!--QR-->|__[A-Z_]+__", html)
    if left:
        sys.exit(f"模板里还有没填的占位：{sorted(set(left))}")
    open(os.path.join(out, "index.html"), "w", encoding="utf-8").write(html)
    shutil.copy(os.path.join(HERE, "ar", "poster.webp"), img)
    print(f"ar → {os.path.relpath(out, ROOT)}：网页模型 {mb(size)} MB / {tris / 1e4:.1f} 万面，链接指向 {sha[:7]}，二维码 {q.version} 版")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("what", nargs="?", choices=["site", "ar", "film", "all"], default="all")
    ap.add_argument("--sha", help="AR 文件所在的提交（默认取最后一次修改 AR 文件的提交）")
    a = ap.parse_args()
    if a.what in ("site", "all"):
        build_site()
    if a.what in ("ar", "all"):
        build_ar(a.sha)
    if a.what == "film":
        build_film()
