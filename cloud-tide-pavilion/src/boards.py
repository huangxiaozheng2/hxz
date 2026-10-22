"""竞赛展板排版（A1 横版 ×2）。用法：python boards.py 渲染目录 分析图目录 输出目录"""
import os
import sys

from PIL import Image, ImageDraw, ImageFilter, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plan as P  # noqa: E402

FONT = "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"
W, H = 4200, 2970
M = 110                 # 外边距
G = 28                  # 图间距
PAPER = (247, 244, 238)
INK = (43, 42, 39)
MUTED = (128, 121, 113)
RULE = (214, 206, 195)
ACCENT = (200, 116, 58)


def font(size):
    return ImageFont.truetype(FONT, size)


def cover(im, w, h):
    r = w / h
    if im.width / im.height > r:
        nw = int(im.height * r); x = (im.width - nw) // 2
        im = im.crop((x, 0, x + nw, im.height))
    else:
        nh = int(im.width / r); y = (im.height - nh) // 2
        im = im.crop((0, y, im.width, y + nh))
    return im.resize((w, h), Image.LANCZOS)


def fit(im, w, h, bg=PAPER):
    s = min(w / im.width, h / im.height)
    im = im.resize((int(im.width * s), int(im.height * s)), Image.LANCZOS)
    out = Image.new("RGB", (w, h), bg)
    out.paste(im, ((w - im.width) // 2, (h - im.height) // 2))
    return out


def label(im, text, size=34):
    f = font(size)
    pad = int(size * 0.7)
    sh = Image.new("RGBA", im.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(sh)
    x, y = pad, im.height - pad - size
    d.text((x, y + 2), text, font=f, fill=(0, 0, 0, 160))
    sh = sh.filter(ImageFilter.GaussianBlur(size * 0.12))
    out = Image.alpha_composite(im.convert("RGBA"), sh)
    ImageDraw.Draw(out).text((x, y), text, font=f, fill=(255, 255, 255, 255))
    return out.convert("RGB")


def wrap(d, text, f, width):
    lines, cur = [], ""
    for ch in text:
        if ch == "\n":
            lines.append(cur); cur = ""
            continue
        if d.textlength(cur + ch, font=f) > width and cur:
            if ch in "，。、；：）」』":
                cur += ch; lines.append(cur); cur = ""
                continue
            lines.append(cur); cur = ch
        else:
            cur += ch
    if cur:
        lines.append(cur)
    return lines


def para(d, xy, text, size, width, color=INK, lh=1.62):
    f = font(size)
    x, y = xy
    for line in wrap(d, text, f, width):
        d.text((x, y), line, font=f, fill=color)
        y += int(size * lh)
    return y


def header(d, title, sub, board_no):
    d.text((M, 70), title, font=font(118), fill=INK)
    tw = d.textlength(title, font=font(118))
    d.text((M + tw + 46, 120), sub, font=font(40), fill=MUTED)
    d.text((W - M, 112), board_no, font=font(36), fill=MUTED, anchor="ra")
    d.line([(M, 230), (W - M, 230)], fill=RULE, width=3)


def load(path):
    return Image.open(path).convert("RGB")


def shot(R, name):
    """渲染图：优先读 JPG，其次 PNG。"""
    for ext in ("jpg", "png"):
        p = os.path.join(R, f"{name}.{ext}")
        if os.path.exists(p):
            return load(p)
    raise FileNotFoundError(name)


def trimmed(path, pad=30):
    """去掉分析图四周的纸色留白。"""
    from PIL import ImageChops
    im = load(path)
    bg = Image.new("RGB", im.size, PAPER)
    diff = ImageChops.difference(im, bg).convert("L").point(lambda v: 255 if v > 12 else 0)
    box = diff.getbbox()
    if not box:
        return im
    x0, y0, x1, y1 = box
    return im.crop((max(0, x0 - pad), max(0, y0 - pad), min(im.width, x1 + pad), min(im.height, y1 + pad)))


def board1(R, D, out):
    im = Image.new("RGB", (W, H), PAPER)
    d = ImageDraw.Draw(im)
    header(d, "云上潮汐馆", "CLOUD TIDE PAVILION  ·  从潮到云", "01 / 03  概念与形象")
    top = 290
    left_w = 1060
    x0 = M + left_w + 70
    right_w = W - M - x0
    # 主视图
    hero_h = int(right_w / 2.4)
    im.paste(label(cover(shot(R, 'hero'), right_w, hero_h), "整体外观 · 冬至傍晚 · 高潮位", 40), (x0, top))
    # 三张分图
    y1 = top + hero_h + G
    pw = (right_w - 2 * G) // 3
    ph = H - M - y1
    for k, (name, txt) in enumerate((("axis", "日落轴线：夕阳沿水口照进潮池"), ("moon", "月洞框景：冬至夕阳穿洞入室"),
                                     ("aerial", "鸟瞰：三片月牙环抱潮池"))):
        im.paste(label(cover(shot(R, name), pw, ph), txt, 32), (x0 + k * (pw + G), y1))
    # 左栏：设计说明
    y = top + 6
    d.text((M, y), "设计说明", font=font(54), fill=INK); y += 100
    texts = [
        "三片月牙屋顶围绕一座潮池交错环抱。潮池经西南的水口与外部水面相通，一日两涨两落：涨潮时池水漫上台阶，"
        "落潮时露出石底与潮痕。同一个庭院，一天里两次改变模样——潮汐不再是比喻，而是空间本身。",
        "参观路线从潮走到云。一层贴近水面，展廊外墙为象牙白实墙，只开四处深框景窗，天光从屋面光缝顺墙洗下；"
        "1:12 无障碍坡道绕潮池上升近两圈，抵达屋面平台，终点是次屋顶下的观景厅。",
        "水口、潮池与平台西端的月洞墙同指冬至日落方位 242.6°（北纬 30°）。冬至傍晚，夕阳沿水口照进潮池，"
        "同一时刻也正好穿过月洞，落在观景厅里。",
    ]
    for t in texts:
        y = para(d, (M, y), t, 38, left_w) + 34
    # 关键数据
    y += 20
    d.line([(M, y), (M + left_w, y)], fill=RULE, width=2); y += 36
    d.text((M, y), "关键数据", font=font(46), fill=INK); y += 84
    enclosed = P.E1.area + P.E2.area + P.E3.area
    rows = [
        ("围合面积", f"约 {enclosed:.0f} m²（展廊 · 门厅 · 观景厅）"),
        ("潮位", "设计高潮位 ±0.00 · 低潮位 -0.95 · 潮差 0.95 m"),
        ("标高", f"首层 +{P.FFL:.2f} · 屋面平台 +{P.DECK:.2f} · 主屋顶翼尖 +11.2"),
        ("坡道", f"1:12 · {P.RAMP_SEGMENTS} 段 × 升 {P.RAMP_SEG_RISE:.2f} m · 平台 1.5 m · 全长 {P.RAMP_LEN:.0f} m"),
        ("轴线", f"冬至日落 {P.SOLSTICE_SUNSET_AZ}° · 夕阳高度角 4°"),
        ("天光", f"屋面光缝 {len(P.SKYLIGHTS)} 段 · 宽 {P.SKYLIGHT_W:.1f} m · 沿外墙洗墙"),
    ]
    for k, v in rows:
        d.text((M, y), k, font=font(34), fill=ACCENT)
        y = para(d, (M + 190, y), v, 34, left_w - 190, lh=1.5) + 18
    # 沿轴线外景
    y += 24
    rem = H - M - y
    if rem > 300:
        im.paste(label(cover(shot(R, 'axis_out'), left_w, rem), "沿日落轴线看水口与 Y 形斜柱", 30), (M, y))
    im.save(out, quality=92)


def board2(R, D, out):
    im = Image.new("RGB", (W, H), PAPER)
    d = ImageDraw.Draw(im)
    header(d, "云上潮汐馆", "空间  ·  潮汐  ·  构造", "02 / 03  技术与细部")
    top = 290
    # 第一行：总平面 + 长剖面
    row1_h = 1180
    plan_w = 1200
    im.paste(fit(trimmed(f"{D}/plan_axis.png"), plan_w, row1_h), (M, top))
    sec_w = W - 2 * M - plan_w - G
    im.paste(fit(trimmed(f"{D}/section.png"), sec_w, row1_h), (M + plan_w + G, top))
    # 第二行：涨落潮对比 + 潮位曲线
    y2 = top + row1_h + G
    row2_h = 640
    cw = 1060
    for k, (name, txt) in enumerate((("court_ht", "涨潮 ±0.00：池水漫至第三级台阶"), ("court_lt", "落潮 -0.95：池底与潮痕露出"))):
        im.paste(label(cover(shot(R, name), cw, row2_h), txt, 32), (M + k * (cw + G), y2))
    tx = M + 2 * (cw + G)
    im.paste(fit(trimmed(f"{D}/tide_curve.png"), W - M - tx, row2_h), (tx, y2))
    # 第三行：细部 + 策略文字
    y3 = y2 + row2_h + G
    row3_h = H - M - y3
    text_w = 1120
    pics = (("interior", "展廊：实墙与框景窗"), ("stair", "悬浮螺旋阶梯"), ("structure", "Y 形斜柱与胶合木梁"),
            ("dusk", "黄昏：让建筑自己发光"))
    pw = (W - 2 * M - text_w - 4 * G) // 4
    for k, (name, txt) in enumerate(pics):
        im.paste(label(cover(shot(R, name), pw, row3_h), txt, 28), (M + k * (pw + G), y3))
    xt = M + 4 * (pw + G) + 10
    yy = y3 + 4
    blocks = [
        ("结构", "胶合木主梁约 1.5 m 一根，落在柱顶钢环梁上；柱、承重外墙、电梯筒、Y 形斜柱与水中石墩"
                "共同支承三片屋面，屋面任一点距最近支承不超过 5.8 m。次屋顶的柱与一层柱上下对齐。详见第三张展板。"),
        ("光与热", "展廊朝外为实墙，西晒与直射光不进展厅；屋面光缝提供漫射天光；深挑檐为庭院一侧的玻璃遮阳。"),
        ("水与韧性", "首层高出设计高潮位 0.85 m；潮池与水口由石材砌筑，可承受日常淹没，潮痕即是材料的记录。"),
        ("无障碍", "坡道 1:12，每升 0.68 m 设 1.5 m 平台；电梯筒连通首层、屋面平台与观景厅。"),
    ]
    for k, v in blocks:
        d.text((xt, yy), k, font=font(36), fill=ACCENT)
        yy = para(d, (xt, yy + 54), v, 30, text_w - 20, lh=1.55) + 22
    im.save(out, quality=92)


def axo_with_labels(png, w, h, label_w=360, fs=30, margin=0):
    """结构轴测图 + 两侧标注。图先裁到内容范围并缩放，标注按锚点高度排开，折线引线连到构件。"""
    import json
    raw = Image.open(png).convert("RGBA")
    labels = json.load(open(png.replace(".png", "_labels.json"), encoding="utf-8"))
    x0, y0, x1, y1 = raw.split()[3].getbbox()
    raw = raw.crop((x0, y0, x1, y1))
    s = min((w - 2 * label_w) / raw.width, h / raw.height)
    iw, ih = int(raw.width * s), int(raw.height * s)
    canvas = Image.new("RGBA", (w, h), PAPER + (255,))
    ox, oy = (w - iw) // 2, (h - ih) // 2
    canvas.alpha_composite(raw.resize((iw, ih), Image.LANCZOS), (ox, oy))
    canvas = canvas.convert("RGB")
    d = ImageDraw.Draw(canvas)
    f = font(fs)
    pts = {k: (ox + (x - x0) * s, oy + (y - y0) * s) for k, (x, y) in labels.items()}
    by_x = sorted(pts, key=lambda k: pts[k][0])
    half = (len(by_x) + 1) // 2
    sp, lead = int(fs * 2.1), (60, 57, 52)
    for side, keys in (("l", by_x[:half]), ("r", by_x[half:])):
        keys = sorted(keys, key=lambda k: pts[k][1])
        ty = [pts[k][1] for k in keys]
        for i in range(1, len(ty)):
            ty[i] = max(ty[i], ty[i - 1] + sp)
        ty[-1] = min(ty[-1], h - fs)
        for i in range(len(ty) - 2, -1, -1):
            ty[i] = min(ty[i], ty[i + 1] - sp)
        for k, t in zip(keys, ty):
            ax_, ay = pts[k]
            tw = d.textlength(k, font=f)
            if side == "l":
                tx, lx, ex = margin, margin + tw + 14, label_w - 10
            else:
                tx, lx, ex = w - margin - tw, w - margin - tw - 14, w - label_w + 10
            d.text((tx, t - fs * 0.62), k, font=f, fill=INK)
            d.line([(lx, t), (ex, t), (ax_, ay)], fill=lead, width=2, joint="curve")
            d.ellipse([ax_ - 9, ay - 9, ax_ + 9, ay + 9], fill=PAPER)
            d.ellipse([ax_ - 6, ay - 6, ax_ + 6, ay + 6], fill=ACCENT)
    return canvas


# 主要构件（与 architecture.py / structure.py 中的尺寸一致）
MEMBERS = [
    ("主梁", "胶合木 GL28h，宽 160；高 200（檐口）→ 600 / 520 / 480（A / B / C 支座）"),
    ("次肋", "木格栅 100 深，间距约 240 mm"),
    ("柱顶环梁", "钢箱梁 220 宽，梁底与主梁底平"),
    ("柱", "钢管 φ194×10，外张柱头，顶板 φ500 承于主梁与环梁下"),
    ("Y 形斜柱", "变截面钢管 φ460 → φ320，铸钢节点，立于 φ2.1 m 石墩"),
    ("电梯筒", "钢筋混凝土筒 φ2.3 m，由基础贯通至次屋顶"),
    ("月洞墙 / 展廊外墙", "钢筋混凝土承重墙，外饰矿物质涂层"),
    ("水中石墩", "花岗岩外包 φ1.24 m，内为钢筋混凝土墩，高出设计高潮位 0.6 m"),
    ("坡道", "螺旋钢箱梁 260 高，外侧 φ180 钢柱每 30° 一根"),
    ("基础", "钻孔灌注桩 + 承台，基座为钢筋混凝土箱体"),
]


def board3(R, D, out):
    import structure as ST
    im = Image.new("RGB", (W, H), PAPER)
    d = ImageDraw.Draw(im)
    header(d, "云上潮汐馆", "结构体系  ·  荷载路径  ·  初步校核", "03 / 03  结构")
    top = 290
    left_w = 2560
    sm = trimmed(f"{D}/support_map.png")
    row2_h = min(int(left_w * sm.height / sm.width), 1100)
    y2 = H - M - row2_h
    row1_h = y2 - G - top
    im.paste(axo_with_labels(f"{R}/struct_axo.png", left_w, row1_h), (M, top))
    im.paste(fit(sm, left_w, row2_h), (M, y2))
    # 右栏：荷载路径 → 支承布置 → 主要构件 → 初步校核
    x0 = M + left_w + 90
    tw = W - M - x0
    y = top + 10
    d.text((x0, y), "荷载路径", font=font(50), fill=INK); y += 86
    steps = ["屋面板：轻钢龙骨 + 20 mm GRC 面层",
             "次肋：100 mm 木格栅，传力给主梁",
             "主梁：胶合木，约 1.5 m 一根，与幕墙分格对齐；截面由檐口向支座加深",
             "柱顶环梁：钢箱梁，收集主梁荷载传给柱",
             "柱 / 承重墙 / 电梯筒 / Y 形斜柱 / 水中石墩",
             "钻孔灌注桩：基座与石墩之下"]
    for k, t in enumerate(steps):
        d.text((x0, y), f"{k + 1}", font=font(32), fill=ACCENT)
        y = para(d, (x0 + 46, y), t, 30, tw - 46, lh=1.42) + 8
    y += 34
    d.text((x0, y), "支承布置", font=font(50), fill=INK); y += 86
    for t in ["次屋顶的柱立在主屋顶结构层上，与一层内侧柱上下对齐，荷载直落基础。",
              "主屋顶西翼尖由水中石墩上的 Y 形斜柱托起；低檐东翼尖由两处水中石墩承托，翼尖下不再设基座。",
              "屋面上任一点到最近支承不超过 5.8 m（左下图），悬挑与跨度都落在胶合木梁的常规范围内。"]:
        y = para(d, (x0, y), t, 30, tw, lh=1.42) + 12
    y += 34
    d.text((x0, y), "主要构件", font=font(50), fill=INK); y += 86
    for a, b in MEMBERS:
        d.text((x0, y), a, font=font(28), fill=ACCENT)
        y = para(d, (x0 + 270, y), b, 28, tw - 270, lh=1.38) + 8
    # 初步校核：与支承距离图同一行起
    chk = ST.prelim_checks()
    yy = max(y + 40, y2)
    d.text((x0, yy), "初步校核", font=font(50), fill=INK); yy += 80
    yy = para(d, (x0, yy), "屋面恒载 1.5 / 活载 0.5 kPa；屋面平台恒载 3.0 / 活载 3.5 kPa；分项系数 1.3 / 1.5。"
              "方案阶段手算，用于确认构件尺度，深化阶段需结构工程师建模复核。", 26, tw, color=MUTED, lh=1.42) + 18
    names = {"Roof_A_Main": "A 主梁悬挑", "Roof_B_Upper": "B 主梁悬挑", "Roof_C_Low": "C 主梁悬挑"}
    rows = []
    for k, v in chk.items():
        if k in names:
            rows.append((names[k], f"{v['L']:.1f} m · 梁高 {v['h'] * 1000:.0f}",
                         f"应力比 {v['util']:.2f} · 挠度 L/{v['ratio']:.0f}"))
    v = chk["deck_beam"]
    rows.append(("平台主梁跨中", f"{v['L']:.1f} m · 梁高 {P.ROOF_A.beam_core * 1000:.0f}",
                 f"应力比 {v['util']:.2f} · 挠度 L/{v['ratio']:.0f}"))
    v = chk["column"]
    rows.append(("一层柱 φ194×10", f"轴力 {v['N']:.0f} kN", f"稳定承载 {v['Nb']:.0f} kN · 应力比 {v['util']:.2f}"))
    f1, f2 = font(30), font(28)
    for a, b, c in rows:
        d.text((x0, yy), a, font=f1, fill=ACCENT)
        d.text((x0 + 300, yy), b, font=f2, fill=INK)
        d.text((x0 + 300 + 330, yy), c, font=f2, fill=MUTED)
        yy += 56
    im.save(out, quality=92)


if __name__ == "__main__":
    R, D, O = sys.argv[1], sys.argv[2], sys.argv[3]
    os.makedirs(O, exist_ok=True)
    board1(R, D, os.path.join(O, "board_01.jpg"))
    board2(R, D, os.path.join(O, "board_02.jpg"))
    board3(R, D, os.path.join(O, "board_03.jpg"))
    print("boards ->", O)
