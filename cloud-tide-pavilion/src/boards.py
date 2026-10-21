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
    header(d, "云上潮汐馆", "CLOUD TIDE PAVILION  ·  从潮到云", "01 / 02  概念与形象")
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
    header(d, "云上潮汐馆", "空间  ·  潮汐  ·  构造", "02 / 02  技术与细部")
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
        ("结构", "主屋顶由展廊外墙、内侧柱列与电梯筒共同支承；西翼尖由水中石墩上的 Y 形斜柱托起。"
                "次屋顶落在电梯筒、月洞墙与观景厅东端弧墙上。檐下胶合木径向梁由檐口 100 mm 加深到支座处 340 mm。"),
        ("光与热", "展廊朝外为实墙，西晒与直射光不进展厅；屋面光缝提供漫射天光；深挑檐为庭院一侧的玻璃遮阳。"),
        ("水与韧性", "首层高出设计高潮位 0.85 m；潮池与水口由石材砌筑，可承受日常淹没，潮痕即是材料的记录。"),
        ("无障碍", "坡道 1:12，每升 0.68 m 设 1.5 m 平台；电梯筒连通首层、屋面平台与观景厅。"),
    ]
    for k, v in blocks:
        d.text((xt, yy), k, font=font(36), fill=ACCENT)
        yy = para(d, (xt, yy + 54), v, 30, text_w - 20, lh=1.55) + 22
    im.save(out, quality=92)


if __name__ == "__main__":
    R, D, O = sys.argv[1], sys.argv[2], sys.argv[3]
    os.makedirs(O, exist_ok=True)
    board1(R, D, os.path.join(O, "board_01.jpg"))
    board2(R, D, os.path.join(O, "board_02.jpg"))
    print("boards ->", O)
