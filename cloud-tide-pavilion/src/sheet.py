"""把渲染图排版为展板（仿参考图：顶部标题 + 主视图 + 两行分图）。
用法：python sheet.py 渲染目录 输出.jpg
"""
import os
import sys
from PIL import Image, ImageDraw, ImageFilter, ImageFont

FONT = "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"
PANELS = [
    ("aerial", "鸟瞰全貌"), ("rear", "背面视角"), ("court", "中央庭院"), ("interior", "室内空间"),
    ("terrace", "屋顶平台"), ("moon", "框景开口"), ("stair", "悬浮螺旋阶梯"), ("detail", "材质细节"),
]
BG = (246, 243, 238)
INK = (70, 64, 58)


def label(img, text, size):
    """左下角白色标签，带柔和投影。"""
    font = ImageFont.truetype(FONT, size)
    pad = int(size * 0.7)
    shadow = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(shadow)
    x, y = pad, img.height - pad - size
    d.text((x, y + 2), text, font=font, fill=(0, 0, 0, 150))
    shadow = shadow.filter(ImageFilter.GaussianBlur(size * 0.12))
    out = Image.alpha_composite(img.convert("RGBA"), shadow)
    ImageDraw.Draw(out).text((x, y), text, font=font, fill=(255, 255, 255, 255))
    return out.convert("RGB")


def main(src, dst, W=2400, gap=10):
    head = 150
    hero = Image.open(os.path.join(src, "hero.jpg")).convert("RGB")
    hero = hero.resize((W, int(W * hero.height / hero.width)), Image.LANCZOS)
    pw = (W - 3 * gap) // 4
    ph = int(pw / 1.6)
    H = head + hero.height + 2 * (gap + ph) + 70
    sheet = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(sheet)
    t1 = ImageFont.truetype(FONT, 58)
    t2 = ImageFont.truetype(FONT, 22)
    title = "云 上 潮 汐 馆"
    tw = d.textlength(title, font=t1)
    d.text(((W - tw) / 2, 26), title, font=t1, fill=INK)
    d.line([(W / 2 - 230, 104), (W / 2 + 230, 104)], fill=(190, 182, 172), width=2)
    sub = "建 筑 概 念 效 果 展 示  ·  CLOUD TIDE PAVILION"
    sw = d.textlength(sub, font=t2)
    d.text(((W - sw) / 2, 114), sub, font=t2, fill=(120, 112, 104))
    sheet.paste(label(hero, "整体外观", 34), (0, head))
    y0 = head + hero.height + gap
    for k, (name, text) in enumerate(PANELS):
        p = os.path.join(src, f"{name}.jpg")
        if not os.path.exists(p):
            continue
        im = Image.open(p).convert("RGB")
        # 居中裁切到面板比例
        r = pw / ph
        if im.width / im.height > r:
            nw = int(im.height * r); x = (im.width - nw) // 2
            im = im.crop((x, 0, x + nw, im.height))
        else:
            nh = int(im.width / r); y = (im.height - nh) // 2
            im = im.crop((0, y, im.width, y + nh))
        im = im.resize((pw, ph), Image.LANCZOS)
        im = label(im, text, 26)
        col, row = k % 4, k // 4
        sheet.paste(im, (col * (pw + gap), y0 + row * (ph + gap)))
    foot = ImageFont.truetype(FONT, 20)
    note = "悬浮月牙 · 三片屋顶交错环抱 · 弧形坡道绕庭院上升 · 象牙白矿物墙面 / 浅木细肋 / 拉丝金属 / 低对比石材"
    nw = d.textlength(note, font=foot)
    d.text(((W - nw) / 2, H - 50), note, font=foot, fill=(130, 122, 114))
    sheet.save(dst, quality=93)
    print("saved", dst, sheet.size)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
