"""展廊里的六幅画：以展馆自身的题材（湖山、冬至落日、月洞、潮痕、夜湖与对岸塔影、芦汀白鹭）用程序画成的原创作品。
水墨两幅用“墨色浓度场 + 宣纸纹理”合成；油画两幅先做底色稿，再按流向场一笔一笔铺色块；另两幅为矿物色与抽象色带。
画幅与展墙上的画框一致（1.15 × 1.45 m，竖幅），输出 1024 × 1291 像素 JPEG。
用法：python paintings.py 输出目录
"""
import json
import math
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

W, H = 1024, 1291
FONT = "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"
TITLES = [
    ("湖山清远", "水墨。远山四重，近岸松林，一叶小舟。"),
    ("冬至落日", "油彩。冬至傍晚，太阳贴着湖面落下，水面被拉出一道金色。"),
    ("月洞", "矿物色。青绿底上一轮圆洞，洞里是落日与水平线。"),
    ("潮痕", "丙烯。高低潮位留在石头上的一道道色带。"),
    ("夜湖灯影", "油彩。月夜湖面，对岸城灯与七层塔的倒影。"),
    ("芦汀白鹭", "水墨淡彩。秋芦、白鹭与一轮淡月。"),
]


# ---------------------------------------------------------------- 基础工具
def fbm(h, w, rng, octaves=6, base=4, persistence=0.55, stretch=(1.0, 1.0)):
    """分形噪声：逐级放大的随机格网叠加，结果在 0–1。stretch 拉长纹理（如水平方向的笔触）。"""
    out = np.zeros((h, w))
    amp, tot = 1.0, 0.0
    for o in range(octaves):
        gh = max(2, int(base * 2 ** o / stretch[0]))
        gw = max(2, int(base * 2 ** o / stretch[1]))
        g = rng.random((gh, gw)).astype(np.float32)
        im = Image.fromarray(g).resize((w, h), Image.BICUBIC)
        out += amp * np.asarray(im)
        tot += amp
        amp *= persistence
    out /= tot
    return (out - out.min()) / (out.max() - out.min() + 1e-9)


def ridge(n, rng, rough=0.55, octaves=7):
    """一维分形起伏（山脊线），0–1。"""
    x = np.zeros(n)
    amp, tot = 1.0, 0.0
    for o in range(octaves):
        k = 2 + 2 ** o
        pts = rng.random(k)
        x += amp * np.interp(np.linspace(0, k - 1, n), np.arange(k), pts)
        tot += amp
        amp *= rough
    x /= tot
    return (x - x.min()) / (x.max() - x.min() + 1e-9)


def blur(a, r):
    from scipy.ndimage import gaussian_filter
    if a.ndim == 2:
        return gaussian_filter(a, r)
    return np.stack([gaussian_filter(a[..., c], r) for c in range(a.shape[2])], -1)


def paper(rng, base=(0.93, 0.90, 0.83), stains=0.035, fibers=900):
    """宣纸：暖白底、淡淡的水渍与纤维。"""
    img = np.ones((H, W, 3)) * np.array(base)
    img *= 1 - stains * fbm(H, W, rng, 5, 3)[..., None]
    img *= 1 - 0.02 * fbm(H, W, rng, 3, 64)[..., None]
    lay = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(lay)
    for _ in range(fibers):
        x, y = rng.random() * W, rng.random() * H
        a, L = rng.random() * math.pi, 6 + rng.random() * 22
        d.line([(x, y), (x + math.cos(a) * L, y + math.sin(a) * L)], fill=int(18 + rng.random() * 30), width=1)
    img *= 1 - 0.06 * (np.asarray(lay.filter(ImageFilter.GaussianBlur(0.6))) / 255.0)[..., None]
    return img


def canvas_texture(img, rng, k=0.045):
    """画布纹理：细密的经纬格纹 + 颗粒。"""
    yy, xx = np.mgrid[0:H, 0:W]
    weave = np.sin(xx * 2 * math.pi / 3.1) * np.sin(yy * 2 * math.pi / 3.3)
    grain = fbm(H, W, rng, 2, 180) - 0.5
    return np.clip(img * (1 + k * weave[..., None] + 0.05 * grain[..., None]), 0, 1)


def ink(img, density, color=(0.07, 0.07, 0.09), rng=None, granulate=0.25):
    """按墨色浓度叠墨：浓度略做晕开，并带颗粒（墨在纸纤维里的不匀）。"""
    d = np.clip(density, 0, 1)
    if rng is not None:
        g = fbm(H, W, rng, 3, 90)
        d = d * (1 - granulate / 2 + granulate * g)
    d = np.clip(blur(d, 0.7), 0, 1)[..., None]
    return img * (1 - d) + np.array(color) * d


def seal(img, text, x, y, size=58, rng=None, color=(0.78, 0.16, 0.12)):
    """朱文/白文印：红色方印，白字，边缘与印泥略有残缺。"""
    im = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))
    s = Image.new("L", (size, size), 0)
    d = ImageDraw.Draw(s)
    d.rectangle([0, 0, size - 1, size - 1], fill=255)
    font = ImageFont.truetype(FONT, int(size * 0.42))
    for i, ch in enumerate(text[:4]):
        cx = size * (0.72 if i < 2 else 0.28)
        cy = size * (0.29 if i % 2 == 0 else 0.71)
        if len(text) <= 2:
            cx, cy = size * 0.5, size * (0.29 if i == 0 else 0.71)
        d.text((cx, cy), ch, fill=0, font=font, anchor="mm")
    m = np.asarray(s).astype(float) / 255
    if rng is not None:
        m *= (rng.random(m.shape) > 0.07)
        m = blur(m, 0.6)
    a = np.clip(m * 0.92, 0, 1)[..., None]
    region = np.asarray(im).astype(float)[y:y + size, x:x + size] / 255
    region = region * (1 - a) + np.array(color) * a
    out = np.asarray(im).astype(float) / 255
    out[y:y + size, x:x + size] = region
    return out


def strokes(target, rng, passes, flow, alpha=225, wiggle=0.25):
    """油画笔触：在底色稿上按流向场铺长条色块，先大后小。"""
    base = Image.fromarray((np.clip(blur(target, 6), 0, 1) * 255).astype(np.uint8))
    d = ImageDraw.Draw(base, "RGBA")
    for r, length, jitter in passes:
        step = max(2, int(r * 0.85))
        ys, xs = np.mgrid[step // 2:H:step, step // 2:W:step]
        pts = np.stack([xs.ravel() + rng.normal(0, step * 0.4, xs.size), ys.ravel() + rng.normal(0, step * 0.4, ys.size)], 1)
        rng.shuffle(pts)
        for x, y in pts:
            xi, yi = int(np.clip(x, 0, W - 1)), int(np.clip(y, 0, H - 1))
            col = np.clip(target[yi, xi] + rng.normal(0, jitter, 3), 0, 1)
            a = flow(xi, yi) + rng.normal(0, wiggle)
            L = r * length * (0.7 + 0.6 * rng.random())
            w2 = r * (0.35 + 0.25 * rng.random())
            ca, sa = math.cos(a), math.sin(a)
            poly = [(x - ca * L / 2 - sa * w2 * 0.7, y - sa * L / 2 + ca * w2 * 0.7), (x + ca * L / 2 - sa * w2, y + sa * L / 2 + ca * w2),
                    (x + ca * L / 2 + sa * w2, y + sa * L / 2 - ca * w2), (x - ca * L / 2 + sa * w2 * 0.7, y - sa * L / 2 - ca * w2 * 0.7)]
            d.polygon(poly, fill=(*[int(c * 255) for c in col], alpha))
    return np.asarray(base).astype(float) / 255


def vignette(img, k=0.12):
    yy, xx = np.mgrid[0:H, 0:W]
    r = np.hypot((xx - W / 2) / (W / 2), (yy - H / 2) / (H / 2))
    return img * (1 - k * np.clip(r - 0.55, 0, 1)[..., None])


def grad(stops, t):
    """沿 t（0–1）的多段色阶。stops: [(t, (r,g,b)), ...]"""
    t = np.clip(t, 0, 1)
    out = np.zeros(t.shape + (3,))
    ts = [s[0] for s in stops]
    for c in range(3):
        out[..., c] = np.interp(t, ts, [s[1][c] for s in stops])
    return out


# ---------------------------------------------------------------- 1 湖山清远（水墨）
def p_ink_landscape(rng):
    img = paper(rng)
    yy, xx = np.mgrid[0:H, 0:W]
    dens = np.zeros((H, W))
    layers = [(0.30, 0.20, 0.16, 0.18), (0.40, 0.26, 0.30, 0.13), (0.52, 0.22, 0.48, 0.10), (0.63, 0.18, 0.72, 0.075)]
    mist = fbm(H, W, rng, 5, 3, stretch=(1, 4))
    for i, (base, amp, inkv, fade) in enumerate(layers):
        prof = ridge(W, rng, 0.58)
        if i == 0:                                            # 远处主峰偏左，竖幅里撑起上半幅
            prof = np.maximum(prof * 0.6, np.exp(-((np.arange(W) - W * 0.34) / (W * 0.16)) ** 2) * 1.35)
        if i == 1:
            prof = np.maximum(prof * 0.7, np.exp(-((np.arange(W) - W * 0.75) / (W * 0.14)) ** 2) * 1.1)
        top = (base - amp * prof) * H
        below = yy - top[None, :]
        layer = np.where(below > 0, np.exp(-below / (fade * H)), 0.0)
        # 皴擦：山脊下方的短笔触纹理
        tex = fbm(H, W, rng, 4, 40, stretch=(3, 1))
        layer *= 0.55 + 0.9 * tex
        edge = np.where(below > 0, np.exp(-below / 6.0), 0.0)
        dens = dens * (1 - layer * 0.85) + layer * inkv + edge * inkv * 0.5
        # 山间雾带
        band = np.exp(-((yy - (base + 0.035) * H) / (0.035 * H)) ** 2) * (0.6 + 0.4 * mist)
        dens *= 1 - 0.75 * band
    water = yy > 0.70 * H
    dens = np.where(water, dens * np.clip(1 - (yy - 0.70 * H) / (0.02 * H), 0, 1), dens)
    img = ink(img, dens * 0.95, rng=rng)
    # 近岸松树
    im = Image.fromarray((img * 255).astype(np.uint8))
    d = ImageDraw.Draw(im, "RGBA")
    for px, py, hh in ((0.14, 0.70, 0.20), (0.22, 0.705, 0.15), (0.83, 0.69, 0.17)):
        x0, y0, th = px * W, py * H, hh * H
        pts = [(x0 + math.sin(t * 2.4) * 10, y0 - t * th) for t in np.linspace(0, 1, 12)]
        d.line(pts, fill=(25, 23, 22, 235), width=5)
        for t in np.linspace(0.35, 1.0, 7):                  # 一层层的松针团
            cx, cy = x0 + math.sin(t * 2.4) * 10, y0 - t * th
            span = (1.1 - t) * 70 + 22
            for _ in range(60):
                u = rng.normal(0, 0.45)
                x = cx + u * span
                y = cy + rng.normal(0, 5) - abs(u) * 6
                a = rng.random() * math.pi
                L = 5 + rng.random() * 8
                d.line([(x - math.cos(a) * L, y - math.sin(a) * L), (x + math.cos(a) * L, y + math.sin(a) * L)],
                       fill=(18, 20, 18, int(140 + rng.random() * 90)), width=2)
    # 水纹与小舟
    for _ in range(26):
        y = (0.74 + rng.random() * 0.22) * H
        x = rng.random() * W * 0.9
        L = 30 + rng.random() * 110
        d.line([(x, y), (x + L, y + rng.normal(0, 1.5))], fill=(40, 40, 42, int(30 + rng.random() * 40)), width=1)
    bx, by = 0.62 * W, 0.83 * H
    d.polygon([(bx - 46, by), (bx + 52, by), (bx + 38, by + 9), (bx - 34, by + 9)], fill=(28, 26, 24, 230))
    d.line([(bx + 10, by), (bx + 12, by - 34)], fill=(28, 26, 24, 220), width=2)
    d.ellipse([bx - 12, by - 22, bx - 2, by - 12], fill=(28, 26, 24, 230))
    d.polygon([(bx - 16, by - 12), (bx + 2, by - 12), (bx + 4, by), (bx - 18, by)], fill=(28, 26, 24, 230))
    img = np.asarray(im).astype(float) / 255
    return seal(img, "潮生", int(0.08 * W), int(0.88 * H), 62, rng)


# ---------------------------------------------------------------- 2 冬至落日（油彩）
def p_sunset(rng):
    yy, xx = np.mgrid[0:H, 0:W].astype(float)
    hz = 0.60 * H
    t = yy / hz
    sky = grad([(0, (0.16, 0.19, 0.38)), (0.35, (0.42, 0.30, 0.46)), (0.68, (0.84, 0.45, 0.36)),
                (0.88, (0.98, 0.64, 0.32)), (1.0, (1.0, 0.80, 0.48))], t)
    sx, sy, sr = 0.40 * W, hz - 0.045 * H, 0.052 * W
    rr = np.hypot(xx - sx, (yy - sy) * 1.08)
    glow = np.exp(-rr / (0.22 * W))[..., None]
    sky = sky + glow * np.array([0.35, 0.22, 0.05])
    cl = fbm(H, W, rng, 5, 3, stretch=(1, 6))
    cloud = np.clip((cl - 0.55) * 3, 0, 1) * np.exp(-((yy - 0.38 * H) / (0.13 * H)) ** 2)
    sky = sky * (1 - cloud[..., None] * 0.6) + cloud[..., None] * np.array([0.88, 0.48, 0.42]) * 0.6
    sky = np.where((rr < sr)[..., None], np.array([1.0, 0.93, 0.72]), sky)
    # 水面：倒映天色，越近越暗；落日的金色倒影带
    tw = (yy - hz) / (H - hz)
    refl = grad([(0, (0.95, 0.66, 0.36)), (0.25, (0.62, 0.38, 0.36)), (0.6, (0.28, 0.22, 0.32)), (1.0, (0.12, 0.12, 0.22))], tw)
    rip = fbm(H, W, rng, 5, 6, stretch=(1, 10))
    col = np.exp(-((xx - sx) / (0.06 * W + tw * 0.10 * W)) ** 2) * np.clip(rip * 1.6 - 0.3, 0, 1) * (1 - tw * 0.6)
    refl = refl + col[..., None] * np.array([0.45, 0.32, 0.10])
    img = np.where((yy < hz)[..., None], sky, refl)
    # 远岸：一道低低的暗色树线
    shore = hz - (0.006 + 0.012 * ridge(W, rng, 0.6)) * H
    land = (yy > shore[None, :]) & (yy < hz + 2)
    img = np.where(land[..., None], np.array([0.22, 0.15, 0.20]), img)
    img = np.clip(img, 0, 1)

    def flow(x, y):
        if y > hz:
            return 0.0
        dx, dy = x - sx, y - sy
        r = math.hypot(dx, dy)
        if r < sr * 2.2:
            return math.atan2(dy, dx) + math.pi / 2
        return 0.08 * math.sin(x / 140 + y / 90)
    out = strokes(img, rng, [(26, 3.2, 0.03), (15, 3.4, 0.03), (8, 3.6, 0.025)], flow)
    out = np.where((rr < sr * 0.92)[..., None], np.clip(out * 0.3 + np.array([1.0, 0.93, 0.74]) * 0.7, 0, 1), out)
    return vignette(canvas_texture(out, rng), 0.18)


# ---------------------------------------------------------------- 3 月洞（矿物色）
def p_moon_gate(rng):
    yy, xx = np.mgrid[0:H, 0:W].astype(float)
    m = fbm(H, W, rng, 6, 3)
    bg = grad([(0, (0.10, 0.24, 0.36)), (0.5, (0.14, 0.34, 0.38)), (1.0, (0.22, 0.46, 0.38))], m)
    bg *= 1 - 0.18 * fbm(H, W, rng, 4, 24, stretch=(1, 3))[..., None]
    cx, cy, R = 0.5 * W, 0.42 * H, 0.33 * W
    r = np.hypot(xx - cx, yy - cy)
    v = (yy - (cy - R)) / (2 * R)
    inside = grad([(0, (0.95, 0.93, 0.86)), (0.55, (0.97, 0.86, 0.62)), (0.62, (0.93, 0.70, 0.46)),
                   (0.64, (0.44, 0.52, 0.56)), (1.0, (0.62, 0.70, 0.72))], v)
    inside *= 1 - 0.06 * fbm(H, W, rng, 4, 12)[..., None]
    img = np.where((r < R)[..., None], inside, bg)
    sun = np.hypot(xx - (cx + 0.18 * R), yy - (cy + 0.19 * R)) < 0.07 * R
    img = np.where(sun[..., None], np.array([0.84, 0.30, 0.22]), img)
    # 圆环：一圈有浓淡、起收笔的墨青色笔痕
    im = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))
    d = ImageDraw.Draw(im, "RGBA")
    a0 = math.radians(-58)
    for k in range(1400):
        u = k / 1400
        a = a0 + u * math.radians(338)
        wdt = 9 + 13 * math.sin(u * math.pi) ** 0.6 + rng.normal(0, 1.2)
        rr_ = R + rng.normal(0, 1.5)
        x, y = cx + rr_ * math.cos(a), cy + rr_ * math.sin(a)
        alpha = int(150 + 80 * (1 - u) ** 0.4)
        d.ellipse([x - wdt / 2, y - wdt / 2, x + wdt / 2, y + wdt / 2], fill=(18, 30, 42, alpha))
    for _ in range(9):                                         # 圆下方的几道台阶与水纹
        y = cy + R + 40 + rng.random() * 0.25 * H
        x = cx - 0.42 * W + rng.random() * 0.3 * W
        d.line([(x, y), (x + 0.3 * W + rng.random() * 0.3 * W, y)], fill=(225, 220, 200, int(40 + rng.random() * 50)), width=2)
    for _ in range(260):                                       # 金箔碎点
        x, y = rng.random() * W, rng.random() * H
        if math.hypot(x - cx, y - cy) < R + 14:
            continue
        s = 1 + rng.random() * 3
        d.rectangle([x, y, x + s, y + s * (0.5 + rng.random())], fill=(236, 205, 120, int(120 + rng.random() * 120)))
    img = np.asarray(im).astype(float) / 255
    img = canvas_texture(img, rng, 0.03)
    return seal(img, "月洞", int(0.83 * W), int(0.90 * H), 56, rng)


# ---------------------------------------------------------------- 4 潮痕（抽象色带）
def p_tide_lines(rng):
    yy, xx = np.mgrid[0:H, 0:W].astype(float)
    warp = (fbm(H, W, rng, 5, 3, stretch=(1, 3)) - 0.5) * 0.10 * H
    y = (yy + warp) / H
    bands = [(0.0, (0.87, 0.86, 0.81)), (0.20, (0.84, 0.83, 0.78)), (0.24, (0.16, 0.23, 0.40)), (0.40, (0.20, 0.30, 0.46)),
             (0.43, (0.95, 0.92, 0.84)), (0.455, (0.94, 0.90, 0.80)), (0.47, (0.72, 0.54, 0.31)), (0.64, (0.62, 0.40, 0.24)),
             (0.67, (0.48, 0.28, 0.18)), (0.82, (0.30, 0.21, 0.16)), (0.85, (0.20, 0.16, 0.14)), (1.0, (0.16, 0.13, 0.12))]
    img = grad(bands, y)
    dry = fbm(H, W, rng, 4, 30, stretch=(1, 18))               # 干笔横向擦痕
    img *= (0.86 + 0.24 * dry)[..., None]
    scr = fbm(H, W, rng, 3, 160, stretch=(1, 30))
    img = img + (np.clip(scr - 0.72, 0, 1) * 1.4)[..., None] * np.array([0.25, 0.22, 0.18])
    for yc, col, wd in ((0.235, (0.92, 0.88, 0.78), 0.004), (0.665, (0.88, 0.80, 0.62), 0.003)):   # 潮位线
        line = np.exp(-((y - yc) / wd) ** 2) * (0.6 + 0.4 * dry)
        img = img * (1 - line[..., None]) + np.array(col) * line[..., None]
    return vignette(canvas_texture(np.clip(img, 0, 1), rng, 0.05), 0.1)


# ---------------------------------------------------------------- 5 夜湖灯影（油彩）
def p_night_lake(rng):
    yy, xx = np.mgrid[0:H, 0:W].astype(float)
    hz = 0.56 * H
    t = yy / hz
    sky = grad([(0, (0.04, 0.06, 0.15)), (0.7, (0.09, 0.11, 0.24)), (0.95, (0.22, 0.18, 0.28)), (1.0, (0.30, 0.22, 0.26))], t)
    mx, my, mr = 0.26 * W, 0.16 * H, 0.045 * W
    rm = np.hypot(xx - mx, yy - my)
    sky = sky + np.exp(-rm / (0.10 * W))[..., None] * np.array([0.12, 0.12, 0.10])
    sky = np.where((rm < mr)[..., None], np.array([0.94, 0.92, 0.80]), sky)
    # 对岸：低矮城区天际线 + 右侧七层塔
    prof = np.full(W, hz)
    x = 0
    while x < W:
        bw = int(30 + rng.random() * 70)
        prof[x:x + bw] = hz - (0.01 + rng.random() * 0.04) * H
        x += bw
    px, base_w = int(0.74 * W), 46
    for k in range(7):
        w2 = int(base_w * (1 - k * 0.09))
        y0 = hz - 0.03 * H - k * 0.021 * H
        prof[px - w2 // 2:px + w2 // 2] = np.minimum(prof[px - w2 // 2:px + w2 // 2], y0)
    prof[px - 2:px + 2] = np.minimum(prof[px - 2:px + 2], hz - 0.03 * H - 7 * 0.021 * H - 0.03 * H)
    land = (yy > prof[None, :]) & (yy <= hz)
    img = np.where((yy < hz)[..., None], sky, 0)
    img = np.where(land[..., None], np.array([0.06, 0.06, 0.10]), img)
    tw = (yy - hz) / (H - hz)
    water = grad([(0, (0.16, 0.14, 0.24)), (0.4, (0.08, 0.09, 0.18)), (1.0, (0.03, 0.04, 0.10))], tw)
    img = np.where((yy >= hz)[..., None], water, img)
    img = np.clip(img, 0, 1)
    im = Image.fromarray((img * 255).astype(np.uint8))
    d = ImageDraw.Draw(im, "RGBA")
    lights = []
    for _ in range(140):                                       # 城灯
        lx = rng.random() * W
        ly = prof[int(lx)] + 3 + rng.random() * (hz - prof[int(lx)] - 4)
        if ly >= hz - 1:
            continue
        c = (255, int(190 + rng.random() * 50), int(110 + rng.random() * 60))
        d.rectangle([lx, ly, lx + 2, ly + 2], fill=(*c, 230))
        lights.append((lx, c))
    for k in range(7):                                         # 塔檐下的灯
        y0 = hz - 0.03 * H - k * 0.021 * H + 4
        w2 = int(base_w * (1 - k * 0.09)) // 2
        for xx_ in range(px - w2 + 4, px + w2 - 4, 7):
            d.rectangle([xx_, y0, xx_ + 2, y0 + 2], fill=(255, 205, 120, 235))
        lights.append((px, (255, 205, 120)))
    for _ in range(90):                                        # 星
        sx, sy = rng.random() * W, rng.random() * 0.4 * H
        d.point((sx, sy), fill=(230, 230, 255, int(60 + rng.random() * 150)))
    img = np.asarray(im).astype(float) / 255
    base = img.copy()

    def flow(x, y):
        return 0.0 if y > hz else 0.06 * math.sin(x / 120)
    out = strokes(base, rng, [(22, 3.0, 0.02), (12, 3.2, 0.02), (6, 3.0, 0.015)], flow, alpha=210)
    out[:int(hz)] = np.where((rm < mr * 0.95)[:int(hz), :, None], base[:int(hz)], out[:int(hz)])
    im = Image.fromarray((np.clip(out, 0, 1) * 255).astype(np.uint8))
    d = ImageDraw.Draw(im, "RGBA")
    for lx, c in lights[::2]:                                  # 灯影：竖向的碎笔
        y = hz + 4
        while y < H * 0.97:
            L = 4 + rng.random() * 9
            w2 = 1.5 + rng.random() * 2.5 + (y - hz) / H * 4
            a = int(170 * max(0.0, 1 - (y - hz) / (0.45 * H)) * (0.4 + 0.6 * rng.random()))
            if a > 8:
                jx = rng.normal(0, 2)
                d.rectangle([lx - w2 + jx, y, lx + w2 + jx, y + L * 0.5], fill=(*c, a))
            y += L + rng.random() * 6
    for k in range(60):                                        # 月影
        y = hz + 10 + k * (0.38 * H / 60)
        w2 = 6 + k * 0.5 + rng.random() * 8
        d.rectangle([mx - w2, y, mx + w2, y + 2 + rng.random() * 3], fill=(235, 232, 210, int(120 * (1 - k / 60))))
    out = np.asarray(im).astype(float) / 255
    return vignette(canvas_texture(out, rng, 0.05), 0.25)


# ---------------------------------------------------------------- 6 芦汀白鹭（水墨淡彩）
def p_reeds_egret(rng):
    img = paper(rng, base=(0.92, 0.89, 0.81))
    yy, xx = np.mgrid[0:H, 0:W].astype(float)
    wash = np.clip((yy - 0.62 * H) / (0.38 * H), 0, 1) ** 1.2 * (0.6 + 0.4 * fbm(H, W, rng, 4, 4, stretch=(1, 5)))
    img = img * (1 - 0.22 * wash[..., None]) + 0.22 * wash[..., None] * np.array([0.55, 0.62, 0.66])
    mx, my, mr = 0.70 * W, 0.20 * H, 0.10 * W
    rm = np.hypot(xx - mx, yy - my)
    img = img * (1 - 0.05 * np.exp(-((rm - mr) / 3) ** 2))[..., None] + np.where((rm < mr)[..., None], 0.03, 0)
    im = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))
    d = ImageDraw.Draw(im, "RGBA")
    for k in range(16):                                        # 芦苇：弯秆、长叶、芦花
        x0 = (0.02 + 0.55 * rng.random()) * W if k < 11 else (0.80 + 0.18 * rng.random()) * W
        h_ = (0.35 + 0.35 * rng.random()) * H
        bend = rng.normal(0, 0.12) * W
        pts = [(x0 + bend * t ** 2, H - t * h_) for t in np.linspace(0, 1, 30)]
        ink_a = int(150 + rng.random() * 90)
        for i in range(len(pts) - 1):
            wdt = max(1, int(5 * (1 - i / len(pts)) + 1))
            d.line([pts[i], pts[i + 1]], fill=(32, 34, 30, ink_a), width=wdt)
        for _ in range(3):                                     # 叶
            t = 0.25 + 0.5 * rng.random()
            bx, by = pts[int(t * 29)]
            side = 1 if rng.random() < 0.5 else -1
            L = 90 + rng.random() * 160
            leaf = []
            for s in np.linspace(0, 1, 14):
                leaf.append((bx + side * s * L * 0.8, by - s * L * 0.55 + s * s * L * 0.55))
            back = [(x + side * 0, y + (1 - s) * 7) for (x, y), s in zip(leaf[::-1], np.linspace(1, 0, 14))]
            d.polygon(leaf + back, fill=(40, 46, 36, int(ink_a * 0.8)))
        tx, ty = pts[-1]                                       # 芦花
        for _ in range(70):
            a = rng.normal(-math.pi / 2 + 0.6, 0.5)
            L = 6 + rng.random() * 26
            d.line([(tx, ty), (tx + math.cos(a) * L, ty + math.sin(a) * L)], fill=(150, 120, 90, int(40 + rng.random() * 80)), width=1)
    # 白鹭：留白的身体，淡墨勾边，S 形颈
    ex, ey = 0.60 * W, 0.66 * H
    body = [(ex + 70 * math.cos(a), ey + 30 * math.sin(a) - 14 * math.cos(a)) for a in np.linspace(0, 2 * math.pi, 40)]
    d.polygon(body, fill=(247, 245, 238, 255), outline=(110, 110, 105, 200))
    neck = [(ex - 50 + 14 * math.sin(t * 3.2), ey - 10 - t * 110) for t in np.linspace(0, 1, 20)]
    for i in range(len(neck) - 1):
        d.line([neck[i], neck[i + 1]], fill=(247, 245, 238, 255), width=13)
    for i in range(len(neck) - 1):
        a, b = neck[i], neck[i + 1]
        d.line([(a[0] - 7, a[1]), (b[0] - 7, b[1])], fill=(120, 120, 115, 150), width=1)
    hx, hy = neck[-1]
    d.ellipse([hx - 13, hy - 10, hx + 13, hy + 9], fill=(247, 245, 238, 255), outline=(110, 110, 105, 200))
    d.polygon([(hx - 10, hy - 2), (hx - 58, hy + 3), (hx - 10, hy + 4)], fill=(196, 160, 70, 235))
    d.ellipse([hx - 4, hy - 4, hx, hy], fill=(30, 30, 30, 255))
    for lx in (ex - 6, ex + 14):                               # 腿
        d.line([(lx, ey + 26), (lx + 4, ey + 150)], fill=(40, 40, 38, 230), width=3)
    d.line([(ex + 20, ey - 8), (ex + 100, ey + 22)], fill=(150, 150, 140, 120), width=1)   # 尾羽
    img = np.asarray(im).astype(float) / 255
    return seal(img, "白鹭", int(0.07 * W), int(0.06 * H), 54, rng)


def main(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    makers = [p_ink_landscape, p_sunset, p_moon_gate, p_tide_lines, p_night_lake, p_reeds_egret]
    meta = []
    for i, fn in enumerate(makers):
        rng = np.random.default_rng(20261102 + i * 17)
        img = np.clip(fn(rng), 0, 1)
        path = os.path.join(out_dir, f"painting_{i}.jpg")
        Image.fromarray((img * 255).astype(np.uint8)).save(path, quality=90, optimize=True)
        meta.append(dict(file=f"painting_{i}.jpg", title=TITLES[i][0], note=TITLES[i][1]))
        print("saved", path)
    with open(os.path.join(out_dir, "paintings.json"), "w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "../assets/paintings")
