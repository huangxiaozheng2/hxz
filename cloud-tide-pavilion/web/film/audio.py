"""导览影片的配乐：程序合成，不含任何现成音乐。
柔和的和弦铺底（随段落换和弦）、每段字幕出现时一声轻钟、极淡的湖水声，整体加混响，开头结尾淡入淡出。
    ../venv/bin/python web/film/audio.py 输出.wav [时长 秒]
"""
import sys

import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, fftconvolve, sosfilt

SR = 48000
DUR = float(sys.argv[2]) if len(sys.argv) > 2 else 110.0
N = int(DUR * SR)
t = np.arange(N) / SR
rng = np.random.default_rng(20261102)


def hz(note):
    """'A4' / 'F#3' → 频率"""
    names = {"C": -9, "C#": -8, "D": -7, "D#": -6, "E": -5, "F": -4, "F#": -3, "G": -2, "G#": -1, "A": 0, "A#": 1, "B": 2}
    return 440.0 * 2 ** ((names[note[:-1]] + 12 * (int(note[-1]) - 4)) / 12)


def env(t0, t1, att, rel):
    """t0 起音（att 秒升满），t1 起释音（rel 秒衰尽）"""
    e = np.clip((t - t0) / att, 0, 1) * np.clip(1 - (t - t1) / rel, 0, 1)
    return np.where(t < t0, 0, e) ** 1.5


# 和弦铺底：每个音是基音 + 弱的二、三次谐波，左右声道各带一点失谐，起落都很慢
CHORDS = [  # (起, 止, 音)
    (0.0, 14.0, ["D3", "A3", "E4", "F#4"]),
    (13.0, 26.5, ["B2", "F#3", "D4", "A4"]),
    (25.5, 35.5, ["G2", "D3", "F#3", "B3"]),
    (34.5, 46.0, ["A2", "E3", "B3", "E4"]),
    (45.0, 60.0, ["E2", "B2", "G3", "D4", "F#4"]),
    (59.0, 74.0, ["D3", "A3", "E4", "F#4"]),
    (73.0, 86.0, ["G2", "D3", "A3", "B3", "F#4"]),
    (85.0, 100.0, ["B2", "F#3", "C#4", "D4"]),
    (99.0, DUR, ["D3", "A3", "E4", "F#4", "A4"]),
]
pad = np.zeros((N, 2))
for t0, t1, notes in CHORDS:
    e = env(t0, t1, 3.0, 4.0)
    for k, n in enumerate(notes):
        f = hz(n)
        for ch, det in ((0, 0.9993), (1, 1.0007)):
            ph = rng.uniform(0, 2 * np.pi)
            w = 2 * np.pi * f * det * t + ph + 0.002 * np.sin(2 * np.pi * 0.21 * t + k)   # 很轻的颤音
            tone = np.sin(w) + 0.22 * np.sin(2 * w) + 0.06 * np.sin(3 * w)
            pad[:, ch] += tone * e * (0.9 if f < 200 else 0.6) / len(notes)
pad = np.stack([sosfilt(butter(2, 2400, fs=SR, output="sos"), pad[:, c]) for c in range(2)], 1)

# 钟音：每段字幕出现时一声，用五声音阶上的音；泛音按钟的比例
BELLS = [(1.0, "A5"), (7.0, "F#5"), (26.6, "D6"), (35.6, "E5"), (46.2, "B5"), (60.4, "F#5"), (74.4, "A5"), (86.4, "D6"), (100.4, "A5")]
bell = np.zeros((N, 2))
for tb, n in BELLS:
    f = hz(n)
    i0 = int(tb * SR)
    tt = t[i0:] - tb
    s = sum(a * np.sin(2 * np.pi * f * r * tt) * np.exp(-tt / d) for r, a, d in ((1, 1.0, 2.6), (2.76, 0.32, 1.1), (5.4, 0.12, 0.5)))
    s *= np.clip(tt / 0.006, 0, 1)
    pan = rng.uniform(-0.35, 0.35)
    bell[i0:, 0] += s * (1 - pan) * 0.5
    bell[i0:, 1] += s * (1 + pan) * 0.5

# 湖水：低通的布朗噪声，随几道缓慢的“涌”起伏
white = rng.normal(size=(N, 2))
brown = np.cumsum(white, axis=0)                           # 布朗噪声；带通滤波同时去掉了低频漂移
water = np.stack([sosfilt(butter(2, [90, 700], btype="band", fs=SR, output="sos"), brown[:, c]) for c in range(2)], 1)
swell = 0.55 + 0.45 * (0.5 * np.sin(2 * np.pi * 0.09 * t + 1.3) + 0.3 * np.sin(2 * np.pi * 0.17 * t + 0.4) + 0.2 * np.sin(2 * np.pi * 0.31 * t + 2.2))
water *= swell[:, None] / (np.abs(water).max() + 1e-9)

# 混响：指数衰减的噪声做冲激响应（约 3 秒），左右声道不同
L = int(3.2 * SR)
ir = rng.normal(size=(L, 2)) * np.exp(-np.arange(L) / (0.9 * SR))[:, None]
ir[:int(0.012 * SR)] = 0
ir /= np.sqrt((ir ** 2).sum(0))


def reverb(x, wet):
    return np.stack([(1 - wet) * x[:, c] + wet * fftconvolve(x[:, c], ir[:, c])[:N] for c in range(2)], 1)


norm = lambda x: x / (np.abs(x).max() + 1e-9)
mix = 0.55 * norm(reverb(pad, 0.45)) + 0.32 * norm(reverb(bell, 0.55)) + 0.10 * norm(water)
fade = np.clip(t / 2.0, 0, 1) * np.clip((DUR - t) / 3.0, 0, 1)
mix *= fade[:, None]
rms0 = np.sqrt((mix[int(2 * SR):int((DUR - 3) * SR)] ** 2).mean())
mix *= min(10 ** (-20 / 20) / rms0, 10 ** (-3 / 20) / np.abs(mix).max())   # 平均约 −20 dBFS（铺底不抢解说），峰值不超过 −3 dBFS
wavfile.write(sys.argv[1], SR, (mix * 32767).astype(np.int16))
rms = 20 * np.log10(np.sqrt((mix ** 2).mean()) + 1e-12)
peak = 20 * np.log10(np.abs(mix).max())
print(f"audio → {sys.argv[1]}：{DUR:.1f} s，峰值 {peak:.1f} dBFS，平均 {rms:.1f} dBFS")
