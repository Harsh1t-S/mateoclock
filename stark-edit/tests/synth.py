import json
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np

WIDTH = 1920
HEIGHT = 1080
SHIFT = 4
FIXED = 1 << SHIFT
BITS = 12
SAMPLE_RATE = 48000
TEMPO = 128.0
SONG_OFFSET = 0.371
DROP_BAR = 8
HIGH_BARS = 16
TOTAL_BARS = 32


def point(x, y):
    return int(round(x * FIXED)), int(round(y * FIXED))


def make_background(rng, hue, height=1500, width=2700):
    low = rng.random((18, 32, 3)).astype(np.float32)
    big = np.clip(cv2.resize(low, (width, height), interpolation=cv2.INTER_CUBIC), 0.0, 1.0)
    hsv = np.zeros((height, width, 3), np.float32)
    hsv[..., 0] = (hue + 30.0 * (big[..., 0] - 0.5)) % 180.0
    hsv[..., 1] = (0.35 + 0.4 * big[..., 1]) * 255.0
    hsv[..., 2] = (0.22 + 0.4 * big[..., 2]) * 255.0
    image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    for _ in range(320):
        x = int(rng.integers(0, width))
        y = int(rng.integers(0, height))
        r = int(rng.integers(4, 18))
        color = tuple(int(v) for v in rng.integers(25, 170, 3))
        cv2.circle(image, (x, y), r, color, -1, cv2.LINE_AA)
    for _ in range(40):
        x = int(rng.integers(0, width))
        y = int(rng.integers(0, height))
        w = int(rng.integers(30, 140))
        h = int(rng.integers(6, 30))
        color = tuple(int(v) for v in rng.integers(25, 150, 3))
        cv2.rectangle(image, (x, y), (x + w, y + h), color, -1)
    return image


def draw_subject(image, cx, cy, d, color, code):
    dark = tuple(int(c * 0.45) for c in color)
    cv2.rectangle(image, point(cx - 2.3 * d, cy + 2.35 * d), point(cx + 2.3 * d, cy + 6.5 * d), dark, -1, cv2.LINE_AA, SHIFT)
    cv2.ellipse(image, point(cx, cy + 0.35 * d), (int(1.55 * d * FIXED), int(1.95 * d * FIXED)), 0, 0, 360, color, -1, cv2.LINE_AA, SHIFT)
    marks = [(-1.1, -0.9), (1.0, -1.0), (-1.2, 0.5), (1.2, 0.45), (0.0, -1.2), (-0.6, 1.75), (0.6, 1.75), (0.0, 0.42)]
    for mx, my in marks:
        cv2.circle(image, point(cx + mx * d, cy + my * d), int(0.07 * d * FIXED), dark, -1, cv2.LINE_AA, SHIFT)
    cv2.line(image, point(cx - 0.5 * d, cy + 1.55 * d), point(cx + 0.5 * d, cy + 1.55 * d), dark, max(1, int(0.05 * d)), cv2.LINE_AA, SHIFT)
    for sx in (-0.5, 0.5):
        cv2.circle(image, point(cx + sx * d, cy), int(0.16 * d * FIXED), (255, 255, 255), -1, cv2.LINE_AA, SHIFT)
    cv2.circle(image, point(cx, cy + 3.2 * d), int(0.3 * d * FIXED), (255, 255, 215), -1, cv2.LINE_AA, SHIFT)
    for mx, my in [(-1.4, 2.9), (1.4, 2.9), (-1.6, 4.2), (1.6, 4.2), (0.0, 4.4), (-0.8, 5.3), (0.8, 5.3)]:
        cv2.rectangle(image, point(cx + (mx - 0.12) * d, cy + (my - 0.12) * d), point(cx + (mx + 0.12) * d, cy + (my + 0.12) * d), color, -1, cv2.LINE_AA, SHIFT)
    block = 0.2 * d
    y0 = cy + 0.95 * d - block / 2
    x0 = cx - BITS * block / 2
    cv2.rectangle(image, point(x0 - 0.06 * d, y0 - 0.06 * d), point(x0 + BITS * block + 0.06 * d, y0 + block + 0.06 * d), (0, 0, 0), -1, cv2.LINE_AA, SHIFT)
    for bit in range(BITS):
        if (code >> (BITS - 1 - bit)) & 1:
            cv2.rectangle(image, point(x0 + bit * block + 0.02 * d, y0 + 0.02 * d), point(x0 + (bit + 1) * block - 0.02 * d, y0 + block - 0.02 * d), (255, 255, 255), -1, cv2.LINE_AA, SHIFT)


def read_code(image, x, y, d):
    block = 0.2 * d
    value = 0
    half = max(1, int(block * 0.2))
    for bit in range(BITS):
        px = int(round(x - BITS * block / 2 + (bit + 0.5) * block))
        py = int(round(y + 0.95 * d))
        if py - half < 0 or px - half < 0 or py + half >= image.shape[0] or px + half >= image.shape[1]:
            return None
        patch = image[py - half : py + half + 1, px - half : px + half + 1]
        value = (value << 1) | int(patch.mean() > 110)
    return value


class Shot:
    def __init__(self, rng, length, hue, color, base, drift, size, growth, wobble):
        self.length = int(length)
        self.background = make_background(rng, hue)
        self.color = color
        self.base = base
        self.drift = drift
        self.size = size
        self.growth = growth
        self.wobble = wobble
        self.pan = (float(rng.uniform(-1.6, 1.6)), float(rng.uniform(-0.6, 0.6)))
        self.origin = (float(rng.uniform(200, 500)), float(rng.uniform(120, 260)))

    def anchor(self, k):
        cx = self.base[0] + self.drift[0] * k + self.wobble[0] * np.sin(2.0 * np.pi * k / 37.0)
        cy = self.base[1] + self.drift[1] * k + self.wobble[1] * np.sin(2.0 * np.pi * k / 29.0 + 0.7)
        d = self.size * (1.0 + self.growth * k)
        return float(cx), float(cy), float(d)

    def frame(self, k, code, bars):
        matrix = np.float32([[1, 0, -(self.origin[0] + self.pan[0] * k)], [0, 1, -(self.origin[1] + self.pan[1] * k)]])
        image = cv2.warpAffine(self.background, matrix, (WIDTH, HEIGHT), flags=cv2.INTER_LINEAR)
        cx, cy, d = self.anchor(k)
        draw_subject(image, cx, cy, d, self.color, code)
        top, bottom, left, right = bars
        if top:
            image[:top] = 0
        if bottom:
            image[HEIGHT - bottom :] = 0
        if left:
            image[:, :left] = 0
        if right:
            image[:, WIDTH - right :] = 0
        return image


def encode(path, frames, rate, tagged, audio=None, gop=48):
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{WIDTH}x{HEIGHT}", "-framerate", rate, "-i", "pipe:0"]
    if audio is not None:
        cmd += ["-i", str(audio)]
    cmd += ["-vf", "scale=in_range=pc:out_range=tv:out_color_matrix=bt709:flags=bicubic+accurate_rnd+full_chroma_int,format=yuv420p"]
    cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "14", "-g", str(gop), "-bf", "2", "-pix_fmt", "yuv420p"]
    if tagged:
        cmd += ["-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709", "-color_range", "tv"]
    if audio is not None:
        cmd += ["-c:a", "aac", "-b:a", "160k", "-shortest"]
    cmd += [str(path)]
    process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    count = 0
    for frame in frames:
        process.stdin.write(frame.tobytes())
        count += 1
    process.stdin.close()
    error = process.stderr.read().decode(errors="replace")
    if process.wait() != 0:
        raise RuntimeError(error)
    return count


def envelope(length, decay):
    t = np.arange(length) / SAMPLE_RATE
    return np.exp(-t / decay)


def kick(level=1.0):
    n = int(0.32 * SAMPLE_RATE)
    t = np.arange(n) / SAMPLE_RATE
    freq = 48.0 + 110.0 * np.exp(-t / 0.035)
    phase = 2.0 * np.pi * np.cumsum(freq) / SAMPLE_RATE
    body = np.sin(phase) * envelope(n, 0.11)
    click = np.zeros(n)
    click[: int(0.004 * SAMPLE_RATE)] = np.hanning(int(0.008 * SAMPLE_RATE))[int(0.004 * SAMPLE_RATE) :]
    return level * (0.95 * body + 0.5 * click)


def snare(rng, level=1.0):
    n = int(0.22 * SAMPLE_RATE)
    t = np.arange(n) / SAMPLE_RATE
    noise = rng.standard_normal(n)
    noise = np.diff(noise, prepend=0.0)
    tone = np.sin(2.0 * np.pi * 185.0 * t)
    return level * (0.5 * noise * envelope(n, 0.07) + 0.4 * tone * envelope(n, 0.05))


def hat(rng, level=1.0):
    n = int(0.06 * SAMPLE_RATE)
    noise = rng.standard_normal(n)
    noise = np.diff(np.diff(noise, prepend=0.0), prepend=0.0)
    return level * 0.18 * noise * envelope(n, 0.018)


def tone(freq, seconds, level, harmonics=4):
    n = int(seconds * SAMPLE_RATE)
    t = np.arange(n) / SAMPLE_RATE
    out = np.zeros(n)
    for h in range(1, harmonics + 1):
        out += np.sin(2.0 * np.pi * freq * h * t) / h
    fade = int(0.01 * SAMPLE_RATE)
    window = np.ones(n)
    window[:fade] = np.linspace(0.0, 1.0, fade)
    window[-fade:] = np.linspace(1.0, 0.0, fade)
    return level * out * window


def add(buffer, start, sound):
    i = int(round(start * SAMPLE_RATE))
    if i >= len(buffer):
        return
    j = min(len(buffer), i + len(sound))
    buffer[i:j] += sound[: j - i]


def make_song(folder, rng):
    beat = 60.0 / TEMPO
    bar = 4 * beat
    total = SONG_OFFSET + TOTAL_BARS * bar + 1.0
    mix = np.zeros(int(total * SAMPLE_RATE))
    roots = [55.0, 43.65, 65.41, 49.0]
    for b in range(TOTAL_BARS):
        start = SONG_OFFSET + b * bar
        high = DROP_BAR <= b < DROP_BAR + HIGH_BARS
        build = b == DROP_BAR - 1
        root = roots[b % 4]
        if high:
            add(mix, start, tone(root, bar, 0.16, 5))
            add(mix, start, tone(root * 4, bar, 0.035, 2))
            add(mix, start, tone(root * 5, bar, 0.03, 2))
            add(mix, start, tone(root * 6, bar, 0.03, 2))
        else:
            add(mix, start, tone(root * 2, bar, 0.035, 3))
            add(mix, start, tone(root * 3, bar, 0.02, 2))
        for k in range(4):
            t = start + k * beat
            if high:
                add(mix, t, kick(0.9))
                if k in (1, 3):
                    add(mix, t, snare(rng, 0.55))
                add(mix, t, hat(rng, 0.9))
                add(mix, t + beat / 2, hat(rng, 1.3))
            elif build:
                add(mix, t, hat(rng, 0.8))
                add(mix, t + beat / 2, hat(rng, 0.8))
            else:
                add(mix, t, kick(0.22 if k == 0 else 0.12))
                add(mix, t + beat / 2, hat(rng, 0.5))
        if build:
            n = int(bar * SAMPLE_RATE)
            rise = np.linspace(0.0, 1.0, n) ** 2
            noise = np.diff(rng.standard_normal(n), prepend=0.0)
            add(mix, start, 0.09 * noise * rise)
        if b in (DROP_BAR, DROP_BAR + 8):
            n = int(1.2 * SAMPLE_RATE)
            crash = np.diff(rng.standard_normal(n), prepend=0.0) * envelope(n, 0.35)
            add(mix, start, 0.22 * crash)
    mix = np.tanh(mix * 1.1) * 0.85
    stereo = np.stack([mix, mix], axis=1).astype(np.float32)
    import soundfile

    wav = folder / "song_source.wav"
    soundfile.write(str(wav), stereo, SAMPLE_RATE, subtype="PCM_16")
    target = folder / "audio" / "My Song (Official Video) [abc123].mp4"
    target.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i", f"color=c=black:s=320x180:r=5:d={total:.3f}", "-i", str(wav), "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-shortest", str(target)],
        check=True,
    )
    wav.unlink()
    beats = [SONG_OFFSET + i * beat for i in range(TOTAL_BARS * 4)]
    return {
        "file": str(target.relative_to(folder)),
        "tempo": TEMPO,
        "beat": beat,
        "beats": beats,
        "downbeats": beats[::4],
        "drop": SONG_OFFSET + DROP_BAR * bar,
        "high_end": SONG_OFFSET + (DROP_BAR + HIGH_BARS) * bar,
    }


def voice_track(path, seconds, bursts):
    data = np.zeros(int(seconds * SAMPLE_RATE))
    for start, freq in bursts:
        add(data, start, tone(freq, 0.18, 0.4, 3))
    import soundfile

    soundfile.write(str(path), np.stack([data, data], axis=1).astype(np.float32), SAMPLE_RATE, subtype="PCM_16")
    return path


CLIPS = [
    {"name": "001 - scope clip (1080p) [xYz].mp4", "key": "scope", "rate": "24000/1001", "bars": (138, 138, 0, 0), "tagged": True, "lengths": [72, 60, 84, 48, 96], "audio": True, "y": 462, "size": (70, 82)},
    {"name": "flat_185_clip.mp4", "key": "flat", "rate": "24", "bars": (21, 21, 0, 0), "tagged": False, "lengths": [80, 64, 72, 90], "audio": False, "y": 440, "size": (78, 96)},
    {"name": "Clip PAL 25fps.mp4", "key": "pal", "rate": "25", "bars": (0, 0, 0, 0), "tagged": True, "lengths": [75, 60, 90, 70], "audio": False, "y": 430, "size": (80, 100)},
    {"name": "ntsc_true30.mp4", "key": "ntsc", "rate": "30000/1001", "bars": (138, 138, 0, 0), "tagged": False, "lengths": [90, 75, 110], "audio": False, "y": 462, "size": (68, 80)},
    {"name": "telecined 29.97 upload.mp4", "key": "telecine", "rate": "30000/1001", "bars": (138, 138, 0, 0), "tagged": True, "lengths": [72, 96, 64], "audio": False, "y": 462, "size": (70, 80), "telecine": True},
    {"name": "sixty_fps.mp4", "key": "sixty", "rate": "60", "bars": (138, 138, 0, 0), "tagged": True, "lengths": [240, 200], "audio": False, "y": 462, "size": (70, 80), "fast": True},
]


def build_clip(folder, spec, rng, hue_start):
    bars = spec["bars"]
    shots = []
    for n, length in enumerate(spec["lengths"]):
        hue = (hue_start + 47 * n) % 180
        color = tuple(int(v) for v in cv2.cvtColor(np.uint8([[[((hue + 90) % 180), 190, 215]]]), cv2.COLOR_HSV2BGR)[0, 0])
        speed = 2.4 if spec.get("fast") else 1.0
        base = (float(rng.uniform(700, 1220)), float(spec["y"] + rng.uniform(-6, 6)))
        drift = (float(rng.uniform(-0.9, 0.9) * speed), float(rng.uniform(-0.05, 0.05)))
        size = float(rng.uniform(*spec["size"]))
        growth = float(rng.uniform(-0.0004, 0.0006))
        wobble = (float(rng.uniform(4, 14) * speed), float(rng.uniform(0.5, 2.5)))
        shots.append(Shot(rng, length, hue, color, base, drift, size, growth, wobble))

    truth_frames = []
    source_index = []
    for s, shot in enumerate(shots):
        for k in range(shot.length):
            source_index.append((s, k))
    order = {item: n for n, item in enumerate(source_index)}
    if spec.get("telecine"):
        emitted = [source_index[int(m * 0.8)] for m in range(int(len(source_index) / 0.8))]
    else:
        emitted = source_index

    def frames():
        for code, (s, k) in enumerate(emitted):
            shot = shots[s]
            unique = order[(s, k)]
            cx, cy, d = shot.anchor(k)
            truth_frames.append({"shot": s, "k": k, "code": unique, "eyes": [cx, cy, d], "reactor": [cx, cy + 3.2 * d, 0.6 * d]})
            yield shot.frame(k, unique, bars)

    path = folder / "clips" / spec["name"]
    path.parent.mkdir(parents=True, exist_ok=True)
    audio = None
    bursts = []
    if spec["audio"]:
        seconds = len(emitted) / float(Fraction(spec["rate"]))
        bursts = [(0.5, 440.0), (0.9, 554.0), (1.3, 659.0), (2.2, 880.0)]
        audio = voice_track(folder / "voice.wav", seconds + 0.5, bursts)
    encode(path, frames(), spec["rate"], spec["tagged"], audio)
    if audio is not None:
        Path(audio).unlink()
    boundaries = []
    previous = None
    for i, item in enumerate(truth_frames):
        if previous is not None and item["shot"] != previous:
            boundaries.append(i)
        previous = item["shot"]
    return {
        "file": str(path.relative_to(folder)),
        "key": spec["key"],
        "rate": spec["rate"],
        "bars": list(bars),
        "active": [bars[2], bars[0], WIDTH - bars[3], HEIGHT - bars[1]],
        "cuts": boundaries,
        "frames": truth_frames,
        "bursts": bursts,
        "telecine": bool(spec.get("telecine")),
    }


def build_face_clip(folder, portrait):
    image = cv2.imread(str(portrait))
    if image is None:
        return None
    big = cv2.resize(image, None, fx=4.0, fy=4.0, interpolation=cv2.INTER_LANCZOS4)
    bars = (138, 138, 0, 0)
    plans = [
        {"length": 60, "scale": (0.62, 0.66), "at": (880.0, 300.0), "move": (0.5, 0.1), "flip": False},
        {"length": 60, "scale": (0.80, 0.78), "at": (1080.0, 310.0), "move": (-0.4, 0.05), "flip": True},
    ]
    truth = []

    def frames():
        for s, plan in enumerate(plans):
            source = cv2.flip(big, 1) if plan["flip"] else big
            for k in range(plan["length"]):
                u = k / max(1, plan["length"] - 1)
                scale = plan["scale"][0] + (plan["scale"][1] - plan["scale"][0]) * u
                tx = plan["at"][0] + plan["move"][0] * k - scale * source.shape[1] * 0.44
                ty = plan["at"][1] + plan["move"][1] * k - scale * source.shape[0] * 0.2
                matrix = np.float32([[scale, 0, tx], [0, scale, ty]])
                blurred = cv2.GaussianBlur(cv2.resize(source, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA), (0, 0), 25)
                frame = cv2.warpAffine(source, matrix, (WIDTH, HEIGHT), dst=blurred, flags=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC, borderMode=cv2.BORDER_TRANSPARENT)
                frame[: bars[0]] = 0
                frame[HEIGHT - bars[1] :] = 0
                truth.append({"shot": s, "k": k, "matrix": [float(scale), float(tx), float(ty)], "flip": plan["flip"], "source_width": int(source.shape[1])})
                yield frame

    path = folder / "clips" / "portrait closeup.mkv"
    encode(path, frames(), "24000/1001", True)
    return {
        "file": str(path.relative_to(folder)),
        "key": "portrait",
        "rate": "24000/1001",
        "bars": list(bars),
        "active": [0, bars[0], WIDTH, HEIGHT - bars[1]],
        "cuts": [plans[0]["length"]],
        "frames": truth,
        "telecine": False,
    }


def build(folder, portrait=None, seed=7):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    truth = {"song": make_song(folder, rng), "clips": {}}
    for n, spec in enumerate(CLIPS):
        truth["clips"][spec["key"]] = build_clip(folder, spec, rng, 20 + 31 * n)
    if portrait is not None and Path(portrait).exists():
        face = build_face_clip(folder, portrait)
        if face is not None:
            truth["clips"]["portrait"] = face
    (folder / "truth.json").write_text(json.dumps(truth))
    return truth


if __name__ == "__main__":
    target = sys.argv[1]
    picture = sys.argv[2] if len(sys.argv) > 2 else None
    result = build(target, picture)
    for key, clip in result["clips"].items():
        print(key, clip["file"], len(clip["frames"]), "frames", "cuts", clip["cuts"])
    print("song", result["song"]["file"], "drop", round(result["song"]["drop"], 3))
