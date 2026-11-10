import math

import cv2
import numpy as np

LUMA = np.array([0.0722, 0.7152, 0.2126], dtype=np.float32)


def measure(frames):
    luma_values = []
    chroma_values = []
    for frame in frames:
        small = cv2.resize(frame, None, fx=0.125, fy=0.125, interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        luma = small @ LUMA
        spread = small.max(axis=2) - small.min(axis=2)
        middle = (luma > 0.08) & (luma < 0.92)
        luma_values.append(luma.ravel())
        if middle.any():
            chroma_values.append(spread[middle])
    luma = np.concatenate(luma_values) if luma_values else np.array([0.5])
    chroma = np.concatenate(chroma_values) if chroma_values else np.array([0.2])
    return {
        "black": float(np.percentile(luma, 1.0)),
        "white": float(np.percentile(luma, 99.5)),
        "mid": float(np.median(luma)),
        "chroma": float(np.mean(chroma)),
    }


def solve(stats, look, override=None):
    override = override or {}
    automatic = override.get("auto", True)
    strength = float(look["normalize"]) if automatic else 0.0
    black_target = float(look["black_target"])
    white_target = float(look["white_target"])
    black_in = black_target + (stats["black"] - black_target) * strength
    white_in = white_target + (stats["white"] - white_target) * strength
    gain = (white_target - black_target) / max(white_in - black_in, 0.05)
    gain = float(np.clip(gain, 0.8, 1.25))
    mid = (stats["mid"] - black_in) * gain + black_target
    mid = float(np.clip(mid, 0.03, 0.97))
    gamma = 1.0
    if automatic and float(look["mid_strength"]) > 0:
        wanted = mid + (float(look["mid_target"]) - mid) * float(look["mid_strength"])
        wanted = float(np.clip(wanted, 0.03, 0.97))
        gamma = float(np.clip(math.log(wanted) / math.log(mid), 0.82, 1.22))
    saturation = float(look["saturation"])
    if automatic and stats["chroma"] > 0.01:
        ideal = float(look["sat_target"]) / stats["chroma"]
        saturation *= float(np.clip(1.0 + (ideal - 1.0) * float(look["sat_strength"]), 0.85, 1.2))
    saturation *= float(override.get("saturation", 1.0))
    return {
        "black_in": round(black_in, 4),
        "gain": round(gain * (2.0 ** float(override.get("exposure", 0.0))), 4),
        "gamma": round(gamma, 4),
        "saturation": round(saturation, 4),
        "contrast": round(float(look["contrast"]) + float(override.get("contrast", 0.0)), 4),
        "split": float(look["split_tone"]),
        "temperature": float(override.get("temperature", 0.0)),
        "black_target": black_target,
    }


def tables(params):
    x = np.arange(256, dtype=np.float64) / 255.0
    y = (x - params["black_in"]) * params["gain"] + params["black_target"]
    y = np.clip(y, 0.0, 1.0) ** params["gamma"]
    y = y + params["contrast"] * (y * y * (3.0 - 2.0 * y) - y)
    high = y * y
    low = (1.0 - y) ** 2
    tone = params["split"] * (high - low)
    warm = params["temperature"]
    blue = np.clip(y - tone - warm, 0.0, 1.0)
    green = np.clip(y, 0.0, 1.0)
    red = np.clip(y + tone + warm, 0.0, 1.0)
    lut = np.stack([blue, green, red], axis=1).reshape(256, 1, 3)
    lut = np.round(lut * 255.0).astype(np.uint8)
    s = params["saturation"]
    matrix = (1.0 - s) * np.tile(LUMA, (3, 1)) + s * np.eye(3, dtype=np.float32)
    return lut, matrix.astype(np.float32)


class Look:
    def __init__(self, settings, width, height):
        self.settings = settings
        self.width = width
        self.height = height
        self.vignette = None
        amount = float(settings.get("vignette", 0.0))
        if amount > 0:
            ys, xs = np.mgrid[0:height, 0:width].astype(np.float32)
            radius = np.hypot((xs - width / 2.0) / (width / 2.0), (ys - height / 2.0) / (height / 2.0)) / math.sqrt(2.0)
            mask = 1.0 - amount * radius**2.2
            self.vignette = cv2.merge([np.round(mask * 255.0).astype(np.uint8)] * 3)
        self.grain = []
        level = float(settings.get("grain", 0.0))
        if level > 0:
            rng = np.random.default_rng(1977)
            for _ in range(6):
                noise = rng.standard_normal((height, width)).astype(np.float32)
                noise = cv2.GaussianBlur(noise, (0, 0), max(0.6, 0.6 * height / 1920.0))
                noise *= level * 255.0 / max(float(noise.std()), 1e-6)
                positive = np.clip(noise, 0, 255).astype(np.uint8)
                negative = np.clip(-noise, 0, 255).astype(np.uint8)
                self.grain.append((cv2.merge([positive] * 3), cv2.merge([negative] * 3)))

    def apply(self, frame, lut, matrix, flash=0.0, number=0, finish=True):
        out = frame
        sharpen = float(self.settings.get("sharpen", 0.0))
        if finish and sharpen > 0:
            blurred = cv2.GaussianBlur(out, (0, 0), max(0.8, 1.1 * self.height / 1920.0))
            out = cv2.addWeighted(out, 1.0 + sharpen, blurred, -sharpen, 0)
        out = cv2.LUT(out, lut)
        out = cv2.transform(out, matrix)
        if flash > 0:
            out = cv2.convertScaleAbs(out, alpha=1.0 + 1.1 * flash, beta=255.0 * 0.5 * flash)
        if self.vignette is not None:
            out = cv2.multiply(out, self.vignette, scale=1.0 / 255.0)
        if self.grain:
            positive, negative = self.grain[number % len(self.grain)]
            turn = (number // len(self.grain)) % 4
            if turn == 1:
                positive, negative = positive[::-1], negative[::-1]
            elif turn == 2:
                positive, negative = positive[:, ::-1], negative[:, ::-1]
            elif turn == 3:
                positive, negative = positive[::-1, ::-1], negative[::-1, ::-1]
            out = cv2.subtract(cv2.add(out, np.ascontiguousarray(positive)), np.ascontiguousarray(negative))
        return out
