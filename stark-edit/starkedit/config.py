import copy
import json
from fractions import Fraction
from pathlib import Path

DEFAULTS = {
    "aspect": "9:16",
    "short_side": 1080,
    "fps": "auto",
    "paths": {
        "clips": "clips",
        "audio": "audio",
        "work": "work",
        "output": "output",
        "models": "models",
        "edl": "edl.json",
        "clips_index": "clips.json",
    },
    "length": {
        "min_seconds": 20.0,
        "max_seconds": 30.0,
        "target_seconds": 26.0,
        "cold_open_seconds": [1.0, 2.4],
        "tail_seconds": 0.75,
    },
    "song": {
        "beats_per_bar": 4,
        "beat_period_range": [0.34, 0.72],
        "section_start": None,
        "downbeat_shift": None,
        "cut_bias_frames": -0.2,
    },
    "scan": {
        "analysis_width": 480,
        "cut_ratio": 3.0,
        "cut_min_content": 12.0,
        "cut_hard_content": 38.0,
        "cut_min_hist": 0.2,
        "bar_threshold": 20.0,
        "bar_inset": 2,
        "profile_step": 4,
    },
    "framing": {
        "max_boost": 1.35,
        "track_sigma": 2.0,
        "track_pad_frames": 6,
        "upscale_warn": 3.2,
    },
    "targets": {
        "eyes": {"x": 0.5, "y": 0.40, "size": 0.105},
        "helmet": {"x": 0.5, "y": 0.40, "size": 0.118},
        "reactor": {"x": 0.5, "y": 0.52, "size": 0.070},
        "frame": {"x": 0.5, "y": 0.5, "size": 1.0},
    },
    "look": {
        "normalize": 0.6,
        "black_target": 0.03,
        "white_target": 0.93,
        "mid_target": 0.40,
        "mid_strength": 0.35,
        "sat_target": 0.24,
        "sat_strength": 0.3,
        "contrast": 0.10,
        "saturation": 1.04,
        "split_tone": 0.035,
        "vignette": 0.14,
        "grain": 0.018,
        "sharpen": 0.30,
    },
    "render": {
        "final": {"scale": 1.0, "crf": 16, "preset": "slow"},
        "preview": {"scale": 0.5, "crf": 24, "preset": "veryfast"},
        "audio_bitrate": "256k",
        "sample_rate": 48000,
        "loudness": -14.0,
        "peak_db": -1.0,
    },
}

COMMON_RATES = [
    Fraction(24000, 1001),
    Fraction(24),
    Fraction(25),
    Fraction(30000, 1001),
    Fraction(30),
    Fraction(48),
    Fraction(50),
    Fraction(60000, 1001),
    Fraction(60),
]


def merge(base, override):
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def parse_rate(value):
    if isinstance(value, Fraction):
        return value
    if isinstance(value, (int, float)):
        return snap_rate(float(value))
    text = str(value).strip()
    if "/" in text:
        num, den = text.split("/")
        return Fraction(int(num), int(den))
    return snap_rate(float(text))


def snap_rate(value):
    for rate in COMMON_RATES:
        if abs(float(rate) - value) / float(rate) < 0.0004:
            return rate
    return Fraction(value).limit_denominator(1001)


def rate_text(rate):
    rate = Fraction(rate)
    if rate.denominator == 1:
        return str(rate.numerator)
    return f"{rate.numerator}/{rate.denominator}"


def canvas_size(aspect, short_side):
    left, right = str(aspect).split(":")
    aw, ah = float(left), float(right)
    if aw <= 0 or ah <= 0:
        raise ValueError(f"invalid aspect {aspect}")
    short = int(short_side)
    if aw <= ah:
        width = short
        height = int(round(short * ah / aw / 2.0)) * 2
    else:
        height = short
        width = int(round(short * aw / ah / 2.0)) * 2
    return width, height


class Project:
    def __init__(self, root=".", overrides=None):
        self.root = Path(root).resolve()
        self.config_path = self.root / "config.json"
        loaded = {}
        if self.config_path.exists():
            loaded = json.loads(self.config_path.read_text())
        self.cfg = merge(merge(DEFAULTS, loaded), overrides or {})

    def path(self, key):
        return (self.root / self.cfg["paths"][key]).resolve()

    @property
    def clips_dir(self):
        return self.path("clips")

    @property
    def audio_dir(self):
        return self.path("audio")

    @property
    def work(self):
        target = self.path("work")
        target.mkdir(parents=True, exist_ok=True)
        return target

    @property
    def output(self):
        target = self.path("output")
        target.mkdir(parents=True, exist_ok=True)
        return target

    @property
    def models(self):
        target = self.path("models")
        target.mkdir(parents=True, exist_ok=True)
        return target

    @property
    def edl_path(self):
        return self.path("edl")

    @property
    def clips_index_path(self):
        return self.path("clips_index")

    def subdir(self, *parts):
        target = self.work.joinpath(*parts)
        target.mkdir(parents=True, exist_ok=True)
        return target

    @property
    def canvas(self):
        return canvas_size(self.cfg["aspect"], self.cfg["short_side"])

    def relative(self, path):
        path = Path(path).resolve()
        try:
            return str(path.relative_to(self.root))
        except ValueError:
            return str(path)


def load_json(path, default=None):
    path = Path(path)
    if not path.exists():
        return default
    return json.loads(path.read_text())


def _compact(value, indent, level, inline_depth):
    pad = " " * (indent * level)
    inner = " " * (indent * (level + 1))
    if isinstance(value, dict):
        if not value:
            return "{}"
        if _is_flat(value) and len(json.dumps(value)) <= inline_depth:
            return json.dumps(value)
        parts = [f"{inner}{json.dumps(str(k))}: {_compact(v, indent, level + 1, inline_depth)}" for k, v in value.items()]
        return "{\n" + ",\n".join(parts) + f"\n{pad}}}"
    if isinstance(value, (list, tuple)):
        if not value:
            return "[]"
        if all(_is_scalar(v) for v in value) and len(json.dumps(list(value))) <= inline_depth:
            return json.dumps(list(value))
        if all(_is_scalar(v) for v in value):
            rows = []
            row = []
            width = 0
            for item in value:
                text = json.dumps(item)
                if row and width + len(text) + 2 > inline_depth:
                    rows.append(", ".join(row))
                    row = []
                    width = 0
                row.append(text)
                width += len(text) + 2
            if row:
                rows.append(", ".join(row))
            return "[\n" + ",\n".join(inner + r for r in rows) + f"\n{pad}]"
        parts = [f"{inner}{_compact(v, indent, level + 1, inline_depth)}" for v in value]
        return "[\n" + ",\n".join(parts) + f"\n{pad}]"
    return json.dumps(value)


def _is_scalar(value):
    return not isinstance(value, (dict, list, tuple))


def _is_flat(value):
    for item in value.values():
        if isinstance(item, dict):
            return False
        if isinstance(item, (list, tuple)) and not all(_is_scalar(v) for v in item):
            return False
    return True


def dump_json(data, width=110):
    return _compact(data, 2, 0, width) + "\n"


def save_json(path, data, width=110):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dump_json(data, width))
