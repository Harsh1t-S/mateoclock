import hashlib
import math
from collections import Counter
from fractions import Fraction

import numpy as np
import scipy.ndimage

from . import anchors, clipscan
from .config import load_json, merge, parse_rate, rate_text
from .media import SourcePool


class EdlError(ValueError):
    pass


def load_edl(project, path=None):
    path = path or project.edl_path
    edl = load_json(path)
    if edl is None:
        raise EdlError(f"no EDL found at {path}")
    if "segments" not in edl or not edl["segments"]:
        raise EdlError("the EDL has no segments")
    if "audio" not in edl or "beat_times" not in edl["audio"]:
        raise EdlError("the EDL needs audio.beat_times")
    seen = set()
    for number, segment in enumerate(edl["segments"]):
        segment.setdefault("id", f"s{number + 1:02d}")
        if segment["id"] in seen:
            raise EdlError(f"duplicate segment id {segment['id']}")
        seen.add(segment["id"])
        if ("beats" in segment) == ("seconds" in segment):
            raise EdlError(f"segment {segment['id']} needs exactly one of beats or seconds")
    return edl


def stable_seed(*parts):
    digest = hashlib.sha1("|".join(str(p) for p in parts).encode()).hexdigest()
    return int(digest[:8], 16)


def curve(spec, default):
    if spec is None:
        return lambda x: np.full_like(np.asarray(x, dtype=np.float64), float(default))
    if isinstance(spec, (int, float)):
        return lambda x: np.full_like(np.asarray(x, dtype=np.float64), float(spec))
    points = sorted((float(a), float(b)) for a, b in spec)
    xs = np.array([p[0] for p in points])
    vs = np.array([p[1] for p in points])
    return lambda x: np.interp(np.asarray(x, dtype=np.float64), xs, vs)


def ease(x, kind):
    x = np.clip(np.asarray(x, dtype=np.float64), 0.0, 1.0)
    if kind == "in":
        return x * x
    if kind == "out":
        return 1.0 - (1.0 - x) ** 2
    if kind == "inout":
        return x * x * (3.0 - 2.0 * x)
    return x


def advance(speed, frames):
    sub = 16
    grid = np.linspace(0.0, 1.0, frames * sub + 1)
    values = speed(grid)
    steps = 0.5 * (values[1:] + values[:-1]) / sub
    total = np.concatenate([[0.0], np.cumsum(steps)])
    return total[::sub][:frames], float(total[-1]), lambda x: float(np.interp(x, grid, total))


class BeatMap:
    def __init__(self, beat_times):
        self.times = np.asarray(beat_times, dtype=np.float64)
        if len(self.times) < 2:
            raise EdlError("audio.beat_times needs at least two beats")
        self.last = float(self.times[-1] - self.times[-2])
        self.first = float(self.times[1] - self.times[0])

    def time(self, position):
        position = float(position)
        top = len(self.times) - 1
        if position <= 0:
            return float(self.times[0] + position * self.first)
        if position >= top:
            return float(self.times[-1] + (position - top) * self.last)
        base = int(math.floor(position))
        return float(self.times[base] + (position - base) * (self.times[base + 1] - self.times[base]))


def effect_list(segment):
    items = [dict(item) for item in segment.get("fx", [])]
    head = {key: segment[key] for key in ("punch", "flash", "shake") if key in segment}
    if head:
        head["at"] = 0
        items.insert(0, head)
    return items


def number_or(value, key, default):
    if isinstance(value, dict):
        return float(value.get(key, default))
    return float(default)


def effect_tracks(segment, frames, fps, offset_of, height):
    punch = np.ones(frames)
    flash = np.zeros(frames)
    shake = np.zeros((frames, 3))
    marks = []
    for number, item in enumerate(effect_list(segment)):
        start = int(offset_of(float(item.get("at", 0.0))))
        if start >= frames or start < 0:
            continue
        marks.append(start)
        index = np.arange(start, frames)
        tau = (index - start) / fps
        if item.get("punch"):
            value = item["punch"]
            amount = float(value["amount"]) if isinstance(value, dict) else float(value)
            duration = number_or(value, "duration", 0.26)
            shape = value.get("shape", "decay") if isinstance(value, dict) else "decay"
            if shape == "bump":
                peak = max(1.5 / fps, duration / 5.0)
                gain = (tau / peak) * np.exp(1.0 - tau / peak)
            else:
                gain = np.exp(-tau / (duration / 3.0))
            punch[index] *= 1.0 + amount * gain
        if item.get("flash"):
            value = item["flash"]
            count = int(value["frames"]) if isinstance(value, dict) else int(value)
            strength = number_or(value, "strength", 0.8)
            for step in range(count):
                if start + step < frames:
                    flash[start + step] = max(flash[start + step], strength * (0.45**step))
        if item.get("shake"):
            value = item["shake"]
            amount = float(value["amount"]) if isinstance(value, dict) else float(value)
            duration = number_or(value, "duration", 0.36)
            rng = np.random.default_rng(stable_seed(segment["id"], number, "shake"))
            envelope = np.exp(-tau / (duration / 3.0))
            angle = rng.uniform(0.0, 2.0 * math.pi)
            for step, k in enumerate(index):
                angle += math.pi + rng.uniform(-0.9, 0.9)
                reach = amount * height * envelope[step]
                if reach < 0.25:
                    break
                shake[k, 0] += reach * math.cos(angle)
                shake[k, 1] += reach * math.sin(angle)
                shake[k, 2] += (1 if step % 2 == 0 else -1) * amount * 28.0 * envelope[step] * rng.uniform(0.5, 1.0)
    return punch, flash, shake, marks


def corner_offsets(width, height, target, rotation, flip):
    corners = np.array([[0.0, 0.0], [width, 0.0], [0.0, height], [width, height]])
    relative = corners[None, :, :] - target[:, None, :]
    radians = np.radians(-rotation)
    cos, sin = np.cos(radians)[:, None], np.sin(radians)[:, None]
    dx = relative[:, :, 0] * cos - relative[:, :, 1] * sin
    dy = relative[:, :, 0] * sin + relative[:, :, 1] * cos
    return dx * flip, dy


def required_scale(width, height, target, rotation, flip, anchor, active):
    dx, dy = corner_offsets(width, height, target, rotation, flip)
    x0, y0, x1, y1 = active
    right = np.maximum(x1 - anchor[:, 0], 1e-9)[:, None]
    left = np.maximum(anchor[:, 0] - x0, 1e-9)[:, None]
    down = np.maximum(y1 - anchor[:, 1], 1e-9)[:, None]
    up = np.maximum(anchor[:, 1] - y0, 1e-9)[:, None]
    need_x = np.where(dx > 0, dx / right, -dx / left)
    need_y = np.where(dy > 0, dy / down, -dy / up)
    need = np.maximum(need_x, need_y).max(axis=1)
    outside = (anchor[:, 0] <= x0) | (anchor[:, 0] >= x1) | (anchor[:, 1] <= y0) | (anchor[:, 1] >= y1)
    need[outside] = np.inf
    return need


def clamp_target(width, height, target, scale, rotation, flip, anchor, active):
    x0, y0, x1, y1 = active
    scale = max(scale, 1.0005 * max(width / (x1 - x0), height / (y1 - y0)))
    tx, ty = float(target[0]), float(target[1])
    for _ in range(60):
        left = anchor[0] - x0
        right = x1 - anchor[0]
        if flip < 0:
            left, right = right, left
        low_x, high_x = width - scale * right, scale * left
        low_y, high_y = height - scale * (y1 - anchor[1]), scale * (anchor[1] - y0)
        cx = min(max(tx, low_x), high_x) if low_x <= high_x else 0.5 * (low_x + high_x)
        cy = min(max(ty, low_y), high_y) if low_y <= high_y else 0.5 * (low_y + high_y)
        need = required_scale(width, height, np.array([[cx, cy]]), np.array([rotation]), flip, np.array([anchor]), active)[0]
        if need <= scale * 1.0002:
            return scale, cx, cy
        scale *= 1.01
    return scale, cx, cy


def build_matrices(scale, rotation, flip, anchor, target):
    radians = np.radians(rotation)
    cos, sin = np.cos(radians), np.sin(radians)
    a = scale * cos * flip
    b = -scale * sin
    c = scale * sin * flip
    d = scale * cos
    matrices = np.zeros((len(scale), 2, 3))
    matrices[:, 0, 0] = a
    matrices[:, 0, 1] = b
    matrices[:, 1, 0] = c
    matrices[:, 1, 1] = d
    matrices[:, 0, 2] = target[:, 0] - (a * anchor[:, 0] + b * anchor[:, 1])
    matrices[:, 1, 2] = target[:, 1] - (c * anchor[:, 0] + d * anchor[:, 1])
    return matrices


def visible_rect(matrix, width, height):
    linear = matrix[:, :2]
    inverse = np.linalg.inv(linear)
    corners = np.array([[0.0, 0.0], [width, 0.0], [0.0, height], [width, height]])
    points = (corners - matrix[:, 2]) @ inverse.T
    return [float(points[:, 0].min()), float(points[:, 1].min()), float(points[:, 0].max()), float(points[:, 1].max())]


class Segment:
    def __init__(self, spec, index):
        self.spec = spec
        self.index = index
        self.id = spec["id"]
        self.clip = spec.get("clip")
        self.issues = []
        self.info = {}
        self.positions = None
        self.matrices = None
        self.flash = None
        self.anchor_canvas = None
        self.anchor_size = None

    @property
    def frames(self):
        return self.frame_out - self.frame_in

    def issue(self, level, message):
        self.issues.append({"segment": self.id, "level": level, "message": message})


class Resolved:
    def __init__(self):
        self.segments = []
        self.issues = []
        self.fps = None
        self.width = 0
        self.height = 0
        self.total_frames = 0
        self.beat_zero = 0.0
        self.beats = None
        self.edl = None
        self.targets = {}

    def all_issues(self):
        out = list(self.issues)
        for segment in self.segments:
            out.extend(segment.issues)
        return out

    def errors(self):
        return [i for i in self.all_issues() if i["level"] == "error"]

    def segment_at(self, frame):
        for segment in self.segments:
            if segment.frame_in <= frame < segment.frame_out:
                return segment
        return None

    def cuts(self):
        return [s.frame_in for s in self.segments[1:]]


def choose_rate(project, edl, scans):
    declared = edl.get("timeline", {}).get("fps") or project.cfg.get("fps")
    if declared and declared != "auto":
        return parse_rate(declared)
    votes = Counter()
    for segment in edl["segments"]:
        clip = segment.get("clip")
        if clip and clip in scans:
            votes[scans[clip]["cadence"]["unique_rate"]] += float(segment.get("beats", 1.0))
    if not votes:
        return Fraction(24000, 1001)
    return parse_rate(votes.most_common(1)[0][0])


def resolve(project, edl=None, pool=None, finder=None, log=None):
    edl = edl or load_edl(project)
    own_pool = pool is None
    pool = pool or SourcePool()
    out = Resolved()
    out.edl = edl
    width, height = project.canvas
    out.width, out.height = width, height
    entries = {entry["id"]: entry for entry in clipscan.load_index(project)}
    overrides = edl.get("clips", {})
    scans = {}
    for spec in edl["segments"]:
        clip = spec.get("clip")
        if clip and clip not in scans:
            if clip not in entries:
                raise EdlError(f"segment {spec['id']} uses unknown clip {clip}")
            scans[clip] = clipscan.scan_clip(project, entries[clip], sheets=False, log=lambda *_: None)
    fps = choose_rate(project, edl, scans)
    out.fps = fps
    rate = float(fps)
    beats = BeatMap(edl["audio"]["beat_times"])
    out.beats = beats
    bias = float(edl.get("timeline", {}).get("cut_bias_frames", project.cfg["song"]["cut_bias_frames"]))
    segments = [Segment(spec, n) for n, spec in enumerate(edl["segments"])]
    out.segments = segments
    first_beat = next((n for n, s in enumerate(segments) if "beats" in s.spec), None)
    last_beat = max((n for n, s in enumerate(segments) if "beats" in s.spec), default=None)
    if first_beat is None:
        raise EdlError("the EDL has no beat-locked segments")
    for n, segment in enumerate(segments):
        if "seconds" in segment.spec and first_beat < n < last_beat:
            raise EdlError(f"segment {segment.id}: 'seconds' segments are only allowed before the first or after the last beat-locked segment")
    lead_frames = 0
    for segment in segments[:first_beat]:
        count = max(1, int(round(float(segment.spec["seconds"]) * rate)))
        segment.frame_in = lead_frames
        segment.frame_out = lead_frames + count
        segment.beat_in = segment.beat_out = None
        lead_frames += count
    zero = lead_frames / rate
    out.beat_zero = zero

    def beat_frame(position):
        return int(math.floor((zero + beats.time(position) - beats.time(0.0)) * rate + 0.5 + bias))

    position = 0.0
    cursor = lead_frames
    for segment in segments[first_beat : last_beat + 1]:
        segment.beat_in = position
        position += float(segment.spec["beats"])
        segment.beat_out = position
        segment.frame_in = cursor
        cursor = max(beat_frame(position), 0)
        segment.frame_out = cursor
        if segment.frame_out - segment.frame_in < 2:
            raise EdlError(f"segment {segment.id} is shorter than two frames")
    for segment in segments[last_beat + 1 :]:
        count = max(1, int(round(float(segment.spec["seconds"]) * rate)))
        segment.frame_in = cursor
        segment.frame_out = cursor + count
        segment.beat_in = segment.beat_out = None
        cursor += count
    out.total_frames = cursor
    for segment in segments:
        segment.time_in = segment.frame_in / rate
        segment.time_out = segment.frame_out / rate
        if segment.beat_in is not None:
            segment.ideal_in = zero + beats.time(segment.beat_in) - beats.time(0.0)
            segment.ideal_out = zero + beats.time(segment.beat_out) - beats.time(0.0)
        else:
            segment.ideal_in = segment.time_in
            segment.ideal_out = segment.time_out
    defaults = edl.get("defaults", {})
    targets = merge(project.cfg["targets"], edl.get("targets", {}))
    pending = {}
    for segment in segments:
        spec = segment.spec
        if not segment.clip:
            continue
        entry = entries[segment.clip]
        scan = scans[segment.clip]
        clip_settings = overrides.get(segment.clip, {})
        source = pool.get(clipscan.clip_path(project, entry))
        n = segment.frames
        segment.entry = entry
        segment.scan = scan
        segment.path = clipscan.clip_path(project, entry)
        source_rate = float(source.fps)
        ratio = source_rate / rate
        conform = spec.get("conform", clip_settings.get("conform", 0.9 <= ratio <= 1.3))
        step = 1.0 if conform else ratio
        speed = curve(spec.get("speed", defaults.get("speed")), 1.0)
        moved, total, moved_at = advance(speed, n)
        constant = isinstance(spec.get("speed", 1.0), (int, float)) and float(spec.get("speed", 1.0)) == 1.0
        if "sync" in spec:
            sync = spec["sync"]
            if "beat" in sync:
                frame_at = beat_frame(segment.beat_in + float(sync["beat"])) - segment.frame_in
                where = min(max(frame_at / float(n), 0.0), 1.0)
            else:
                where = float(sync.get("at", 0.0))
            event = source.position_at(float(sync["src"]))
            start = event - step * moved_at(where)
        elif "src_in" in spec:
            start = float(source.index_at(float(spec["src_in"])))
        else:
            raise EdlError(f"segment {segment.id} needs src_in or sync")
        if constant and abs(step - 1.0) < 1e-9:
            start = float(round(start))
        positions = start + step * moved
        segment.positions = positions
        segment.step = step
        segment.conform = bool(conform)
        segment.source_rate = source_rate
        segment.src_in = source.time_of(max(0.0, positions[0]))
        segment.src_out = source.time_of(min(source.count - 1, positions[-1] + step * float(speed(np.array([1.0]))[0])))
        low = int(math.floor(positions.min() + 1e-6))
        high = int(math.ceil(positions.max() - 1e-6))
        if low < 0 or high > source.count - 1:
            segment.issue("error", f"source range frames {low}-{high} is outside the clip (0-{source.count - 1})")
            positions = np.clip(positions, 0, source.count - 1)
            segment.positions = positions
            low, high = max(low, 0), min(high, source.count - 1)
        shot = clipscan.shot_at(scan, int(round(float(positions[0]))))
        bounds = (shot["start"], shot["end"])
        crossing = [c for c in scan["cuts"] if low < c <= high]
        if crossing:
            where = ", ".join(f"{source.times[c]:.3f}s" for c in crossing)
            level = "warn" if spec.get("allow_cut") else "error"
            segment.issue(level, f"source range {source.times[low]:.3f}-{source.times[high]:.3f}s crosses a detected cut at {where}")
            bounds = (low, high + 1)
        active = spec.get("active") or clip_settings.get("active") or shot["active"]
        segment.active = [float(v) for v in active]
        segment.bounds = bounds
        anchor_spec = spec.get("anchor") or {"type": "frame"}
        segment.anchor_type = anchor_spec.get("type", "frame")
        path = anchors.build_anchor(project, source, entry, [int(v) for v in active], anchor_spec, low, high, finder, bounds)
        values = path.at(positions)
        segment.anchor_values = values
        segment.info["anchor"] = path.source
        segment.info["anchor_coverage"] = round(path.coverage, 3)
        if path.coverage < 0.8:
            segment.issue("warn", f"anchor found in only {path.coverage * 100:.0f}% of frames")
        target_spec = spec.get("target", segment.anchor_type)
        if isinstance(target_spec, dict):
            name = f"@{segment.id}"
            targets[name] = merge(targets.get(segment.anchor_type, targets["frame"]), target_spec)
        else:
            name = target_spec
            if name not in targets:
                raise EdlError(f"segment {segment.id} uses unknown target {name}")
        segment.target_name = name
        x = (np.arange(n) / float(n)) if n > 1 else np.zeros(1)
        zoom_spec = spec.get("zoom")
        if zoom_spec:
            zoom = float(zoom_spec.get("from", 1.0)) + (float(zoom_spec.get("to", 1.0)) - float(zoom_spec.get("from", 1.0))) * ease(x * n / max(1, n - 1), zoom_spec.get("ease", "linear"))
        else:
            zoom = np.ones(n)

        def offset_of(at, segment=segment):
            if segment.beat_in is None:
                return int(round(at * rate))
            return beat_frame(segment.beat_in + at) - segment.frame_in

        punch, flash, shake, marks = effect_tracks(spec, n, rate, offset_of, height)
        nudge = spec.get("nudge", {})
        segment.flash = flash
        segment.fx_marks = marks
        segment.flip = -1.0 if spec.get("flip") else 1.0
        segment.gain = zoom * punch * float(nudge.get("scale", 1.0))
        segment.shake = shake
        segment.nudge = (float(nudge.get("dx", 0.0)), float(nudge.get("dy", 0.0)))
        segment.rotation = -values[:, 3] + shake[:, 2]
        segment.interp = spec.get("interp", defaults.get("interp", "auto"))
        pending.setdefault(name, []).append(segment)
    for name, members in pending.items():
        target = targets[name]
        if target.get("size") != "auto":
            continue
        needs = []
        for segment in members:
            base = np.tile(np.array([[target["x"] * width + segment.nudge[0], target["y"] * height + segment.nudge[1]]]), (segment.frames, 1))
            need = required_scale(width, height, base, -segment.anchor_values[:, 3], segment.flip, segment.anchor_values[:, :2], segment.active)
            size = need * segment.anchor_values[:, 2] / segment.gain
            finite = size[np.isfinite(size)]
            if len(finite) == 0:
                continue
            ends = [v for v in (size[0], size[-1]) if np.isfinite(v)]
            needs.append(max(ends + [float(np.median(finite))]))
        value = (max(needs) * 1.004 / height) if needs else targets["frame"]["size"]
        if target.get("max_size"):
            value = min(value, float(target["max_size"]))
        if target.get("min_size"):
            value = max(value, float(target["min_size"]))
        targets[name] = merge(target, {"size": round(float(value), 5), "auto": True})
    out.targets = targets
    limit = float(project.cfg["framing"]["max_boost"])
    for segment in segments:
        if not segment.clip:
            continue
        n = segment.frames
        target = targets[segment.target_name]
        values = segment.anchor_values
        anchor = values[:, :2]
        base = np.zeros((n, 2))
        base[:, 0] = target["x"] * width + segment.nudge[0] + segment.shake[:, 0]
        base[:, 1] = target["y"] * height + segment.nudge[1] + segment.shake[:, 1]
        wanted = float(target["size"]) * height / values[:, 2] * segment.gain
        need = required_scale(width, height, base, segment.rotation, segment.flip, anchor, segment.active)
        boost = np.maximum(1.0, need / wanted)
        capped = np.minimum(boost, limit)
        if n >= 5:
            spread = scipy.ndimage.gaussian_filter1d(scipy.ndimage.maximum_filter1d(capped, 9, mode="nearest"), 2.5, mode="nearest")
            ramp = np.minimum(1.0, np.minimum(np.arange(n), np.arange(n)[::-1]) / 6.0)
            capped = np.maximum(capped, capped + (spread - capped) * ramp)
        scale = wanted * capped
        final = base.copy()
        shift = np.zeros(n)
        for k in np.where(need > scale * 1.0002)[0]:
            fixed, cx, cy = clamp_target(width, height, base[k], float(scale[k]), float(segment.rotation[k]), segment.flip, anchor[k], segment.active)
            shift[k] = math.hypot(cx - base[k, 0], cy - base[k, 1])
            scale[k] = fixed
            final[k] = [cx, cy]
        segment.matrices = build_matrices(scale, segment.rotation, segment.flip, anchor, final)
        segment.scale = scale
        segment.anchor_canvas = final
        segment.anchor_size = values[:, 2] * scale
        segment.target_point = (target["x"] * width + segment.nudge[0], target["y"] * height + segment.nudge[1])
        segment.target_size = float(target["size"]) * height
        ratio_in = float(scale[0] / wanted[0])
        ratio_out = float(scale[-1] / wanted[-1])
        segment.info.update(
            {
                "boost_in": round(ratio_in, 4),
                "boost_out": round(ratio_out, 4),
                "boost_max": round(float((scale / wanted).max()), 4),
                "shift_in": round(float(shift[0]), 2),
                "shift_out": round(float(shift[-1]), 2),
                "shift_max": round(float(shift.max()), 2),
                "upscale": round(float(scale.max()), 3),
                "target": segment.target_name,
            }
        )
        if max(ratio_in, ratio_out) > 1.02:
            segment.issue("warn", f"subject is {max(ratio_in, ratio_out):.2f}x the target size at a cut (no room in the source frame for an exact size match)")
        if max(shift[0], shift[-1]) > 2.0:
            segment.issue("warn", f"anchor sits {max(shift[0], shift[-1]):.0f}px off its target at a cut (subject too close to the source frame edge)")
        if float(scale.max()) > float(project.cfg["framing"]["upscale_warn"]):
            segment.issue("warn", f"source is enlarged {float(scale.max()):.2f}x (soft picture)")
        if "duplicate_frames" in segment.scan["flags"]:
            segment.issue("warn", "clip has duplicated frames (run the prepare step)")
        for rect in overrides.get(segment.clip, {}).get("avoid", []) + [o for o in segment.scan.get("overlays", [])]:
            for k in (0, n // 2, n - 1):
                view = visible_rect(segment.matrices[k], width, height)
                if view[0] < rect[2] and rect[0] < view[2] and view[1] < rect[3] and rect[1] < view[3]:
                    segment.issue("warn", f"frame shows a region marked to avoid {rect}")
                    break
            else:
                continue
            break
    if own_pool:
        pool.close()
    return out


def describe_effects(spec):
    parts = []
    for item in effect_list(spec):
        at = float(item.get("at", 0.0))
        label = "" if at == 0 else f"@{at:g}b "
        bits = []
        for key in ("punch", "flash", "shake"):
            if item.get(key):
                value = item[key]
                amount = value.get("amount", value.get("frames")) if isinstance(value, dict) else value
                bits.append(f"{key} {amount:g}")
        if bits:
            parts.append(label + "+".join(bits))
    if isinstance(spec.get("speed"), list):
        low = min(v for _, v in spec["speed"])
        parts.append(f"speed ramp (min {low:g}x)")
    elif spec.get("speed") not in (None, 1, 1.0):
        parts.append(f"speed {spec['speed']:g}x")
    if spec.get("zoom"):
        parts.append(f"push {spec['zoom'].get('from', 1.0):g}->{spec['zoom'].get('to', 1.0):g}")
    if spec.get("flip"):
        parts.append("flip")
    if spec.get("audio"):
        parts.append("source audio")
    if spec.get("music") == "stop":
        parts.append("music stops")
    return ", ".join(parts) or "-"


def cut_list(resolved):
    rate = float(resolved.fps)
    rows = []
    for segment in resolved.segments:
        row = {
            "id": segment.id,
            "time_in": round(segment.time_in, 3),
            "time_out": round(segment.time_out, 3),
            "frame_in": segment.frame_in,
            "frame_out": segment.frame_out,
            "frames": segment.frames,
            "beat_in": segment.beat_in,
            "beats": segment.spec.get("beats"),
            "sync_error_ms": round((segment.time_in - segment.ideal_in) * 1000.0, 1),
            "clip": segment.clip or "black",
            "effects": describe_effects(segment.spec),
            "note": segment.spec.get("note", ""),
        }
        if segment.clip:
            row.update(
                {
                    "src_in": round(segment.src_in, 3),
                    "src_out": round(segment.src_out, 3),
                    "anchor": segment.anchor_type,
                    "target": segment.target_name,
                    **segment.info,
                }
            )
        rows.append(row)
    return {"fps": rate_text(resolved.fps), "frame_ms": round(1000.0 / rate, 2), "total_frames": resolved.total_frames, "duration": round(resolved.total_frames / rate, 3), "canvas": [resolved.width, resolved.height], "segments": rows}


def cut_list_markdown(data, issues):
    lines = [
        f"# Cut list ({data['canvas'][0]}x{data['canvas'][1]}, {data['fps']} fps, {data['duration']} s, {data['total_frames']} frames)",
        "",
        "| # | time | frames | beat | len | clip | source in-out | anchor | effects | sync |",
        "|---|------|--------|------|-----|------|---------------|--------|---------|------|",
    ]
    for number, row in enumerate(data["segments"]):
        beat = "-" if row["beat_in"] is None else f"{row['beat_in']:g}"
        length = f"{row['beats']:g}b" if row["beats"] is not None else f"{row['frames']}f"
        source = f"{row['src_in']:.2f}-{row['src_out']:.2f}" if "src_in" in row else "-"
        anchor = row.get("anchor", "-")
        lines.append(f"| {number + 1} | {row['time_in']:.3f} | {row['frame_in']}-{row['frame_out']} | {beat} | {length} | {row['clip']} | {source} | {anchor} | {row['effects']} | {row['sync_error_ms']:+.1f} ms |")
    if issues:
        lines += ["", "## Notes from the resolver", ""]
        for item in issues:
            lines.append(f"- {item['level']}: {item['segment']}: {item['message']}")
    return "\n".join(lines) + "\n"
