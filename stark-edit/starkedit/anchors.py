import atexit
import hashlib
import json
import math
import subprocess
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import scipy.signal

from .config import load_json, save_json
from .media import file_signature, slug

MODELS = {
    "landmarker": ("face_landmarker.task", "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"),
    "yunet": ("face_detection_yunet_2023mar.onnx", "https://huggingface.co/opencv/face_detection_yunet/resolve/main/face_detection_yunet_2023mar.onnx"),
}
FACE_VERSION = 2
RIGHT_EYE = (33, 133)
LEFT_EYE = (362, 263)
NOSE_TIP = 1


def ensure_model(project, name):
    filename, url = MODELS[name]
    target = project.models / filename
    if target.exists() and target.stat().st_size > 1000:
        return target
    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            data = response.read()
        target.write_bytes(data)
    except Exception:
        subprocess.run(["curl", "-sS", "-L", "--max-time", "120", "-o", str(target), url], check=True)
    if not target.exists() or target.stat().st_size < 1000:
        raise RuntimeError(f"could not download model {filename} from {url}")
    return target


class FaceFinder:
    def __init__(self, project):
        self.project = project
        self.landmarker = None
        self.detector = None
        self.image_class = None
        self.format = None

    def _load_landmarker(self):
        if self.landmarker is not None:
            return
        import mediapipe
        from mediapipe.tasks import python as tasks
        from mediapipe.tasks.python import vision

        options = vision.FaceLandmarkerOptions(
            base_options=tasks.BaseOptions(model_asset_path=str(ensure_model(self.project, "landmarker"))),
            num_faces=4,
            min_face_detection_confidence=0.35,
            min_face_presence_confidence=0.35,
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(options)
        self.image_class = mediapipe.Image
        self.format = mediapipe.ImageFormat.SRGB
        atexit.register(self.close)

    def close(self):
        if self.landmarker is not None:
            try:
                self.landmarker.close()
            except Exception:
                pass
            self.landmarker = None

    def _load_detector(self):
        if self.detector is None:
            self.detector = cv2.FaceDetectorYN.create(str(ensure_model(self.project, "yunet")), "", (640, 360), 0.55, 0.3, 50)

    def _landmarks(self, image):
        self._load_landmarker()
        rgb = np.ascontiguousarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        result = self.landmarker.detect(self.image_class(image_format=self.format, data=rgb))
        height, width = image.shape[:2]
        return [np.array([[p.x * width, p.y * height] for p in face], dtype=np.float64) for face in result.face_landmarks]

    def _boxes(self, image):
        self._load_detector()
        height, width = image.shape[:2]
        scale = min(1.0, 960.0 / max(height, width))
        small = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1.0 else image
        self.detector.setInputSize((small.shape[1], small.shape[0]))
        _, faces = self.detector.detect(small)
        if faces is None:
            return []
        return [(float(f[0] / scale), float(f[1] / scale), float(f[2] / scale), float(f[3] / scale)) for f in faces]

    @staticmethod
    def describe(points):
        right = 0.5 * (points[RIGHT_EYE[0]] + points[RIGHT_EYE[1]])
        left = 0.5 * (points[LEFT_EYE[0]] + points[LEFT_EYE[1]])
        middle = 0.5 * (right + left)
        delta = left - right
        size = float(np.hypot(delta[0], delta[1]))
        low = points.min(axis=0)
        high = points.max(axis=0)
        unit = delta / max(size, 1e-6)
        yaw = float((points[NOSE_TIP] - middle) @ unit) / max(size, 1e-6)
        return {
            "x": float(middle[0]),
            "y": float(middle[1]),
            "size": size,
            "angle": float(math.degrees(math.atan2(delta[1], delta[0]))),
            "yaw": yaw,
            "box": [float(low[0]), float(low[1]), float(high[0]), float(high[1])],
        }

    def find(self, image, active=None, thorough=False):
        x0, y0, x1, y1 = active if active is not None else (0, 0, image.shape[1], image.shape[0])
        crop = image[y0:y1, x0:x1]
        faces = []
        for points in self._landmarks(crop):
            points[:, 0] += x0
            points[:, 1] += y0
            faces.append(self.describe(points))
        if faces and not thorough:
            return faces
        for bx, by, bw, bh in self._boxes(crop):
            cx = bx + bw / 2.0 + x0
            cy = by + bh / 2.0 + y0
            if any(f["box"][0] <= cx <= f["box"][2] and f["box"][1] <= cy <= f["box"][3] for f in faces):
                continue
            half = 1.25 * max(bw, bh)
            rx0 = int(max(0, cx - half))
            ry0 = int(max(0, cy - half))
            rx1 = int(min(image.shape[1], cx + half))
            ry1 = int(min(image.shape[0], cy + half))
            region = image[ry0:ry1, rx0:rx1]
            if region.shape[0] < 24 or region.shape[1] < 24:
                continue
            factor = 384.0 / max(region.shape[:2])
            resized = cv2.resize(region, None, fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC if factor > 1 else cv2.INTER_AREA)
            found = self._landmarks(resized)
            if not found:
                continue
            points = found[0] / factor
            points[:, 0] += rx0
            points[:, 1] += ry0
            faces.append(self.describe(points))
        return faces


class FaceCache:
    def __init__(self, project, entry, path):
        self.file = project.subdir("tracks") / f"{slug(Path(entry['file']).stem)}-{file_signature(path)}-faces.json"
        data = load_json(self.file, {})
        self.frames = data.get("frames", {}) if data.get("version") == FACE_VERSION else {}
        self.dirty = False

    def get(self, index):
        return self.frames.get(str(int(index)))

    def put(self, index, faces):
        compact = [{k: (round(v, 3) if isinstance(v, float) else [round(c, 2) for c in v]) for k, v in face.items()} for face in faces]
        self.frames[str(int(index))] = compact
        self.dirty = True

    def save(self):
        if self.dirty:
            self.file.write_text(json.dumps({"version": FACE_VERSION, "frames": self.frames}))
            self.dirty = False


def detect_range(finder, cache, source, first, last, active):
    for index in range(int(first), int(last) + 1):
        if cache.get(index) is None:
            cache.put(index, finder.find(source.get(index), active))
    cache.save()


def _closeness(face, reference):
    distance = math.hypot(face["x"] - reference[0], face["y"] - reference[1]) / max(reference[2], 1.0)
    return distance + abs(math.log(max(face["size"], 1e-3) / max(reference[2], 1e-3)))


def select_faces(cache, first, last, hint=None):
    frames = np.arange(int(first), int(last) + 1)
    n = len(frames)
    values = np.full((n, 4), np.nan)
    start = None
    best = None
    for k, index in enumerate(frames):
        for face in cache.get(index) or []:
            if hint is not None:
                rank = -math.hypot(face["x"] - hint[0], face["y"] - hint[1]) / max(face["size"], 1.0)
            else:
                rank = face["size"]
            if best is None or rank > best[0]:
                best = (rank, k, face)
        if best is not None and hint is not None and k - best[1] > 12:
            break
    if best is None:
        return frames, values
    start = best[1]
    seed = best[2]
    values[start] = [seed["x"], seed["y"], seed["size"], seed["angle"]]
    for direction in (1, -1):
        reference = (seed["x"], seed["y"], seed["size"])
        k = start + direction
        misses = 0
        while 0 <= k < n:
            faces = cache.get(frames[k]) or []
            choice = min(faces, key=lambda f: _closeness(f, reference)) if faces else None
            limit = 0.7 + 0.15 * min(misses, 6)
            if choice is not None and _closeness(choice, reference) < limit:
                values[k] = [choice["x"], choice["y"], choice["size"], choice["angle"]]
                reference = (choice["x"], choice["y"], choice["size"])
                misses = 0
            else:
                misses += 1
            k += direction
    return frames, values


def fill_gaps(values):
    out = values.copy()
    n = len(out)
    index = np.arange(n)
    for column in range(out.shape[1]):
        good = ~np.isnan(out[:, column])
        if good.sum() == 0:
            continue
        out[:, column] = np.interp(index, index[good], out[good, column])
    return out


def smooth_path(values, sigma):
    out = fill_gaps(values)
    n = len(out)
    if n < 3 or sigma <= 0:
        return out
    log_size = np.log(np.maximum(out[:, 2], 1e-3))
    columns = [out[:, 0], out[:, 1], log_size, out[:, 3]]
    result = []
    for column in columns:
        median = scipy.signal.medfilt(np.pad(column, 2, mode="edge"), 5)[2:-2] if n >= 5 else column
        residual = column - median
        spread = 1.4826 * np.median(np.abs(residual)) + 1e-6
        cleaned = np.where(np.abs(residual) > 4.0 * spread + 0.5, median, column)
        window = int(4 * sigma + 1) | 1
        window = min(window, n if n % 2 == 1 else n - 1)
        if window >= 5:
            cleaned = scipy.signal.savgol_filter(cleaned, window, 2, mode="interp")
        elif window >= 3:
            cleaned = scipy.signal.savgol_filter(cleaned, window, 1, mode="interp")
        result.append(cleaned)
    out[:, 0] = result[0]
    out[:, 1] = result[1]
    out[:, 2] = np.exp(result[2])
    out[:, 3] = result[3]
    return out


def refine_blob(image, x, y, radius):
    reach = int(max(6, 2.2 * radius))
    cx, cy = int(round(x)), int(round(y))
    x0, y0 = max(0, cx - reach), max(0, cy - reach)
    x1, y1 = min(image.shape[1], cx + reach + 1), min(image.shape[0], cy + reach + 1)
    if x1 - x0 < 5 or y1 - y0 < 5:
        return None
    region = image[y0:y1, x0:x1].max(axis=2).astype(np.float32)
    region = cv2.GaussianBlur(region, (0, 0), 0.8)
    high = float(np.percentile(region, 99.5))
    low = float(np.median(region))
    if high - low < 25.0:
        return None
    threshold = low + 0.6 * (high - low)
    mask = (region >= threshold).astype(np.uint8)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    best = None
    for label in range(1, count):
        area = stats[label, cv2.CC_STAT_AREA]
        if area < 5:
            continue
        distance = math.hypot(centroids[label][0] + x0 - x, centroids[label][1] + y0 - y)
        if distance > 1.5 * radius + 3:
            continue
        if best is None or distance < best[0]:
            best = (distance, label, area)
    if best is None:
        return None
    weight = np.where(labels == best[1], region - threshold + 1.0, 0.0)
    total = float(weight.sum())
    ys, xs = np.mgrid[0 : region.shape[0], 0 : region.shape[1]]
    return float((weight * xs).sum() / total + x0), float((weight * ys).sum() / total + y0), float(2.0 * math.sqrt(best[2] / math.pi))


def refine_pair(image, x, y, size, angle):
    radians = math.radians(angle)
    ux, uy = math.cos(radians), math.sin(radians)
    first = refine_blob(image, x - 0.5 * size * ux, y - 0.5 * size * uy, 0.28 * size)
    second = refine_blob(image, x + 0.5 * size * ux, y + 0.5 * size * uy, 0.28 * size)
    if first is None or second is None:
        return None
    dx, dy = second[0] - first[0], second[1] - first[1]
    distance = math.hypot(dx, dy)
    if distance < 0.6 * size or distance > 1.6 * size:
        return None
    return 0.5 * (first[0] + second[0]), 0.5 * (first[1] + second[1]), distance, math.degrees(math.atan2(dy, dx))


def propagate(gray_a, gray_b, state):
    x, y, size, angle = state
    half = float(np.clip(1.5 * size, 24.0, 280.0))
    x0, y0 = int(max(0, x - half)), int(max(0, y - half))
    x1, y1 = int(min(gray_a.shape[1], x + half)), int(min(gray_a.shape[0], y + half))
    if x1 - x0 < 12 or y1 - y0 < 12:
        return state, False
    corners = cv2.goodFeaturesToTrack(gray_a[y0:y1, x0:x1], 160, 0.01, 4)
    if corners is None or len(corners) < 6:
        return state, False
    points = corners.reshape(-1, 2).astype(np.float32) + np.float32([x0, y0])
    settings = {"winSize": (21, 21), "maxLevel": 3, "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)}
    forward, ok_forward, _ = cv2.calcOpticalFlowPyrLK(gray_a, gray_b, points, None, **settings)
    backward, ok_backward, _ = cv2.calcOpticalFlowPyrLK(gray_b, gray_a, forward, None, **settings)
    drift = np.linalg.norm(backward - points, axis=1)
    good = (ok_forward.ravel() == 1) & (ok_backward.ravel() == 1) & (drift < 1.0)
    if good.sum() < 6:
        return state, False
    matrix, inliers = cv2.estimateAffinePartial2D(points[good], forward[good], method=cv2.RANSAC, ransacReprojThreshold=2.0)
    if matrix is None or inliers is None or int(inliers.sum()) < 5:
        return state, False
    scale = math.hypot(matrix[0, 0], matrix[1, 0])
    turn = math.degrees(math.atan2(matrix[1, 0], matrix[0, 0]))
    nx = matrix[0, 0] * x + matrix[0, 1] * y + matrix[0, 2]
    ny = matrix[1, 0] * x + matrix[1, 1] * y + matrix[1, 2]
    if not (0.8 < scale < 1.25):
        return state, False
    return (float(nx), float(ny), float(size * scale), float(angle + turn)), True


def normalize_key(key, source):
    if "f" in key:
        frame = float(key["f"])
    else:
        frame = source.position_at(float(key["t"]))
    if "p1" in key and "p2" in key:
        ax, ay = key["p1"]
        bx, by = key["p2"]
        x, y = 0.5 * (ax + bx), 0.5 * (ay + by)
        size = math.hypot(bx - ax, by - ay)
        angle = math.degrees(math.atan2(by - ay, bx - ax))
    else:
        x, y = float(key["x"]), float(key["y"])
        size = float(key["size"])
        angle = float(key.get("angle", 0.0))
    return int(round(frame)), (x, y, size, angle)


def interpolate_keys(keys, frames):
    order = sorted(keys, key=lambda k: k[0])
    positions = np.array([k[0] for k in order], dtype=np.float64)
    table = np.array([k[1] for k in order], dtype=np.float64)
    out = np.zeros((len(frames), 4))
    for column in range(4):
        series = np.log(np.maximum(table[:, column], 1e-3)) if column == 2 else table[:, column]
        values = np.interp(frames, positions, series)
        out[:, column] = np.exp(values) if column == 2 else values
    return out


def track_keys(source, keys, first, last):
    frames = np.arange(int(first), int(last) + 1)
    base = interpolate_keys(keys, frames)
    order = sorted(keys, key=lambda k: k[0])
    grays = {}

    def gray(index):
        if index not in grays:
            grays[index] = cv2.cvtColor(source.get(index), cv2.COLOR_BGR2GRAY)
            if len(grays) > 6:
                grays.pop(next(iter(grays)))
        return grays[index]

    def run(start_frame, state, stop_frame):
        path = {start_frame: state}
        step = 1 if stop_frame >= start_frame else -1
        frame = start_frame
        healthy = True
        while frame != stop_frame:
            following = frame + step
            if following < 0 or following >= source.count:
                break
            if healthy:
                state, healthy = propagate(gray(frame), gray(following), state)
            if not healthy:
                return path, False
            path[following] = state
            frame = following
        return path, True

    result = base.copy()
    tracked = np.zeros(len(frames), dtype=bool)

    def write(frame, state):
        if frame < first or frame > last:
            return
        result[frame - int(first)] = state
        tracked[frame - int(first)] = True

    head_frame, head_state = order[0]
    if head_frame > first:
        path, ok = run(head_frame, head_state, int(first))
        if ok:
            for frame, state in path.items():
                write(frame, state)
    tail_frame, tail_state = order[-1]
    if tail_frame < last:
        path, ok = run(tail_frame, tail_state, int(last))
        if ok:
            for frame, state in path.items():
                write(frame, state)
    for (frame_a, state_a), (frame_b, state_b) in zip(order[:-1], order[1:]):
        if frame_b <= frame_a:
            continue
        ahead, ok_a = run(frame_a, state_a, frame_b)
        behind, ok_b = run(frame_b, state_b, frame_a)
        if not (ok_a and ok_b):
            continue
        for frame in range(frame_a, frame_b + 1):
            u = (frame - frame_a) / float(frame_b - frame_a)
            a = ahead[frame]
            b = behind[frame]
            size = math.exp(math.log(a[2]) * (1.0 - u) + math.log(b[2]) * u)
            write(frame, (a[0] * (1 - u) + b[0] * u, a[1] * (1 - u) + b[1] * u, size, a[3] * (1 - u) + b[3] * u))
    return frames, result, tracked


class AnchorPath:
    def __init__(self, frames, values, source="static", coverage=1.0):
        self.frames = np.asarray(frames, dtype=np.float64)
        self.values = np.asarray(values, dtype=np.float64)
        self.source = source
        self.coverage = float(coverage)

    def at(self, positions):
        positions = np.atleast_1d(np.asarray(positions, dtype=np.float64))
        if len(self.frames) == 1:
            return np.repeat(self.values[:1], len(positions), axis=0)
        out = np.zeros((len(positions), 4))
        for column in range(4):
            series = np.log(np.maximum(self.values[:, column], 1e-3)) if column == 2 else self.values[:, column]
            sampled = np.interp(positions, self.frames, series)
            out[:, column] = np.exp(sampled) if column == 2 else sampled
        return out


def spec_digest(spec, first, last):
    payload = json.dumps({"spec": spec, "first": int(first), "last": int(last)}, sort_keys=True)
    return hashlib.sha1(payload.encode()).hexdigest()[:12]


def build_anchor(project, source, entry, active, spec, first, last, finder=None, bounds=None):
    spec = spec or {}
    kind = spec.get("type", "frame")
    pad = int(project.cfg["framing"]["track_pad_frames"])
    low = int(max(0, first - pad))
    high = int(min(source.count - 1, last + pad))
    if bounds is not None:
        low = max(low, int(bounds[0]))
        high = min(high, int(bounds[1]) - 1)
    low = min(low, int(first))
    high = max(high, int(last))
    sigma = float(spec.get("smooth", project.cfg["framing"]["track_sigma"]))
    keys = [normalize_key(key, source) for key in spec.get("keys", [])]
    info = "static"
    coverage = 1.0
    if keys:
        frames = np.arange(low, high + 1)
        if spec.get("track", False) and high > low:
            cache_file = project.subdir("tracks") / f"{slug(Path(entry['file']).stem)}-{file_signature(source.path)}-{spec_digest(spec, low, high)}.json"
            cached = load_json(cache_file)
            if cached is not None:
                values = np.array(cached["values"])
                coverage = cached["coverage"]
            else:
                _, values, tracked = track_keys(source, keys, low, high)
                coverage = float(tracked.mean())
                save_json(cache_file, {"values": [[round(float(v), 3) for v in row] for row in values], "coverage": coverage})
            info = "tracked"
        else:
            values = interpolate_keys(keys, frames)
            info = "keys"
        refine = spec.get("refine")
        if refine in ("blob", "pair"):
            values = values.copy()
            base = values.copy()
            seed = int(np.clip(min(k[0] for k in keys) - low, 0, len(frames) - 1))
            hits = 0
            for order in (range(seed, len(frames)), range(seed - 1, -1, -1)):
                shift = np.zeros(2)
                ratio = 1.0
                turn = 0.0
                for k in order:
                    image = source.get(int(frames[k]))
                    x = base[k, 0] + shift[0]
                    y = base[k, 1] + shift[1]
                    size = base[k, 2] * ratio
                    angle = base[k, 3] + turn
                    if refine == "pair":
                        found = refine_pair(image, x, y, size, angle)
                        if found is not None:
                            values[k] = found
                            ratio = found[2] / base[k, 2]
                            turn = found[3] - base[k, 3]
                    else:
                        found = refine_blob(image, x, y, float(spec.get("radius", 0.5 * size)))
                        if found is not None:
                            values[k, 0], values[k, 1] = found[0], found[1]
                    if found is None:
                        values[k] = [x, y, size, angle]
                    else:
                        shift = np.array([values[k, 0] - base[k, 0], values[k, 1] - base[k, 1]])
                        hits += 1
            coverage = hits / float(len(frames))
            info = f"{info}+{refine}"
        if info != "keys" and len(frames) >= 3:
            values = smooth_path(values, sigma)
    elif kind == "eyes" or spec.get("detect"):
        if finder is None:
            finder = FaceFinder(project)
        cache = FaceCache(project, entry, source.path)
        detect_range(finder, cache, source, low, high, active)
        hint = spec.get("hint")
        frames, raw = select_faces(cache, low, high, hint)
        coverage = float((~np.isnan(raw[:, 0])).mean())
        if coverage == 0.0:
            raise RuntimeError(f"no face found in {entry['id']} frames {low}-{high}; give the anchor manual keys")
        values = smooth_path(raw, sigma)
        info = "faces"
    else:
        x0, y0, x1, y1 = active
        frames = np.array([low])
        values = np.array([[0.5 * (x0 + x1), 0.5 * (y0 + y1), float(y1 - y0), 0.0]])
    offset = spec.get("offset")
    if offset:
        values = values.copy()
        values[:, 0] += float(offset[0]) * values[:, 2]
        values[:, 1] += float(offset[1]) * values[:, 2]
    if spec.get("scale"):
        values = values.copy()
        values[:, 2] *= float(spec["scale"])
    if not spec.get("rotate", False):
        values = values.copy()
        values[:, 3] = 0.0
    return AnchorPath(frames, values, info, coverage)
