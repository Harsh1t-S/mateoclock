from pathlib import Path

import cv2
import numpy as np

from . import draw
from .config import load_json, rate_text, save_json, snap_rate
from .media import VIDEO_EXTENSIONS, FrameSource, file_signature, frame_to_bgr, list_media, probe_clip, slug

SCAN_VERSION = 5


def load_index(project, create=True):
    path = project.clips_index_path
    data = load_json(path)
    files = list_media(project.clips_dir, VIDEO_EXTENSIONS)
    if data is None:
        if not create:
            return []
        data = {"clips": []}
    known = {entry["file"] for entry in data["clips"]}
    used = {entry["id"] for entry in data["clips"]}
    changed = False
    number = len(data["clips"])
    for file in files:
        relative = project.relative(file)
        if relative in known:
            continue
        number += 1
        clip_id = f"c{number:02d}"
        while clip_id in used:
            number += 1
            clip_id = f"c{number:02d}"
        used.add(clip_id)
        data["clips"].append({"id": clip_id, "file": relative, "label": "", "notes": ""})
        changed = True
    if changed and create:
        save_json(path, data)
    return data["clips"]


def find_entry(project, clip_id):
    for entry in load_index(project):
        if entry["id"] == clip_id:
            return entry
    raise KeyError(f"unknown clip id {clip_id}")


def clip_path(project, entry):
    return (project.root / entry["file"]).resolve()


def scan_paths(project, entry):
    path = clip_path(project, entry)
    stem = f"{slug(Path(entry['file']).stem)}-{file_signature(path)}"
    folder = project.subdir("scan")
    return folder / f"{stem}.json", folder / f"{stem}.npz"


def load_scan(project, entry):
    summary, _ = scan_paths(project, entry)
    data = load_json(summary)
    if data is None or data.get("version") != SCAN_VERSION:
        return None
    return data


def _moving_mean_excluding(values, radius):
    n = len(values)
    padded = np.concatenate([np.zeros(radius), values, np.zeros(radius)])
    counts = np.concatenate([np.zeros(radius), np.ones(n), np.zeros(radius)])
    kernel = np.ones(2 * radius + 1)
    sums = np.convolve(padded, kernel, mode="valid") - values
    weights = np.convolve(counts, kernel, mode="valid") - 1.0
    return sums / np.maximum(weights, 1.0)


def detect_cuts(content, hist, lag, luma, luma_std, settings):
    n = len(content)
    if n < 3:
        return [], []
    neighbours = _moving_mean_excluding(content, 2)
    ratio = content / (neighbours + 0.5)
    floor = float(settings["cut_min_content"])
    adaptive = (ratio >= settings["cut_ratio"]) & (content >= floor)
    faint = (ratio >= 2.0 * settings["cut_ratio"]) & (content >= 0.5 * floor)
    hard = (content >= settings["cut_hard_content"]) & (hist >= 0.45)
    candidate = adaptive | faint | hard
    candidate[0] = False
    cuts = []
    flashes = []
    i = 1
    while i < n:
        if candidate[i]:
            j = i
            while j + 1 < n and candidate[j + 1]:
                j += 1
            best = i + int(np.argmax(content[i : j + 1]))
            after = min(n - 1, j + 2)
            back = after - 3
            transient = back >= 0 and back < i and lag[after] < 0.4 * content[best] and lag[after] < floor
            if transient:
                flashes.append(best)
            else:
                cuts.append(best)
            i = j + 1
        else:
            i += 1
    black = (luma < 14.0) & (luma_std < 7.0)
    fades = []
    i = 0
    while i < n:
        if black[i]:
            j = i
            while j + 1 < n and black[j + 1]:
                j += 1
            if i > 0:
                fades.append(i)
            if j + 1 < n:
                fades.append(j + 1)
            i = j + 1
        else:
            i += 1
    merged = sorted(set(cuts) | set(fades))
    return merged, sorted(flashes)


def cross_check(path, cuts):
    try:
        import scenedetect
    except Exception:
        return cuts, None
    try:
        scenes = scenedetect.detect(str(path), scenedetect.AdaptiveDetector(min_scene_len=3))
    except Exception:
        return cuts, None
    extra = []
    starts = [int(scene[0].frame_num) for scene in scenes[1:]]
    for start in starts:
        if not any(abs(start - c) <= 1 for c in cuts):
            extra.append(start)
    return sorted(set(cuts) | set(extra)), extra


def measure_bars(rows, cols, threshold):
    def edges(profile):
        lit = np.where(profile >= threshold)[0]
        if len(lit) == 0:
            return None
        return int(lit[0]), int(lit[-1]) + 1

    vertical = edges(rows)
    horizontal = edges(cols)
    if vertical is None or horizontal is None:
        return None
    return [horizontal[0], vertical[0], horizontal[1], vertical[1]]


def cadence(pixel_diff, cut_set, fps):
    n = len(pixel_diff)
    if n < 40:
        return {"dup_ratio": 0.0, "unique_rate": rate_text(fps), "considered": 0}
    values = pixel_diff.copy()
    for c in cut_set:
        if 0 <= c < n:
            values[c] = np.nan
    window = 11
    half = window // 2
    padded = np.pad(values, half, mode="edge")
    stacked = np.lib.stride_tricks.sliding_window_view(padded, window)
    local = np.nanmedian(stacked, axis=1)
    local_max = np.nanpercentile(stacked, 80, axis=1)
    moving = (local_max > 0.6) & ~np.isnan(values)
    duplicate = moving & (values < 0.25 * local_max) & (values < 0.45)
    considered = int(moving.sum())
    ratio = float(duplicate.sum() / considered) if considered else 0.0
    unique = fps
    if ratio > 0.03:
        known = min((0.04, 0.2, 0.5, 0.6), key=lambda r: abs(r - ratio))
        unique = snap_rate(float(fps) * (1.0 - (known if abs(known - ratio) < 0.05 else ratio)))
    return {"dup_ratio": round(ratio, 4), "unique_rate": rate_text(unique), "considered": considered, "median_motion": round(float(np.nanmedian(local)), 3)}


def scan_clip(project, entry, force=False, sheets=True, log=print):
    settings = project.cfg["scan"]
    summary_path, arrays_path = scan_paths(project, entry)
    if not force:
        cached = load_scan(project, entry)
        if cached is not None and arrays_path.exists():
            return cached
    path = clip_path(project, entry)
    info = probe_clip(path)
    source = FrameSource(path)
    n = source.count
    aw = int(settings["analysis_width"])
    ah = max(2, int(round(aw * source.height / source.width / 2.0)) * 2)
    step = int(settings["profile_step"])
    content = np.zeros(n, np.float32)
    lag = np.zeros(n, np.float32)
    history = []
    hist = np.zeros(n, np.float32)
    pixel = np.zeros(n, np.float32)
    luma = np.zeros(n, np.float32)
    luma_std = np.zeros(n, np.float32)
    sharp = np.zeros(n, np.float32)
    profile_index = []
    row_profiles = []
    col_profiles = []
    edge_hits = np.zeros((ah, aw), np.float32)
    edge_frames = 0
    previous_hsv = None
    previous_gray = None
    previous_hist = None
    seen = 0
    for position, frame in source.sequence():
        small = frame_to_bgr(frame, aw, ah)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV).astype(np.int16)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        histogram = cv2.calcHist([gray], [0], None, [64], [0, 256]).ravel()
        histogram /= max(1.0, histogram.sum())
        if previous_hsv is not None:
            delta = np.abs(hsv - previous_hsv)
            content[position] = float(delta.mean())
            hist[position] = float(0.5 * np.abs(histogram - previous_hist).sum())
            pixel[position] = float(np.abs(gray.astype(np.int16) - previous_gray).mean())
        if len(history) == 3:
            lag[position] = float(np.abs(hsv - history[0]).mean())
            history.pop(0)
        history.append(hsv)
        luma[position] = float(gray.mean())
        luma_std[position] = float(gray.std())
        sharp[position] = float(cv2.Laplacian(gray, cv2.CV_32F).var())
        previous_hsv = hsv
        previous_gray = gray.astype(np.int16)
        previous_hist = histogram
        if position % step == 0:
            full = cv2.cvtColor(frame_to_bgr(frame), cv2.COLOR_BGR2GRAY)
            row_profiles.append(np.percentile(full[:, ::6], 98, axis=1).astype(np.float32))
            col_profiles.append(np.percentile(full[::6, :], 98, axis=0).astype(np.float32))
            profile_index.append(position)
            edges = cv2.Canny(gray, 60, 150)
            edge_hits += (edges > 0).astype(np.float32)
            edge_frames += 1
        seen += 1
    cuts, flashes = detect_cuts(content, hist, lag, luma, luma_std, settings)
    cuts, extra = cross_check(path, cuts) if settings.get("cross_check", True) else (cuts, None)
    bounds = [0] + cuts + [n]
    profile_index = np.array(profile_index, dtype=np.int64)
    rows = np.array(row_profiles) if row_profiles else np.zeros((0, source.height), np.float32)
    cols = np.array(col_profiles) if col_profiles else np.zeros((0, source.width), np.float32)
    threshold = float(settings["bar_threshold"])
    shots = []
    for s in range(len(bounds) - 1):
        i0, i1 = int(bounds[s]), int(bounds[s + 1])
        if i1 <= i0:
            continue
        inside = (profile_index >= i0) & (profile_index < i1)
        bars = None
        reliable = False
        if inside.any():
            row_max = rows[inside].max(axis=0)
            col_max = cols[inside].max(axis=0)
            bars = measure_bars(row_max, col_max, threshold)
            reliable = bool(row_max.max() > 70.0)
        inner = slice(i0 + 1, i1) if i1 - i0 > 1 else slice(i0, i1)
        shots.append(
            {
                "index": len(shots),
                "start": i0,
                "end": i1,
                "t0": round(float(source.times[i0]), 4),
                "t1": round(float(source.times[i1 - 1] + 1.0 / float(source.fps)), 4),
                "frames": i1 - i0,
                "sharp": round(float(np.median(sharp[i0:i1])), 2),
                "luma": round(float(np.median(luma[i0:i1])), 2),
                "motion": round(float(np.median(content[inner])), 3),
                "bars": bars,
                "bars_reliable": reliable,
            }
        )
    reliable_bars = np.array([s["bars"] for s in shots if s["bars"] is not None and s["bars_reliable"]], dtype=np.float64)
    if len(reliable_bars) == 0:
        reliable_bars = np.array([s["bars"] for s in shots if s["bars"] is not None], dtype=np.float64)
    inset = int(settings["bar_inset"])
    if len(reliable_bars):
        outer = [float(np.percentile(reliable_bars[:, 0], 10)), float(np.percentile(reliable_bars[:, 1], 10)), float(np.percentile(reliable_bars[:, 2], 90)), float(np.percentile(reliable_bars[:, 3], 90))]
        spread = float(np.max(np.percentile(reliable_bars, 90, axis=0) - np.percentile(reliable_bars, 10, axis=0)))
    else:
        outer = [0.0, 0.0, float(source.width), float(source.height)]
        spread = 0.0

    def inset_rect(rect):
        x0, y0, x1, y1 = rect
        x0 = int(round(x0)) + (inset if x0 > 0.5 else 0)
        y0 = int(round(y0)) + (inset if y0 > 0.5 else 0)
        x1 = int(round(x1)) - (inset if x1 < source.width - 0.5 else 0)
        y1 = int(round(y1)) - (inset if y1 < source.height - 0.5 else 0)
        return [x0, y0, x1, y1]

    active = inset_rect(outer)
    varying = spread > 8.0
    for shot in shots:
        if varying and shot["bars"] is not None and shot["bars_reliable"]:
            shot["active"] = inset_rect(shot["bars"])
        else:
            shot["active"] = active
    edge_ratio = edge_hits / max(1, edge_frames)
    static = (edge_ratio > 0.8).astype(np.uint8)
    sx = source.width / aw
    sy = source.height / ah
    margin = 3
    for rect in {tuple(s["active"]) for s in shots}:
        for y in (rect[1], rect[3]):
            r = int(round(y / sy))
            static[max(0, r - margin) : min(ah, r + margin + 1)] = 0
        for x in (rect[0], rect[2]):
            c = int(round(x / sx))
            static[:, max(0, c - margin) : min(aw, c + margin + 1)] = 0
    overlays = []
    if len(shots) >= 3 and edge_frames >= 12:
        dilated = cv2.dilate(static, np.ones((5, 5), np.uint8))
        count, labels, stats, _ = cv2.connectedComponentsWithStats(dilated, 8)
        for label in range(1, count):
            x, y, w, h, area = stats[label]
            if int(static[labels == label].sum()) < 12:
                continue
            overlays.append([int(x * sx), int(y * sy), int((x + w) * sx), int((y + h) * sy)])
    beat = cadence(pixel, set(cuts), source.fps)
    flags = []
    if info.get("hdr"):
        flags.append("hdr")
    if info.get("field_order") not in (None, "progressive", "unknown"):
        flags.append("interlaced")
    if info.get("sar") not in (None, "1", "1/1"):
        flags.append("non_square_pixels")
    if not source.constant_rate:
        flags.append("variable_frame_rate")
    if beat["dup_ratio"] > 0.03:
        flags.append("duplicate_frames")
    if overlays:
        flags.append("static_overlay")
    result = {
        "version": SCAN_VERSION,
        "id": entry["id"],
        "file": entry["file"],
        "info": info,
        "width": source.width,
        "height": source.height,
        "frames": n,
        "fps": rate_text(source.fps),
        "duration": round(source.duration, 4),
        "start_time": round(source.start_time, 6),
        "active": active,
        "bars_vary": bool(varying),
        "cuts": [int(c) for c in cuts],
        "flashes": [int(f) for f in flashes],
        "cross_check_extra": extra,
        "shots": shots,
        "cadence": beat,
        "overlays": overlays,
        "flags": flags,
    }
    np.savez_compressed(arrays_path, content=content, hist=hist, pixel=pixel, luma=luma, luma_std=luma_std, sharp=sharp)
    save_json(summary_path, result)
    if sheets:
        make_sheets(project, entry, result, source)
    source.close()
    log(f"scanned {entry['id']}: {n} frames, {len(shots)} shots, active {active}, {rate_text(source.fps)} fps, flags {flags or '-'}")
    return result


def make_sheets(project, entry, scan, source=None, columns=5, rows=6, tile_width=384):
    own = source is None
    if own:
        source = FrameSource(clip_path(project, entry))
    folder = project.subdir("sheets")
    for old in folder.glob(f"{entry['id']}_*.jpg"):
        old.unlink()
    tiles = []
    for shot in scan["shots"]:
        middle = (shot["start"] + shot["end"] - 1) // 2
        tile = draw.fit(source.get(middle), width=tile_width).copy()
        draw.text(tile, f"#{shot['index']}  {shot['t0']:.2f}-{shot['t1']:.2f}s  {shot['frames']}f", (5, 16), 0.45)
        draw.text(tile, f"sharp {shot['sharp']:.0f}  luma {shot['luma']:.0f}", (5, tile.shape[0] - 7), 0.4, draw.CYAN)
        tiles.append(tile)
    per = columns * rows
    paths = []
    for page in range(0, max(1, len(tiles)), per):
        image = draw.sheet(tiles[page : page + per], columns)
        header = np.full((26, image.shape[1], 3), 18, np.uint8)
        draw.text(header, f"{entry['id']}  {Path(entry['file']).name}  {scan['width']}x{scan['height']} {scan['fps']}fps  {scan['duration']:.1f}s  active {scan['active']}  flags {','.join(scan['flags']) or '-'}", (6, 18), 0.5, draw.YELLOW, background=False)
        paths.append(draw.save(folder / f"{entry['id']}_{page // per + 1:02d}.jpg", np.vstack([header, image]), 88))
    if own:
        source.close()
    return paths


def shot_at(scan, frame_index):
    for shot in scan["shots"]:
        if shot["start"] <= frame_index < shot["end"]:
            return shot
    return scan["shots"][-1] if scan["shots"] else None


def export_frames(project, entry, times, grid=True, crop=None, zoom=1.0, step=None, width=None, name=None):
    source = FrameSource(clip_path(project, entry))
    folder = project.subdir("frames")
    outputs = []
    for seconds in times:
        index = source.index_at(seconds)
        image = source.get(index)
        origin = (0.0, 0.0)
        if crop is not None:
            x0, y0, x1, y1 = [int(v) for v in crop]
            x0, y0 = max(0, x0), max(0, y0)
            x1, y1 = min(source.width, x1), min(source.height, y1)
            image = image[y0:y1, x0:x1]
            origin = (float(x0), float(y0))
        scale = float(zoom)
        if width is not None:
            scale = float(width) / image.shape[1]
        if abs(scale - 1.0) > 1e-6:
            image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA)
        else:
            image = image.copy()
        if grid:
            spacing = step or _grid_step(image.shape[1] / scale)
            image = draw.coordinate_grid(image, origin, scale, spacing)
        draw.text(image, f"{entry['id']} t={source.times[index]:.3f}s frame {index}", (6, image.shape[0] - 24), 0.5, draw.YELLOW)
        stem = name or f"{entry['id']}_{source.times[index]:08.3f}"
        if len(times) > 1 and name:
            stem = f"{name}_{len(outputs):02d}"
        outputs.append(draw.save(folder / f"{stem}.jpg", image, 90))
    source.close()
    return outputs


def _grid_step(span):
    for candidate in (10, 20, 25, 50, 100, 200):
        if span / candidate <= 20:
            return candidate
    return 200


def export_strip(project, entry, start, end, count=8, crop=None, tile_width=384, name=None):
    source = FrameSource(clip_path(project, entry))
    i0 = source.index_at(start)
    i1 = max(i0, source.index_at(end))
    picks = sorted(set(int(round(v)) for v in np.linspace(i0, i1, count)))
    tiles = []
    for index in picks:
        image = source.get(index)
        if crop is not None:
            x0, y0, x1, y1 = [int(v) for v in crop]
            image = image[max(0, y0) : y1, max(0, x0) : x1]
        tile = draw.fit(image, width=tile_width).copy()
        draw.text(tile, f"{source.times[index]:.3f}s  f{index}", (5, 16), 0.45)
        tiles.append(tile)
    source.close()
    columns = min(len(tiles), 4)
    path = project.subdir("frames") / f"{name or entry['id'] + '_strip_' + format(start, '07.2f')}.jpg"
    return draw.save(path, draw.sheet(tiles, columns), 88)
