import math
from pathlib import Path

import cv2
import numpy as np

from . import anchors, draw, song
from .config import save_json
from .media import FrameSource, extract_wav, probe_clip, read_wav


def scan_output(path, keep):
    source = FrameSource(path, cache_bytes=64 * 1024 * 1024)
    n = source.count
    stats = {
        "luma": np.zeros(n),
        "bright": np.zeros(n),
        "diff": np.zeros(n),
        "hist": np.zeros(n),
        "bars": np.zeros((n, 4)),
    }
    kept = {}
    previous = None
    previous_hist = None
    from .media import frame_to_bgr

    for position, frame in source.sequence():
        image = frame_to_bgr(frame)
        if position in keep:
            kept[position] = image
        small = cv2.resize(image, (135, int(round(135.0 * image.shape[0] / image.shape[1]))), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        stats["luma"][position] = float(gray.mean())
        stats["bright"][position] = float(np.percentile(gray, 99))
        histogram = cv2.calcHist([gray], [0], None, [48], [0, 256]).ravel()
        histogram /= max(1.0, histogram.sum())
        if previous is not None:
            stats["diff"][position] = float(np.abs(gray.astype(np.int16) - previous).mean())
            stats["hist"][position] = float(0.5 * np.abs(histogram - previous_hist).sum())
        previous = gray.astype(np.int16)
        previous_hist = histogram
        rows = gray.max(axis=1)
        cols = gray.max(axis=0)
        lit_rows = np.where(rows > 18)[0]
        lit_cols = np.where(cols > 18)[0]
        if len(lit_rows) and len(lit_cols):
            stats["bars"][position] = [lit_rows[0] / len(rows), 1.0 - (lit_rows[-1] + 1) / len(rows), lit_cols[0] / len(cols), 1.0 - (lit_cols[-1] + 1) / len(cols)]
        else:
            stats["bars"][position] = [1.0, 1.0, 1.0, 1.0]
    info = {"frames": n, "fps": source.fps, "times": source.times + source.start_time, "width": source.width, "height": source.height}
    source.close()
    return stats, kept, info


def audio_alignment(project, path, resolved):
    wav = extract_wav(path, project.subdir("qc") / "output_audio.wav", song.ANALYSIS_RATE, 1, aligned=True)
    data, rate = read_wav(wav)
    y = data[:, 0]
    rise = song.attack_envelope(y, rate)
    zero = resolved.beat_zero
    beats = resolved.beats
    origin = beats.time(0.0)
    count = len(beats.times)
    expected = np.array([zero + beats.time(p) - origin for p in range(count)])
    music_out = resolved.total_frames / float(resolved.fps)
    for segment in resolved.segments:
        if segment.spec.get("music") == "stop":
            music_out = segment.time_in
            break
    measured, strength = song.locate_attacks(rise, expected)
    clear = (strength > 0.25) & (expected < music_out - 0.05) & (~np.isnan(measured))
    offsets = (measured - expected)[clear]
    if len(offsets) == 0:
        return {"offset_ms": None, "spread_ms": None, "clear_beats": 0, "beat_time": lambda p: zero + beats.time(p) - origin, "rms_after_stop": None}
    offset = float(np.median(offsets))
    stop_index = int(music_out * rate)
    after = y[stop_index + int(0.02 * rate) :]
    before = y[max(0, stop_index - int(0.5 * rate)) : stop_index]
    silence = float(np.sqrt(np.mean(after.astype(np.float64) ** 2))) if len(after) else 0.0
    level = float(np.sqrt(np.mean(before.astype(np.float64) ** 2))) if len(before) else 0.0
    return {
        "offset_ms": round(offset * 1000.0, 2),
        "spread_ms": round(float(np.std(offsets)) * 1000.0, 2),
        "clear_beats": int(clear.sum()),
        "beat_time": lambda p: zero + beats.time(p) - origin + offset,
        "rms_after_stop": round(silence, 6),
        "rms_before_stop": round(level, 6),
    }


def measure_anchor(finder, image, segment, expected, size, scale):
    spec = segment.spec.get("anchor") or {}
    kind = segment.anchor_type
    if kind == "eyes" and not spec.get("keys"):
        faces = finder.find(image)
        if not faces:
            return None
        face = min(faces, key=lambda f: math.hypot(f["x"] - expected[0], f["y"] - expected[1]))
        if math.hypot(face["x"] - expected[0], face["y"] - expected[1]) > 1.2 * size:
            return None
        return face["x"], face["y"], face["size"], "face"
    if spec.get("refine") == "pair":
        found = anchors.refine_pair(image, expected[0], expected[1], size, 0.0)
        return None if found is None else (found[0], found[1], found[2], "pair")
    if spec.get("refine") == "blob":
        found = anchors.refine_blob(image, expected[0], expected[1], 0.5 * size)
        return None if found is None else (found[0], found[1], None, "blob")
    return None


def motion_vector(first, second):
    a = cv2.cvtColor(cv2.resize(first, (180, 320), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    b = cv2.cvtColor(cv2.resize(second, (180, 320), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 3, 21, 3, 5, 1.1, 0)
    return float(np.median(flow[..., 0])) / 180.0, float(np.median(flow[..., 1])) / 320.0


def pair_image(first, second, a, b, scale, title, lines):
    width = 360
    factor = width / first.shape[1]
    left = draw.fit(first, width=width).copy()
    right = draw.fit(second, width=width).copy()
    blend = draw.onion(draw.fit(first, width=width), draw.fit(second, width=width))
    if a is not None:
        draw.guides(left, a[0] * factor, a[1] * factor, draw.YELLOW)
    if b is not None:
        draw.guides(right, b[0] * factor, b[1] * factor, draw.CYAN)
    draw.text(left, "out (last frame)", (6, left.shape[0] - 8), 0.45, draw.YELLOW)
    draw.text(right, "in (first frame)", (6, right.shape[0] - 8), 0.45, draw.CYAN)
    draw.text(blend, "both overlaid", (6, blend.shape[0] - 8), 0.45)
    top = draw.hstack([left, right, blend])
    focus = b if b is not None else a
    if focus is None:
        focus = (first.shape[1] / 2.0, first.shape[0] * 0.4, first.shape[1] * 0.2)
    half = int(max(60 * scale, min(first.shape[1] / 2.0, 1.9 * (focus[2] if len(focus) > 2 and focus[2] else first.shape[1] * 0.2))))
    cx = int(min(max(focus[0], half), first.shape[1] - half))
    cy = int(min(max(focus[1], half), first.shape[0] - half))
    zoom_a = cv2.resize(first[cy - half : cy + half, cx - half : cx + half], (width, width), interpolation=cv2.INTER_CUBIC)
    zoom_b = cv2.resize(second[cy - half : cy + half, cx - half : cx + half], (width, width), interpolation=cv2.INTER_CUBIC)
    edges = draw.edge_overlay(zoom_a, zoom_b)
    ratio = width / (2.0 * half)
    if a is not None:
        draw.guides(zoom_a, (a[0] - cx + half) * ratio, (a[1] - cy + half) * ratio, draw.YELLOW)
    if b is not None:
        draw.guides(zoom_b, (b[0] - cx + half) * ratio, (b[1] - cy + half) * ratio, draw.CYAN)
    draw.text(edges, "edges: out=red in=cyan", (6, width - 8), 0.42)
    bottom = draw.hstack([zoom_a, zoom_b, edges])
    header = np.full((22 + 18 * len(lines), top.shape[1], 3), 18, np.uint8)
    draw.text(header, title, (6, 16), 0.5, draw.YELLOW, background=False)
    for n, line in enumerate(lines):
        draw.text(header, line, (6, 34 + 18 * n), 0.43, draw.WHITE, background=False)
    return draw.vstack([header, top, bottom])


def run(project, resolved, video, finder=None, images=True, log=print):
    folder = project.subdir("qc")
    for old in folder.glob("cut_*.jpg"):
        old.unlink()
    for old in folder.glob("cuts_sheet_*.jpg"):
        old.unlink()
    for old in folder.glob("review_*.jpg"):
        old.unlink()
    rate = float(resolved.fps)
    cuts = [s.frame_in for s in resolved.segments[1:]]
    keep = set()
    for cut in cuts:
        keep.update({cut - 2, cut - 1, cut, cut + 1})
    for segment in resolved.segments:
        keep.update({segment.frame_in, (segment.frame_in + segment.frame_out - 1) // 2, segment.frame_out - 1})
    stats, kept, info = scan_output(video, keep)
    probe = probe_clip(video)
    scale = info["width"] / float(resolved.width)
    finder = finder or anchors.FaceFinder(project)
    report = {"video": project.relative(video), "issues": [], "cuts": []}
    spec_ok = True
    expected_width = int(round(resolved.width * scale / 2.0)) * 2
    if info["frames"] != resolved.total_frames:
        report["issues"].append({"level": "error", "check": "length", "message": f"file has {info['frames']} frames, the edit list resolves to {resolved.total_frames}"})
        spec_ok = False
    if abs(float(info["fps"]) - rate) > 1e-6:
        report["issues"].append({"level": "error", "check": "frame rate", "message": f"file is {info['fps']} fps, expected {resolved.fps}"})
    deltas = np.diff(info["times"])
    if len(deltas) and float(np.max(np.abs(deltas - 1.0 / rate))) > 0.002:
        report["issues"].append({"level": "error", "check": "frame rate", "message": "frame timestamps are not evenly spaced"})
    report["file"] = {
        "width": info["width"],
        "height": info["height"],
        "frames": info["frames"],
        "fps": str(info["fps"]),
        "duration": round(info["frames"] / rate, 3),
        "video_codec": probe.get("codec"),
        "audio_codec": probe.get("audio_codec"),
        "pix_fmt": probe.get("pix_fmt"),
    }
    sync = audio_alignment(project, video, resolved)
    report["audio"] = {k: v for k, v in sync.items() if k != "beat_time"}
    if sync["offset_ms"] is not None and abs(sync["offset_ms"]) > 12.0:
        report["issues"].append({"level": "warn", "check": "audio", "message": f"music in the file sits {sync['offset_ms']:+.1f} ms from where the edit list places it"})
    if sync.get("rms_after_stop") is not None and sync["rms_before_stop"] > 0 and sync["rms_after_stop"] > 0.02 * sync["rms_before_stop"]:
        report["issues"].append({"level": "warn", "check": "audio", "message": "audio is not silent after the music stop"})
    frame_ms = 1000.0 / rate
    tiles = []
    for number, cut in enumerate(cuts):
        before = resolved.segments[number]
        after = resolved.segments[number + 1]
        row = {"cut": number + 1, "frame": cut, "out": before.id, "in": after.id, "beat": after.beat_in}
        video_time = float(info["times"][cut]) if cut < len(info["times"]) else cut / rate
        row["time"] = round(video_time, 4)
        if after.beat_in is not None:
            beat_time = sync["beat_time"](after.beat_in)
            error = (video_time - beat_time) * 1000.0
            row["beat_time"] = round(beat_time, 4)
            row["sync_ms"] = round(error, 1)
            row["sync_frames"] = round(error / frame_ms, 2)
            if abs(error) > frame_ms:
                report["issues"].append({"level": "error", "check": "beat sync", "message": f"cut {number + 1} ({before.id}>{after.id}) is {error:+.0f} ms from its beat (more than one frame)"})
        lo = max(1, cut - 3)
        hi = min(info["frames"] - 1, cut + 3)
        window = stats["diff"][lo : hi + 1]
        seen = lo + int(np.argmax(window))
        around = np.delete(window, cut - lo) if lo <= cut <= hi else window
        clear = stats["diff"][cut] > 2.5 * (np.median(around) + 0.3)
        row["visible_at"] = int(seen)
        row["clear"] = bool(clear)
        if clear and seen != cut and stats["diff"][seen] > 1.5 * stats["diff"][cut]:
            report["issues"].append({"level": "error", "check": "cut frame", "message": f"cut {number + 1} shows on frame {seen}, expected {cut}"})
        first = kept.get(cut - 1)
        second = kept.get(cut)
        a_point = b_point = None
        lines = []
        if first is not None and second is not None:
            if before.clip:
                a_expect = before.anchor_canvas[-1] * scale
                a_size = float(before.anchor_size[-1]) * scale
                a_point = (a_expect[0], a_expect[1], a_size)
            if after.clip:
                b_expect = after.anchor_canvas[0] * scale
                b_size = float(after.anchor_size[0]) * scale
                b_point = (b_expect[0], b_expect[1], b_size)
            matched = before.clip and after.clip and before.target_name == after.target_name
            row["match_cut"] = bool(matched)
            if before.clip and after.clip:
                found_a = measure_anchor(finder, first, before, a_expect, a_size, scale)
                found_b = measure_anchor(finder, second, after, b_expect, b_size, scale)
                if found_a is not None:
                    row["out_off_target_px"] = round(math.hypot(found_a[0] - a_expect[0], found_a[1] - a_expect[1]) / scale, 1)
                if found_b is not None:
                    row["in_off_target_px"] = round(math.hypot(found_b[0] - b_expect[0], found_b[1] - b_expect[1]) / scale, 1)
                if matched and found_a is not None and found_b is not None:
                    drift = math.hypot(found_b[0] - found_a[0], found_b[1] - found_a[1]) / scale
                    row["drift_px"] = round(drift, 1)
                    planned = b_size / a_size
                    row["size_planned"] = round(planned, 3)
                    if found_a[2] and found_b[2]:
                        row["size_ratio"] = round(found_b[2] / found_a[2], 3)
                    limit = max(6.0, 0.06 * before.target_size)
                    if drift > limit:
                        report["issues"].append({"level": "warn", "check": "match cut", "message": f"cut {number + 1} ({before.id}>{after.id}): anchor jumps {drift:.0f}px across the cut"})
                    if row.get("size_ratio") and abs(math.log(row["size_ratio"] / planned)) > 0.05:
                        report["issues"].append({"level": "warn", "check": "match cut", "message": f"cut {number + 1} ({before.id}>{after.id}): subject size changes {row['size_ratio']:.2f}x across the cut, the edit list plans {planned:.2f}x"})
                elif matched:
                    row["drift_px"] = None
                first_prev = kept.get(cut - 2)
                second_next = kept.get(cut + 1)
                if first_prev is not None and second_next is not None:
                    va = motion_vector(first_prev, first)
                    vb = motion_vector(second, second_next)
                    ma, mb = math.hypot(*va), math.hypot(*vb)
                    row["motion_out"] = [round(va[0] * 100, 2), round(va[1] * 100, 2)]
                    row["motion_in"] = [round(vb[0] * 100, 2), round(vb[1] * 100, 2)]
                    if ma > 0.006 and mb > 0.006:
                        cosine = (va[0] * vb[0] + va[1] * vb[1]) / (ma * mb)
                        row["motion_agreement"] = round(cosine, 2)
            if images:
                beat_text = "-" if after.beat_in is None else f"{after.beat_in:g}"
                title = f"cut {number + 1}: {before.id} > {after.id}   t={video_time:.3f}s  frame {cut}  beat {beat_text}  sync {row.get('sync_ms', 0):+.1f} ms"
                if "drift_px" in row and row["drift_px"] is not None:
                    lines.append(f"measured anchor jump {row['drift_px']} px, size ratio {row.get('size_ratio', '-')} (planned {row.get('size_planned', '-')})")
                elif row.get("match_cut"):
                    lines.append("match cut: check alignment by eye (no automatic measure for this anchor)")
                if "motion_agreement" in row:
                    lines.append(f"motion out {row['motion_out']} in {row['motion_in']} agreement {row['motion_agreement']}")
                image = pair_image(first, second, a_point, b_point, scale, title, lines)
                draw.save(folder / f"cut_{number + 1:02d}.jpg", image, 88)
                tiles.append(draw.fit(image, width=540))
        report["cuts"].append(row)
    if images:
        for page in range(0, len(tiles), 6):
            draw.save(folder / f"cuts_sheet_{page // 6 + 1:02d}.jpg", draw.sheet(tiles[page : page + 6], 3), 85)
    for segment in resolved.segments:
        lo, hi = segment.frame_in, min(segment.frame_out, info["frames"])
        if hi <= lo:
            continue
        luma = stats["luma"][lo:hi]
        bright = stats["bright"][lo:hi]
        dark = (luma < 10.0) & (bright < 32.0)
        if segment.clip:
            if dark.any():
                where = lo + int(np.argmax(dark))
                report["issues"].append({"level": "error" if dark.mean() > 0.2 else "warn", "check": "black frames", "message": f"{segment.id}: {int(dark.sum())} black frame(s) starting at frame {where}"})
            bars = stats["bars"][lo:hi][~dark]
            if len(bars):
                typical = np.median(bars, axis=0)
                for name, value in zip(("top", "bottom", "left", "right"), typical):
                    if value > 0.012:
                        report["issues"].append({"level": "warn", "check": "letterbox", "message": f"{segment.id}: dark band on the {name} edge ({value * 100:.1f}% of the frame)"})
            diff = stats["diff"][lo:hi].copy()
            hist = stats["hist"][lo:hi]
            skip = {0}
            for mark in getattr(segment, "fx_marks", []):
                skip.update({mark, mark + 1, mark + 2, mark + 3})
            flashes = np.where(segment.flash > 0)[0]
            for f in flashes:
                skip.update({int(f), int(f) + 1, int(f) + 2})
            base = np.median(diff[1:]) if len(diff) > 2 else 0.0
            for k in range(1, len(diff)):
                if k in skip:
                    continue
                near = np.concatenate([diff[max(1, k - 3) : k], diff[k + 1 : k + 4]])
                local = float(np.median(near)) if len(near) else base
                if diff[k] > 12.0 and diff[k] > 5.0 * (local + 0.5) and hist[k] > 0.12:
                    report["issues"].append({"level": "error", "check": "stray cut", "message": f"{segment.id}: picture jumps at frame {lo + k} ({(lo + k) / rate:.3f}s) inside the segment"})
                    break
            if segment.anchor_type == "eyes" and not (segment.spec.get("anchor") or {}).get("keys"):
                for position in (segment.frame_in, (segment.frame_in + segment.frame_out - 1) // 2, segment.frame_out - 1):
                    image = kept.get(position)
                    if image is None:
                        continue
                    faces = finder.find(image)
                    if not faces:
                        continue
                    k = min(position - segment.frame_in, segment.frames - 1)
                    aim = segment.anchor_canvas[k] * scale
                    face = min(faces, key=lambda f: math.hypot(f["x"] - aim[0], f["y"] - aim[1]))
                    x0, y0, x1, y1 = face["box"]
                    edges = []
                    if x0 < 2:
                        edges.append("left")
                    if x1 > image.shape[1] - 2:
                        edges.append("right")
                    if y0 < 2:
                        edges.append("top")
                    if y1 > image.shape[0] - 2:
                        edges.append("bottom")
                    if edges:
                        report["issues"].append({"level": "warn", "check": "face crop", "message": f"{segment.id}: face runs off the {'/'.join(edges)} edge at frame {position}"})
                        break
        else:
            if (~dark).any():
                report["issues"].append({"level": "warn", "check": "black frames", "message": f"{segment.id}: expected black but {int((~dark).sum())} frame(s) are not"})
    if images:
        rows = []
        for segment in resolved.segments:
            picks = [segment.frame_in, (segment.frame_in + segment.frame_out - 1) // 2, segment.frame_out - 1]
            shots = []
            for position in picks:
                image = kept.get(position)
                if image is None:
                    continue
                tile = draw.fit(image, height=300).copy()
                draw.text(tile, f"{segment.id} f{position}", (4, 14), 0.4)
                shots.append(tile)
            if shots:
                rows.append(draw.hstack(shots, gap=2))
        for page in range(0, len(rows), 9):
            draw.save(folder / f"review_{page // 9 + 1:02d}.jpg", draw.sheet(rows[page : page + 9], 3, gap=10), 85)
    sync_values = [abs(c["sync_ms"]) for c in report["cuts"] if "sync_ms" in c]
    drifts = [c["drift_px"] for c in report["cuts"] if c.get("drift_px") is not None]
    report["summary"] = {
        "cuts": len(cuts),
        "frame_ms": round(frame_ms, 2),
        "max_sync_ms": round(max(sync_values), 1) if sync_values else None,
        "mean_sync_ms": round(float(np.mean(sync_values)), 1) if sync_values else None,
        "match_cuts": int(sum(1 for c in report["cuts"] if c.get("match_cut"))),
        "match_cuts_measured": len(drifts),
        "max_drift_px": max(drifts) if drifts else None,
        "errors": sum(1 for i in report["issues"] if i["level"] == "error"),
        "warnings": sum(1 for i in report["issues"] if i["level"] == "warn"),
        "spec_ok": spec_ok,
    }
    save_json(folder / "report.json", report)
    (folder / "report.md").write_text(markdown(report))
    if log:
        s = report["summary"]
        log(f"  qc: {s['cuts']} cuts, max sync {s['max_sync_ms']} ms (frame = {s['frame_ms']} ms), {s['match_cuts_measured']}/{s['match_cuts']} match cuts measured, max drift {s['max_drift_px']} px, {s['errors']} errors, {s['warnings']} warnings")
    return report


def markdown(report):
    f = report["file"]
    s = report["summary"]
    lines = [
        f"# QC report: {report['video']}",
        "",
        f"- file: {f['width']}x{f['height']}, {f['fps']} fps, {f['frames']} frames, {f['duration']} s, {f['video_codec']} + {f['audio_codec']}",
        f"- cuts: {s['cuts']}, largest distance from a beat {s['max_sync_ms']} ms (one frame = {s['frame_ms']} ms)",
        f"- music offset in the file: {report['audio'].get('offset_ms')} ms over {report['audio'].get('clear_beats')} beats",
        f"- match cuts: {s['match_cuts']} ({s['match_cuts_measured']} measured automatically, largest anchor jump {s['max_drift_px']} px)",
        f"- errors: {s['errors']}, warnings: {s['warnings']}",
        "",
        "| cut | out > in | time | frame | beat | sync ms | jump px | size | motion |",
        "|-----|----------|------|-------|------|---------|---------|------|--------|",
    ]
    for c in report["cuts"]:
        beat = "-" if c.get("beat") is None else f"{c['beat']:g}"
        sync = "-" if "sync_ms" not in c else f"{c['sync_ms']:+.1f}"
        jump = "-" if c.get("drift_px") is None else f"{c['drift_px']}"
        if c.get("match_cut") and c.get("drift_px") is None:
            jump = "by eye"
        size = "-" if "size_ratio" not in c else f"{c['size_ratio']} (plan {c.get('size_planned', '-')})"
        motion = "-" if "motion_agreement" not in c else f"{c['motion_agreement']:+.2f}"
        lines.append(f"| {c['cut']} | {c['out']} > {c['in']} | {c['time']:.3f} | {c['frame']} | {beat} | {sync} | {jump} | {size} | {motion} |")
    if report["issues"]:
        lines += ["", "## Issues", ""]
        for item in report["issues"]:
            lines.append(f"- {item['level']}: {item['check']}: {item['message']}")
    return "\n".join(lines) + "\n"


def pairs_from_renderer(project, resolved, renderer, only=None):
    folder = project.subdir("pairs")
    for old in folder.glob("pair_*.jpg"):
        old.unlink()
    scale = renderer.width / float(resolved.width)
    outputs = []
    tiles = []
    for number, after in enumerate(resolved.segments[1:]):
        before = resolved.segments[number]
        if only and before.id not in only and after.id not in only:
            continue
        if not before.clip and not after.clip:
            continue
        cut = after.frame_in
        first = renderer.frame(cut - 1)
        second = renderer.frame(cut)
        a_point = b_point = None
        if before.clip:
            a_point = (before.anchor_canvas[-1][0] * scale, before.anchor_canvas[-1][1] * scale, float(before.anchor_size[-1]) * scale)
        if after.clip:
            b_point = (after.anchor_canvas[0][0] * scale, after.anchor_canvas[0][1] * scale, float(after.anchor_size[0]) * scale)
        beat_text = "-" if after.beat_in is None else f"{after.beat_in:g}"
        title = f"cut {number + 1}: {before.id} > {after.id}   frame {cut}  beat {beat_text}"
        image = pair_image(first, second, a_point, b_point, scale, title, [])
        outputs.append(draw.save(folder / f"pair_{number + 1:02d}.jpg", image, 88))
        tiles.append(draw.fit(image, width=540))
    for page in range(0, len(tiles), 6):
        draw.save(folder / f"pairs_sheet_{page // 6 + 1:02d}.jpg", draw.sheet(tiles[page : page + 6], 3), 85)
    return outputs
