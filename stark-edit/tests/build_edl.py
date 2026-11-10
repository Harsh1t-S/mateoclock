import json
from pathlib import Path

import numpy as np

from starkedit import clipscan, song
from starkedit.config import save_json


def name_clips(project, truth):
    entries = clipscan.load_index(project)
    by_file = {clip["file"]: key for key, clip in truth["clips"].items()}
    for entry in entries:
        entry["id"] = by_file.get(entry["file"], entry["id"])
    save_json(project.clips_index_path, {"clips": entries})
    return {entry["id"]: entry for entry in entries}


def shot_bounds(truth_clip, shot):
    frames = [n for n, item in enumerate(truth_clip["frames"]) if item["shot"] == shot]
    return frames[0], frames[-1]


def keyed(truth_clip, shot, kind, rng, noise, refine=None, track=False):
    first, last = shot_bounds(truth_clip, shot)
    keys = []
    for frame in (first, (first + last) // 2, last):
        x, y, size = truth_clip["frames"][frame][kind]
        keys.append({"f": frame, "x": round(x + float(rng.uniform(-noise, noise)), 2), "y": round(y + float(rng.uniform(-noise, noise)), 2), "size": round(size * float(rng.uniform(0.97, 1.03)), 2)})
    spec = {"type": "eyes" if kind == "eyes" else "reactor", "keys": keys}
    if refine:
        spec["refine"] = refine
    if track:
        spec["track"] = True
    return spec


def time_of(truth_clip, shot, offset_frames):
    from fractions import Fraction

    first, last = shot_bounds(truth_clip, shot)
    offset = min(offset_frames, max(0, last - first - 30))
    return round((first + offset) / float(Fraction(truth_clip["rate"])), 4)


def make_edl(project, truth, seed=11):
    rng = np.random.default_rng(seed)
    analysis = song.analyze(project)
    section = analysis["section"]
    clips = truth["clips"]
    segments = []

    def eyes(clip, shot, offset=4, noise=5.0):
        return {"clip": clip, "src_in": time_of(clips[clip], shot, offset), "anchor": keyed(clips[clip], shot, "eyes", rng, noise, "pair"), "target": "eyes"}

    def reactor(clip, shot, offset=4, track=False):
        spec = keyed(clips[clip], shot, "reactor", rng, 1.5 if track else 4.0, None if track else "blob", track)
        return {"clip": clip, "src_in": time_of(clips[clip], shot, offset), "anchor": spec, "target": "reactor"}

    segments.append({"id": "open", "beats": section["lead_beats"], **eyes("scope", 0, 6), "zoom": {"from": 1.0, "to": 1.08}, "audio": {"gain_db": 0.0, "fade_out": 0.05}, "note": "cold open with source audio"})
    order = [("scope", 1), ("pal", 0), ("flat", 0), ("ntsc", 0), ("scope", 2), ("pal", 1), ("flat", 1), ("ntsc", 1)]
    for n, (clip, shot) in enumerate(order):
        item = {"id": f"match{n + 1}", "beats": 1, **eyes(clip, shot, 6 + 3 * n)}
        if n == 0:
            item.update({"punch": 0.12, "flash": 2, "shake": 0.012, "note": "drop"})
        elif n % 2 == 0:
            item["punch"] = {"amount": 0.06, "shape": "bump"}
        segments.append(item)
    if "portrait" in clips:
        segments.append({"id": "face1", "beats": 2, "clip": "portrait", "src_in": time_of(clips["portrait"], 0, 4), "anchor": {"type": "eyes"}, "target": "portrait"})
        segments.append({"id": "face2", "beats": 2, "clip": "portrait", "src_in": time_of(clips["portrait"], 1, 4), "anchor": {"type": "eyes"}, "target": "portrait", "punch": 0.08})
    else:
        segments.append({"id": "face1", "beats": 2, **eyes("scope", 3)})
        segments.append({"id": "face2", "beats": 2, **eyes("pal", 2)})
    run = [("flat", 2, False), ("pal", 2, True), ("scope", 3, False), ("ntsc", 2, False), ("sixty", 0, False)]
    for n, (clip, shot, track) in enumerate(run):
        item = {"id": f"suit{n + 1}", "beats": 2, **reactor(clip, shot, 5, track)}
        if n in (0, 3):
            item["punch"] = 0.1
        if clip == "sixty":
            item["conform"] = False
        segments.append(item)
    sixty_rate = 60.0
    first, _ = shot_bounds(clips["sixty"], 1)
    segments.append(
        {
            "id": "hero1",
            "beats": 4,
            "clip": "sixty",
            "conform": False,
            "sync": {"src": round((first + 110) / sixty_rate, 4), "beat": 2},
            "speed": [[0.0, 1.0], [0.2, 1.0], [0.45, 0.3], [0.5, 0.3], [0.52, 1.6], [0.7, 1.0], [1.0, 1.0]],
            "interp": "flow",
            "anchor": keyed(clips["sixty"], 1, "eyes", rng, 5.0, "pair"),
            "target": "eyes",
            "fx": [{"at": 0, "punch": 0.08}, {"at": 2, "punch": 0.14, "flash": 2, "shake": 0.014}],
            "note": "speed ramp into beat 2",
        }
    )
    segments.append(
        {
            "id": "hero2",
            "beats": 4,
            **reactor("telecine", 1, 6),
            "zoom": {"from": 1.0, "to": 1.1, "ease": "out"},
            "fx": [{"at": 0, "punch": 0.1, "flash": 1}, {"at": 1, "punch": 0.05}, {"at": 2, "punch": 0.05}, {"at": 3, "punch": 0.05}],
        }
    )
    burst = [("scope", 4), ("pal", 3), ("flat", 3), ("ntsc", 0), ("scope", 0), ("pal", 0), ("flat", 0), ("scope", 2)]
    for n, (clip, shot) in enumerate(burst):
        segments.append({"id": f"burst{n + 1}", "beats": 0.5, **eyes(clip, shot, 30 + n)})
    used = sum(float(s["beats"]) for s in segments)
    total = len(section["beat_times"]) - 1
    fill = [("flat", 2), ("scope", 4), ("pal", 3), ("ntsc", 2), ("flat", 3), ("scope", 3), ("pal", 2), ("flat", 1), ("scope", 1)]
    n = 0
    while used + 2 <= total:
        clip, shot = fill[n % len(fill)]
        item = {"id": f"tail{n + 1}", "beats": 2, **(reactor(clip, shot, 28) if n % 2 else eyes(clip, shot, 34))}
        if n == 0:
            item["flip"] = True
        segments.append(item)
        used += 2
        n += 1
    if used < total:
        segments.append({"id": "last", "beats": total - used, **eyes("scope", 0, 40)})
    segments.append({"id": "black", "seconds": project.cfg["length"]["tail_seconds"], "clip": None, "music": "stop", "note": "cut to black, music stops"})
    edl = {
        "version": 1,
        "title": "synthetic pipeline check",
        "timeline": {"fps": "auto"},
        "audio": {
            "song": analysis["file"],
            "song_start": section["song_start"],
            "beat_times": section["beat_times"],
            "lead_beats": section["lead_beats"],
            "gain_db": 0.0,
            "duck": [{"from_beat": 0, "to_beat": section["lead_beats"] - 0.25, "gain_db": -9.0}],
        },
        "targets": {
            "eyes": {"x": 0.5, "y": 0.4, "size": "auto"},
            "reactor": {"x": 0.5, "y": 0.52, "size": "auto"},
            "portrait": {"x": 0.5, "y": 0.3, "size": "auto"},
        },
        "segments": segments,
    }
    return edl


def write_edl(project, truth):
    name_clips(project, truth)
    edl = make_edl(project, truth)
    save_json(project.edl_path, edl)
    return edl


if __name__ == "__main__":
    import sys

    from starkedit.config import Project

    root = Path(sys.argv[1])
    write_edl(Project(root), json.loads((root / "truth.json").read_text()))
