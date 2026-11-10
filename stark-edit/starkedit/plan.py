from . import song
from .config import save_json


def skeleton(project, analysis=None):
    analysis = analysis or song.analyze(project)
    section = analysis["section"]
    hits = set(section["biggest_hits"][:6])
    segments = []
    for number, slot in enumerate(section["grid"]):
        beat = slot["beat"]
        note = f"bar {slot['bar'] + 1}, {slot['level']} energy" if slot["bar"] >= 0 else "cold open"
        if int(round(beat)) in hits and abs(beat - round(beat)) < 1e-6:
            note += ", big hit"
        item = {"id": "open" if number == 0 else f"s{number:02d}", "beats": slot["beats"], "clip": "TODO", "src_in": 0.0, "anchor": {"type": "frame"}, "note": note}
        if number == 1:
            item.update({"punch": 0.12, "flash": 2, "shake": 0.012})
        segments.append(item)
    segments.append({"id": "black", "seconds": project.cfg["length"]["tail_seconds"], "clip": None, "music": "stop", "note": "cut to black, music stops dead"})
    return {
        "version": 1,
        "title": "Tony Stark x Iron Man",
        "timeline": {"fps": "auto"},
        "audio": {
            "song": analysis["file"],
            "song_start": section["song_start"],
            "beat_times": section["beat_times"],
            "lead_beats": section["lead_beats"],
            "bars": section["bars"],
            "tempo": analysis["tempo"],
            "gain_db": 0.0,
            "duck": [],
        },
        "targets": project.cfg["targets"],
        "clips": {},
        "segments": segments,
    }


def write(project, force=False):
    target = project.edl_path
    if target.exists() and not force:
        raise FileExistsError(f"{target} already exists (use --force to replace it)")
    data = skeleton(project)
    save_json(target, data)
    return target
