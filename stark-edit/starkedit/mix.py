import json
import re
import subprocess
from fractions import Fraction
from pathlib import Path

import numpy as np
import scipy.signal

from . import clipscan
from .media import extract_wav, file_signature, read_wav, slug, write_wav
from .song import find_song, song_wav


def decibels(value):
    return 10.0 ** (float(value) / 20.0)


def clip_wav(project, path, rate):
    target = project.subdir("audio") / f"{slug(Path(path).stem)}-{file_signature(path)}.wav"
    if not target.exists():
        extract_wav(path, target, rate, 2, aligned=True)
    return target


def fade(data, rate, fade_in, fade_out):
    out = data.copy()
    a = min(len(out), int(round(fade_in * rate)))
    b = min(len(out), int(round(fade_out * rate)))
    if a > 1:
        out[:a] *= np.linspace(0.0, 1.0, a, dtype=np.float32)[:, None]
    if b > 1:
        out[-b:] *= np.linspace(1.0, 0.0, b, dtype=np.float32)[:, None]
    return out


def place(buffer, data, start):
    begin = int(start)
    if begin < 0:
        data = data[-begin:]
        begin = 0
    end = min(len(buffer), begin + len(data))
    if end > begin:
        buffer[begin:end] += data[: end - begin]


def stretch(data, factor):
    if abs(factor - 1.0) < 0.002:
        return data
    ratio = Fraction(factor).limit_denominator(400)
    return scipy.signal.resample_poly(data, ratio.numerator, ratio.denominator, axis=0).astype(np.float32)


def measure_loudness(path):
    result = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", str(path), "-af", "loudnorm=print_format=json", "-f", "null", "-"], capture_output=True, text=True)
    found = re.search(r"\{[^{}]*\"input_i\"[^{}]*\}", result.stderr, re.S)
    if not found:
        return None
    data = json.loads(found.group(0))
    try:
        return {"integrated": float(data["input_i"]), "peak": float(data["input_tp"])}
    except ValueError:
        return None


def build_mix(project, resolved, target, pool=None, log=print):
    cfg = project.cfg["render"]
    rate = int(cfg["sample_rate"])
    fps = float(resolved.fps)
    edl = resolved.edl
    audio = edl["audio"]
    total = int(round(resolved.total_frames / fps * rate))
    mix = np.zeros((total, 2), dtype=np.float32)
    report = {"events": []}
    song_path = project.root / audio["song"] if audio.get("song") else find_song(project)
    song, song_rate = read_wav(song_wav(project, song_path))
    if song_rate != rate:
        song = stretch(song, rate / song_rate)
    zero = resolved.beat_zero
    beats = resolved.beats
    origin = beats.time(0.0)

    def beat_seconds(position):
        return zero + beats.time(float(position)) - origin

    music_in = beat_seconds(audio.get("music_in_beat", 0.0)) - float(audio.get("lead_seconds", 0.0))
    music_in = max(0.0, music_in)
    music_out = resolved.total_frames / fps
    for segment in resolved.segments:
        if segment.spec.get("music") == "stop":
            music_out = segment.time_in
            break
    first = int(round(music_in * rate))
    last = int(round(music_out * rate))
    source_first = int(round((float(audio.get("song_start", 0.0)) + (music_in - zero)) * rate))
    length = last - first
    piece = np.zeros((length, 2), dtype=np.float32)
    take_from = max(0, source_first)
    take_to = min(len(song), source_first + length)
    if take_to > take_from:
        piece[take_from - source_first : take_to - source_first] = song[take_from:take_to]
    envelope = np.full(length, float(audio.get("gain_db", 0.0)), dtype=np.float32)
    for duck in audio.get("duck", []):
        start = beat_seconds(duck["from_beat"]) if "from_beat" in duck else float(duck.get("from", 0.0))
        stop = beat_seconds(duck["to_beat"]) if "to_beat" in duck else float(duck.get("to", music_out))
        attack = float(duck.get("attack", 0.04))
        release = float(duck.get("release", 0.08))
        times = np.arange(length) / rate + music_in
        shape = np.clip((times - (start - attack)) / max(attack, 1e-4), 0.0, 1.0) * np.clip(((stop + release) - times) / max(release, 1e-4), 0.0, 1.0)
        envelope += float(duck["gain_db"]) * shape.astype(np.float32)
    piece *= (10.0 ** (envelope / 20.0))[:, None]
    piece = fade(piece, rate, float(audio.get("fade_in", 0.004)), float(audio.get("stop_fade", 0.006)))
    place(mix, piece, first)
    report["music"] = {"in": round(music_in, 4), "out": round(music_out, 4), "song_from": round(source_first / rate, 4)}
    entries = {entry["id"]: entry for entry in clipscan.load_index(project)}

    def add_event(path, start_time, source_start, source_length, speed, options, label):
        data, clip_rate = read_wav(clip_wav(project, path, rate))
        a = int(round(source_start * clip_rate))
        b = int(round((source_start + source_length) * clip_rate))
        chunk = data[max(0, a) : max(0, b)]
        if len(chunk) == 0:
            report["events"].append({"label": label, "skipped": "no audio in range"})
            return
        chunk = stretch(chunk, 1.0 / speed)
        chunk = chunk * decibels(options.get("gain_db", 0.0))
        chunk = fade(chunk, rate, float(options.get("fade_in", 0.012)), float(options.get("fade_out", 0.05)))
        place(mix, chunk, int(round(start_time * rate)))
        report["events"].append({"label": label, "at": round(start_time, 4), "source": round(source_start, 4), "seconds": round(len(chunk) / rate, 4), "speed": round(speed, 4)})

    for segment in resolved.segments:
        options = segment.spec.get("audio")
        if not options or not segment.clip:
            continue
        if not segment.scan["info"].get("has_audio"):
            report["events"].append({"label": segment.id, "skipped": "clip has no audio"})
            continue
        if options is True:
            options = {}
        speed = (segment.source_rate / fps) if segment.conform else 1.0
        lead = float(options.get("lead", 0.0))
        tail = float(options.get("tail", 0.0))
        seconds = (segment.frames / fps + lead + tail) * speed
        source_start = segment.scan["start_time"] + segment.src_in - lead * speed
        add_event(segment.path, segment.time_in - lead, source_start, seconds, speed, options, segment.id)
    for number, event in enumerate(audio.get("events", [])):
        entry = entries[event["clip"]]
        path = clipscan.clip_path(project, entry)
        scan = clipscan.load_scan(project, entry) or {"start_time": 0.0}
        at = beat_seconds(event["at_beat"]) if "at_beat" in event else float(event.get("at", 0.0))
        add_event(path, at, scan["start_time"] + float(event["src_in"]), float(event["src_out"]) - float(event["src_in"]), 1.0, event, f"event{number + 1}")
    scratch = Path(target).with_suffix(".premaster.wav")
    write_wav(scratch, mix, rate)
    loudness = measure_loudness(scratch)
    gain_db = 0.0
    if loudness is not None and np.isfinite(loudness["integrated"]):
        gain_db = float(cfg["loudness"]) - loudness["integrated"]
        peak = float(np.max(np.abs(mix))) if len(mix) else 0.0
        if peak > 0:
            headroom = float(cfg["peak_db"]) - 20.0 * np.log10(peak)
            gain_db = min(gain_db, headroom)
        mix *= decibels(gain_db)
    scratch.unlink(missing_ok=True)
    write_wav(target, mix, rate)
    report["loudness"] = {"measured": loudness, "gain_db": round(float(gain_db), 2), "target": cfg["loudness"]}
    report["seconds"] = round(total / rate, 4)
    if log:
        log(f"  audio: music {music_in:.3f}-{music_out:.3f}s, {len(report['events'])} source-audio events, gain {gain_db:+.1f} dB")
    return report
