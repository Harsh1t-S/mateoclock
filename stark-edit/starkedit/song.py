from pathlib import Path

import numpy as np
import scipy.ndimage
import scipy.signal

from .config import load_json, save_json
from .media import AUDIO_EXTENSIONS, VIDEO_EXTENSIONS, MediaError, extract_wav, file_signature, list_media

SONG_VERSION = 5
ANALYSIS_RATE = 22050
HOP = 256
ATTACK_BIAS = 0.004


def find_song(project):
    files = list_media(project.audio_dir, AUDIO_EXTENSIONS | VIDEO_EXTENSIONS)
    if not files:
        raise MediaError(f"no song found in {project.audio_dir}")
    return files[0]


def song_wav(project, path=None):
    path = Path(path) if path else find_song(project)
    target = project.subdir("audio") / f"song-{file_signature(path)}.wav"
    if not target.exists():
        extract_wav(path, target, project.cfg["render"]["sample_rate"], 2, aligned=True)
    return target


def attack_envelope(y, rate):
    window = max(3, int(0.006 * rate))
    energy = scipy.ndimage.uniform_filter1d(y.astype(np.float64) ** 2, window, mode="nearest")
    step = rate / 1000.0
    points = np.round(np.arange(0, len(y) / step) * step).astype(np.int64)
    points = points[points < len(y)]
    level = np.log10(energy[points] + 1e-7)
    rise = np.zeros_like(level)
    rise[8:] = np.maximum(0.0, level[8:] - level[:-8])
    return rise


def locate_attacks(rise, beats, radius=0.06):
    times = np.full(len(beats), np.nan)
    strength = np.zeros(len(beats))
    span = int(radius * 1000)
    for k, beat in enumerate(beats):
        centre = int(round(beat * 1000))
        lo = max(0, centre - span)
        hi = min(len(rise), centre + span + 1)
        if hi - lo < 5:
            continue
        window = rise[lo:hi]
        peak = int(np.argmax(window))
        strength[k] = float(window[peak])
        left = window[peak - 1] if peak > 0 else window[peak]
        right = window[peak + 1] if peak + 1 < len(window) else window[peak]
        denominator = left - 2.0 * window[peak] + right
        shift = 0.5 * (left - right) / denominator if abs(denominator) > 1e-9 else 0.0
        times[k] = (lo + peak + float(np.clip(shift, -0.5, 0.5))) / 1000.0 - ATTACK_BIAS
    return times, strength


def regularize(beats, attacks, strength, radius=12, tolerance=0.022):
    n = len(beats)
    refined = np.array(beats, dtype=np.float64)
    clear = (~np.isnan(attacks)) & (strength > 0.2)
    offsets = attacks - beats
    usable = clear & (np.abs(offsets) < 0.055)
    spread = np.zeros(n)
    index = np.arange(n, dtype=np.float64)
    fallback = float(np.median(offsets[usable])) if usable.any() else 0.0
    for k in range(n):
        lo = max(0, k - radius)
        hi = min(n, k + radius + 1)
        pick = np.where(usable[lo:hi])[0] + lo
        if len(pick) < 6:
            refined[k] = beats[k] + fallback
            spread[k] = np.nan
            continue
        x = index[pick]
        t = attacks[pick]
        for _ in range(3):
            slope, intercept = np.polyfit(x, t, 1)
            residual = t - (slope * x + intercept)
            keep = np.abs(residual) < max(tolerance, 2.5 * np.std(residual))
            if keep.all() or keep.sum() < 6:
                break
            x, t = x[keep], t[keep]
        slope, intercept = np.polyfit(x, t, 1)
        refined[k] = slope * k + intercept
        spread[k] = float(np.std(t - (slope * x + intercept)))
    for k in range(1, n):
        if refined[k] <= refined[k - 1] + 0.05:
            refined[k] = max(refined[k], refined[k - 1] + 0.05)
    return refined, spread


def band_rise(y, rate, low, high, beats):
    sos = scipy.signal.butter(4, [low, high], btype="bandpass", fs=rate, output="sos")
    band = scipy.signal.sosfiltfilt(sos, y)
    window = max(3, int(0.012 * rate))
    energy = scipy.ndimage.uniform_filter1d(band.astype(np.float64) ** 2, window, mode="nearest")
    out = np.zeros(len(beats))
    span = int(0.07 * rate)
    for k, beat in enumerate(beats):
        centre = int(beat * rate)
        before = energy[max(0, centre - span) : max(1, centre - int(0.01 * rate))]
        after = energy[centre : min(len(energy), centre + span)]
        if len(before) == 0 or len(after) == 0:
            continue
        out[k] = 10.0 * np.log10((after.max() + 1e-9) / (before.mean() + 1e-9))
    return out


def zscore(values):
    values = np.asarray(values, dtype=np.float64)
    spread = np.nanstd(values)
    if spread < 1e-9:
        return np.zeros_like(values)
    return (values - np.nanmean(values)) / spread


def normalize_period(beats, strength, limits):
    beats = np.asarray(beats, dtype=np.float64)
    strength = np.asarray(strength, dtype=np.float64)
    low, high = limits
    factor = 1.0
    for _ in range(3):
        period = float(np.median(np.diff(beats)))
        if period < low and len(beats) > 8:
            first = float(np.mean(strength[0::2]))
            second = float(np.mean(strength[1::2]))
            start = 0 if first >= second else 1
            beats = beats[start::2]
            strength = strength[start::2]
            factor *= 2.0
        elif period > high:
            middle = 0.5 * (beats[:-1] + beats[1:])
            merged = np.empty(len(beats) + len(middle))
            merged[0::2] = beats
            merged[1::2] = middle
            beats = merged
            strength = np.repeat(strength, 2)[: len(merged)]
            factor *= 0.5
        else:
            break
    return beats, factor


def beat_features(y, rate, beats, onset, chroma):
    n = len(beats)
    energy = np.zeros(n)
    hit = np.zeros(n)
    change = np.zeros(n)
    edges = np.append(beats, beats[-1] + (beats[-1] - beats[-2]))
    frames = np.round(edges * rate / HOP).astype(int)
    chroma_frames = np.round(edges * rate / 512).astype(int)
    vectors = np.zeros((n, chroma.shape[0]))
    for k in range(n):
        a = int(edges[k] * rate)
        b = int(edges[k + 1] * rate)
        chunk = y[a:b]
        energy[k] = 10.0 * np.log10(float(np.mean(chunk.astype(np.float64) ** 2)) + 1e-9) if len(chunk) else -90.0
        lo = max(0, frames[k] - 3)
        hi = min(len(onset), frames[k] + 4)
        hit[k] = float(onset[lo:hi].max()) if hi > lo else 0.0
        ca = min(chroma.shape[1] - 1, chroma_frames[k])
        cb = max(ca + 1, min(chroma.shape[1], chroma_frames[k + 1]))
        vectors[k] = np.median(chroma[:, ca:cb], axis=1)
    for k in range(1, n):
        a = vectors[k - 1]
        b = vectors[k]
        norm = np.linalg.norm(a) * np.linalg.norm(b)
        change[k] = 1.0 - float(a @ b) / norm if norm > 1e-9 else 0.0
    return energy, hit, change


def downbeat_phase(low, change, energy, per_bar):
    n = len(low)
    jump = np.zeros(n)
    for k in range(4, n - 8):
        jump[k] = float(np.mean(energy[k : k + 8]) - np.mean(energy[k - 4 : k]))
    positive = np.maximum(jump, 0.0) ** 2
    total = positive.sum() + 1e-9
    zl = zscore(low)
    zc = zscore(change)
    scores = []
    for phase in range(per_bar):
        picks = np.arange(phase, n, per_bar)
        scores.append(float(np.mean(zl[picks]) + np.mean(zc[picks]) + 2.0 * positive[picks].sum() / total))
    order = np.argsort(scores)[::-1]
    best = int(order[0])
    confidence = float(scores[order[0]] - scores[order[1]]) if per_bar > 1 else 1.0
    return best, scores, confidence, jump


def choose_lead(period, limits, per_bar):
    low, high = limits
    for count in (per_bar, 2 * per_bar, per_bar // 2, 3 * per_bar // 2, per_bar // 4 or 1):
        if count >= 1 and low <= count * period <= high:
            return int(count)
    return int(max(1, round(0.5 * (low + high) / period)))


def score_sections(beats, energy, hit, jump, phase, per_bar, length_cfg, period):
    n = len(beats)
    lead = choose_lead(period, length_cfg["cold_open_seconds"], per_bar)
    tail = float(length_cfg["tail_seconds"])
    ze = zscore(energy)
    zh = zscore(hit)
    results = []
    for drop in range(phase, n, per_bar):
        if drop - lead < 0:
            continue
        for bars in range(4, 49):
            end = drop + bars * per_bar
            if end >= n:
                break
            total = float(beats[end] - beats[drop - lead]) + tail
            if total < length_cfg["min_seconds"] or total > length_cfg["max_seconds"]:
                continue
            main = ze[drop:end]
            by_bar = main.reshape(bars, per_bar).mean(axis=1)
            level = float(main.mean())
            sag = float(max(0.0, level - by_bar.min()))
            contrast = float(np.mean(energy[drop : drop + 2 * per_bar]) - np.mean(energy[drop - lead : drop]))
            shape = 0.2 if bars % 4 == 0 else (0.1 if bars % 2 == 0 else 0.0)
            near = -0.25 * abs(total - length_cfg["target_seconds"]) / max(1.0, length_cfg["max_seconds"] - length_cfg["min_seconds"])
            score = level + 0.5 * float(np.clip(contrast / 6.0, -1.0, 2.0)) + 0.3 * float(zh[drop]) + 0.15 * float(zh[end]) - 0.6 * sag + shape + near
            results.append(
                {
                    "drop_beat": int(drop),
                    "start_beat": int(drop - lead),
                    "end_beat": int(end),
                    "lead_beats": int(lead),
                    "bars": int(bars),
                    "start": round(float(beats[drop - lead]), 4),
                    "drop": round(float(beats[drop]), 4),
                    "end": round(float(beats[end]), 4),
                    "seconds": round(total, 3),
                    "level": round(level, 3),
                    "contrast_db": round(contrast, 2),
                    "sag": round(sag, 3),
                    "score": round(float(score), 4),
                }
            )
    results.sort(key=lambda r: -r["score"])
    return results


def cut_grid(section, energy, hit, half_strength, per_bar):
    drop = section["drop_beat"]
    start = section["start_beat"]
    end = section["end_beat"]
    lead = section["lead_beats"]
    main = energy[drop:end]
    by_bar = main.reshape(section["bars"], per_bar).mean(axis=1)
    centre = float(np.median(by_bar))
    spread = float(np.std(by_bar)) + 1e-6
    zh = zscore(hit[start : end + 1])
    slots = [{"beat": 0.0, "beats": float(lead), "bar": -1, "level": "open", "hit": round(float(zh[0]), 2)}]
    for bar in range(section["bars"]):
        first = lead + bar * per_bar
        relative = (by_bar[bar] - centre) / spread
        beat_hits = hit[drop + bar * per_bar : drop + (bar + 1) * per_bar]
        halves = half_strength[drop + bar * per_bar : drop + (bar + 1) * per_bar]
        busy = float(np.mean(halves)) >= 0.6 * float(np.mean(beat_hits) + 1e-9)
        if spread < 0.4 or relative >= -0.35:
            level = "high"
            pattern = [1.0] * per_bar
            if busy and bar % 4 == 3:
                pattern = [1.0] * (per_bar - 2) + [0.5] * 4
        elif relative >= -1.2:
            level = "mid"
            pattern = [2.0] * (per_bar // 2) if per_bar % 2 == 0 else [float(per_bar)]
        else:
            level = "low"
            pattern = [float(per_bar)]
        position = float(first)
        for duration in pattern:
            index = int(round(position))
            slots.append({"beat": position, "beats": duration, "bar": bar, "level": level, "hit": round(float(zh[min(len(zh) - 1, index)]), 2)})
            position += duration
    return slots


def analyze(project, force=False, path=None, log=print):
    import librosa

    source = Path(path) if path else find_song(project)
    cache = project.work / f"song-{file_signature(source)}.json"
    settings = project.cfg["song"]
    lengths = project.cfg["length"]
    key = {"version": SONG_VERSION, "song": settings, "length": lengths}
    if not force:
        cached = load_json(cache)
        if cached is not None and cached.get("key") == key:
            return cached
    wav = song_wav(project, source)
    y, rate = librosa.load(str(wav), sr=ANALYSIS_RATE, mono=True)
    duration = len(y) / rate
    onset = librosa.onset.onset_strength(y=y, sr=rate, hop_length=HOP)
    tempo, frames = librosa.beat.beat_track(onset_envelope=onset, sr=rate, hop_length=HOP, trim=False)
    raw = librosa.frames_to_time(frames, sr=rate, hop_length=HOP)
    if len(raw) < 16:
        raise MediaError("beat tracker found too few beats")
    rise = attack_envelope(y, rate)
    _, raw_strength = locate_attacks(rise, raw)
    tracked, factor = normalize_period(raw, raw_strength, settings["beat_period_range"])
    attacks, strength = locate_attacks(rise, tracked)
    beats, spread = regularize(tracked, attacks, strength)
    period = float(np.median(np.diff(beats)))
    chroma = librosa.feature.chroma_cqt(y=y, sr=rate, hop_length=512)
    energy, hit, change = beat_features(y, rate, beats, onset, chroma)
    low = band_rise(y, rate, 30.0, 160.0, beats)
    halves = 0.5 * (beats[:-1] + beats[1:])
    _, half_strength = locate_attacks(rise, halves, 0.04)
    half_strength = np.append(half_strength, 0.0)
    per_bar = int(settings["beats_per_bar"])
    phase, phase_scores, confidence, jump = downbeat_phase(low, change, energy, per_bar)
    forced_start = settings.get("section_start")
    if settings.get("downbeat_shift") is not None:
        phase = int(settings["downbeat_shift"]) % per_bar
    candidates = score_sections(beats, energy, hit, jump, phase, per_bar, lengths, period)
    if confidence < 0.15 and settings.get("downbeat_shift") is None:
        strongest = int(np.argmax(jump))
        alternative = strongest % per_bar
        if alternative != phase:
            other = score_sections(beats, energy, hit, jump, alternative, per_bar, lengths, period)
            if other and (not candidates or other[0]["score"] > candidates[0]["score"] + 0.3):
                phase = alternative
                candidates = other
    if forced_start is not None:
        nearest = int(np.argmin(np.abs(beats - float(forced_start))))
        forced_phase = nearest % per_bar
        forced = [c for c in score_sections(beats, energy, hit, jump, forced_phase, per_bar, lengths, period) if c["drop_beat"] == nearest]
        if forced:
            phase = forced_phase
            candidates = forced + [c for c in candidates if c["drop_beat"] != nearest]
    if not candidates:
        raise MediaError("no section of the song fits the requested length")
    chosen = candidates[0]
    start = chosen["start_beat"]
    end = chosen["end_beat"]
    song_start = float(beats[start])
    timeline = [round(float(b - song_start), 5) for b in beats[start : end + 1]]
    grid = cut_grid(chosen, energy, hit, half_strength, per_bar)
    impact = 0.6 * zscore(hit) + zscore(low) + zscore(np.maximum(jump, 0.0))
    impact[phase::per_bar] += 0.4
    section_impact = impact[start : end + 1]
    order = []
    for candidate in np.argsort(section_impact)[::-1]:
        if all(abs(int(candidate) - chosen_hit) > 2 for chosen_hit in order):
            order.append(int(candidate))
        if len(order) == 8:
            break
    result = {
        "key": key,
        "file": project.relative(source),
        "wav": project.relative(wav),
        "duration": round(duration, 3),
        "tempo": round(60.0 / period, 3),
        "tracker_tempo": round(float(np.atleast_1d(tempo)[0]), 3),
        "period": round(period, 5),
        "period_factor": factor,
        "beats_per_bar": per_bar,
        "downbeat_phase": int(phase),
        "phase_scores": [round(s, 3) for s in phase_scores],
        "phase_confidence": round(confidence, 3),
        "beats": [round(float(b), 5) for b in beats],
        "beat_spread_ms": [None if np.isnan(s) else round(float(s) * 1000.0, 2) for s in spread],
        "attack_clear": [bool(v > 0.2) for v in strength],
        "energy_db": [round(float(v), 2) for v in energy],
        "hit": [round(float(v), 3) for v in hit],
        "impact": [round(float(v), 3) for v in impact],
        "candidates": candidates[:8],
        "section": {
            **chosen,
            "song_start": round(song_start, 5),
            "beat_times": timeline,
            "biggest_hits": [int(i) for i in order[:8]],
            "grid": grid,
        },
    }
    save_json(cache, result)
    plot(project, result, y, rate, rise)
    log(f"song: {result['tempo']:.2f} BPM, section {chosen['start']:.2f}-{chosen['end']:.2f}s ({chosen['bars']} bars after a {chosen['lead_beats']}-beat open), downbeat phase {phase} (confidence {confidence:.2f})")
    return result


def plot(project, result, y, rate, rise):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    beats = np.array(result["beats"])
    energy = np.array(result["energy_db"])
    hit = np.array(result["impact"])
    section = result["section"]
    per_bar = result["beats_per_bar"]
    phase = result["downbeat_phase"]
    figure, axes = plt.subplots(4, 1, figsize=(18, 13))
    ax = axes[0]
    ax.plot(beats, energy, color="#2b6cb0", linewidth=1.0)
    for rank, candidate in enumerate(result["candidates"][:5]):
        ax.axvspan(candidate["start"], candidate["end"], color="#e53e3e" if rank == 0 else "#a0aec0", alpha=0.28 if rank == 0 else 0.12)
        ax.text(candidate["start"], energy.max() - 1.5 * rank, f"#{rank + 1} {candidate['score']:.2f}", fontsize=8)
    ax.set_title(f"{result['file']}  {result['tempo']:.2f} BPM  energy per beat (dB), top section candidates")
    ax.set_xlim(0, result["duration"])
    ax = axes[1]
    a = max(0.0, section["start"] - 6.0)
    b = min(result["duration"], section["end"] + 4.0)
    i0, i1 = int(a * rate), int(b * rate)
    step = max(1, (i1 - i0) // 6000)
    chunk = np.abs(y[i0:i1])
    usable = (len(chunk) // step) * step
    peaks = chunk[:usable].reshape(-1, step).max(axis=1)
    ax.fill_between(a + np.arange(len(peaks)) * step / rate, -peaks, peaks, color="#4a5568", linewidth=0)
    for k, beat in enumerate(beats):
        if a <= beat <= b:
            down = k % per_bar == phase
            ax.axvline(beat, color="#e53e3e" if down else "#ed8936", linewidth=1.3 if down else 0.5, alpha=0.9 if down else 0.6)
    ax.axvline(section["drop"], color="#38a169", linewidth=2.5)
    ax.axvspan(section["start"], section["end"], color="#e53e3e", alpha=0.08)
    ax.set_xlim(a, b)
    ax.set_title("chosen section: waveform, beats (orange), downbeats (red), drop (green)")
    ax = axes[2]
    start = section["start_beat"]
    end = section["end_beat"]
    positions = np.arange(end - start + 1)
    colors = ["#e53e3e" if (start + p) % per_bar == phase else "#3182ce" for p in positions]
    ax.bar(positions, hit[start : end + 1], color=colors, width=0.8)
    for slot in section["grid"]:
        ax.axvline(slot["beat"] - 0.5, color="#1a202c", linewidth=0.9, alpha=0.8)
    ax.set_xlim(-1, len(positions))
    ax.set_title("impact per beat in the section (onset + low-end rise + energy jump; red = downbeat), vertical lines = suggested cuts")
    ax = axes[3]
    centre = section["drop"]
    lo = int((centre - 0.25) * 1000)
    hi = int((centre + 0.75) * 1000)
    ax.plot(np.arange(lo, hi) / 1000.0, rise[lo:hi], color="#2d3748", linewidth=0.8)
    for beat in beats:
        if lo / 1000.0 <= beat <= hi / 1000.0:
            ax.axvline(beat, color="#e53e3e", linewidth=1.2)
    ax.set_xlim(lo / 1000.0, hi / 1000.0)
    ax.set_title("attack envelope around the drop, red = beat grid (should sit on the rising edge)")
    figure.tight_layout()
    target = project.work / "song.png"
    figure.savefig(target, dpi=70)
    plt.close(figure)
    return target
