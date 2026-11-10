from pathlib import Path

from . import clipscan
from .config import load_json, parse_rate, save_json
from .media import file_signature, run, slug


def filter_chain(scan):
    info = scan["info"]
    flags = set(scan["flags"])
    chain = []
    reasons = []
    if "interlaced" in flags:
        chain.append("bwdif=mode=send_frame")
        reasons.append("deinterlace")
    if "non_square_pixels" in flags:
        chain.append("scale=trunc(iw*sar/2)*2:ih,setsar=1")
        reasons.append("square pixels")
    if "hdr" in flags:
        chain.append("zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,tonemap=tonemap=hable:desat=0,zscale=t=bt709:m=bt709:r=tv,format=yuv420p")
        reasons.append("tone-map HDR to SDR")
    rate = float(parse_rate(scan["fps"]))
    if "variable_frame_rate" in flags:
        chain.append(f"fps={scan['fps']}")
        reasons.append("constant frame rate")
    if "duplicate_frames" in flags:
        ratio = scan["cadence"]["dup_ratio"]
        if abs(ratio - 0.2) < 0.06:
            chain.append("decimate=cycle=5")
            reasons.append("drop 1 duplicate in 5")
        elif abs(ratio - 0.6) < 0.07:
            chain.append("select='not(mod(n\\,2))',decimate=cycle=5")
            reasons.append("drop 3 duplicates in 5")
        elif abs(ratio - 0.5) < 0.05:
            chain.append("select='not(mod(n\\,2))'")
            reasons.append("drop every other frame")
        else:
            chain.append("mpdecimate")
            reasons.append("drop duplicate frames")
        chain.append(f"setpts=N/({scan['cadence']['unique_rate']})/TB")
    return chain, reasons, rate


def prepare_clip(project, entry, force=False, log=print):
    if entry.get("prepared") and not force:
        target = project.root / entry["prepared"]
        if target.exists():
            return target
    original = (project.root / entry["file"]).resolve()
    plain = dict(entry)
    plain.pop("prepared", None)
    scan = clipscan.scan_clip(project, plain, sheets=False, log=lambda *_: None)
    chain, reasons, _ = filter_chain(scan)
    if not chain:
        return None
    target = project.subdir("mezz") / f"{slug(original.stem)}-{file_signature(original)}.mp4"
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", original, "-vf", ",".join(chain)]
    if "duplicate_frames" in scan["flags"]:
        cmd += ["-r", scan["cadence"]["unique_rate"]]
    cmd += ["-c:v", "libx264", "-preset", "fast", "-crf", "12", "-pix_fmt", "yuv420p", "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709", "-color_range", "tv"]
    if scan["info"].get("has_audio"):
        cmd += ["-c:a", "aac", "-b:a", "256k"]
    cmd += [target]
    run(cmd)
    index = load_json(project.clips_index_path)
    for item in index["clips"]:
        if item["id"] == entry["id"]:
            item["prepared"] = project.relative(target)
            item["prepared_for"] = reasons
    save_json(project.clips_index_path, index)
    if log:
        log(f"prepared {entry['id']}: {', '.join(reasons)} -> {project.relative(target)}")
    return target


def prepare_all(project, only=None, force=False, log=print):
    results = {}
    for entry in clipscan.load_index(project):
        if only and entry["id"] not in only:
            continue
        results[entry["id"]] = prepare_clip(project, entry, force, log)
    return results
