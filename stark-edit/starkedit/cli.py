import argparse
import sys
from pathlib import Path

from .config import Project, save_json


def emit(text=""):
    print(text, flush=True)


def need_media(project):
    from .media import AUDIO_EXTENSIONS, VIDEO_EXTENSIONS, list_media

    clips = list_media(project.clips_dir, VIDEO_EXTENSIONS)
    songs = list_media(project.audio_dir, AUDIO_EXTENSIONS | VIDEO_EXTENSIONS)
    missing = []
    if not clips:
        missing.append(f"no clips in {project.clips_dir}")
    if not songs:
        missing.append(f"no song in {project.audio_dir}")
    return clips, songs, missing


def command_probe(project, args):
    from . import clipscan
    from .media import probe_clip

    clips, songs, missing = need_media(project)
    for line in missing:
        emit(f"MISSING: {line}")
    for entry in clipscan.load_index(project) if clips else []:
        info = probe_clip(clipscan.clip_path(project, entry))
        emit(f"{entry['id']:>12}  {info.get('width')}x{info.get('height')}  {info.get('avg_rate')} fps  {info['duration']:7.2f}s  {info.get('codec')}/{info.get('pix_fmt')}  audio {info.get('audio_codec', '-')}  {entry['file']}")
    for path in songs:
        info = probe_clip(path)
        emit(f"{'song':>12}  {info['duration']:7.2f}s  audio {info.get('audio_codec', '-')} {info.get('audio_rate', '-')} Hz  {project.relative(path)}")
    return 1 if missing else 0


def command_setup(project, args):
    from . import anchors

    for name in anchors.MODELS:
        emit(f"model {name}: {anchors.ensure_model(project, name)}")
    return 0


def command_scan(project, args):
    from . import clipscan

    for entry in clipscan.load_index(project):
        if args.clip and entry["id"] not in args.clip:
            continue
        scan = clipscan.scan_clip(project, entry, force=args.force, log=emit)
        if scan["flags"]:
            emit(f"   flags: {', '.join(scan['flags'])}  cadence {scan['cadence']}  overlays {scan['overlays']}")
    emit(f"contact sheets: {project.subdir('sheets')}")
    return 0


def command_prepare(project, args):
    from . import prepare

    results = prepare.prepare_all(project, args.clip, args.force, emit)
    if not any(results.values()):
        emit("no clip needs preparing")
    return 0


def command_frames(project, args):
    from . import clipscan

    entry = clipscan.find_entry(project, args.clip)
    crop = [float(v) for v in args.crop.split(",")] if args.crop else None
    if args.strip:
        path = clipscan.export_strip(project, entry, args.strip[0], args.strip[1], args.count, crop)
        emit(str(path))
        return 0
    for path in clipscan.export_frames(project, entry, args.at, not args.no_grid, crop, args.zoom, args.step):
        emit(str(path))
    return 0


def command_faces(project, args):
    import cv2

    from . import anchors, clipscan, draw
    from .media import FrameSource

    entry = clipscan.find_entry(project, args.clip)
    scan = clipscan.scan_clip(project, entry, sheets=False, log=lambda *_: None)
    source = FrameSource(clipscan.clip_path(project, entry))
    finder = anchors.FaceFinder(project)
    first = source.index_at(args.start)
    last = source.index_at(args.end)
    step = max(1, (last - first) // max(1, args.count - 1))
    tiles = []
    for index in range(first, last + 1, step):
        image = source.get(index).copy()
        shot = clipscan.shot_at(scan, index)
        faces = finder.find(image, [int(v) for v in shot["active"]], thorough=True)
        for face in faces:
            draw.crosshair(image, face["x"], face["y"], face["size"] * 0.5, draw.YELLOW, 2, ring=face["size"] * 0.5)
            cv2.rectangle(image, (int(face["box"][0]), int(face["box"][1])), (int(face["box"][2]), int(face["box"][3])), draw.GREEN, 2)
            draw.text(image, f"eyes {face['x']:.0f},{face['y']:.0f} d={face['size']:.0f} yaw={face['yaw']:+.2f}", (int(face["box"][0]), int(face["box"][3]) + 26), 0.9, draw.YELLOW, 2)
        tile = draw.fit(image, width=640).copy()
        draw.text(tile, f"{source.times[index]:.3f}s f{index}  faces {len(faces)}", (5, 16), 0.45)
        tiles.append(tile)
    source.close()
    path = draw.save(project.subdir("frames") / f"{entry['id']}_faces_{args.start:07.2f}.jpg", draw.sheet(tiles, 3), 88)
    emit(str(path))
    return 0


def command_song(project, args):
    from . import song

    result = song.analyze(project, force=args.force, log=emit)
    section = result["section"]
    emit(f"tempo {result['tempo']} BPM, {len(result['beats'])} beats, downbeat phase {result['downbeat_phase']} (confidence {result['phase_confidence']})")
    for rank, candidate in enumerate(result["candidates"][:5]):
        emit(f"  #{rank + 1}: {candidate['start']:.2f}-{candidate['end']:.2f}s, drop at {candidate['drop']:.2f}s, {candidate['bars']} bars, {candidate['seconds']}s, contrast {candidate['contrast_db']} dB, score {candidate['score']}")
    emit(f"chosen: song {section['start']:.3f}-{section['end']:.3f}s, {section['lead_beats']}-beat open then {section['bars']} bars; biggest hits at section beats {section['biggest_hits']}")
    emit(f"plot: {project.work / 'song.png'}")
    return 0


def command_plan(project, args):
    from . import plan

    emit(f"wrote {plan.write(project, args.force)}")
    return 0


def resolve_project(project, quiet=False):
    from . import timeline

    resolved = timeline.resolve(project)
    data = timeline.cut_list(resolved)
    issues = resolved.all_issues()
    save_json(project.output / "cut_list.json", data)
    (project.output / "cut_list.md").write_text(timeline.cut_list_markdown(data, issues))
    if not quiet:
        emit(timeline.cut_list_markdown(data, issues))
        emit(f"targets: { {k: v for k, v in resolved.targets.items() if not k.startswith('@')} }")
    return resolved


def command_check(project, args):
    resolved = resolve_project(project)
    errors = resolved.errors()
    emit(f"{len(errors)} error(s), {len(resolved.all_issues()) - len(errors)} warning(s)")
    return 1 if errors else 0


def command_pairs(project, args):
    from . import qc, render

    resolved = resolve_project(project, quiet=True)
    renderer = render.Renderer(project, resolved, "preview")
    outputs = qc.pairs_from_renderer(project, resolved, renderer, set(args.only) if args.only else None)
    renderer.close()
    emit(f"{len(outputs)} cut pairs in {project.subdir('pairs')}")
    return 0


def command_still(project, args):
    from . import draw, render

    resolved = resolve_project(project, quiet=True)
    renderer = render.Renderer(project, resolved, args.mode)
    folder = project.subdir("stills")
    for segment in resolved.segments:
        if segment.id not in args.segment:
            continue
        picks = {"first": 0, "mid": segment.frames // 2, "last": segment.frames - 1}
        for label in args.pos:
            offset = picks[label] if label in picks else int(label)
            frame = renderer.frame(segment.frame_in + offset)
            emit(str(draw.save(folder / f"{segment.id}_{label}.jpg", frame, 92)))
    renderer.close()
    return 0


def command_render(project, args):
    from . import mix, qc, render

    mode = "final" if args.final else "preview"
    resolved = resolve_project(project, quiet=True)
    errors = resolved.errors()
    if errors and not args.force:
        for item in errors:
            emit(f"error: {item['segment']}: {item['message']}")
        emit("fix the edit list or pass --force")
        return 1
    target = project.output / ("edit_final.mp4" if args.final else "preview.mp4")
    wav = project.work / f"mix_{mode}.wav"
    emit(f"rendering {mode}: {resolved.total_frames} frames at {resolved.fps} fps")
    report = mix.build_mix(project, resolved, wav, log=emit)
    renderer = render.Renderer(project, resolved, mode)
    renderer.render(target, wav, emit)
    save_json(project.work / f"render_{mode}.json", {"audio": report, "grades": renderer.grade_report()})
    renderer.close()
    if not args.no_qc:
        qc.run(project, resolved, target, log=emit)
        emit(f"qc report: {project.subdir('qc') / 'report.md'}")
    return 0


def command_qc(project, args):
    from . import qc

    resolved = resolve_project(project, quiet=True)
    video = Path(args.video) if args.video else project.output / "edit_final.mp4"
    if not video.exists():
        video = project.output / "preview.mp4"
    report = qc.run(project, resolved, video, log=emit)
    emit((project.subdir("qc") / "report.md").read_text())
    return 1 if report["summary"]["errors"] else 0


def build_parser():
    parser = argparse.ArgumentParser(prog="starkedit", description="Beat-synced, match-cut fan edit pipeline driven by an edit decision list.")
    parser.add_argument("--project", default=".", help="folder holding config.json, clips/, audio/ (default: current folder)")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("probe", help="list the clips and the song")
    commands.add_parser("setup", help="download the face models")
    scan = commands.add_parser("scan", help="detect shots, letterbox and cadence; write contact sheets")
    scan.add_argument("--clip", nargs="*")
    scan.add_argument("--force", action="store_true")
    prepare = commands.add_parser("prepare", help="clean up clips with duplicate frames, HDR, interlacing or odd pixels")
    prepare.add_argument("--clip", nargs="*")
    prepare.add_argument("--force", action="store_true")
    frames = commands.add_parser("frames", help="export frames with a coordinate grid")
    frames.add_argument("--clip", required=True)
    frames.add_argument("--at", nargs="*", type=float, default=[])
    frames.add_argument("--strip", nargs=2, type=float)
    frames.add_argument("--count", type=int, default=8)
    frames.add_argument("--crop")
    frames.add_argument("--zoom", type=float, default=1.0)
    frames.add_argument("--step", type=int)
    frames.add_argument("--no-grid", action="store_true")
    faces = commands.add_parser("faces", help="show detected eye anchors over a time range")
    faces.add_argument("--clip", required=True)
    faces.add_argument("--start", type=float, required=True)
    faces.add_argument("--end", type=float, required=True)
    faces.add_argument("--count", type=int, default=6)
    song = commands.add_parser("song", help="analyse the song: beats, downbeats, section, cut grid")
    song.add_argument("--force", action="store_true")
    plan = commands.add_parser("plan", help="write a skeleton edit list from the song analysis")
    plan.add_argument("--force", action="store_true")
    commands.add_parser("check", help="resolve the edit list and print the cut list and problems")
    pairs = commands.add_parser("pairs", help="render only the frames either side of each cut")
    pairs.add_argument("--only", nargs="*")
    still = commands.add_parser("still", help="render single frames of a segment")
    still.add_argument("--segment", nargs="+", required=True)
    still.add_argument("--pos", nargs="*", default=["first", "mid", "last"])
    still.add_argument("--mode", default="preview", choices=["preview", "final"])
    render = commands.add_parser("render", help="render the preview or the final video, then run QC")
    render.add_argument("--final", action="store_true")
    render.add_argument("--force", action="store_true")
    render.add_argument("--no-qc", action="store_true")
    qc = commands.add_parser("qc", help="check a rendered file")
    qc.add_argument("--video")
    return parser


HANDLERS = {
    "probe": command_probe,
    "setup": command_setup,
    "scan": command_scan,
    "prepare": command_prepare,
    "frames": command_frames,
    "faces": command_faces,
    "song": command_song,
    "plan": command_plan,
    "check": command_check,
    "pairs": command_pairs,
    "still": command_still,
    "render": command_render,
    "qc": command_qc,
}


def main(argv=None):
    args = build_parser().parse_args(argv)
    project = Project(args.project)
    if args.command not in ("setup",):
        _, _, missing = need_media(project)
        if missing:
            for line in missing:
                emit(f"MISSING: {line}")
            if args.command != "probe":
                return 2
    try:
        return HANDLERS[args.command](project, args)
    except (ValueError, KeyError, FileExistsError, RuntimeError) as error:
        emit(f"error: {error}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
