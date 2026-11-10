import hashlib
import json
import subprocess
import tempfile
from collections import OrderedDict
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
from av.video.reformatter import ColorRange, Colorspace

from .config import rate_text, snap_rate

VIDEO_EXTENSIONS = {".mp4", ".mkv", ".mov", ".webm", ".avi", ".m4v", ".ts", ".flv", ".wmv", ".mpg", ".mpeg"}
AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".opus", ".aac", ".wma", ".aiff"}
HDR_TRANSFERS = {"smpte2084", "arib-std-b67"}


class MediaError(RuntimeError):
    pass


def run(cmd):
    result = subprocess.run([str(c) for c in cmd], capture_output=True, text=True)
    if result.returncode != 0:
        tail = "\n".join(result.stderr.strip().splitlines()[-12:])
        raise MediaError(f"{cmd[0]} failed ({result.returncode}): {tail}")
    return result


def ffprobe(path):
    result = run(["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", path])
    return json.loads(result.stdout)


def list_media(folder, extensions):
    folder = Path(folder)
    if not folder.exists():
        return []
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in extensions)


def file_signature(path):
    path = Path(path)
    stat = path.stat()
    digest = hashlib.sha1(f"{path.name}:{stat.st_size}:{int(stat.st_mtime)}".encode()).hexdigest()
    return digest[:10]


def slug(text):
    keep = [c.lower() if c.isalnum() else "_" for c in str(text)]
    out = "".join(keep)
    while "__" in out:
        out = out.replace("__", "_")
    return out.strip("_")[:48] or "clip"


def _fraction(text):
    if not text or text in ("0/0", "N/A"):
        return None
    if "/" in text:
        num, den = text.split("/")
        if int(den) == 0:
            return None
        return Fraction(int(num), int(den))
    return Fraction(text)


def probe_clip(path):
    data = ffprobe(path)
    video = next((s for s in data["streams"] if s.get("codec_type") == "video" and s.get("disposition", {}).get("attached_pic", 0) == 0), None)
    audio = next((s for s in data["streams"] if s.get("codec_type") == "audio"), None)
    fmt = data.get("format", {})
    info = {
        "file": str(path),
        "duration": float(fmt.get("duration", 0.0) or 0.0),
        "bitrate": int(fmt.get("bit_rate", 0) or 0),
        "has_video": video is not None,
        "has_audio": audio is not None,
    }
    if video is not None:
        avg = _fraction(video.get("avg_frame_rate"))
        real = _fraction(video.get("r_frame_rate"))
        sar = _fraction((video.get("sample_aspect_ratio") or "1:1").replace(":", "/")) or Fraction(1)
        rotation = 0
        for side in video.get("side_data_list", []) or []:
            if "rotation" in side:
                rotation = int(side["rotation"])
        info.update(
            {
                "width": int(video["width"]),
                "height": int(video["height"]),
                "codec": video.get("codec_name"),
                "pix_fmt": video.get("pix_fmt"),
                "avg_rate": rate_text(avg) if avg else None,
                "real_rate": rate_text(real) if real else None,
                "frames_declared": int(video["nb_frames"]) if str(video.get("nb_frames", "")).isdigit() else None,
                "color_space": video.get("color_space"),
                "color_transfer": video.get("color_transfer"),
                "color_primaries": video.get("color_primaries"),
                "color_range": video.get("color_range"),
                "field_order": video.get("field_order", "progressive"),
                "sar": rate_text(sar),
                "rotation": rotation,
                "hdr": video.get("color_transfer") in HDR_TRANSFERS,
            }
        )
    if audio is not None:
        info.update(
            {
                "audio_codec": audio.get("codec_name"),
                "audio_rate": int(audio.get("sample_rate", 0) or 0),
                "audio_channels": int(audio.get("channels", 0) or 0),
            }
        )
    return info


def _source_colorspace(frame):
    value = frame.colorspace
    if value == 1:
        return Colorspace.ITU709
    if value in (5, 6):
        return Colorspace.ITU601
    if value == 7:
        return Colorspace.SMPTE240M
    if value in (9, 10):
        return Colorspace.BT2020
    if value == 4:
        return Colorspace.FCC
    return Colorspace.ITU709 if frame.height >= 700 or frame.width >= 1200 else Colorspace.ITU601


def _source_range(frame):
    if frame.color_range == 2 or frame.format.name.startswith("yuvj"):
        return ColorRange.JPEG
    return ColorRange.MPEG


def frame_to_bgr(frame, width=None, height=None, interpolation=None):
    name = frame.format.name
    kwargs = {"format": "bgr24"}
    if width is not None and height is not None:
        kwargs["width"] = int(width)
        kwargs["height"] = int(height)
        kwargs["interpolation"] = interpolation or "AREA"
    if not (name.startswith("rgb") or name.startswith("bgr") or name.startswith("gbr") or name.startswith("pal")):
        kwargs["src_colorspace"] = _source_colorspace(frame)
        kwargs["src_color_range"] = _source_range(frame)
        kwargs["dst_color_range"] = ColorRange.JPEG
    return frame.to_ndarray(**kwargs)


class FrameSource:
    def __init__(self, path, cache_bytes=900 * 1024 * 1024):
        self.path = str(path)
        self.container = av.open(self.path)
        if not self.container.streams.video:
            raise MediaError(f"no video stream in {self.path}")
        self.stream = self.container.streams.video[0]
        self.stream.codec_context.thread_type = "AUTO"
        self.time_base = self.stream.time_base
        self.width = self.stream.codec_context.width
        self.height = self.stream.codec_context.height
        stamps = []
        for packet in self.container.demux(self.stream):
            if packet.size == 0 or packet.is_discard:
                continue
            stamp = packet.pts if packet.pts is not None else packet.dts
            if stamp is None:
                continue
            stamps.append(stamp)
        if not stamps:
            raise MediaError(f"no decodable frames in {self.path}")
        self.pts = np.unique(np.array(stamps, dtype=np.int64))
        self.count = int(len(self.pts))
        self.start_time = float(self.pts[0] * self.time_base)
        self.times = (self.pts - self.pts[0]).astype(np.float64) * float(self.time_base)
        deltas = np.diff(self.times) if self.count > 1 else np.array([1.0 / 24.0])
        median = float(np.median(deltas))
        measured = (self.count - 1) / float(self.times[-1]) if self.count > 1 and self.times[-1] > 0 else 24.0
        declared = self.stream.average_rate or self.stream.guessed_rate
        if declared and abs(float(declared) - measured) / measured < 0.005:
            self.fps = snap_rate(float(declared))
        else:
            self.fps = snap_rate(measured)
        self.constant_rate = bool(np.mean(np.abs(deltas - median) > 0.25 * median) < 0.01)
        self.duration = float(self.times[-1] + 1.0 / float(self.fps))
        frame_bytes = max(1, self.width * self.height * 3)
        self.cache_limit = max(8, int(cache_bytes // frame_bytes))
        self.cache = OrderedDict()
        self.cursor = None
        self.generator = None
        self.pending = None
        self.sequential_gap = 90

    def close(self):
        self.drop_cache()
        self.generator = None
        self.container.close()

    def drop_cache(self):
        self.cache.clear()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def index_at(self, seconds):
        seconds = float(seconds)
        i = int(np.searchsorted(self.times, seconds))
        if i <= 0:
            return 0
        if i >= self.count:
            return self.count - 1
        if abs(self.times[i] - seconds) < abs(self.times[i - 1] - seconds):
            return i
        return i - 1

    def position_at(self, seconds):
        return float(np.interp(float(seconds), self.times, np.arange(self.count, dtype=np.float64)))

    def time_of(self, position):
        return float(np.interp(float(position), np.arange(self.count, dtype=np.float64), self.times))

    def _restart(self, index):
        self.container.seek(int(self.pts[index]), stream=self.stream, backward=True, any_frame=False)
        self.generator = self.container.decode(self.stream)
        self.cursor = None
        self.pending = None

    def _seek(self, index):
        for back in (0.0, 1.0, 4.0, 12.0):
            self._restart(self.index_at(max(0.0, self.times[index] - back)))
            first = self._pull()
            if first is not None and first[0] <= index:
                self.pending = first
                return
        self._restart(0)
        self.pending = self._pull()

    def _pull(self):
        while self.generator is not None:
            try:
                frame = next(self.generator)
            except (StopIteration, av.error.EOFError):
                self.generator = None
                self.cursor = None
                return None
            stamp = frame.pts if frame.pts is not None else frame.dts
            if stamp is None:
                continue
            position = int(np.searchsorted(self.pts, stamp))
            if position >= self.count or self.pts[position] != stamp:
                continue
            self.cursor = position + 1
            return position, frame
        return None

    def _store(self, position, frame):
        if position in self.cache:
            self.cache.move_to_end(position)
            return
        image = frame_to_bgr(frame)
        image.flags.writeable = False
        self.cache[position] = image
        while len(self.cache) > self.cache_limit:
            self.cache.popitem(last=False)

    def get(self, index):
        index = int(min(max(index, 0), self.count - 1))
        cached = self.cache.get(index)
        if cached is not None:
            self.cache.move_to_end(index)
            return cached
        sequential = self.generator is not None and self.cursor is not None and 0 <= index - self.cursor <= self.sequential_gap
        if sequential:
            keep_from = self.cursor
            self.pending = None
        else:
            keep_from = index
            self._seek(index)
        last = None
        while True:
            item = self.pending if self.pending is not None else self._pull()
            self.pending = None
            if item is None:
                break
            position, frame = item
            if position >= keep_from:
                self._store(position, frame)
                last = position
            if position >= index:
                break
        if index in self.cache:
            return self.cache[index]
        if last is not None and last in self.cache:
            return self.cache[last]
        raise MediaError(f"could not decode frame {index} of {self.path}")

    def get_at(self, seconds):
        return self.get(self.index_at(seconds))

    def sequence(self):
        container = av.open(self.path)
        stream = container.streams.video[0]
        stream.codec_context.thread_type = "AUTO"
        try:
            for frame in container.decode(stream):
                stamp = frame.pts if frame.pts is not None else frame.dts
                if stamp is None:
                    continue
                position = int(np.searchsorted(self.pts, stamp))
                if position >= self.count or self.pts[position] != stamp:
                    continue
                yield position, frame
        finally:
            container.close()


class SourcePool:
    def __init__(self, cache_bytes=900 * 1024 * 1024):
        self.cache_bytes = cache_bytes
        self.sources = {}
        self.active = None

    def get(self, path):
        key = str(path)
        if key not in self.sources:
            self.sources[key] = FrameSource(key, self.cache_bytes)
        if self.active is not None and self.active != key and self.active in self.sources:
            self.sources[self.active].drop_cache()
        self.active = key
        return self.sources[key]

    def close(self):
        for source in self.sources.values():
            source.close()
        self.sources = {}
        self.active = None


def extract_wav(source, target, sample_rate=48000, channels=2, aligned=False):
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", source, "-vn", "-map", "0:a:0", "-ac", channels, "-ar", sample_rate]
    if aligned:
        cmd += ["-af", "aresample=async=1:first_pts=0"]
    cmd += ["-c:a", "pcm_f32le", target]
    run(cmd)
    return target


def read_wav(path):
    import soundfile

    data, rate = soundfile.read(str(path), dtype="float32", always_2d=True)
    return data, int(rate)


def write_wav(path, data, rate):
    import soundfile

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    soundfile.write(str(path), data, int(rate), subtype="PCM_24")


class VideoWriter:
    def __init__(self, path, width, height, fps, crf=16, preset="slow", audio=None, audio_bitrate="256k", sample_rate=48000):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.width = int(width)
        self.height = int(height)
        self.log = tempfile.NamedTemporaryFile(prefix="starkedit_ffmpeg_", suffix=".log", delete=False)
        rate = rate_text(fps)
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{self.width}x{self.height}", "-framerate", rate, "-i", "pipe:0",
        ]
        if audio is not None:
            cmd += ["-i", str(audio)]
        cmd += [
            "-map", "0:v:0",
        ]
        if audio is not None:
            cmd += ["-map", "1:a:0"]
        cmd += [
            "-vf", "scale=in_range=pc:out_range=tv:out_color_matrix=bt709:flags=bicubic+accurate_rnd+full_chroma_int,format=yuv420p",
            "-c:v", "libx264", "-preset", str(preset), "-crf", str(crf), "-profile:v", "high", "-pix_fmt", "yuv420p",
            "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709", "-color_range", "tv",
            "-fps_mode", "cfr", "-r", rate,
        ]
        if audio is not None:
            cmd += ["-c:a", "aac", "-b:a", str(audio_bitrate), "-ar", str(sample_rate), "-ac", "2"]
        cmd += ["-movflags", "+faststart", str(self.path)]
        self.process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=self.log)
        self.frames = 0

    def write(self, frame):
        if frame.shape[0] != self.height or frame.shape[1] != self.width:
            raise MediaError(f"frame size {frame.shape[1]}x{frame.shape[0]} does not match {self.width}x{self.height}")
        try:
            self.process.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())
        except BrokenPipeError:
            self.close()
        self.frames += 1

    def close(self):
        if self.process.stdin and not self.process.stdin.closed:
            self.process.stdin.close()
        code = self.process.wait()
        self.log.close()
        text = Path(self.log.name).read_text(errors="replace")
        Path(self.log.name).unlink(missing_ok=True)
        if code != 0:
            raise MediaError(f"ffmpeg encode failed ({code}): {text[-1500:]}")
        return self.path
