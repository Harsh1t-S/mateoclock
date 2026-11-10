import math
import time

import cv2
import numpy as np

from . import look as grading
from .interp import FlowInterpolator, blend
from .media import SourcePool, VideoWriter


class Renderer:
    def __init__(self, project, resolved, mode="final", pool=None):
        self.project = project
        self.resolved = resolved
        self.mode = mode
        settings = project.cfg["render"][mode]
        self.settings = settings
        self.scale = float(settings.get("scale", 1.0))
        self.width = int(round(resolved.width * self.scale / 2.0)) * 2
        self.height = int(round(resolved.height * self.scale / 2.0)) * 2
        self.pool = pool or SourcePool()
        self.look = grading.Look(project.cfg["look"], self.width, self.height)
        self.flow = FlowInterpolator()
        self.grades = {}
        self.sharp = cv2.INTER_LANCZOS4 if self.scale >= 0.99 else cv2.INTER_LINEAR

    def close(self):
        self.pool.close()

    def source_frame(self, segment, j):
        source = self.pool.get(segment.path)
        position = float(segment.positions[j])
        base = int(math.floor(position + 1e-6))
        fraction = position - base
        mode = segment.interp
        if fraction < 0.1 or mode == "nearest":
            return source.get(int(round(position)))
        if fraction > 0.9:
            return source.get(base + 1)
        if mode == "auto":
            n = len(segment.positions)
            if n > 1:
                step = segment.positions[min(j + 1, n - 1)] - segment.positions[max(j - 1, 0)]
                step /= float(min(j + 1, n - 1) - max(j - 1, 0))
            else:
                step = 1.0
            if step >= 0.95:
                return source.get(int(round(position)))
            mode = "flow"
        first = source.get(base)
        second = source.get(min(base + 1, source.count - 1))
        if mode == "blend":
            return blend(first, second, fraction)
        return self.flow.between(first, second, fraction, key=(segment.path, base))

    def warp(self, segment, j, image):
        matrix = segment.matrices[j] * self.scale
        x0, y0, x1, y1 = [int(round(v)) for v in segment.active]
        crop = image[y0:y1, x0:x1]
        matrix = matrix.copy()
        matrix[:, 2] += matrix[:, :2] @ np.array([x0, y0], dtype=np.float64)
        factor = math.sqrt(abs(matrix[0, 0] * matrix[1, 1] - matrix[0, 1] * matrix[1, 0]))
        flags = self.sharp
        if factor < 0.8:
            crop = cv2.resize(crop, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA)
            matrix[:, :2] /= factor
            flags = cv2.INTER_LINEAR
        return cv2.warpAffine(crop, matrix, (self.width, self.height), flags=flags, borderMode=cv2.BORDER_REPLICATE)

    def plain(self, segment, j):
        return self.warp(segment, j, self.source_frame(segment, j))

    def grade_for(self, segment):
        if segment.id not in self.grades:
            n = segment.frames
            picks = sorted({0, n // 2, n - 1})
            saved = segment.interp
            segment.interp = "nearest"
            samples = [self.plain(segment, j) for j in picks]
            segment.interp = saved
            stats = grading.measure(samples)
            params = grading.solve(stats, self.project.cfg["look"], segment.spec.get("grade"))
            self.grades[segment.id] = (grading.tables(params), stats, params)
        return self.grades[segment.id][0]

    def frame(self, number, graded=True):
        segment = self.resolved.segment_at(number)
        if segment is None or not segment.clip:
            return np.zeros((self.height, self.width, 3), np.uint8)
        j = number - segment.frame_in
        image = self.plain(segment, j)
        if not graded:
            return image
        lut, matrix = self.grade_for(segment)
        return self.look.apply(image, lut, matrix, float(segment.flash[j]), number, finish=self.mode == "final")

    def render(self, target, audio=None, log=print):
        cfg = self.project.cfg["render"]
        writer = VideoWriter(target, self.width, self.height, self.resolved.fps, self.settings["crf"], self.settings["preset"], audio, cfg["audio_bitrate"], cfg["sample_rate"])
        total = self.resolved.total_frames
        started = time.time()
        try:
            for number in range(total):
                writer.write(self.frame(number))
                if log and (number + 1) % 120 == 0:
                    log(f"  rendered {number + 1}/{total} frames ({(number + 1) / (time.time() - started):.1f} fps)")
        finally:
            writer.close()
        if log:
            log(f"  wrote {target} ({total} frames, {time.time() - started:.0f}s)")
        return target

    def grade_report(self):
        return {key: {"stats": {k: round(v, 4) for k, v in value[1].items()}, "params": value[2]} for key, value in self.grades.items()}
