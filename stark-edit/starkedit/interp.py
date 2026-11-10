import cv2
import numpy as np


class FlowInterpolator:
    def __init__(self, work_scale=0.5):
        self.work_scale = float(work_scale)
        self.engine = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        self.key = None
        self.pair = None

    def _flows(self, first, second, key):
        if key is not None and key == self.key:
            return self.pair
        scale = self.work_scale
        a = cv2.cvtColor(cv2.resize(first, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        b = cv2.cvtColor(cv2.resize(second, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        forward = self.engine.calc(a, b, None)
        backward = self.engine.calc(b, a, None)
        size = (first.shape[1], first.shape[0])
        forward = cv2.resize(forward, size, interpolation=cv2.INTER_LINEAR) / scale
        backward = cv2.resize(backward, size, interpolation=cv2.INTER_LINEAR) / scale
        height, width = first.shape[:2]
        grid_x, grid_y = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
        self.key = key
        self.pair = (forward, backward, grid_x, grid_y)
        return self.pair

    @staticmethod
    def _project(flow, grid_x, grid_y, amount):
        guess = flow
        for _ in range(3):
            guess = cv2.remap(flow, grid_x - amount * guess[..., 0], grid_y - amount * guess[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        return guess

    def between(self, first, second, fraction, key=None):
        fraction = float(fraction)
        if fraction <= 0.0:
            return first
        if fraction >= 1.0:
            return second
        forward, backward, grid_x, grid_y = self._flows(first, second, key)
        from_first = self._project(forward, grid_x, grid_y, fraction)
        from_second = self._project(backward, grid_x, grid_y, 1.0 - fraction)
        map_ax = grid_x - fraction * from_first[..., 0]
        map_ay = grid_y - fraction * from_first[..., 1]
        map_bx = grid_x - (1.0 - fraction) * from_second[..., 0]
        map_by = grid_y - (1.0 - fraction) * from_second[..., 1]
        warped_a = cv2.remap(first, map_ax, map_ay, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        warped_b = cv2.remap(second, map_bx, map_by, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        back_at_a = cv2.remap(backward, map_ax + from_first[..., 0], map_ay + from_first[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        forward_at_b = cv2.remap(forward, map_bx + from_second[..., 0], map_by + from_second[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        error_a = np.hypot(from_first[..., 0] + back_at_a[..., 0], from_first[..., 1] + back_at_a[..., 1])
        error_b = np.hypot(from_second[..., 0] + forward_at_b[..., 0], from_second[..., 1] + forward_at_b[..., 1])
        weight_a = (1.0 - fraction) * np.exp(-((error_a / 3.0) ** 2)) + 1e-4
        weight_b = fraction * np.exp(-((error_b / 3.0) ** 2)) + 1e-4
        total = weight_a + weight_b
        mixed = warped_a.astype(np.float32) * (weight_a / total)[..., None] + warped_b.astype(np.float32) * (weight_b / total)[..., None]
        return np.clip(mixed + 0.5, 0, 255).astype(np.uint8)


def blend(first, second, fraction):
    return cv2.addWeighted(first, 1.0 - float(fraction), second, float(fraction), 0)
