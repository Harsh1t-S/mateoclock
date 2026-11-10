from pathlib import Path

import cv2
import numpy as np

FONT = cv2.FONT_HERSHEY_SIMPLEX
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
YELLOW = (0, 230, 255)
CYAN = (255, 230, 0)
MAGENTA = (255, 0, 255)
GREEN = (80, 255, 80)
RED = (60, 60, 255)


def save(path, image, quality=90):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() in (".jpg", ".jpeg"):
        cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    else:
        cv2.imwrite(str(path), image)
    return path


def fit(image, width=None, height=None):
    h, w = image.shape[:2]
    if width is None and height is None:
        return image
    if width is not None and height is not None:
        scale = min(width / w, height / h)
    elif width is not None:
        scale = width / w
    else:
        scale = height / h
    size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
    mode = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    return cv2.resize(image, size, interpolation=mode)


def text(image, message, origin=(6, 18), scale=0.5, color=WHITE, thickness=1, background=True):
    x, y = int(origin[0]), int(origin[1])
    if background:
        (tw, th), base = cv2.getTextSize(message, FONT, scale, thickness)
        cv2.rectangle(image, (x - 3, y - th - 4), (x + tw + 3, y + base + 2), BLACK, -1)
    cv2.putText(image, message, (x, y), FONT, scale, color, thickness, cv2.LINE_AA)
    return image


def crosshair(image, x, y, radius=20, color=YELLOW, thickness=1, ring=None):
    cx, cy = int(round(x)), int(round(y))
    r = int(round(radius))
    cv2.line(image, (cx - r, cy), (cx + r, cy), color, thickness, cv2.LINE_AA)
    cv2.line(image, (cx, cy - r), (cx, cy + r), color, thickness, cv2.LINE_AA)
    if ring:
        cv2.circle(image, (cx, cy), int(round(ring)), color, thickness, cv2.LINE_AA)
    return image


def guides(image, x, y, color=YELLOW, thickness=1):
    h, w = image.shape[:2]
    cv2.line(image, (0, int(round(y))), (w - 1, int(round(y))), color, thickness, cv2.LINE_AA)
    cv2.line(image, (int(round(x)), 0), (int(round(x)), h - 1), color, thickness, cv2.LINE_AA)
    return image


def pad_to(image, width, height, fill=18):
    h, w = image.shape[:2]
    canvas = np.full((height, width, 3), fill, dtype=np.uint8)
    y0 = (height - h) // 2
    x0 = (width - w) // 2
    canvas[y0 : y0 + h, x0 : x0 + w] = image[: height - y0, : width - x0]
    return canvas


def sheet(tiles, columns, gap=4, fill=18):
    if not tiles:
        return np.full((32, 32, 3), fill, dtype=np.uint8)
    th = max(t.shape[0] for t in tiles)
    tw = max(t.shape[1] for t in tiles)
    rows = (len(tiles) + columns - 1) // columns
    out = np.full((rows * th + (rows + 1) * gap, columns * tw + (columns + 1) * gap, 3), fill, dtype=np.uint8)
    for i, tile in enumerate(tiles):
        r, c = divmod(i, columns)
        y = gap + r * (th + gap)
        x = gap + c * (tw + gap)
        out[y : y + th, x : x + tw] = pad_to(tile, tw, th, fill)
    return out


def hstack(images, gap=4, fill=18):
    height = max(i.shape[0] for i in images)
    parts = []
    for n, image in enumerate(images):
        parts.append(pad_to(image, image.shape[1], height, fill))
        if n < len(images) - 1:
            parts.append(np.full((height, gap, 3), fill, dtype=np.uint8))
    return np.hstack(parts)


def vstack(images, gap=4, fill=18):
    width = max(i.shape[1] for i in images)
    parts = []
    for n, image in enumerate(images):
        parts.append(pad_to(image, width, image.shape[0], fill))
        if n < len(images) - 1:
            parts.append(np.full((gap, width, 3), fill, dtype=np.uint8))
    return np.vstack(parts)


def coordinate_grid(image, origin=(0.0, 0.0), zoom=1.0, step=100, label_every=1, color=(120, 255, 120)):
    out = image.copy()
    h, w = out.shape[:2]
    ox, oy = origin
    first_x = int(np.ceil(ox / step)) * step
    first_y = int(np.ceil(oy / step)) * step
    overlay = out.copy()
    n = 0
    value = first_x
    while (value - ox) * zoom < w:
        px = int(round((value - ox) * zoom))
        cv2.line(overlay, (px, 0), (px, h - 1), color, 1)
        if n % label_every == 0:
            text(out, str(int(value)), (px + 3, 14), 0.42, color)
            text(out, str(int(value)), (px + 3, h - 6), 0.42, color)
        value += step
        n += 1
    n = 0
    value = first_y
    while (value - oy) * zoom < h:
        py = int(round((value - oy) * zoom))
        cv2.line(overlay, (0, py), (w - 1, py), color, 1)
        if n % label_every == 0:
            text(out, str(int(value)), (3, py - 4 if py > 20 else py + 14), 0.42, color)
            text(out, str(int(value)), (w - 44, py - 4 if py > 20 else py + 14), 0.42, color)
        value += step
        n += 1
    blended = cv2.addWeighted(overlay, 0.45, out, 0.55, 0)
    mask = np.any(out != image, axis=2)
    blended[mask] = out[mask]
    return blended


def onion(a, b):
    return cv2.addWeighted(a, 0.5, b, 0.5, 0)


def edge_overlay(a, b):
    ga = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    gb = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    ea = cv2.Canny(cv2.GaussianBlur(ga, (0, 0), 1.5), 40, 110)
    eb = cv2.Canny(cv2.GaussianBlur(gb, (0, 0), 1.5), 40, 110)
    base = (cv2.addWeighted(a, 0.5, b, 0.5, 0) * 0.35).astype(np.uint8)
    base[ea > 0] = (60, 60, 255)
    base[eb > 0] = (255, 220, 0)
    both = (ea > 0) & (eb > 0)
    base[both] = (255, 255, 255)
    return base
