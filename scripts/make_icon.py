#!/usr/bin/env python3
"""Draw the AnyRAG app icon with no image library (stdlib PNG writer).

  .venv/bin/python scripts/make_icon.py

Writes assets/anyrag.png (1024 px) and, on macOS, assets/anyrag.icns via
sips + iconutil (anyrag.spec uses the .icns when it exists). The design: a
teal rounded square, a white document page with text lines, and a
magnifying glass over its lower right corner.
"""
import math
import shutil
import struct
import subprocess
import sys
import tempfile
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "assets"
N = 1024
SS = 2                                        # supersampling per axis


def _rrect(x, y, x0, y0, x1, y1, r):
    """Signed distance to a rounded rectangle (negative inside)."""
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    hx, hy = (x1 - x0) / 2 - r, (y1 - y0) / 2 - r
    dx, dy = abs(x - cx) - hx, abs(y - cy) - hy
    out = math.hypot(max(dx, 0), max(dy, 0))
    return out + min(max(dx, dy), 0) - r


def _seg(x, y, ax, ay, bx, by):
    px, py, vx, vy = x - ax, y - ay, bx - ax, by - ay
    t = max(0.0, min(1.0, (px * vx + py * vy) / (vx * vx + vy * vy)))
    return math.hypot(px - vx * t, py - vy * t)


def _mix(c, d, t):
    return tuple(a + (b - a) * t for a, b in zip(c, d))


TEAL_TOP, TEAL_BOT = (32, 160, 160), (14, 92, 110)
PAPER, LINE, RING = (250, 252, 252), (150, 196, 200), (255, 196, 64)
GLASS = (210, 240, 238)


def color_at(x, y):
    """RGBA (0-255 floats) of one sample point in a 1024 grid."""
    if _rrect(x, y, 64, 64, 960, 960, 200) > 0:      # outside the tile
        return (0, 0, 0, 0)
    col = _mix(TEAL_TOP, TEAL_BOT, (y - 64) / 896)
    # page shadow, page
    if _rrect(x, y, 262, 222, 722, 842, 40) < 0:
        col = _mix(col, (0, 0, 0), 0.18)
    if _rrect(x, y, 250, 200, 710, 820, 40) < 0:
        col = PAPER
        for i, (a, b) in enumerate(((320, 640), (320, 600), (320, 640),
                                    (320, 520), (320, 460))):
            ly = 300 + i * 82
            if _seg(x, y, a, ly, b, ly) < 16:
                col = LINE
    # magnifying glass: handle, ring, lens
    cx, cy, r = 640, 640, 150
    if _seg(x, y, 765, 765, 860, 860) < 38:
        col = RING
    d = math.hypot(x - cx, y - cy)
    if d < r - 26:
        col = _mix(col, GLASS, 0.55)
    elif d < r + 26:
        col = RING
    return (*col, 255)


def render(n=N):
    rows = []
    step = N / n
    for py in range(n):
        row = bytearray([0])                       # PNG filter: none
        for px in range(n):
            acc = [0.0, 0.0, 0.0, 0.0]
            for sy in range(SS):
                for sx in range(SS):
                    r, g, b, a = color_at((px + (sx + .5) / SS) * step,
                                          (py + (sy + .5) / SS) * step)
                    acc[0] += r * a; acc[1] += g * a; acc[2] += b * a
                    acc[3] += a
            a = acc[3]
            if a:
                row += bytes((round(acc[0] / a), round(acc[1] / a),
                              round(acc[2] / a), round(a / (SS * SS))))
            else:
                row += b"\0\0\0\0"
        rows.append(bytes(row))
    return b"".join(rows)


def write_png(path, raw, n=N):
    def chunk(tag, data):
        c = struct.pack(">I", len(data)) + tag + data
        return c + struct.pack(">I", zlib.crc32(tag + data) & 0xffffffff)
    png = (b"\x89PNG\r\n\x1a\n"
           + chunk(b"IHDR", struct.pack(">IIBBBBB", n, n, 8, 6, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))
    path.write_bytes(png)


def make_icns(png, icns):
    sizes = [16, 32, 64, 128, 256, 512, 1024]
    with tempfile.TemporaryDirectory() as tmp:
        iconset = Path(tmp) / "anyrag.iconset"
        iconset.mkdir()
        for s in sizes:
            for scale, name in ((1, f"icon_{s}x{s}.png"),
                                (2, f"icon_{s // 2}x{s // 2}@2x.png")):
                px = s if scale == 1 else s
                if scale == 2 and s < 32:
                    continue
                if scale == 1 and s == 1024:
                    continue
                subprocess.run(["sips", "-z", str(px), str(px), str(png),
                                "--out", str(iconset / name)],
                               check=True, capture_output=True)
        subprocess.run(["iconutil", "-c", "icns", str(iconset), "-o",
                        str(icns)], check=True)


def main():
    OUT.mkdir(exist_ok=True)
    png = OUT / "anyrag.png"
    print("drawing 1024 px icon …")
    write_png(png, render())
    print(f"  -> {png}")
    if sys.platform == "darwin" and shutil.which("iconutil"):
        make_icns(png, OUT / "anyrag.icns")
        print(f"  -> {OUT / 'anyrag.icns'}")
    else:
        print("(not macOS: .icns skipped)")


if __name__ == "__main__":
    main()
