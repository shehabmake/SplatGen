"""Draw the SplatGen mark - two rotated ellipses - as an RGBA PNG.

Pure stdlib: the pixels are computed here and zlib writes the PNG, so the
generator needs nothing Blender does not already ship.
"""

import math
import struct
import sys
import zlib
from pathlib import Path

SIZE = 256
SS = 4  # supersampling factor per axis, for smooth edges

ORANGE = (255, 85, 0)
BLUE = (15, 53, 232)

#: (centre x, centre y, semi-major, semi-minor, rotation degrees, colour)
#: Both ellipses share a tilt; the blue one sits lower-left and draws on top,
#: which is what gives the mark its interlocking look.
SHAPES = (
    (0.525, 0.330, 0.400, 0.215, -28.0, ORANGE),
    (0.475, 0.670, 0.400, 0.215, -28.0, BLUE),
)


def coverage(shape, x, y):
    """1.0 inside the ellipse, 0.0 outside, sampled at a point in 0..1."""
    cx, cy, a, b, angle, _colour = shape
    radians = math.radians(angle)
    cos_a, sin_a = math.cos(radians), math.sin(radians)
    dx, dy = x - cx, y - cy
    u = dx * cos_a + dy * sin_a
    v = -dx * sin_a + dy * cos_a
    return 1.0 if (u / a) ** 2 + (v / b) ** 2 <= 1.0 else 0.0


def render():
    """Supersampled RGBA rows, blue composited over orange."""
    rows = []
    for py in range(SIZE):
        row = bytearray()
        for px in range(SIZE):
            acc = [0.0, 0.0, 0.0, 0.0]
            for sy in range(SS):
                for sx in range(SS):
                    x = (px + (sx + 0.5) / SS) / SIZE
                    y = (py + (sy + 0.5) / SS) / SIZE
                    # Painter's order: later shapes cover earlier ones.
                    sample = None
                    for shape in SHAPES:
                        if coverage(shape, x, y) > 0.0:
                            sample = shape[5]
                    if sample is not None:
                        acc[0] += sample[0]
                        acc[1] += sample[1]
                        acc[2] += sample[2]
                        acc[3] += 255.0
            taken = SS * SS
            alpha = acc[3] / taken
            if alpha <= 0.0:
                row += bytes((0, 0, 0, 0))
                continue
            # Un-premultiply so partly covered edge pixels keep their hue.
            scale = acc[3] / 255.0
            row += bytes((
                min(255, round(acc[0] / scale)),
                min(255, round(acc[1] / scale)),
                min(255, round(acc[2] / scale)),
                min(255, round(alpha)),
            ))
        rows.append(bytes(row))
    return rows


def write_png(path, rows):
    raw = b"".join(b"\x00" + row for row in rows)

    def chunk(tag, payload):
        body = tag + payload
        return (struct.pack(">I", len(payload)) + body
                + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF))

    header = struct.pack(">IIBBBBB", SIZE, SIZE, 8, 6, 0, 0, 0)
    png = (b"\x89PNG\r\n\x1a\n"
           + chunk(b"IHDR", header)
           + chunk(b"IDAT", zlib.compress(raw, 9))
           + chunk(b"IEND", b""))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(png)
    return len(png)


if __name__ == "__main__":
    target = Path(sys.argv[1])
    size = write_png(target, render())
    print(f"wrote {target} ({size:,} bytes, {SIZE}x{SIZE})")
