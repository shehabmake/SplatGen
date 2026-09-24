"""Exact SplatGen identity colors, shared by UI glyphs and overlays."""

ORANGE = '#FF5701'
BLUE = '#043BFD'
YELLOW = '#E1FF01'
RED = '#D3331A'
GREEN = '#6CFF27'
COLORS = (ORANGE, BLUE, YELLOW, RED, GREEN)
NEUTRAL = '#E6E8EC'
ICON_COLORS = (*COLORS, NEUTRAL)


def rgb(value):
    return tuple(int(value[i:i + 2], 16) / 255.0 for i in (1, 3, 5))
