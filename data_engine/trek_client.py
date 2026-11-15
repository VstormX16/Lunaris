"""NASA Moon Trek WMTS basemap: tile addressing for the south-polar stereographic layer.

Verified live (HTTP 200, image/png, CORS `*`, levels 0-9) - the browser can load tiles directly.
The layer uses the same south-polar stereographic metres as our LOLA DEM grid.
"""
import math

from computation_core.coordinates import polar_stereo_forward

_TREK = "https://trek.nasa.gov/tiles/Moon/SP"
LAYERS = {
    "lola_color_shade": {
        "title": "LRO LOLA color shaded relief, south pole (87.5S-90S, 5 m/px source)",
        "id": "LRO_LOLA_ClrShade_SPole875_5mp_v04",
        "format": "png",
        "top_left_m": (-1095930.0, 1095930.0),
        "scale_denominator_0": 3.0578404017857101e07,
        "max_level": 9,
        "extent_m": 75840.0,  # data only within |x|, |y| <= 75.84 km of the pole
    },
}
_PIXEL_MM = 0.28e-3  # OGC standardized rendering pixel
TILE_PX = 256


def resolution_m(layer, z):
    return LAYERS[layer]["scale_denominator_0"] / 2 ** z * _PIXEL_MM


def tile_index(layer, z, x_m, y_m):
    """(row, col) of the tile containing projected point (x, y) at level z."""
    x0, y0 = LAYERS[layer]["top_left_m"]
    span = TILE_PX * resolution_m(layer, z)
    return int(math.floor((y0 - y_m) / span)), int(math.floor((x_m - x0) / span))


def tile_url(layer, z, row, col):
    L = LAYERS[layer]
    if not 0 <= z <= L["max_level"]:
        raise ValueError(f"level must be 0..{L['max_level']}")
    return f"{_TREK}/{L['id']}/1.0.0/default/default028mm/{z}/{row}/{col}.{L['format']}"


def tile_for_latlon(layer, z, lat_deg, lon_deg):
    x, y = polar_stereo_forward(lat_deg, lon_deg)
    row, col = tile_index(layer, z, float(x), float(y))
    return {"row": row, "col": col, "url": tile_url(layer, z, row, col)}


def basemap_config():
    """Everything a map widget needs to address the tiles itself."""
    return {name: {**{k: v for k, v in L.items()},
                   "url_template": f"{_TREK}/{L['id']}/1.0.0/default/default028mm/{{z}}/{{row}}/{{col}}.{L['format']}",
                   "resolutions_m": [resolution_m(name, z) for z in range(L["max_level"] + 1)],
                   "tile_px": TILE_PX}
            for name, L in LAYERS.items()}
