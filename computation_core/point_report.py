"""Everything the Moon Map says about one chosen point on one chosen date, from real data.

Sun and Earth directions come from NASA SPICE (DE440). Inside the south-pole LOLA grid (85S to 90S) the terrain
is real at 40 m. Elsewhere the coarser whole-Moon LOLA grid (64 px/deg) gives regional slope and skyline. Without
either, the report says so and treats the ground as a smooth sphere (no slope, no hiding ridges).
"""
import numpy as np

from data_engine import spice_loader as sl

from .constants import MOON_RADIUS_M
from .coordinates import enu_basis, unit_to_polar_xy
from .earth_visibility import dte_visible
from .horizon import HorizonProfile, compute_horizon
from .solar_illumination import lit_fraction

SLOPE_BASELINE_M = 80.0  # slope = height change over +/- this distance in each direction


def has_terrain(dem, lat, lon) -> bool:
    return lat <= -85.0 and not np.isnan(float(dem.height_at_latlon(lat, lon % 360.0)))


def _height_at_offset(dem, lat, lon, direction, distance_m):
    _, _, up = enu_basis(lat, lon)
    d = distance_m / MOON_RADIUS_M
    q = up * np.cos(d) + direction * np.sin(d)
    x, y = unit_to_polar_xy(q[None, :])
    return float(dem.height_nearest(x, y)[0])


def surface(dem, lat, lon) -> dict | None:
    """Height, slope and downhill direction at the point (None without terrain data)."""
    if not has_terrain(dem, lat, lon):
        return None
    east, north, _ = enu_basis(lat, lon)
    b = SLOPE_BASELINE_M
    h = lambda v, s: _height_at_offset(dem, lat, lon, v, s * b)  # noqa: E731
    g_e = (h(east, 1) - h(east, -1)) / (2 * b)
    g_n = (h(north, 1) - h(north, -1)) / (2 * b)
    if np.isnan(g_e) or np.isnan(g_n):
        return None
    return {
        "height_m": float(dem.height_at_latlon(lat, lon % 360.0)),
        "slope_deg": float(np.degrees(np.arctan(np.hypot(g_e, g_n)))),
        "downhill_azimuth_deg": float(np.degrees(np.arctan2(-g_e, -g_n)) % 360.0),
        "grad_east": float(g_e), "grad_north": float(g_n),
        "baseline_m": SLOPE_BASELINE_M,
    }


def _incidence_on_slope(az, el, surf) -> float:
    """Angle between the Sun's rays and the surface normal; above 90 the ground faces away from the Sun."""
    a, e = np.radians(az), np.radians(el)
    sun = np.array([np.cos(e) * np.sin(a), np.cos(e) * np.cos(a), np.sin(e)])  # east, north, up
    n = np.array([-surf["grad_east"], -surf["grad_north"], 1.0])
    n /= np.linalg.norm(n)
    return float(np.degrees(np.arccos(np.clip(sun @ n, -1.0, 1.0))))


def point_report(dem, lat, lon, utc, observer_height_m=2.0, gdem=None):
    spice = sl
    et0 = spice.utc_to_et(utc)
    polar = has_terrain(dem, lat, lon)
    terrain = polar or gdem is not None
    if polar:
        horizon = compute_horizon(dem, lat, lon % 360.0, observer_height_m)
    elif gdem is not None:
        horizon = gdem.horizon(lat, lon, observer_height_m)
    else:
        az = np.linspace(0.0, 360.0, 720, endpoint=False)
        horizon = HorizonProfile(az, np.zeros_like(az), None)
    surf = surface(dem, lat, lon) if polar else (gdem.surface(lat, lon) if gdem is not None else None)

    day0 = spice.utc_to_et(utc[:10] + "T00:00:00")  # the chosen date, from 00:00 UTC
    et = np.concatenate([[et0], day0 + np.arange(0, 25) * 3600.0])
    s_az, s_el, s_d = spice.topocentric("SUN", lat, lon, et)
    e_az, e_el, e_d = spice.topocentric("EARTH", lat, lon, et)
    frac = lit_fraction(s_az, s_el, s_d, horizon)
    link = dte_visible(e_az, e_el, horizon)

    sun_h = float(horizon.at(s_az[0]))
    earth_h = float(horizon.at(e_az[0]))
    sun = {
        "azimuth_deg": float(s_az[0]), "elevation_deg": float(s_el[0]),
        "incidence_flat_deg": float(90.0 - s_el[0]),  # from the local vertical (a flat floor)
        "incidence_slope_deg": _incidence_on_slope(s_az[0], s_el[0], surf) if surf else None,
        "skyline_deg": sun_h, "lit_fraction": float(frac[0]), "lit": bool(frac[0] >= 0.5),
        "distance_km": float(s_d[0]),
    }
    earth = {
        "azimuth_deg": float(e_az[0]), "elevation_deg": float(e_el[0]), "distance_km": float(e_d[0]),
        "skyline_deg": earth_h, "visible": bool(link[0]),
        "below_horizon": bool(e_el[0] <= 0.0),
        "blocked_by_terrain": bool(e_el[0] > 0.0 and not link[0]),
        "margin_deg": float(e_el[0] - max(earth_h, 0.0)),
        "blocker_distance_m": (None if not terrain or np.isnan(horizon.blocker_distance_m(e_az[0]))
                               else float(horizon.blocker_distance_m(e_az[0]))),
    }
    day = slice(1, 25)  # 00:00 .. 23:00 UTC of the chosen date
    return {
        "utc": utc,
        "lat": lat, "lon": lon, "terrain": "lola" if polar else ("global" if gdem is not None else "smooth"),
        "surface": ({k: v for k, v in surf.items() if not k.startswith("grad")} if surf else None),
        "sun": sun, "earth": earth,
        "day": {
            "hours_utc": list(range(24)),
            "sun_elevation_deg": s_el[day].tolist(), "sun_lit": (frac[day] >= 0.5).tolist(),
            "earth_elevation_deg": e_el[day].tolist(), "earth_visible": link[day].tolist(),
            "lit_hours": int((frac[day] >= 0.5).sum()), "link_hours": int(link[day].sum()),
        },
    }
