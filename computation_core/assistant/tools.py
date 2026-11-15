"""The assistant's hands: every number it says comes from these functions, i.e. from the real engine (SPICE + LOLA).

Scans are made cheap by computing one long time series per site (the skyline and the ephemeris once) and sliding the
30-day window over it, instead of re-running the whole analysis for every candidate date.
"""
from dataclasses import dataclass
from datetime import date, datetime, timedelta

import numpy as np

from ..earth_visibility import dte_stats
from ..mission_risk import mission_risk
from ..verdict import decide
from ..site_analysis import analyze_site, site_timeseries
from ..solar_illumination import illumination_stats
from ..terrain_slope import site_hazard_stats
from ..vehicle_assessment import assess_vehicle

WINDOW_DAYS = 30
SITE_NAMES = {"shackleton": (-89.54, 0.0), "haworth": (-87.45, -4.5), "malapert": (-85.987, 2.08), "faustini": (-87.3, 84.0), "nobile": (-85.2, 53.5)}


@dataclass
class Scenario:
    date: str          # ISO yyyy-mm-dd, the first day of the 30-day window
    vehicle_id: str
    lat: float
    lon: float

    def start_utc(self):
        return f"{self.date}T00:00:00"


def _add_days(iso: str, n: int) -> str:
    return (datetime.fromisoformat(iso) + timedelta(days=n)).date().isoformat()


def assess(dem, profile, sc: Scenario):
    """The full engine on one scenario: window metrics (with Sun height and when the long night / outage start) + vehicle
    assessment, with the mission success probability and the call (the same ones the page shows)."""
    (iso, m, series), = _windows(dem, sc, sc.date, 1)
    a = assess_vehicle(profile, m, dem, sc.lat, sc.lon, series=series)
    a["risk"] = mission_risk(profile, dem, sc.lat, sc.lon, sc.start_utc(), WINDOW_DAYS, 60)
    a["verdict"] = decide(a, m, a["risk"])
    return m, a


def hours_to_iso(start_iso: str, hours):
    """The UTC moment `hours` after midnight of start_iso, as 'yyyy-mm-dd HH:00'; None passes through."""
    if hours is None:
        return None
    return (datetime.fromisoformat(start_iso) + timedelta(hours=float(hours))).strftime("%Y-%m-%d %H:00")


def _row(a, m, iso):
    return {"date": iso, "index": a["vehicle_index"], "weighted": a["weighted_index"], "limit": a["limiting_factor"],
            "sun": m["illumination_fraction"], "dte": m["dte_fraction"], "dark_h": m["longest_darkness_hours"], "outage_h": m["longest_outage_hours"],
            "slope": m["mean_slope_deg"], "sun_el": m["sun_el_mean_lit"], "sun_el_max": m["sun_el_max"]}


def _longest_false_run_start(ok):
    """Index where the longest run of False begins (hours from the window start), or None when there is none."""
    best, start, cur = (0, None), None, 0
    for i, v in enumerate(np.append(ok, True)):
        if not v:
            start = i if cur == 0 else start
            cur += 1
        else:
            if cur > best[0]:
                best = (cur, start)
            cur = 0
    return best[1]


def _sun_elevation_stats(ts, sl):
    """Mean Sun elevation over the hours it is lit, and its highest value, in a window (degrees above the local horizon)."""
    el, lit = np.asarray(ts["sun_el"])[sl], np.asarray(ts["sun_lit_fraction"])[sl] >= 0.5
    return {"sun_el_mean_lit": float(el[lit].mean()) if lit.any() else 0.0, "sun_el_max": float(el.max())}


def _windows(dem, sc: Scenario, start_iso: str, n_starts: int, step_days: int = 1):
    """Metrics for n_starts consecutive 30-day windows beginning at start_iso, from ONE long series."""
    total_days = (n_starts - 1) * step_days + WINDOW_DAYS
    ts = site_timeseries(dem, sc.lat, sc.lon, f"{start_iso}T00:00:00", total_days, 60)
    hazard = site_hazard_stats(dem, sc.lat, sc.lon, 500.0)
    out, n = [], WINDOW_DAYS * 24
    for k in range(n_starts):
        a = k * step_days * 24
        sl = slice(a, a + n)
        iso = _add_days(start_iso, k * step_days)
        m = {"site": {"lat": sc.lat, "lon": sc.lon}, "window": {"start_utc": f"{iso}T00:00:00", "days": WINDOW_DAYS, "step_minutes": 60},
             **illumination_stats(ts["sun_lit_fraction"][sl], 1.0), **dte_stats(ts["earth_visible"][sl], 1.0), **hazard, **_sun_elevation_stats(ts, sl),
             "dark_start_h": _longest_false_run_start(ts["sun_lit_fraction"][sl] >= 0.5), "outage_start_h": _longest_false_run_start(ts["earth_visible"][sl])}
        out.append((iso, m, {"sun_lit_fraction": ts["sun_lit_fraction"][sl]}))
    return out


def date_scan(dem, profile, sc: Scenario, span_days: int = 60, step_days: int = 1, earliest: str | None = None):
    """Score for every start date from sc.date - span to sc.date + span (the rest of the scenario unchanged).

    `earliest` stops the scan from suggesting dates that are already in the past."""
    start = _add_days(sc.date, -span_days)
    if earliest and start < earliest:
        start = earliest
    n = (datetime.fromisoformat(_add_days(sc.date, span_days)) - datetime.fromisoformat(start)).days // step_days + 1
    rows = []
    for iso, m, series in _windows(dem, sc, start, n, step_days):
        rows.append(_row(assess_vehicle(profile, m, dem, sc.lat, sc.lon, series=series), m, iso))
    return rows


def vehicle_scan(dem, profiles, sc: Scenario):
    """Every vehicle in the project at the scenario's site and date, from one shared time series."""
    (iso, m, series), = _windows(dem, sc, sc.date, 1)
    rows = []
    for p in profiles:
        a = assess_vehicle(p, m, dem, sc.lat, sc.lon, series=series)
        known = sum(a["data_quality"][k] for k in ("inputs_verified", "inputs_unverified"))
        rows.append({"id": p["id"], "name": p["name"], "index": a["vehicle_index"], "limit": a["limiting_factor"], "known_inputs": known})
    return rows


def _offset(lat, lon, dx_m, dy_m):
    """A point dx_m east and dy_m north of the site, on the sphere (works anywhere on the Moon, not only at the pole)."""
    from ..site_dem import tangent_points
    la, lo = tangent_points(lat, lon, np.array(float(dx_m)), np.array(float(dy_m)))
    return float(la), float(lo) % 360.0


def site_scan(dem, profile, sc: Scenario, radius_m: float = 3000.0, step_m: float = 1500.0):
    """The named candidate sites plus a grid of points around the chosen one, for the scenario's date and vehicle."""
    cands = [("here", sc.lat, sc.lon)]
    k = int(radius_m // step_m)
    for i in range(-k, k + 1):
        for j in range(-k, k + 1):
            if i or j:
                cands.append((f"{step_m * np.hypot(i, j) / 1000:.1f} km {'N' if j > 0 else 'S' if j < 0 else ''}{'E' if i > 0 else 'W' if i < 0 else ''}",) + _offset(sc.lat, sc.lon, i * step_m, j * step_m))
    for name, (la, lo) in SITE_NAMES.items():
        cands.append((name, la, lo % 360.0))
    rows = []
    for label, la, lo in cands:
        if np.isnan(float(dem.height_at_latlon(la, lo))):
            continue
        s2 = Scenario(sc.date, sc.vehicle_id, la, lo)
        (iso, m, series), = _windows(dem, s2, sc.date, 1)
        a = assess_vehicle(profile, m, dem, la, lo, series=series)
        rows.append({"label": label, "lat": la, "lon": lo, **_row(a, m, sc.date), "near": not label[0].isalpha() or label == "here"})
    return rows


def requirements(dem, profiles, sc: Scenario, n_check: int = 8):
    """What a vehicle must have to do well at this place in this window, read off the real Sun / Earth / terrain, and which of
    the project's vehicles already have it (the n_check best by score, each run through the mission simulation)."""
    (iso, m, series), = _windows(dem, sc, sc.date, 1)
    slopes = site_hazard_stats(dem, sc.lat, sc.lon, 500.0, with_slopes=True)["slopes"]
    scan = vehicle_scan(dem, profiles, sc)
    by_id = {p["id"]: p for p in profiles}
    top = sorted(scan, key=lambda r: -r["index"])[:n_check]
    fits = []
    for r in top:
        risk = mission_risk(by_id[r["id"]], dem, sc.lat, sc.lon, sc.start_utc(), WINDOW_DAYS, 60)
        fits.append({"id": r["id"], "name": r["name"], "index": r["index"], "p_success": risk["p_success"], "days_p50": risk["operating_days"]["p50"],
                     "confidence": risk["confidence"]["level"]})
    fits.sort(key=lambda f: (-f["p_success"], -f["index"]))
    return {"date": iso, "sunlit": m["illumination_fraction"], "dte": m["dte_fraction"], "dark_h": m["longest_darkness_hours"],
            "dark_start": hours_to_iso(iso, m["dark_start_h"]), "outage_h": m["longest_outage_hours"], "sun_el_mean": m["sun_el_mean_lit"],
            "sun_el_max": m["sun_el_max"], "slope_p90": float(np.quantile(slopes, 0.9)), "slope_mean": float(np.mean(slopes)), "fits": fits}


def latitude_sweep(sc: Scenario, steps_deg=(0.0, 1.0, 2.0, 3.0, 5.0)):
    """Sun elevation on FLAT ground at the same longitude but nearer the equator (the only real lever on the Sun's height).

    Terrain is not included, so this is the geometry of the Sun alone: how high it climbs and for how much of the window it is up."""
    import numpy as _np
    from data_engine import spice_loader as sl
    et = sl.time_grid(sc.start_utc(), WINDOW_DAYS, 60)
    rows = []
    for d in steps_deg:
        lat = sc.lat + d
        _, el, _ = sl.topocentric("SUN", lat, sc.lon, et)
        up = el > 0
        rows.append({"lat": lat, "offset_deg": d, "el_max": float(el.max()), "el_mean_up": float(el[up].mean()) if up.any() else 0.0, "up_fraction": float(up.mean())})
    return rows


def battery_sweep(dem, profile, sc: Scenario, multipliers=(1, 2, 4, 8)):
    """What a bigger battery buys: the load it can carry through this window, and the score if the survival load is known."""
    from ..vehicle_assessment import sustainable_load_w
    (iso, m, series), = _windows(dem, sc, sc.date, 1)
    bat, sol = profile["battery_wh"], profile["solar_w"]
    if bat["value"] is None or sol["value"] is None:
        return None
    frac = np.asarray(series["sun_lit_fraction"], float)
    rows = []
    for mult in multipliers:
        p = {**profile, "battery_wh": {**bat, "value": bat["value"] * mult}}
        a = assess_vehicle(p, m, dem, sc.lat, sc.lon, series=series)
        rows.append({"mult": mult, "battery_wh": bat["value"] * mult, "sustainable_w": sustainable_load_w(frac, 1.0, bat["value"] * mult, sol["value"]),
                     "index": a["vehicle_index"], "blackout_h": a["checks"]["energy"].get("blackout_hours")})
    return rows
