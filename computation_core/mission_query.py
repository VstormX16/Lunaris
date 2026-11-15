"""Mission query: "At this site on this date, what is happening, why, and what comes next?"

Deterministic physics (real SPICE ephemeris + real LOLA horizon) plus explicit
rules that turn numbers into findings. No language model is involved.
"""
import numpy as np

from data_engine import spice_loader

from .site_analysis import site_timeseries
from .timeseries import runs

# Rule thresholds (engineering judgment; tune per mission profile).
DARK_WARN_H, DARK_CRIT_H = 72.0, 240.0       # battery / thermal survival
OUTAGE_WARN_H, OUTAGE_CRIT_H = 24.0, 168.0   # direct-to-Earth comms gap
LOW_LIGHT_FRACTION = 0.2
DEGRADE_DROP = 0.3                           # first-week vs last-week fraction drop


def _instant(az, el, horizon, visible):
    """State of one body at one instant, with the reason when it is not visible."""
    hz = float(horizon.at(az))
    out = {"azimuth_deg": float(az), "elevation_deg": float(el), "terrain_horizon_deg": hz}
    if visible:
        out["state"] = "visible"
    elif el <= 0.0 and el <= hz:
        out["state"] = "below_local_horizon"
    else:
        out["state"] = "terrain_blocked"
        out["blocker_distance_km"] = float(horizon.blocker_distance_m(az)) / 1000.0
    return out


def _next_change(mask, et):
    """UTC and hours-from-now of the first step whose state differs from step 0."""
    diff = np.flatnonzero(mask != mask[0])
    if diff.size == 0:
        return None
    i = int(diff[0])
    return {"utc": spice_loader.et_to_utc(et[i]), "in_hours": float((et[i] - et[0]) / 3600.0)}


def _longest(mask, et, step_h):
    starts, lengths = runs(mask)
    if lengths.size == 0:
        return {"hours": 0.0, "starts_utc": None}
    j = int(lengths.argmax())
    return {"hours": float(lengths[j] * step_h), "starts_utc": spice_loader.et_to_utc(et[starts[j]])}


def _weekly(mask, steps_per_week):
    return [float(mask[i : i + steps_per_week].mean()) for i in range(0, mask.size, steps_per_week)]


def mission_query(dem, lat_deg, lon_deg, utc, days=30, step_minutes=30, observer_height_m=2.0):
    ts = site_timeseries(dem, lat_deg, lon_deg, utc, days, step_minutes, observer_height_m)
    et, hz = ts["et"], ts["horizon"]
    step_h = step_minutes / 60.0
    lit = ts["sun_lit_fraction"] >= 0.5
    link = ts["earth_visible"]

    sun_now = _instant(ts["sun_az"][0], ts["sun_el"][0], hz, lit[0])
    sun_now["lit_fraction"] = float(ts["sun_lit_fraction"][0])
    earth_now = _instant(ts["earth_az"][0], ts["earth_el"][0], hz, link[0])

    dark = _longest(~lit, et, step_h)
    outage = _longest(~link, et, step_h)
    per_week = max(1, int(round(7 * 24 / step_h)))
    sun_weekly, link_weekly = _weekly(lit, per_week), _weekly(link, per_week)

    report = {
        "site": {"lat": lat_deg, "lon": lon_deg, "elevation_m": float(dem.height_at_latlon(lat_deg, lon_deg))},
        "query_utc": spice_loader.et_to_utc(et[0]),
        "window_days": days,
        "now": {"sun": sun_now, "earth": earth_now},
        "next": {
            "sun_change": _next_change(lit, et),    # sunset if lit now, sunrise if dark now
            "earth_change": _next_change(link, et),  # loss of signal if linked now, acquisition otherwise
        },
        "window": {
            "sunlit_fraction": float(lit.mean()),
            "dte_fraction": float(link.mean()),
            "longest_darkness": dark,
            "longest_dte_outage": outage,
            "weekly_sunlit_fraction": sun_weekly,
            "weekly_dte_fraction": link_weekly,
        },
    }
    report["findings"] = _findings(report)
    return report


def _findings(r):
    """Rule engine: numbers -> ordered list of {severity, message}."""
    f = []

    def add(sev, msg):
        f.append({"severity": sev, "message": msg})

    sun, earth, nxt, w = r["now"]["sun"], r["now"]["earth"], r["next"], r["window"]

    if sun["state"] == "visible":
        add("info", f"Sun is up at {sun['elevation_deg']:.2f} deg elevation (azimuth {sun['azimuth_deg']:.0f} deg).")
    elif sun["state"] == "terrain_blocked":
        add("warning", f"Sun is blocked by terrain {sun['blocker_distance_km']:.1f} km away at azimuth "
                       f"{sun['azimuth_deg']:.0f} deg (terrain {sun['terrain_horizon_deg']:.2f} deg vs Sun {sun['elevation_deg']:.2f} deg).")
    else:
        add("warning", f"Sun is below the local horizon ({sun['elevation_deg']:.2f} deg).")
    if nxt["sun_change"]:
        word = "Sunset/shadow" if sun["state"] == "visible" else "Next sunlight"
        add("info", f"{word} in {nxt['sun_change']['in_hours']:.1f} h ({nxt['sun_change']['utc']}).")
    else:
        add("critical" if sun["state"] != "visible" else "info",
            f"Sun state does not change for the whole {r['window_days']}-day window.")

    if earth["state"] == "visible":
        add("info", f"Direct-to-Earth link available (Earth at {earth['elevation_deg']:.2f} deg).")
    elif earth["state"] == "terrain_blocked":
        add("warning", f"Earth link blocked by terrain {earth['blocker_distance_km']:.1f} km away at azimuth {earth['azimuth_deg']:.0f} deg.")
    else:
        add("warning", f"Earth is below the local horizon ({earth['elevation_deg']:.2f} deg) due to libration; no direct link.")
    if nxt["earth_change"]:
        word = "Loss of signal" if earth["state"] == "visible" else "Next Earth contact"
        add("info", f"{word} in {nxt['earth_change']['in_hours']:.1f} h ({nxt['earth_change']['utc']}).")

    d = w["longest_darkness"]
    if d["hours"] >= DARK_CRIT_H:
        add("critical", f"Darkness of {d['hours']:.0f} h starting {d['starts_utc']}: power/thermal survival risk.")
    elif d["hours"] >= DARK_WARN_H:
        add("warning", f"Darkness of {d['hours']:.0f} h starting {d['starts_utc']}.")
    if w["sunlit_fraction"] < LOW_LIGHT_FRACTION:
        add("critical", f"Only {100 * w['sunlit_fraction']:.0f}% sunlight in the window: unsuitable for solar power.")

    o = w["longest_dte_outage"]
    if w["dte_fraction"] == 0.0:
        add("critical", "No direct-to-Earth contact in the whole window: a relay satellite is required.")
    elif o["hours"] >= OUTAGE_CRIT_H:
        add("critical", f"Earth link outage of {o['hours']:.0f} h starting {o['starts_utc']}.")
    elif o["hours"] >= OUTAGE_WARN_H:
        add("warning", f"Earth link outage of {o['hours']:.0f} h starting {o['starts_utc']}.")

    for name, series in (("Sunlight", w["weekly_sunlit_fraction"]), ("Earth link", w["weekly_dte_fraction"])):
        if len(series) >= 2 and series[0] - series[-1] >= DEGRADE_DROP:
            add("warning", f"{name} degrades over the window: {100 * series[0]:.0f}% in week 1 -> {100 * series[-1]:.0f}% in the last week.")

    order = {"critical": 0, "warning": 1, "info": 2}
    return sorted(f, key=lambda x: order[x["severity"]])
