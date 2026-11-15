"""Mission planner: the full analysis for every possible start date in a range, at one place, for one vehicle.

Each start date gets exactly what the analysis page shows for it: the 0-100 score, the mission success probability and the call
(go / possible / not recommended / do not proceed). The Sun and Earth sequence (real SPICE) and the terrain slopes (real LOLA) are
computed once for the whole range; every date is then evaluated on its own 30-day slice with the same code and the same runs as
the analysis page, so a date shows the same numbers in both places.
"""
import numpy as np

from .earth_visibility import dte_stats
from .mission_risk import N_SAMPLES, SLIP_DAYS, mission_risk
from .site_analysis import site_timeseries
from .solar_illumination import illumination_stats
from .terrain_slope import site_hazard_stats
from .vehicle_assessment import assess_vehicle
from .verdict import ORDER, decide

WINDOW_DAYS = 30
LEVEL = {k: len(ORDER) - 1 - i for i, k in enumerate(ORDER)}  # go 3, conditional 2, weak 1, stop 0


def plan(profile, dem, lat, lon, start_utc, span_days, n=N_SAMPLES, lock=None):
    """Per start date: score, success probability, the call, and the numbers behind them (sunlight, Earth in view, typical working days)."""
    total = span_days + WINDOW_DAYS + int(SLIP_DAYS) + 1
    if lock is None:
        ts = site_timeseries(dem, lat, lon, start_utc, total, 60)
    else:
        with lock:  # SPICE is not thread-safe; the simulations below do not use it, so other pages are not kept waiting
            ts = site_timeseries(dem, lat, lon, start_utc, total, 60)
    lit = np.asarray(ts["sun_lit_fraction"], float)
    link = np.asarray(ts["earth_visible"], bool)
    hazard = site_hazard_stats(dem, lat, lon, 500.0, with_slopes=True)
    slopes = hazard["slopes"]
    hazard = {k: v for k, v in hazard.items() if k != "slopes"}
    need = int((WINDOW_DAYS + SLIP_DAYS) * 24)
    days = []
    for k in range(span_days):
        a = k * 24
        win = slice(a, a + WINDOW_DAYS * 24)
        m = {"site": {"lat": lat, "lon": lon}, "window": {"start_utc": start_utc, "days": WINDOW_DAYS, "step_minutes": 60},
             **illumination_stats(lit[win], 1.0), **dte_stats(link[win], 1.0), **hazard}
        assess = assess_vehicle(profile, m, dem, lat, lon, series={"sun_lit_fraction": lit[win]})
        sl = slice(a, a + need)
        risk = mission_risk(profile, dem, lat, lon, start_utc, WINDOW_DAYS, 60, n=n, series={"sun_lit_fraction": lit[sl], "earth_visible": link[sl]}, slopes=slopes)
        v = decide(assess, m, risk)
        days.append({"day": k, "score": assess["vehicle_index"], "p": risk["p_success"], "verdict": v["kind"], "title": v["title"], "days_p50": risk["operating_days"]["p50"],
                     "sun": float((lit[win] >= 0.5).mean()), "earth": float(link[win].mean()), "confidence": risk["confidence"]["level"],
                     "top_failure": risk["failure_modes"][0]["mode"] if risk["failure_modes"] else None})
    return days


def windows(days):
    """The best runs of consecutive start dates: those that get the best call any date gets (longest and highest-scoring first, max 4).
    Empty when even the best call is 'do not proceed'."""
    if not days:
        return []
    best = max(LEVEL[d["verdict"]] for d in days)
    if best == 0:
        return []
    out, start = [], None
    flags = [LEVEL[d["verdict"]] == best for d in days] + [False]
    for i, f in enumerate(flags):
        if f and start is None:
            start = i
        elif not f and start is not None:
            seg = days[start:i]
            at = start + int(np.argmax([d["score"] for d in seg]))
            out.append({"from": start, "to": i - 1, "at": at, "verdict": days[at]["verdict"], "score": days[at]["score"], "p": days[at]["p"]})
            start = None
    out.sort(key=lambda w: (-w["score"], -(w["to"] - w["from"])))
    return out[:4]
