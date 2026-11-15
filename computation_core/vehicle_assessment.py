"""Vehicle-specific assessment: would THIS lander/rover cope at this site in this window?

Inputs: a lander profile (data_engine/lander_profiles.json, every value carries
verified / unverified / unknown provenance) and exact site metrics from analyze_site.
Unknown values are never guessed: the check says it cannot be assessed. Findings that
rely on unverified values say so.
"""
import json
from pathlib import Path

import numpy as np

from .mission_query import OUTAGE_CRIT_H, OUTAGE_WARN_H
from .site_analysis import site_timeseries
from .terrain_slope import site_hazard_stats

PROFILES_PATH = Path(__file__).resolve().parents[1] / "data_engine" / "lander_profiles.json"
DEFAULT_SLOPE_LIMIT_DEG = 15.0  # same as the generic suitability index
_W = {"sun": 0.35, "dte": 0.25, "dark": 0.20, "slope": 0.20, "energy": 0.20}  # energy counts only when the load is known

# Engineering design parameters (assumptions of this engine, not data about any vehicle). They are echoed in the
# result so nobody mistakes them for measured values.
VETO_CAP = 15.0           # a vehicle that cannot land / cannot talk / cannot charge scores at most this, however good the rest is
VETO_AREA_FRACTION = 0.5  # veto the slope gate when under half of the landing area is within the vehicle's slope limit
COLD_LIMIT_H = 72.0       # unheated electronics are assumed lost after this long without sunlight
THERMAL_FLOOR = 0.20      # cap for a vehicle whose very first night is lethal; it rises to 1.0 as the lethal night moves later


def _runs(mask):
    """(start, length) of every run of True in a boolean array."""
    out, start = [], None
    for i, v in enumerate(np.append(mask, False)):
        if v and start is None:
            start = i
        elif not v and start is not None:
            out.append((start, i - start))
            start = None
    return out


def simulate_battery(lit_fraction, step_h, battery_wh, solar_w, survival_w, payload_w=0.0):
    """State of charge (Wh) through the window, starting full.

    Charging is an upper bound: the array is assumed Sun-pointed whenever any of the Sun is up. The survival load (heaters,
    avionics) runs all the time; the payload runs only while the Sun is up and the battery still has charge.
    """
    soc, out, dead = battery_wh, np.empty(lit_fraction.size), np.zeros(lit_fraction.size, bool)
    for i, f in enumerate(lit_fraction):
        load = survival_w + (payload_w if f >= 0.5 else 0.0)
        soc += (solar_w * f - load) * step_h
        if soc <= 0.0:
            soc, dead[i] = 0.0, load > solar_w * f  # empty and still short of power: a blackout step
        soc = min(soc, battery_wh)
        out[i] = soc
    return out, dead


def sustainable_load_w(lit_fraction, step_h, battery_wh, solar_w):
    """The largest constant load the battery and array can carry through the whole window without a blackout."""
    lo, hi = 0.0, solar_w + battery_wh / (lit_fraction.size * step_h) + 1.0
    for _ in range(40):
        mid = (lo + hi) / 2
        _, dead = simulate_battery(lit_fraction, step_h, battery_wh, solar_w, mid)
        lo, hi = (mid, hi) if not dead.any() else (lo, mid)
    return lo


def load_profiles():
    return {p["id"]: p for p in json.loads(PROFILES_PATH.read_text(encoding="utf-8"))}


def _tag(field):
    return "" if field["status"] == "verified" else " [input unverified: user dataset]"


def _field(profile, key):
    return profile.get(key) or {"value": None, "status": "unknown"}  # older profiles may lack the newer fields


def assess_vehicle(profile, metrics, dem, lat_deg, lon_deg, area_radius_m=500.0, series=None):
    """metrics: analyze_site output for the same site/window. Returns checks, findings, index.

    The index is a weighted score that is then capped by veto gates: a vehicle that cannot land safely, cannot talk to
    Earth, cannot charge, or loses its heat early does not get to buy its way out with a good score elsewhere.
    `series` is the site's hourly time series (computed here when not given).
    """
    window_h = metrics["window"]["days"] * 24.0
    w_ = metrics["window"]
    ts = series or site_timeseries(dem, lat_deg, lon_deg, w_["start_utc"], w_["days"], w_["step_minutes"])
    step_h = w_["step_minutes"] / 60.0
    frac = np.asarray(ts["sun_lit_fraction"], float)
    vetoes = []
    dark_h, sunlit = metrics["longest_darkness_hours"], metrics["illumination_fraction"]
    bat, sol, heat = profile["battery_wh"], profile["solar_w"], profile["radioisotope_heat"]
    slope_f, comms = profile["max_slope_deg"], profile["comms"]
    findings, checks = [], {}

    def add(sev, msg):
        findings.append({"severity": sev, "message": msg})

    # --- landing slope against the vehicle's own tolerance ---
    if slope_f["value"] is None:
        limit = DEFAULT_SLOPE_LIMIT_DEG
        hz = site_hazard_stats(dem, lat_deg, lon_deg, area_radius_m, max_slope_deg=limit)
        checks["slope"] = {"assessed": False, "reason": "vehicle slope tolerance unknown", "generic_limit_deg": limit}
        add("info", f"Slope tolerance unknown for this vehicle; generic {limit:.0f} deg used for the index only.")
    else:
        limit = float(slope_f["value"])
        hz = site_hazard_stats(dem, lat_deg, lon_deg, area_radius_m, max_slope_deg=limit)
        checks["slope"] = {"assessed": True, "limit_deg": limit, "mean_slope_deg": hz["mean_slope_deg"],
                           "area_within_limit_fraction": hz["safe_area_fraction"], "status": slope_f["status"]}
        share = 100 * hz["safe_area_fraction"]
        sev = "critical" if share < 50 else "warning" if share < 90 else "info"
        add(sev, f"{share:.0f}% of the {area_radius_m:.0f} m landing area is within the {limit:.0f} deg slope limit "
                 f"(mean slope {hz['mean_slope_deg']:.1f} deg).{_tag(slope_f)}")
    if hz["safe_area_fraction"] < VETO_AREA_FRACTION:
        vetoes.append({"gate": "slope", "cap": VETO_CAP, "reason":
                       f"only {100 * hz['safe_area_fraction']:.0f}% of the landing area is within the {limit:.0f} deg slope limit"})

    # --- energy: solar income and night survival on battery ---
    solar_powered = sol["value"] is not None and sol["value"] > 0
    if sol["value"] is None:
        checks["solar"] = {"assessed": False, "reason": "solar array power unknown"}
        add("info", f"Solar array power unknown: energy income cannot be assessed ({100 * sunlit:.0f}% sunlight).")
    elif solar_powered:
        avg_w = sol["value"] * sunlit
        checks["solar"] = {"assessed": True, "array_w": sol["value"], "average_w_upper_bound": avg_w,
                           "window_energy_kwh_upper_bound": avg_w * window_h / 1000.0, "status": sol["status"]}
        if sunlit == 0:
            add("critical", f"No sunlight in the window: the {sol['value']:.0f} W array cannot recharge.{_tag(sol)}")
        else:
            add("info", f"Solar income at most {avg_w:.0f} W on average ({sol['value']:.0f} W array x "
                        f"{100 * sunlit:.0f}% sunlight, panels assumed Sun-pointed).{_tag(sol)}")
    else:
        checks["solar"] = {"assessed": True, "array_w": 0, "status": sol["status"]}
        add("info", f"Battery-only vehicle: sunlight does not recharge it.{_tag(sol)}")

    if bat["value"] is None:
        checks["night"] = {"assessed": False, "reason": "battery capacity unknown"}
        add("warning", f"Battery capacity unknown: survival of the {dark_h:.0f} h longest darkness cannot be assessed.")
    else:
        span = dark_h if solar_powered else window_h  # battery-only: it must last the whole window
        max_w = bat["value"] / span if span > 0 else float("inf")
        checks["night"] = {"assessed": True, "battery_wh": bat["value"], "hours_on_battery": span,
                           "max_average_draw_w": max_w, "status": bat["status"]}
        what = f"the {dark_h:.0f} h longest darkness" if solar_powered else f"the whole {window_h:.0f} h window"
        add("info", f"To last {what} on its {bat['value']:.0f} Wh battery (full capacity, no reserve), average draw "
                    f"must stay below {max_w:.1f} W.{_tag(bat)}")
    # --- energy through the window: state of charge, not just a division ---
    surv, payl = _field(profile, "survival_power_w"), _field(profile, "payload_power_w")
    energy_scored = False
    if bat["value"] is not None and sol["value"] is not None:
        sust = sustainable_load_w(frac, step_h, float(bat["value"]), float(sol["value"]))
        checks["energy"] = {"assessed": True, "battery_wh": bat["value"], "solar_w": sol["value"],
                            "sustainable_load_w": sust, "status": bat["status"] if bat["status"] == sol["status"] else "unverified"}
        if surv["value"] is None:
            checks["energy"]["load_known"] = False
            add("info", f"Survival load of this vehicle is unknown, so the battery is not scored. For reference the battery and array "
                        f"can carry a constant {sust:.1f} W through this window without a blackout.")
        else:
            soc, dead = simulate_battery(frac, step_h, float(bat["value"]), float(sol["value"]), float(surv["value"]),
                                         float(payl["value"] or 0.0))
            blackout_h = float(dead.sum() * step_h)
            first = next((i * step_h for i in range(dead.size) if dead[i]), None)
            energy_scored = True
            checks["energy"].update({"load_known": True, "survival_power_w": surv["value"], "payload_power_w": payl["value"],
                                     "blackout_hours": blackout_h, "first_blackout_h": first,
                                     "min_state_of_charge": float(soc.min() / bat["value"]) if bat["value"] else 0.0})
            tag = _tag(surv)
            if blackout_h > 0:
                add("critical" if first is not None and first < 72 else "warning",
                    f"Battery runs flat: {blackout_h:.0f} h of blackout in the window, the first at hour {first:.0f} "
                    f"(survival load {surv['value']:.0f} W).{tag}")
            else:
                add("info", f"Battery carries the survival load ({surv['value']:.0f} W) through the whole window; "
                            f"lowest charge {100 * soc.min() / bat['value']:.0f}%.{tag}")
    else:
        checks["energy"] = {"assessed": False, "reason": "battery capacity or solar array power unknown"}

    # --- cold: a long night without a heat source ---
    lit = frac >= 0.5
    lethal = [(a * step_h, n * step_h) for a, n in _runs(~lit) if n * step_h >= COLD_LIMIT_H]
    first_lethal_h = lethal[0][0] if lethal else None
    alive_fraction = 1.0 if first_lethal_h is None else first_lethal_h / window_h
    checks["thermal"] = {"assessed": heat["value"] is not None or not solar_powered, "cold_limit_h": COLD_LIMIT_H,
                         "first_long_night_start_h": first_lethal_h, "first_long_night_hours": lethal[0][1] if lethal else None,
                         "alive_fraction": alive_fraction}
    if heat["value"] is True:
        add("info", f"Radioisotope heat source keeps the vehicle warm through darkness.{_tag(heat)}")
    elif lethal and (solar_powered or heat["value"] is False):  # a battery-only vehicle with no heat source freezes in a long night too
        full_cap = 100.0 * (THERMAL_FLOOR + (1.0 - THERMAL_FLOOR) * alive_fraction)
        known_cold = heat["value"] is False
        cap = full_cap if known_cold else (100.0 + full_cap) / 2  # an unknown heater gets the benefit of the doubt, halfway
        vetoes.append({"gate": "thermal", "cap": cap, "reason":
                       f"a {lethal[0][1]:.0f} h night starts at hour {first_lethal_h:.0f} and no heat source is "
                       f"{'there' if known_cold else 'known'} (limit assumed {COLD_LIMIT_H:.0f} h)"})
        add("critical" if known_cold else "warning",
            f"{lethal[0][1]:.0f} h of darkness from hour {first_lethal_h:.0f} with {'no radioisotope heater' if known_cold else 'no known radioisotope heater'}: "
            f"the vehicle is assumed to survive about {first_lethal_h:.0f} h of the {window_h:.0f} h window.{_tag(heat)}")

    # --- communications ---
    if comms["value"] == "relay":
        checks["comms"] = {"assessed": True, "mode": "relay", "status": comms["status"]}
        add("info", f"Communicates through a relay satellite: direct-to-Earth visibility is not required.{_tag(comms)}")
    elif comms["value"] == "direct_to_earth":
        out, dte_frac = metrics["longest_outage_hours"], metrics["dte_fraction"]
        checks["comms"] = {"assessed": True, "mode": "direct_to_earth", "dte_fraction": dte_frac,
                           "longest_outage_h": out, "status": comms["status"]}
        sev = "critical" if dte_frac == 0 or out >= OUTAGE_CRIT_H else "warning" if out >= OUTAGE_WARN_H else "info"
        add(sev, f"Direct-to-Earth link {100 * dte_frac:.0f}% of the time, longest outage {out:.0f} h.{_tag(comms)}")
        if dte_frac == 0:
            vetoes.append({"gate": "comms", "cap": VETO_CAP, "reason": "Earth is never in line of sight in this window"})
    else:
        checks["comms"] = {"assessed": False, "reason": "communication mode unknown"}

    # --- vehicle-adjusted suitability index (weights renormalized to what applies) ---
    w = dict(_W)
    if comms["value"] == "relay":
        w["dte"] = 0.0
    if sol["value"] == 0:
        w["sun"] = 0.0
    if not energy_scored:
        w["energy"] = 0.0
    elif heat["value"] is True:
        w["dark"] /= 2  # its own heat source takes the cold out of the night; the battery term already counts the power
    total = sum(w.values())
    slope_ok = max(0.0, min(1.0, 1.0 - metrics["mean_slope_deg"] / limit))
    energy_ok = 1.0 - checks["energy"]["blackout_hours"] / window_h if energy_scored else 0.0
    weighted = 100.0 * (w["sun"] * sunlit + w["dte"] * metrics["dte_fraction"] + w["dark"] * (1.0 - dark_h / window_h)
                        + w["slope"] * slope_ok + w["energy"] * energy_ok) / total
    if solar_powered and sunlit == 0:
        vetoes.append({"gate": "power", "cap": VETO_CAP, "reason": "no sunlight in the window: the array cannot recharge"})
    if energy_scored and checks["energy"]["first_blackout_h"] is not None and checks["energy"]["first_blackout_h"] < 72:
        vetoes.append({"gate": "energy", "cap": VETO_CAP * 2, "reason":
                       f"the battery is flat by hour {checks['energy']['first_blackout_h']:.0f}"})
    # veto gates: the index can never exceed the tightest one, whatever the other terms add up to
    index = min([weighted] + [v["cap"] for v in vetoes])
    binding = min(vetoes, key=lambda v: v["cap"]) if vetoes and min(v["cap"] for v in vetoes) < weighted else None
    for v in vetoes:
        add("critical", f"Score capped at {v['cap']:.0f}: {v['reason']}.")

    used = [f for f in (bat, sol, heat, slope_f, comms, surv, payl) if f["value"] is not None]
    order = {"critical": 0, "warning": 1, "info": 2}
    return {
        "vehicle": {"id": profile["id"], "name": profile["name"], "kind": profile["kind"]},
        "vehicle_index": index,
        "weighted_index": weighted,
        "vetoes": vetoes,
        "limiting_factor": binding["gate"] if binding else None,
        "assumptions": {"veto_cap": VETO_CAP, "cold_limit_h": COLD_LIMIT_H, "slope_veto_area_fraction": VETO_AREA_FRACTION},
        "index_weights": {k: v / total for k, v in w.items()},
        "checks": checks,
        "findings": sorted(findings, key=lambda f: order[f["severity"]]),
        "data_quality": {
            "inputs_verified": sum(f["status"] == "verified" for f in used),
            "inputs_unverified": sum(f["status"] == "unverified" for f in used),
            "inputs_unknown": 7 - len(used),
        },
    }
