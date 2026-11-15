"""Mission success probability: the engine's own checks, re-run many times under the uncertainty that is really there.

The deterministic assessment answers "with these exact numbers, what happens?". Real missions do not get exact numbers: the
lander touches down somewhere in its landing area, not on the centre pixel; the landing can slip by a few days; most of the
vehicle figures in the dataset are not verified; nobody knows to the hour how long unheated electronics last in the cold.
Here every one of those is drawn at random (a fixed seed, so the same question always gets the same answer) and the real
physics is evaluated for each draw:

  * touchdown point : uniform over the landing area -> the real LOLA slope under the lander vs. its slope limit
  * landing date    : 0..SLIP_DAYS days late -> the real SPICE Sun / Earth sequence shifts under the mission
  * vehicle figures : battery, solar array and loads scaled by a log-normal factor (verified +-5 %, unverified +-20 %,
                      one sigma); the slope limit +-1 deg (verified) or +-2 deg (unverified)
  * cold survival   : a vehicle with no heat source of its own freezes after COLD_RANGE_H hours without power (uniform)

What happens in one draw, hour by hour: the battery charges from the array and feeds the survival load (and the payload
in daylight). When it is empty and the Sun is not enough, the vehicle goes dark: it hibernates and wakes up when the Sun
returns (SLIM did exactly that), unless the outage outlasts its cold limit, which is the end of it. A radioisotope heat
source never freezes. A vehicle with no array cannot recharge, so its first outage is its last.

A draw is a success when the lander comes down on ground within its slope limit, works (alive, powered and in daylight, when its
instruments run; a powered night is survival, not work) for at least
GOAL_DAYS in the window, and can send its data home: a relay, or Earth in view while it has power for at least MIN_CONTACT
of that time (a long outage only delays the data, it is stored on board). Inputs that are UNKNOWN are never guessed: that
check is reported as not assessed and the confidence drops. Every spread above is an assumption of this engine, not data about any
vehicle, and the result echoes them.
"""
import numpy as np

from .site_analysis import site_timeseries
from .terrain_slope import site_hazard_stats

N_SAMPLES = 400
SEED = 2026
SLIP_DAYS = 3.0
SPREAD = {"verified": 0.05, "unverified": 0.20}        # log-normal sigma of battery / solar / loads
SLOPE_SPREAD = {"verified": 1.0, "unverified": 2.0}    # deg, one sigma
COLD_RANGE_H = (48.0, 120.0)
GOAL_DAYS = 7.0                                        # days of work (alive, powered, in daylight) that make a useful surface mission
MIN_CONTACT = 0.10                                     # of those, the share with Earth in view as well (direct link only)
MODES = ("landing", "cold", "power", "comms")


def _field(profile, key):
    return profile.get(key) or {"value": None, "status": "unknown"}


def _runs_reach(mask, k):
    """Per row, the first step at which a run of True has lasted k steps (T when never). mask: (n, T); k: a number or one per row."""
    n, T = mask.shape
    run = np.zeros(n, int)
    first = np.full(n, T)
    for t in range(T):
        run = (run + 1) * mask[:, t]
        first[(run >= k) & (first == T)] = t
    return first


def _outages(F, step_h, bat, sol, surv, payl):
    """(n, T) True where the vehicle has no power: battery empty and the array short of the load. F: (n, T) lit fraction."""
    n, T = F.shape
    soc, dark = bat.copy(), np.zeros((n, T), bool)
    for t in range(T):
        f = F[:, t]
        load = surv + np.where(f >= 0.5, payl, 0.0)
        soc = soc + (sol * f - load) * step_h
        empty = soc <= 0.0
        dark[:, t] = empty & (load > sol * f)
        soc = np.minimum(np.where(empty, 0.0, soc), bat)
    return dark


def _split(x, outcome):
    """Mean outcome in the lowest and the highest third of a factor."""
    lo, hi = x <= np.quantile(x, 1 / 3), x >= np.quantile(x, 2 / 3)
    return float(outcome[lo].mean()), float(outcome[hi].mean())


def mission_risk(profile, dem, lat, lon, start_utc, days=30, step_minutes=60, area_radius_m=500.0, n=N_SAMPLES, series=None, slopes=None):
    """`series` ({sun_lit_fraction, earth_visible}, hourly, at least days + SLIP_DAYS long from the start) and `slopes` let a caller
    that scans many start dates (the planner) compute the sky and the terrain once."""
    step_minutes = max(step_minutes, 60)  # hourly is fine enough for a 400-run simulation and keeps the run x hour matrix small
    step_h = step_minutes / 60.0
    T = int(round(days * 24 / step_h))
    slip = int(round(SLIP_DAYS * 24 / step_h))
    goal_days = float(min(GOAL_DAYS, days))
    ts = series if series is not None else site_timeseries(dem, lat, lon, start_utc, days + SLIP_DAYS, step_minutes)
    lit_all = np.asarray(ts["sun_lit_fraction"], float)
    link_all = np.asarray(ts["earth_visible"], bool)
    if slopes is None:
        slopes = site_hazard_stats(dem, lat, lon, area_radius_m, with_slopes=True)["slopes"]
    slopes = np.asarray(slopes, float)

    # every draw is made in the same order whatever is known, so the same question always gets the same numbers
    rng = np.random.default_rng(SEED)
    offset = rng.integers(0, slip + 1, n)
    touch = slopes[rng.integers(0, slopes.size, n)]
    z_bat, z_sol, z_load, z_slope = (rng.normal(0.0, 1.0, n) for _ in range(4))
    cold_h = rng.uniform(*COLD_RANGE_H, n)

    idx = np.minimum(offset[:, None] + np.arange(T)[None, :], lit_all.size - 1)
    F, E = lit_all[idx], link_all[idx]

    bat, sol, heat = _field(profile, "battery_wh"), _field(profile, "solar_w"), _field(profile, "radioisotope_heat")
    slope_f, comms = _field(profile, "max_slope_deg"), _field(profile, "comms")
    surv, payl = _field(profile, "survival_power_w"), _field(profile, "payload_power_w")
    factors, not_assessed = {"landing_delay_h": offset * step_h}, []

    def scaled(f, z):
        return float(f["value"]) * np.exp(SPREAD.get(f["status"], SPREAD["unverified"]) * z)

    # --- touchdown ---
    if slope_f["value"] is not None:
        landed = touch <= float(slope_f["value"]) + SLOPE_SPREAD.get(slope_f["status"], 2.0) * z_slope
        factors["touchdown_slope"] = touch
    else:
        landed = np.ones(n, bool)
        not_assessed.append("landing")

    # --- power: hour by hour, with hibernation ---
    if bat["value"] is not None and sol["value"] is not None and surv["value"] is not None:
        b, s = scaled(bat, z_bat), scaled(sol, z_sol)
        load = np.exp(SPREAD.get(surv["status"], 0.2) * z_load)
        off = _outages(F, step_h, b, s, float(surv["value"]) * load, float(payl["value"] or 0.0) * load)
        if float(sol["value"]) == 0:  # nothing can recharge it: the first outage never ends
            off = np.maximum.accumulate(off, axis=1)
        factors.update({"battery": b, "solar": s, "load": load})
        power_known = True
    else:
        off = np.zeros((n, T), bool)
        not_assessed.append("power")
        power_known = False

    # --- cold: an outage (or, with the power unknown, a night) longer than the vehicle can stand unheated ---
    k = np.maximum(1, np.round(cold_h / step_h)).astype(int)
    if heat["value"] is True:
        frozen = np.full(n, T)
    elif heat["value"] is False:
        frozen = _runs_reach(off if power_known else F < 0.5, k)
        factors["cold_limit"] = cold_h
    else:
        frozen = np.full(n, T)
        not_assessed.append("cold")

    alive = (np.arange(T)[None, :] < frozen[:, None]) & landed[:, None]
    up = alive & ~off
    working = up & (F >= 0.5)  # its instruments run in daylight (as in the battery model); a powered night is survival, not work
    op_days = working.sum(axis=1) * step_h / 24.0
    works = op_days >= goal_days

    # --- comms: can it send its data home? ---
    if comms["value"] == "relay":
        talks = np.ones(n, bool)
    elif comms["value"] == "direct_to_earth":  # a long outage only delays the data (it is stored on board); too little contact loses it
        talks = (E & up).sum(axis=1) * step_h >= MIN_CONTACT * goal_days * 24.0
    else:
        talks = np.ones(n, bool)
        not_assessed.append("comms")

    success = works & talks
    lost = "power" if float(sol["value"] or 0) == 0 and power_known else "cold"  # no array: it is the battery running out that ends it
    ended = frozen < T
    cause = np.where(~landed, "landing", np.where(~works & ended, lost, np.where(~works, "power", np.where(~talks, "comms", ""))))
    modes = sorted(({"mode": m, "share": float(np.mean(cause == m))} for m in MODES if np.any(cause == m)), key=lambda d: -d["share"])

    drivers = []
    for name, x in factors.items():
        if np.ptp(x) == 0:
            continue
        p_lo, p_hi = _split(x, success.astype(float))
        d_lo, d_hi = _split(x, op_days)
        drivers.append({"factor": name, "p_low": p_lo, "p_high": p_hi, "effect": p_hi - p_lo, "days_low": d_lo, "days_high": d_hi,
                        "low_value": float(np.quantile(x, 1 / 6)), "high_value": float(np.quantile(x, 5 / 6))})
    drivers.sort(key=lambda d: (-abs(d["effect"]), -abs(d["days_high"] - d["days_low"])))
    drivers = [d for d in drivers if abs(d["effect"]) >= 0.02 or abs(d["days_high"] - d["days_low"]) >= 0.5]

    used = [f for f in (bat, sol, heat, slope_f, comms, surv, payl) if f["value"] is not None]
    unverified = sum(f["status"] != "verified" for f in used)
    reasons = []
    if not_assessed:
        level = "low"
        reasons.append("not assessed for lack of data: " + ", ".join(not_assessed))
    elif unverified > len(used) / 2:
        level = "medium"
    else:
        level = "high"
    if unverified:
        reasons.append(f"{unverified} of {len(used)} known vehicle figures are unverified (a +-{100 * SPREAD['unverified']:.0f}% spread is assumed)")

    return {
        "samples": int(n),
        "seed": SEED,
        "goal_days": goal_days,
        "p_success": float(success.mean()),
        "p_landing": float(landed.mean()),
        "p_works_goal": float(works[landed].mean()) if landed.any() else 0.0,           # given a safe landing
        "p_comms": float(talks[works].mean()) if works.any() else float(talks.mean()),  # given it works
        "p_alive_end": float(alive[:, -1].mean()),
        "milestones": [{"days": d, "p": float(np.mean(op_days >= d))} for d in (1.0, 3.0, 7.0, 14.0, 21.0) if d <= days],
        "assessed": {m: m not in not_assessed for m in MODES},
        "operating_days": {"p10": float(np.quantile(op_days, 0.1)), "p50": float(np.quantile(op_days, 0.5)),
                           "p90": float(np.quantile(op_days, 0.9)), "mean": float(op_days.mean())},
        "survival_curve": [float(alive[:, min(int(d * 24 / step_h), T - 1)].mean()) for d in range(int(days) + 1)],
        "failure_modes": modes,
        "drivers": drivers[:4],
        "confidence": {"level": level, "reasons": reasons},
        "assumptions": {"slip_days": SLIP_DAYS, "spread_verified": SPREAD["verified"], "spread_unverified": SPREAD["unverified"],
                        "slope_spread_deg": SLOPE_SPREAD, "cold_range_h": list(COLD_RANGE_H), "goal_days": GOAL_DAYS,
                        "min_contact": MIN_CONTACT, "area_radius_m": area_radius_m},
    }
