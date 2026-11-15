"""The call: go / possible with risks / not recommended / do not proceed, with its reasons, in one place.

The page, the PDF report, the Landing Map check and the assistant all read this, so they can never disagree. It weighs two
things and takes the more cautious of the two:
  * the vehicle-adjusted score (60 good, 40 fair, 20 weak), which says how good the conditions are, and
  * the mission success probability (mission_risk), which says how likely this mission is to land and do its job when the
    real uncertainty is let in (80 % go, 50 % possible, 20 % weak).
A high score with several real warnings, or with a low confidence (unknown vehicle data), is never a clean go.
"""
ORDER = ["go", "conditional", "weak", "stop"]
TITLE = {"go": "Go: recommended", "conditional": "Possible, with risks", "weak": "Not recommended as it is", "stop": "Do not proceed"}

LIMIT = {
    "slope": "the ground is too steep for this vehicle",
    "comms": "Earth can never be seen from here, so there is no radio link",
    "power": "there is no sunlight to recharge it",
    "thermal": "the nights are long and cold, and it has no known heater",
    "energy": "its battery would run out too early",
}
FIX = {
    "slope": "Pick a flatter landing place.",
    "comms": "Pick a place with a view of Earth, or a vehicle that uses a relay satellite.",
    "power": "Try another start date, or a different place with more sunlight.",
    "thermal": "Try another start date with a shorter night, or a vehicle with a heater.",
    "energy": "Try another start date, or a vehicle with a bigger battery.",
}
MODE_TEXT = {
    "landing": "it touches down on ground steeper than it can stand",
    "cold": "a night without power outlasts what it can stand unheated, and it freezes",
    "power": "the battery cannot keep it working for {goal} days",
    "comms": "it cannot get enough of its data home (Earth is in view too little while it has power)",
}
DRIVER = {
    "landing_delay_h": ("When it lands", "landing on time", "landing up to {v:.0f} h late", "Hold the landing date: even a day or two of delay changes the outcome here."),
    "touchdown_slope": ("Where exactly it touches down", "flat spots", "spots up to {v:.0f} deg steep", "Aim for the flattest part of the area, or a vehicle with a wider slope limit."),
    "battery": ("Battery capacity", "a {v:.0f} Wh battery", "a {v:.0f} Wh battery", "A bigger battery would make this much safer."),
    "solar": ("Solar array output", "a {v:.0f} W array", "a {v:.0f} W array", "More solar power would make this much safer."),
    "load": ("Power draw", "{v:.0%} of the listed load", "{v:.0%} of the listed load", "Cut the power draw (fewer instruments at once, a leaner survival mode)."),
    "cold_limit": ("How long it can stand the cold unpowered", "{v:.0f} h", "{v:.0f} h", "A heat source (or better insulation) would remove the biggest risk."),
}


def _pct(f):
    return f"{100 * f:.0f}%"


def _span(h):
    return f"{h:.0f} hours" if h < 48 else f"{h / 24:.1f} days"


def _worse(a, b):
    return a if ORDER.index(a) >= ORDER.index(b) else b


def drivers_text(risk):
    out = []
    for d in risk.get("drivers", [])[:2]:
        if d["factor"] not in DRIVER or abs(d["effect"]) < 0.05:
            continue
        name, lo, hi, _ = DRIVER[d["factor"]]
        lo_t, hi_t = lo.format(v=d["low_value"]), hi.format(v=d["high_value"])
        out.append(f"{name} {'decides a lot' if abs(d['effect']) >= 0.25 else 'matters'}: {_pct(d['p_low'])} success with {lo_t}, {_pct(d['p_high'])} with {hi_t}")
    return out


def decide(assessment, metrics, risk=None):
    """assessment: assess_vehicle output; metrics: the site metrics of the same window; risk: mission_risk output (optional)."""
    idx, lim = assessment["vehicle_index"], assessment["limiting_factor"]
    checks = assessment["checks"]
    relay = (checks.get("comms") or {}).get("mode") == "relay"
    sun, dte = metrics["illumination_fraction"], metrics["dte_fraction"]
    dark_h, out_h = metrics["longest_darkness_hours"], metrics["longest_outage_hours"]
    slope = checks.get("slope") or {}
    safe = slope.get("area_within_limit_fraction", metrics.get("safe_area_fraction", 1.0))

    good = []
    if sun >= 0.6:
        good.append(f"Plenty of sunlight ({_pct(sun)} of the month)")
    if relay:
        good.append("Talks to Earth through a relay satellite, so Earth being out of view does not matter")
    elif dte >= 0.7:
        good.append(f"Earth is in view most of the time ({_pct(dte)})")
    if safe >= 0.9:
        good.append(f"The landing area is within the vehicle's slope limit ({_pct(safe)})")
    if dark_h < 72:
        good.append(f"Short nights (longest {_span(dark_h)})")

    risks = []
    if lim:
        risks.append(LIMIT[lim][0].upper() + LIMIT[lim][1:])
    if sun < 0.3:
        risks.append(f"Little sunlight ({_pct(sun)} of the month)")
    if dark_h >= 168:
        risks.append(f"A very long night: {_span(dark_h)} without sunlight")
    if not relay and dte < 0.4:
        risks.append(f"Earth is often out of view (visible {_pct(dte)} of the time)")
    if not relay and out_h >= 72:
        risks.append(f"Up to {_span(out_h)} without contact with Earth")
    if safe < 0.9:
        risks.append(f"Only {_pct(safe)} of the landing area is within the vehicle's slope limit")

    kind = "go" if idx >= 60 and not lim else "conditional" if idx >= 40 or (idx >= 60 and lim) else "weak" if idx >= 20 else "stop"
    summary_p, next_ = None, []
    if risk:
        p, goal = risk["p_success"], risk["goal_days"]
        kind = _worse(kind, "go" if p >= 0.8 else "conditional" if p >= 0.5 else "weak" if p >= 0.2 else "stop")
        od = risk["operating_days"]
        summary_p = (f"In {_pct(p)} of {risk['samples']} simulated missions it lands safely, works at least {goal:.0f} days and gets its "
                     f"data home. It typically works {od['p50']:.0f} of the {len(risk['survival_curve']) - 1} days "
                     f"(between {od['p10']:.0f} and {od['p90']:.0f} in 8 of 10 runs).")
        top = next((m for m in risk["failure_modes"] if m["share"] >= 0.05), None)
        if top:
            risks.insert(1 if lim else 0, f"In {_pct(top['share'])} of the runs {MODE_TEXT[top['mode']].format(goal=f'{goal:.0f}')}")
        risks.extend(drivers_text(risk))
        if risk["confidence"]["level"] == "low":
            risks.append("Some details of this vehicle are not published, so part of the risk could not be checked")
        d0 = next((d for d in risk.get("drivers", []) if d["factor"] in DRIVER and abs(d["effect"]) >= 0.1), None)
        if d0 and kind != "go":
            next_.append(DRIVER[d0["factor"]][3])
    elif assessment["data_quality"]["inputs_unknown"] >= 3:
        risks.append("Several details about this vehicle are not published, so the score is less certain")

    if kind == "go" and (len(risks) >= 2 or (risk and risk["confidence"]["level"] == "low")):
        kind = "conditional"  # a high score with several real warnings, or with unknown data, is not a clean go
    summary = {
        "go": "This vehicle should do well at this place and date." + (" One thing to keep an eye on is listed below." if risks else " Nothing in the checks stands in the way."),
        "conditional": f"It can work, but one thing needs attention: {LIMIT[lim]}." if lim else "It can work here, but it will be a hard month. Read the risks below before you commit.",
        "weak": f"As it stands this is a poor choice: {LIMIT[lim]}." if lim else "As it stands this is a poor choice. A different date, place or vehicle would be better.",
        "stop": f"This would very likely fail: {LIMIT[lim]}." if lim else "This would very likely fail. Choose a different date, place or vehicle.",
    }[kind]
    if kind == "go":
        next_ = ["Compare it with another date, place or vehicle to be sure."]
    elif lim and FIX[lim] not in next_:
        next_.insert(0, FIX[lim])
    if not next_:
        next_ = ["Try another start date or another place and compare the scores."]
    return {"kind": kind, "title": TITLE[kind], "summary": summary, "probability_line": summary_p,
            "good": good[:4], "risks": risks[:5], "next": next_[:2],
            "rule": "The call takes the more cautious of two readings: the score (60 or more recommended, 40 to 60 possible, 20 to 40 "
                    "not recommended, below 20 do not proceed) and the simulated success probability (80% / 50% / 20%). Several "
                    "warnings, or vehicle data that is not published, keep a high score from being a clean go. The score rates the whole 30-day "
                    "window (a night over 72 h without a heat source caps it); the probability rates the mission goal (7 days of "
                    "work), where an empty battery is hibernation and only an outage longer than the cold limit ends it."}
