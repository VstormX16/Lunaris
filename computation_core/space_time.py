"""Space Time: real celestial events with countdowns, from the NASA/JPL DE440 ephemeris (SPICE).

Nothing here is tabulated or approximated by formula: every instant is found by root-finding
(or extremum search) on SPICE positions, refined to a fraction of a second.

Conventions (stated, since "sunset" and "phase" have more than one definition):
- Phases: apparent geocentric ecliptic longitude difference Moon - Sun = 0/90/180/270 deg
  (the astronomical definition; light-time and aberration included).
- Site Sun events: the instant half of the solar disk is above the terrain horizon at the site
  (the same rule the analysis engine uses), seen from a 2 m mast over the real LOLA DEM.
- Perigee/apogee: extremum of the true geocentric Earth-Moon distance.
"""
import numpy as np

from data_engine import spice_loader as sl

from .earth_visibility import dte_visible
from .horizon import HorizonProfile, compute_horizon
from .solar_illumination import lit_fraction

DAY = 86400.0
PHASES = {"new_moon": 0.0, "first_quarter": 90.0, "full_moon": 180.0, "last_quarter": 270.0}
PHASE_NAMES = ["New Moon", "Waxing crescent", "First quarter", "Waxing gibbous",
               "Full Moon", "Waning gibbous", "Last quarter", "Waning crescent"]


def _bisect(f, lo, hi, iters=50):
    """Root of a function that changes sign from negative to positive on [lo, hi]."""
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if f(mid) < 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _extremum(f, lo, hi, minimize, iters=60):
    """Golden-section search for the extremum of a smooth scalar function on [lo, hi]."""
    g = (np.sqrt(5) - 1) / 2
    sign = 1.0 if minimize else -1.0
    a, b = lo, hi
    c, d = b - g * (b - a), a + g * (b - a)
    fc, fd = sign * f(c), sign * f(d)
    for _ in range(iters):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - g * (b - a)
            fc = sign * f(c)
        else:
            a, c, fc = c, d, fd
            d = a + g * (b - a)
            fd = sign * f(d)
    return 0.5 * (a + b)


# ---------------- Moon geometry ----------------

def elongation_deg(et):
    """Apparent geocentric ecliptic longitude of the Moon minus that of the Sun, in [0, 360)."""
    sun = sl.rel_positions("SUN", "EARTH", "ECLIPJ2000", et)
    moon = sl.rel_positions("MOON", "EARTH", "ECLIPJ2000", et)
    lon = lambda v: np.degrees(np.arctan2(v[:, 1], v[:, 0]))  # noqa: E731
    return (lon(moon) - lon(sun)) % 360.0


def _signed(et, target):
    return ((elongation_deg(et) - target + 180.0) % 360.0) - 180.0


def next_phase(et0, name, direction=1, horizon_days=32.0):
    """ET of the next (direction=1) or previous (-1) instant at which the elongation hits the phase."""
    target = PHASES[name]
    steps = np.arange(0.0, horizon_days * DAY, 3 * 3600.0)
    grid = et0 + steps if direction > 0 else et0 - steps[::-1]  # always ascending in time
    d = _signed(grid, target)
    up = np.flatnonzero((d[:-1] < 0) & (d[1:] >= 0))  # a true crossing; the +180 -> -180 wrap is not
    roots = [_bisect(lambda x: _signed(np.array([x]), target)[0], grid[i], grid[i + 1]) for i in up]
    roots = [t for t in roots if (t > et0 if direction > 0 else t < et0)]
    if not roots:
        raise RuntimeError(f"no {name} within {horizon_days} days")  # cannot happen for a 29.5-day cycle
    return roots[0] if direction > 0 else roots[-1]


def moon_state(et):
    """Phase and geometry at one instant."""
    et = float(et)
    sun_m = sl.rel_positions("SUN", "MOON", "J2000", et)[0]
    earth_m = sl.rel_positions("EARTH", "MOON", "J2000", et)[0]
    cos_i = np.dot(sun_m, earth_m) / (np.linalg.norm(sun_m) * np.linalg.norm(earth_m))
    phase_angle = float(np.degrees(np.arccos(np.clip(cos_i, -1, 1))))
    elong = float(elongation_deg(np.array([et]))[0])
    last_new = next_phase(et, "new_moon", direction=-1)
    me_earth = sl.body_vectors_me("EARTH", et)[0]
    me_sun = sl.body_vectors_me("SUN", et)[0]
    pt = lambda v: {"lat": float(np.degrees(np.arcsin(v[2] / np.linalg.norm(v)))),  # noqa: E731
                    "lon": float(np.degrees(np.arctan2(v[1], v[0])) % 360.0)}
    return {
        "elongation_deg": elong,
        "phase_angle_deg": phase_angle,
        "illumination": float((1 + np.cos(np.radians(phase_angle))) / 2),
        "phase_name": PHASE_NAMES[int(((elong + 22.5) % 360) // 45)],
        "age_days": float((et - last_new) / DAY),
        "distance_km": float(np.linalg.norm(sl.rel_positions("MOON", "EARTH", "J2000", et, "NONE")[0])),
        "sub_earth": pt(me_earth),   # libration: the point of the Moon facing Earth
        "sub_solar": pt(me_sun),     # where the Sun is overhead
    }


def _scan_extremum(f, et0, days, step_s, kind, prominence_days=0.0):
    """First local min/max of scalar series f over the next `days`, refined by golden-section.

    With `prominence_days`, a candidate must also be the extreme of everything within that many
    days on both sides. Needed where the curve is nearly flat at its peak: there, small monthly
    wiggles (physical libration) create spurious local extrema next to the real one.
    """
    grid = et0 + np.arange(0.0, days * DAY, step_s)
    v = f(grid)
    s = v if kind == "min" else -v  # always look for a minimum
    w = int(prominence_days * DAY / step_s)
    for i in range(1, len(v) - 1):
        if not (s[i - 1] > s[i] <= s[i + 1] and v[i - 1] != v[i + 1]):
            continue
        if w and s[i] > s[max(0, i - w): i + w + 1].min():
            continue
        t = _extremum(lambda x: float(f(np.array([x]))[0]), grid[i - 1], grid[i + 1], kind == "min")
        return t, float(f(np.array([t]))[0])
    return None, None


def _smoothed_extremum(f, et0, days, kind, window_days=27.32, step_s=6 * 3600.0, prominence_days=100.0):
    """Extremum of f after averaging away the monthly wiggle (one sidereal month).

    For a curve that is flat at its peak, the raw argmax is decided by small monthly terms and can
    move by ~10 days. The averaged curve gives a stable estimate (about +-3 days across window
    choices), which is why callers report it as approximate.
    """
    n = int(window_days * DAY / step_s)
    grid = et0 - window_days * DAY / 2 + np.arange(0.0, (days + window_days) * DAY, step_s)
    sm = np.convolve(f(grid), np.ones(n) / n, mode="valid")  # sm[j] is centred on grid[j + n // 2]
    tc = grid[n // 2: n // 2 + len(sm)]
    s = sm if kind == "min" else -sm
    w = int(prominence_days * DAY / step_s)
    for j in range(1, len(sm) - 1):
        if tc[j] <= et0 or not (s[j - 1] > s[j] <= s[j + 1]):
            continue
        if s[j] > s[max(0, j - w): j + w + 1].min():
            continue
        a, b, c = s[j - 1], s[j], s[j + 1]  # parabolic refinement between samples
        shift = 0.5 * (a - c) / (a - 2 * b + c) if (a - 2 * b + c) != 0 else 0.0
        return float(tc[j] + shift * step_s), float(sm[j])
    return None, None


def _distance(et):
    return np.linalg.norm(sl.rel_positions("MOON", "EARTH", "J2000", et, "NONE"), axis=1)


def _sub_earth_lat(et):
    v = sl.body_vectors_me("EARTH", et)
    return np.degrees(np.arcsin(v[:, 2] / np.linalg.norm(v, axis=1)))


def _sub_solar_lat(et):
    v = sl.body_vectors_me("SUN", et)
    return np.degrees(np.arcsin(v[:, 2] / np.linalg.norm(v, axis=1)))


def global_events(et0):
    """Upcoming Moon events independent of any site, soonest first.

    kind "instant": a true crossing (phases), found to a fraction of a second.
    kind "peak": the top of a flat curve; `uncertainty_days` says how well the instant is defined.
    """
    ev = []
    notes = {"new_moon": "Far side fully sunlit", "first_quarter": "Half of the near side lit",
             "full_moon": "Near side fully lit", "last_quarter": "Half of the near side lit"}
    labels = {"new_moon": "New Moon", "first_quarter": "First quarter", "full_moon": "Full Moon", "last_quarter": "Last quarter"}
    for name in PHASES:
        ev.append({"id": name, "label": labels[name], "note": notes[name], "et": next_phase(et0, name),
                   "kind": "instant", "uncertainty_days": 0.0})

    t, d = _scan_extremum(_distance, et0, 30, 3600.0, "min")
    ev.append({"id": "perigee", "label": "Perigee", "note": f"Closest to Earth, {d:,.0f} km", "et": t,
               "kind": "peak", "uncertainty_days": 0.05})
    t, d = _scan_extremum(_distance, et0, 30, 3600.0, "max")
    ev.append({"id": "apogee", "label": "Apogee", "note": f"Farthest from Earth, {d:,.0f} km", "et": t,
               "kind": "peak", "uncertainty_days": 0.05})

    t, lat = _scan_extremum(_sub_earth_lat, et0, 30, 3600.0, "min")
    ev.append({"id": "libration_south", "label": "South-pole Earth view peak",
               "note": f"Earth rises highest in the south-pole sky: {-lat:.1f} deg above the horizon", "et": t,
               "kind": "peak", "uncertainty_days": 0.5})
    t, lat = _smoothed_extremum(_sub_solar_lat, et0, 380, "min")
    if t is not None:
        ev.append({"id": "sun_south_extreme", "label": "Sun farthest south",
                   "note": f"Sun overhead near {abs(lat):.1f} deg {'S' if lat < 0 else 'N'}: the south pole's sunniest season",
                   "et": t, "kind": "peak", "uncertainty_days": 3.0})
    return sorted(ev, key=lambda e: e["et"])


# ---------------- site events ----------------

def _transitions(state_fn, grid, max_events):
    """Instants (refined to ~1 ms) where a boolean series changes, plus its state at grid[0]."""
    s = state_fn(grid)
    flips = np.flatnonzero(s[1:] != s[:-1])
    out = []
    for i in flips[:max_events]:
        before = bool(s[i])
        t = _bisect(lambda x: (1.0 if bool(state_fn(np.array([x]))[0]) != before else -1.0), grid[i], grid[i + 1], iters=40)
        out.append({"et": t, "becomes": (not before)})
    return bool(s[0]), out


def site_events(dem, lat_deg, lon_deg, et0, days=60.0, step_min=10.0, max_events=4, observer_height_m=2.0, horizon=None):
    """Sun and Earth rise/set instants at a lunar site, accounting for the real terrain.

    dem=None means no terrain data for the site (anywhere north of the 85 S DEM): smooth-sphere horizon.
    """
    if horizon is not None:  # a skyline from another source (the whole-Moon grid)
        pass
    elif dem is not None:
        horizon = compute_horizon(dem, lat_deg, lon_deg, observer_height_m)
    else:  # outside the polar DEM there is no terrain data: a smooth sphere, horizon at 0 degrees
        az = np.linspace(0.0, 360.0, 720, endpoint=False)
        horizon = HorizonProfile(az, np.zeros_like(az), None)

    def sun_up(et):
        az, el, dist = sl.topocentric("SUN", lat_deg, lon_deg, et)
        return lit_fraction(az, el, dist, horizon) >= 0.5

    def earth_up(et):
        az, el, _ = sl.topocentric("EARTH", lat_deg, lon_deg, et)
        return dte_visible(az, el, horizon)

    grid = et0 + np.arange(0.0, days * DAY, step_min * 60.0)
    sun_now, sun_ev = _transitions(sun_up, grid, max_events)
    earth_now, earth_ev = _transitions(earth_up, grid, max_events)
    return {"sun": {"up_now": sun_now, "transitions": sun_ev},
            "earth": {"up_now": earth_now, "transitions": earth_ev},
            "searched_days": days}
