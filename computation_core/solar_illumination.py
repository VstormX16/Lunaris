"""Solar illumination of a site: Sun vs. terrain horizon over time."""
import numpy as np

from .constants import SUN_RADIUS_KM
from .horizon import HorizonProfile
from .timeseries import longest_run_hours, runs


def sun_angular_radius_deg(distance_km):
    return np.degrees(np.arcsin(SUN_RADIUS_KM / np.asarray(distance_km)))


def lit_fraction(sun_az, sun_el, sun_dist_km, horizon: HorizonProfile):
    """Fraction of the solar disk above the horizon (0 = full shadow, 1 = full sun).

    Uses the horizon at the Sun-center azimuth and the disk's angular radius.
    """
    rs = sun_angular_radius_deg(sun_dist_km)
    return np.clip((sun_el - horizon.at(sun_az)) / (2.0 * rs) + 0.5, 0.0, 1.0)


def illumination_stats(frac, step_hours, threshold=0.5):
    lit = frac >= threshold
    dark = ~lit
    return {
        "illumination_fraction": float(frac.mean()),
        "longest_darkness_hours": longest_run_hours(dark, step_hours),
        "longest_sunlight_hours": longest_run_hours(lit, step_hours),
        "n_dark_periods": int(runs(dark)[0].size),
    }
