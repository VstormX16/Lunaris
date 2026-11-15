"""Orchestrator: one call -> all physics metrics for a candidate site."""
import numpy as np

from data_engine import spice_loader

from .earth_visibility import dte_stats, dte_visible
from .horizon import compute_horizon
from .solar_illumination import illumination_stats, lit_fraction
from .terrain_slope import site_hazard_stats
from .timeseries import longest_run_rows


def site_timeseries(dem, lat_deg, lon_deg, start_utc, days=30, step_minutes=60,
                    observer_height_m=2.0):
    """Real SPICE ephemeris + real LOLA horizon -> per-step Sun/Earth state."""
    et = spice_loader.time_grid(start_utc, days, step_minutes)
    return exact_series(dem, lat_deg, lon_deg, et, spice_loader.body_vectors_me("SUN", et),
                        spice_loader.body_vectors_me("EARTH", et), observer_height_m)


def exact_series(dem, lat_deg, lon_deg, et, sun_vec, earth_vec, observer_height_m=2.0):
    """Exact per-step state for one site from ephemeris shared by many sites (one code path for
    /analyze, the training targets and map refinement)."""
    horizon = compute_horizon(dem, lat_deg, lon_deg, observer_height_m)
    s_az, s_el, s_d = spice_loader.topocentric("SUN", lat_deg, lon_deg, et, body_vec=sun_vec)
    e_az, e_el, _ = spice_loader.topocentric("EARTH", lat_deg, lon_deg, et, body_vec=earth_vec)
    return {
        "et": et, "horizon": horizon,
        "sun_az": s_az, "sun_el": s_el, "sun_lit_fraction": lit_fraction(s_az, s_el, s_d, horizon),
        "earth_az": e_az, "earth_el": e_el, "earth_visible": dte_visible(e_az, e_el, horizon),
    }


def exact_targets(dem, lat_deg, lon_deg, et, sun_vec, earth_vec, observer_height_m=2.0):
    """The 4 surrogate targets, exactly: sunlit fraction, DTE fraction, longest darkness and
    longest outage (both as a fraction of the window)."""
    ts = exact_series(dem, lat_deg, lon_deg, et, sun_vec, earth_vec, observer_height_m)
    lit = ts["sun_lit_fraction"] >= 0.5
    link = ts["earth_visible"]
    n = float(lit.size)
    return np.array([lit.mean(), link.mean(),
                     longest_run_rows(~lit[None])[0] / n, longest_run_rows(~link[None])[0] / n])


def analyze_site(dem, lat_deg, lon_deg, start_utc, days=30, step_minutes=60,
                 observer_height_m=2.0, area_radius_m=500.0):
    """Illumination, DTE and slope metrics for one site."""
    ts = site_timeseries(dem, lat_deg, lon_deg, start_utc, days, step_minutes, observer_height_m)
    h = step_minutes / 60.0
    return {
        "site": {"lat": lat_deg, "lon": lon_deg},
        "window": {"start_utc": start_utc, "days": days, "step_minutes": step_minutes},
        **illumination_stats(ts["sun_lit_fraction"], h),
        **dte_stats(ts["earth_visible"], h),
        **site_hazard_stats(dem, lat_deg, lon_deg, area_radius_m),
    }
