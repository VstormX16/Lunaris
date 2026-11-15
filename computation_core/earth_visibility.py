"""Direct-to-Earth (DTE) line-of-sight: Earth above both 0 deg and the terrain mask."""
import numpy as np

from .horizon import HorizonProfile
from .timeseries import longest_run_hours, runs


def dte_visible(earth_az, earth_el, horizon: HorizonProfile):
    return earth_el > np.maximum(horizon.at(earth_az), 0.0)


def dte_stats(visible, step_hours):
    return {
        "dte_fraction": float(visible.mean()),
        "longest_outage_hours": longest_run_hours(~visible, step_hours),
        "longest_contact_hours": longest_run_hours(visible, step_hours),
        "n_contact_windows": int(runs(visible)[0].size),
    }
