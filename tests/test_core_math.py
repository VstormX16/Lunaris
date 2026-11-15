"""Math-correctness tests on analytic geometry (not science data)."""
import numpy as np

from computation_core.constants import MOON_RADIUS_M
from computation_core.coordinates import (latlon_to_unit, polar_stereo_forward,
                                          polar_stereo_inverse, unit_to_polar_xy,
                                          vector_to_azel)
from computation_core.dem import LolaDEM
from computation_core.horizon import compute_horizon
from computation_core.timeseries import longest_run_hours, runs


def test_stereo_roundtrip():
    lat, lon = np.array([-89.9, -87.5, -85.2]), np.array([10.0, 200.0, 359.0])
    la, lo = polar_stereo_inverse(*polar_stereo_forward(lat, lon))
    assert np.allclose(la, lat) and np.allclose(lo, lon)


def test_unit_projection_matches_latlon():
    lat, lon = -88.3, 123.0
    assert np.allclose(unit_to_polar_xy(latlon_to_unit(lat, lon)), polar_stereo_forward(lat, lon))


def test_azel_cardinal_directions():
    lat, lon = -80.0, 30.0
    up = latlon_to_unit(lat, lon)
    az, el = vector_to_azel(up, lat, lon)
    assert abs(el - 90) < 1e-9
    zenith_ish = up + 0.01 * np.array([0, 0, 1.0])  # nudged toward north pole axis -> az ~ north
    az, _ = vector_to_azel(zenith_ish, lat, lon)
    assert abs(az - 0) < 1e-6 or abs(az - 360) < 1e-6


def test_runs():
    m = np.array([0, 1, 1, 0, 1, 1, 1, 0], bool)
    s, l = runs(m)
    assert list(s) == [1, 4] and list(l) == [2, 3]
    assert longest_run_hours(m, 2.0) == 6.0


def test_horizon_sees_known_ridge():
    """A 1000 m ridge 10 km toward the pole must appear at az 180 with ~atan(1000/10000)."""
    n, px = 2001, 40.0
    dem = LolaDEM(np.zeros((n, n), np.int16), px)
    lat, lon = -89.0, 0.0
    _, y_site = polar_stereo_forward(lat, lon)
    row = int(round(dem.c0 - (y_site - 10_000.0) / px))
    dem.dn[row - 3 : row + 4, :] = 2000  # DN*0.5 = 1000 m, spans all columns
    hp = compute_horizon(dem, lat, lon, observer_height_m=0.0, n_az=360)
    expected = np.degrees(np.arctan(1000.0 / 10_000.0))
    assert abs(hp.at(180.0) - expected) < 0.3
    assert hp.at(0.0) < 0.5  # open sky on the opposite side


def test_vectorized_run_lengths_match_reference():
    from computation_core.timeseries import count_runs_rows, longest_run_rows
    rng = np.random.default_rng(4)
    m = rng.random((60, 300)) < 0.6
    m[0], m[1] = True, False  # all-true and all-false rows
    longest, count = longest_run_rows(m), count_runs_rows(m)
    for i in range(len(m)):
        starts, lengths = runs(m[i])
        assert longest[i] == (lengths.max() if lengths.size else 0)
        assert count[i] == starts.size
