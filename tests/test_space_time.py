"""Space Time against independent references captured from the US Naval Observatory and JPL Horizons.

The reference numbers below were fetched once (2026-11-14) and frozen, so the tests need no network.
"""
import numpy as np
import pytest

from data_engine import config

pytestmark = pytest.mark.skipif(not (config.SPICE_DIR / "de440s.bsp").exists(), reason="NASA data not downloaded")

START = "2026-11-14T00:00:00"

# USNO moon phases API (minute resolution, UT), api/moon/phases/date?date=2026-11-14&nump=12
USNO = {
    "first_quarter": ["2026-11-17T11:48", "2026-12-17T05:42", "2027-01-15T20:34"],
    "full_moon": ["2026-11-24T14:53", "2026-12-24T01:28", "2027-01-22T12:17"],
    "last_quarter": ["2026-12-01T06:08", "2026-12-30T18:59", "2027-01-29T10:55"],
    "new_moon": ["2026-12-09T00:52", "2027-01-07T20:24", "2027-02-06T15:56"],
}


@pytest.fixture(scope="module")
def sl():
    from data_engine import spice_loader
    spice_loader.load_kernels()
    return spice_loader


def test_all_twelve_phases_match_usno_within_a_minute(sl):
    from computation_core.space_time import next_phase
    for name, expected in USNO.items():
        t = sl.utc_to_et(START)
        for exp in expected:
            t = next_phase(t, name)
            err = abs(t - sl.utc_to_et(exp + ":00"))
            assert err < 75.0, f"{name} {exp}: {err:.0f} s off"  # USNO publishes minutes; both round differently
            t += 86400.0  # search the following cycle


def test_phase_search_goes_forward_and_backward(sl):
    from computation_core.space_time import elongation_deg, next_phase
    et = sl.utc_to_et("2026-11-20T00:00:00")
    nxt, prv = next_phase(et, "new_moon"), next_phase(et, "new_moon", direction=-1)
    assert prv < et < nxt and 28 * 86400 < nxt - prv < 31 * 86400
    for t in (nxt, prv):
        e = float(elongation_deg(np.array([t]))[0])
        assert min(e, 360 - e) < 1e-6  # exactly at the phase, not just near it


def test_moon_state_matches_horizons(sl):
    from computation_core.space_time import moon_state
    s = moon_state(sl.utc_to_et("2026-11-14T12:00:00"))
    assert abs(100 * s["illumination"] - 23.156) < 0.01           # Horizons Illu%
    assert abs(s["sub_earth"]["lon"] - 358.0804) < 0.01 and abs(s["sub_earth"]["lat"] - 3.8761) < 0.01
    assert abs(s["sub_solar"]["lon"] - 120.5239) < 0.01 and abs(s["sub_solar"]["lat"] + 1.5573) < 0.01
    assert s["phase_name"] == "Waxing crescent" and 5.0 < s["age_days"] < 5.4


def test_distance_agrees_with_horizons_once_light_time_is_included(sl):
    """Horizons' range is light-time corrected; with the same correction SPICE matches it to metres.
    (Perigee/apogee use the geometric distance, the almanac convention, so they differ by minutes.)"""
    et = sl.utc_to_et("2026-11-25T21:00:00")
    lt = np.linalg.norm(sl.rel_positions("MOON", "EARTH", "J2000", et, "LT")[0])
    assert abs(lt - 359336.1) < 0.2
    geometric = np.linalg.norm(sl.rel_positions("MOON", "EARTH", "J2000", et, "NONE")[0])
    assert 5 < geometric - lt < 15  # Earth moves ~30 km/s during the light time


def test_perigee_is_the_true_minimum_of_the_geometric_distance(sl):
    from computation_core.space_time import _distance, global_events
    et0 = sl.utc_to_et(START)
    peri = next(e for e in global_events(et0) if e["id"] == "perigee")
    around = peri["et"] + np.arange(-900.0, 901.0, 1.0)  # +-15 min at 1 s
    assert abs(around[int(np.argmin(_distance(around)))] - peri["et"]) < 5.0


def test_sun_south_extreme_is_stable_and_flagged_approximate(sl):
    from computation_core.space_time import global_events
    picks = []
    for start in ("2026-11-20T00:00:00", "2027-02-01T00:00:00", "2027-06-01T00:00:00"):
        ev = next(e for e in global_events(sl.utc_to_et(start)) if e["id"] == "sun_south_extreme")
        assert ev["kind"] == "peak" and ev["uncertainty_days"] >= 3
        picks.append(ev["et"])
    assert max(picks) - min(picks) < 3 * 86400  # three different vantage dates, one answer


def test_events_are_sorted_and_future(sl):
    from computation_core.space_time import global_events
    et0 = sl.utc_to_et(START)
    ev = global_events(et0)
    assert [e["et"] for e in ev] == sorted(e["et"] for e in ev) and all(e["et"] > et0 for e in ev)
    assert {"new_moon", "full_moon", "perigee", "apogee", "libration_south"} <= {e["id"] for e in ev}


@pytest.mark.skipif(not (config.LOLA_DIR / f"{config.LOLA_PRODUCT['name']}.img").exists(), reason="LOLA not downloaded")
def test_site_transitions_flip_exactly_where_reported_and_agree_with_the_analysis_grid(sl):
    from computation_core.mission_query import mission_query
    from computation_core.space_time import site_events
    from data_engine.lola_downloader import load_dem
    dem = load_dem()
    lat, lon = -85.987, 2.08
    et0 = sl.utc_to_et("2026-11-14T12:00:00")
    ev = site_events(dem, lat, lon, et0)["sun"]
    first = ev["transitions"][0]
    assert ev["up_now"] is False and first["becomes"] is True  # a sunrise is next

    from computation_core.horizon import compute_horizon
    from computation_core.solar_illumination import lit_fraction
    hz = compute_horizon(dem, lat, lon, 2.0)
    lit = lambda t: bool(lit_fraction(*sl.topocentric("SUN", lat, lon, np.array([t])), hz)[0] >= 0.5)  # noqa: E731
    assert (lit(first["et"] - 1.0), lit(first["et"] + 1.0)) == (False, True)  # flips within +-1 s

    coarse = mission_query(dem, lat, lon, "2026-11-14T12:00:00", days=10, step_minutes=30)["next"]["sun_change"]
    assert 0 <= coarse["in_hours"] * 3600 - (first["et"] - et0) <= 30 * 60 + 1  # first 30-min sample after the flip


@pytest.mark.skipif(not (config.LOLA_DIR / f"{config.LOLA_PRODUCT['name']}.img").exists(), reason="LOLA not downloaded")
def test_permanently_shadowed_floor_has_no_sunrise(sl):
    from computation_core.space_time import site_events
    from data_engine.lola_downloader import load_dem
    ev = site_events(load_dem(), -89.625, 119.13, sl.utc_to_et(START), days=30)["sun"]
    assert ev["up_now"] is False and ev["transitions"] == []


# USNO rstt/oneday (UT, truncated to the minute), 2026-11-14: (lat, lon), [(time, becomes_up)] in order of occurrence
USNO_EARTH = {
    "Istanbul": ((41.01, 28.98), [("04:50", True), ("14:46", False)]),
    "Reykjavik": ((64.15, -21.94), [("09:53", True), ("16:30", False)]),
    "Sydney": ((-33.87, 151.21), [("08:35", False), ("18:44", True)]),
}


def test_earth_sunrise_and_sunset_match_usno(sl):
    from computation_core.earth_sun import earth_sun
    et0 = sl.utc_to_et("2026-11-14T00:00:00")
    for name, ((lat, lon), ref) in USNO_EARTH.items():
        events = earth_sun(lat, lon, et0, days=1.0)["transitions"]
        assert len(events) >= len(ref), name
        for (hhmm, rises), t in zip(ref, events):
            assert t["becomes"] is rises, f"{name} {hhmm}"
            late = t["et"] - sl.utc_to_et(f"2026-11-14T{hhmm}:00")
            assert 0 <= late < 61, f"{name} {hhmm}: {late:.0f} s"  # USNO drops the seconds, ours are exact


def test_earth_polar_night_and_midnight_sun_have_no_events(sl):
    from computation_core.earth_sun import earth_sun
    night = earth_sun(69.65, 18.96, sl.utc_to_et("2026-12-21T00:00:00"), days=3.0)   # Tromso, midwinter
    day = earth_sun(69.65, 18.96, sl.utc_to_et("2026-06-21T00:00:00"), days=3.0)     # Tromso, midsummer
    assert night["sun_up_now"] is False and night["transitions"] == []
    assert day["sun_up_now"] is True and day["transitions"] == []


def test_earth_sunset_flips_exactly_where_reported(sl):
    from computation_core.earth_sun import _geometry, earth_sun
    et0 = sl.utc_to_et("2026-11-14T00:00:00")
    t = earth_sun(41.01, 28.98, et0, days=1.0)["transitions"][0]
    up = lambda x: bool((lambda g: g[0][0] >= g[1][0])(_geometry(41.01, 28.98, np.array([x]))))  # noqa: E731
    assert (up(t["et"] - 1.0), up(t["et"] + 1.0)) == (False, True)  # sunrise: dark one second before, lit one after


def test_hillshade_image_geometry():
    import io

    from PIL import Image

    from computation_core.dem_view import hillshade_png
    from computation_core.dem import LolaDEM
    rng = np.random.default_rng(0)
    dem = LolaDEM(rng.integers(-3000, 3000, (700, 700)).astype(np.int16), 40.0)  # shape-only check, not science
    png, meta = hillshade_png(dem)
    im = Image.open(io.BytesIO(png))
    a = im.getchannel("A")
    n = im.size[0]
    assert meta == {"size_px": n, "extent_m": 14000.0, "layer": "relief"} and n == 1084  # the size the Space Time page uses
    assert a.getpixel((1, 1)) == 0 and a.getpixel((n // 2, n // 2)) == 255 and a.getpixel((n // 2, 1)) == 255  # disc mask


def test_layer_rendering_geometry_and_legend():
    import io

    from PIL import Image

    from computation_core.dem import LolaDEM
    from computation_core.dem_view import render_layer
    rng = np.random.default_rng(1)
    dem = LolaDEM(rng.integers(-4000, 4000, (600, 600)).astype(np.int16), 40.0)  # geometry check only, not science
    sizes = {}
    for layer in ("relief", "elevation", "slope"):
        data, meta = render_layer(dem, layer, 256)
        im = Image.open(io.BytesIO(data))
        a = im.getchannel("A")
        assert im.size == (256, 256) and a.getpixel((1, 1)) == 0 and a.getpixel((128, 128)) == 255
        assert meta["extent_m"] == 12000.0
        sizes[layer] = len(data)
    lossy, _ = render_layer(dem, "slope", 256, fmt="webp")
    assert len(lossy) < sizes["slope"]
    import pytest
    with pytest.raises(ValueError):
        render_layer(dem, "optical")


def test_day_night_and_far_side_boundary_formula_matches_spice(sl):
    """The Space Time map draws the day/night boundary and the Earth-visible limb with
    tan(lat) = -cos(lon - lon0) / tan(lat0) around the sub-solar / sub-Earth point
    (frontend/src/lib/terminator.ts). Check that rule against SPICE's own elevations on random lunar points."""
    from computation_core.space_time import moon_state
    et = sl.utc_to_et("2026-11-14T12:00:00")
    s = moon_state(et)
    rng = np.random.default_rng(3)

    def beyond_boundary(lat, lon, sub):  # True on the far side of the 90-degree circle from `sub`
        phi = np.radians(sub["lat"] or 0.01)
        t = np.degrees(np.arctan(-np.cos(np.radians(lon - sub["lon"])) / np.tan(phi)))
        return lat < t if sub["lat"] >= 0 else lat > t

    checked = 0
    for _ in range(300):
        lat, lon = float(np.degrees(np.arcsin(rng.uniform(-0.95, 0.95)))), float(rng.uniform(0, 360))
        _, sun_el, _ = sl.topocentric("SUN", lat, lon, np.array([et]))
        _, earth_el, _ = sl.topocentric("EARTH", lat, lon, np.array([et]))
        if abs(sun_el[0]) > 1.5:
            assert beyond_boundary(lat, lon, s["sub_solar"]) == (sun_el[0] < 0), f"night at {lat:.1f},{lon:.1f}"
            checked += 1
        if abs(earth_el[0]) > 1.5:
            assert beyond_boundary(lat, lon, s["sub_earth"]) == (earth_el[0] < 0), f"far side at {lat:.1f},{lon:.1f}"
            checked += 1
    assert checked > 500


def test_site_events_without_terrain_flip_when_the_sun_centre_crosses_the_horizon():
    """Outside the polar DEM the horizon is a smooth sphere: at every reported sunrise/sunset the Sun's centre is
    on the horizon (half its disc, ~0.27 deg, is above it), and the Sun is up exactly between a rise and a set."""
    from computation_core.space_time import site_events
    from data_engine import spice_loader
    spice_loader.load_kernels()
    et0 = spice_loader.utc_to_et("2026-11-14T15:25:32")
    for lat, lon in [(0.0, 0.0), (45.0, 100.0), (-60.0, 200.0), (80.0, 330.0)]:
        ev = site_events(None, lat, lon, et0)
        tr = ev["sun"]["transitions"]
        assert len(tr) >= 3, (lat, lon, tr)  # 60 days is two lunar days at any latitude short of the poles
        for t in tr:
            _, el, _ = spice_loader.topocentric("SUN", lat, lon, np.array([t["et"]]))
            assert abs(el[0]) < 0.35, (lat, lon, el[0])
        assert tr[0]["becomes"] != ev["sun"]["up_now"]
        assert all(a["becomes"] != b["becomes"] for a, b in zip(tr, tr[1:]))
