"""Integration tests on real NASA data (skipped if `python -m data_engine` was not run)."""
import time

import pytest

from data_engine import config

pytestmark = pytest.mark.skipif(
    not (config.LOLA_DIR / f"{config.LOLA_PRODUCT['name']}.img").exists()
    or not (config.SPICE_DIR / "de440s.bsp").exists(),
    reason="NASA data not downloaded",
)


@pytest.fixture(scope="module")
def dem():
    from data_engine.lola_downloader import load_dem
    return load_dem()


def test_sun_elevation_is_shallow_at_pole():
    from data_engine.spice_loader import get_sun_earth_positions
    r = get_sun_earth_positions(-89.9, 0.0, "2026-11-14T00:00:00")
    assert abs(r["sun"]["elevation_deg"]) < 2.0
    assert 1.4e8 < r["sun"]["distance_km"] < 1.6e8


def test_dem_range_matches_pds_label(dem):
    assert dem.dn.min() == -11002 and dem.dn.max() == 14054  # DERIVED_MIN/MAX in .lbl


def test_horizon_under_half_second(dem):
    from computation_core.horizon import compute_horizon
    t = time.perf_counter()
    compute_horizon(dem, -89.9, 0.0)
    assert time.perf_counter() - t < 0.5


def test_full_site_analysis_under_two_seconds(dem):
    from computation_core.site_analysis import analyze_site
    from data_engine.spice_loader import load_kernels
    load_kernels()  # one-time kernel load excluded from the timing
    t = time.perf_counter()
    r = analyze_site(dem, -85.9, 0.0, "2026-11-14T00:00:00", days=30)
    assert time.perf_counter() - t < 2.0
    assert 0.0 <= r["illumination_fraction"] <= 1.0 and 0.0 <= r["dte_fraction"] <= 1.0


@pytest.mark.parametrize("lat,lon,expected", [
    (-89.625, 119.13, "crater_floor_or_basin"),   # Shackleton floor
    (-87.496, 347.64, "crater_floor_or_basin"),   # Haworth floor
    (-87.63, 312.8, "crater_floor_or_basin"),     # Shoemaker floor
    (-85.987, 2.08, "summit_or_massif"),          # Malapert summit
    (-85.23, 36.6, "summit_or_massif"),           # Leibnitz plateau
])
def test_terrain_classifier_known_features(dem, lat, lon, expected):
    from computation_core.terrain_context import classify_terrain
    assert classify_terrain(dem, lat, lon)["class"] == expected


def test_query_shackleton_blocked_by_its_own_rim(dem):
    """Floor of Shackleton (radius ~10.5 km): the Sun must be hidden by terrain roughly one radius away."""
    from computation_core.mission_query import mission_query
    r = mission_query(dem, -89.625, 119.13, "2026-09-12T00:00:00", days=30)
    sun = r["now"]["sun"]
    assert sun["state"] == "terrain_blocked"
    assert 8.0 < sun["blocker_distance_km"] < 13.0
    assert r["window"]["sunlit_fraction"] == 0.0
    assert any(f["severity"] == "critical" for f in r["findings"])


def test_query_now_matches_timeseries(dem):
    from computation_core.mission_query import mission_query
    from computation_core.site_analysis import site_timeseries
    r = mission_query(dem, -85.987, 2.08, "2026-09-12T00:00:00", days=5)
    ts = site_timeseries(dem, -85.987, 2.08, "2026-09-12T00:00:00", days=5, step_minutes=30)
    assert abs(r["now"]["sun"]["elevation_deg"] - ts["sun_el"][0]) < 1e-9
    assert (r["now"]["earth"]["state"] == "visible") == bool(ts["earth_visible"][0])


def test_batched_azel_matches_spice_path(dem):
    """The vectorized az/el behind the features must equal spice_loader.topocentric site by site."""
    import numpy as np
    from computation_core.features import _azel
    from data_engine import spice_loader
    et = spice_loader.time_grid("2026-09-12T00:00:00", 3, 60)
    sun = spice_loader.body_vectors_me("SUN", et)
    lats, lons = np.array([-85.987, -88.0, -89.625]), np.array([2.08, 200.0, 119.13])
    az, el = _azel(sun, lats, lons)
    for i in range(3):
        a, e, _ = spice_loader.topocentric("SUN", lats[i], lons[i], et, body_vec=sun)
        assert np.allclose(az[i], a) and np.allclose(el[i], e)


def test_proxy_horizon_tracks_exact_horizon(dem):
    import numpy as np
    from computation_core.features import proxy_horizon_deg
    from computation_core.horizon import compute_horizon
    errs = []
    for lat, lon in [(-85.987, 2.08), (-87.496, 347.64), (-88.0, 200.0), (-86.5, 120.0)]:
        hp = compute_horizon(dem, lat, lon, 2.0)
        errs.append(proxy_horizon_deg(dem, lat, lon, hp.az_deg) - hp.el_deg)
    assert np.abs(np.concatenate(errs)).mean() < 0.4  # ~0.2 deg measured on random sites


def test_physics_proxies_close_to_exact_targets(dem):
    import numpy as np
    from computation_core.coordinates import polar_stereo_forward
    from computation_core.features import FEATURE_NAMES, STEP_MIN, WINDOW_DAYS, compute_features
    from computation_core.scratch_nn.landing_scorer import PROXY_IDX
    from computation_core.site_analysis import exact_targets
    from data_engine import spice_loader
    et = spice_loader.time_grid("2027-03-01T00:00:00", WINDOW_DAYS, STEP_MIN)
    sun, earth = spice_loader.body_vectors_me("SUN", et), spice_loader.body_vectors_me("EARTH", et)
    lats = np.array([-85.987, -87.496, -88.0, -86.5, -89.2])
    lons = np.array([2.08, 347.64, 200.0, 120.0, 45.0])
    x, y = polar_stereo_forward(lats, lons)
    X = compute_features(dem, x, y, sun, earth)
    Y = np.array([exact_targets(dem, la, lo, et, sun, earth) for la, lo in zip(lats, lons)])
    assert X.shape == (5, len(FEATURE_NAMES)) and np.isfinite(X).all()
    assert np.abs(X[:, PROXY_IDX] - Y).mean() < 0.03


def _model_ready():
    from computation_core.scratch_nn.landing_scorer import MODEL_PATH, Ensemble
    try:
        Ensemble.load(MODEL_PATH)
        return True
    except Exception:  # missing, or from an older feature set
        return False


model_missing = pytest.mark.skipif(not _model_ready(), reason="ensemble surrogate not trained")


@model_missing
def test_region_map_decision_layer_refines_most_uncertain_pixels_exactly(dem):
    import numpy as np
    from computation_core.features import STEP_MIN, WINDOW_DAYS
    from computation_core.region_map import region_map
    from computation_core.scratch_nn.landing_scorer import Ensemble
    from computation_core.site_analysis import exact_targets
    from data_engine import spice_loader
    model, args = Ensemble.load(), (-85.987, 2.08, "2026-09-12T00:00:00")
    m = region_map(dem, model, *args, half_size_km=5, n=12)
    assert m["sunlit_fraction"].shape == (12, 12) and np.nansum(m["source"]) == 0
    s = m["suitability_index"]
    assert np.nanmin(s) >= 0 and np.nanmax(s) <= 100

    r = region_map(dem, model, *args, half_size_km=5, n=12, refine_budget_s=0.3)
    refined = r["source"] == 1
    k = int(refined.sum())
    assert k >= 1
    top_k = np.argsort(-m["uncertainty"].ravel())[:k]  # the decision: most uncertain first
    assert set(top_k) == set(np.flatnonzero(refined.ravel()))
    i, j = np.argwhere(refined)[0]
    et = spice_loader.time_grid(args[2], WINDOW_DAYS, STEP_MIN)
    exact = exact_targets(dem, float(r["lat"][i, j]), float(r["lon"][i, j]), et,
                          spice_loader.body_vectors_me("SUN", et), spice_loader.body_vectors_me("EARTH", et))
    assert abs(r["sunlit_fraction"][i, j] - exact[0]) < 1e-12


def test_vehicle_assessment_uses_vehicle_limits_and_flags_provenance(dem):
    from computation_core.site_analysis import analyze_site
    from computation_core.vehicle_assessment import assess_vehicle, load_profiles
    profiles = load_profiles()
    assert len(profiles) == 27
    m = analyze_site(dem, -85.987, 2.08, "2026-09-12T00:00:00", days=30)

    lm = assess_vehicle(profiles["apollo-15"], m, dem, -85.987, 2.08)  # the five hardware inputs are NASA-verified
    # the two loads come from the 2026-11-15 project dataset: present, but not checked against an agency source
    assert lm["data_quality"] == {"inputs_verified": 5, "inputs_unverified": 2, "inputs_unknown": 0}
    assert lm["checks"]["slope"]["limit_deg"] == 12
    assert abs(lm["checks"]["night"]["max_average_draw_w"] - 56000 / 720) < 1e-9  # battery-only: whole window
    assert lm["index_weights"]["sun"] == 0  # sunlight does not power a battery-only LM

    ch4 = assess_vehicle(profiles["chang-e-4-yutu-2"], m, dem, -85.987, 2.08)
    assert ch4["index_weights"]["dte"] == 0  # relay: Earth in view does not matter
    assert ch4["checks"]["night"]["assessed"] is True  # its battery (2,380 Wh) is now given by the dataset, so the night is checked

    luno = assess_vehicle(profiles["lunokhod-1"], m, dem, -85.987, 2.08)
    assert any("unverified" in f["message"] for f in luno["findings"])
    for r in (lm, ch4, luno):
        assert 0 <= r["vehicle_index"] <= 100


def test_global_lola_matches_known_elevations_and_the_polar_grid():
    """LDEM_64: the Moon's real extremes, Tranquility Base, and agreement with the 40 m polar grid."""
    import numpy as np
    from computation_core.global_dem import load_global_dem
    from data_engine import config
    from data_engine.lola_downloader import load_dem
    g = load_global_dem(config.GLOBAL_DEM_PATH)
    if g is None:
        pytest.skip("LDEM_64 not downloaded")
    assert abs(float(g.dn.min()) * 0.5 + 9114.5) < 1 and abs(float(g.dn.max()) * 0.5 - 10757.0) < 1  # NASA: -9.11 / +10.76 km
    assert -2300 < float(g.height(0.674, 23.473)) < -1600  # Apollo 11, about 1.9 km below the reference sphere
    dem = load_dem()
    for lat, lon in [(-85.987, 2.08), (-86.5, 100.0), (-88.0, 200.0), (-87.0, 300.0)]:
        assert abs(float(g.height(lat, lon)) - float(dem.height_at_latlon(lat, lon))) < 150, (lat, lon)


def test_apollo_landing_sites_match_the_nasa_nssdca_table():
    """Second source for the mission coordinates: NASA NSSDCA 'Apollo Lunar Landings' (DE421 mean-Earth frame)."""
    import json
    from pathlib import Path
    nasa = {"apollo-11": (0.674, 23.473), "apollo-12": (-3.013, -23.422), "apollo-14": (-3.646, -17.472),
            "apollo-15": (26.132, 3.633), "apollo-16": (-8.973, 15.501), "apollo-17": (20.191, 30.772)}
    sites = {s["id"]: s for s in json.loads((Path(config.DATA_DIR).parent / "data_engine" / "lunar_sites.json").read_text(encoding="utf-8"))}
    for k, (lat, lon) in nasa.items():
        assert abs(sites[k]["lat"] - lat) < 0.002 and abs(sites[k]["lon"] - lon) < 0.002, k  # about 60 m


def test_spacecraft_dataset_is_complete_and_every_correction_is_on_record():
    import json
    from pathlib import Path
    root = Path(config.DATA_DIR).parent
    craft = json.loads((root / "frontend" / "src" / "data" / "spacecrafts.json").read_text(encoding="utf-8"))
    assert len(craft) == 52 and len({c["id"] for c in craft}) == 52
    assert all(c["checkLevel"] in ("nasa", "wikipedia", "unchecked") and c["sourceUrl"].startswith("https://") for c in craft)
    by = {c["id"]: c for c in craft}
    # NASA's own numbers, found on the NSSDCA pages: these were wrong in the dataset and are now corrected
    assert by["lro"]["massKg"] == "1846" and by["lunar-orbiter-1"]["massKg"] == "385.6" and by["apollo-8"]["massKg"] == "28817"
    assert by["ranger-7"]["height"] == "3.60" and by["grail-a-b"]["massKg"] == "202.4"
    for c in craft:
        for ch in c["corrections"]:
            assert c[ch["field"]] == ch["to"] and ch["from"] != ch["to"] and ch["why"]


def _fake_metrics(m, **over):
    return {**m, **over}


def test_veto_gates_cap_the_index_whatever_the_other_terms_add_up_to(dem):
    """A great Sun/Earth view cannot buy back ground the vehicle cannot land on, or a night that kills it."""
    import copy

    import numpy as np
    from computation_core.scratch_nn.landing_scorer import suitability_index
    from computation_core.site_analysis import analyze_site
    from computation_core.vehicle_assessment import VETO_CAP, assess_vehicle, load_profiles
    p = load_profiles()["im-1-odysseus"]
    m = analyze_site(dem, -85.987, 2.08, "2026-09-12T00:00:00", days=30)
    base = assess_vehicle(p, m, dem, -85.987, 2.08)
    # same sunlight and Earth view, but a cliff under the lander
    steep = assess_vehicle({**p, "max_slope_deg": {"value": 0.5, "status": "verified"}}, m, dem, -85.987, 2.08)
    assert steep["limiting_factor"] == "slope" and steep["vehicle_index"] <= VETO_CAP < steep["weighted_index"]
    assert any("capped" in f["message"] for f in steep["findings"] if f["severity"] == "critical")
    assert base["vehicle_index"] <= base["weighted_index"] + 1e-9
    # generic scorer: a perfect Sun/Earth site on 30-degree ground is capped, flat ground is not
    pred = np.array([[1.0, 1.0, 0.0, 0.0]])
    assert suitability_index(pred, 30.0)[0] <= 15.0 and suitability_index(pred, 0.0)[0] == 100.0
    # no Earth view at all: a direct-to-Earth vehicle is capped
    dark_link = assess_vehicle(copy.deepcopy(p), _fake_metrics(m, dte_fraction=0.0, longest_outage_hours=720.0), dem, -85.987, 2.08)
    assert dark_link["limiting_factor"] in ("comms", "slope", "thermal") and dark_link["vehicle_index"] <= VETO_CAP


def test_battery_is_simulated_and_changes_the_score_once_the_load_is_known(dem):
    import numpy as np
    from computation_core.site_analysis import analyze_site
    from computation_core.vehicle_assessment import assess_vehicle, load_profiles, simulate_battery, sustainable_load_w
    # arithmetic of the simulation: 100 Wh, 10 W array lit half the time, 4 W load
    lit = np.array([1.0] * 12 + [0.0] * 12)
    soc, dead = simulate_battery(lit, 1.0, 100.0, 10.0, 4.0)
    assert not dead.any() and abs(soc[11] - 100.0) < 1e-9 and abs(soc[-1] - (100.0 - 12 * 4.0)) < 1e-9
    assert sustainable_load_w(lit, 1.0, 100.0, 10.0) > 4.0 and sustainable_load_w(lit, 1.0, 200.0, 10.0) > sustainable_load_w(lit, 1.0, 100.0, 10.0)
    # on a real site: with the survival load known, a bigger battery really does lift the score (and removes blackouts)
    p = load_profiles()["im-1-odysseus"]
    m = analyze_site(dem, -85.987, 2.08, "2026-09-12T00:00:00", days=30)
    small = {**p, "battery_wh": {"value": 100.0, "status": "unverified"}, "survival_power_w": {"value": 60.0, "status": "unverified"}}
    big = {**small, "battery_wh": {"value": 100.0 * 8, "status": "unverified"}}
    a, b = assess_vehicle(small, m, dem, -85.987, 2.08), assess_vehicle(big, m, dem, -85.987, 2.08)
    assert a["checks"]["energy"]["blackout_hours"] > b["checks"]["energy"]["blackout_hours"]
    assert b["weighted_index"] >= a["weighted_index"]
    # the load unknown: the battery is reported (sustainable draw) but not scored, and nothing is invented
    c = assess_vehicle({**p, "survival_power_w": {"value": None, "status": "unknown"}, "payload_power_w": {"value": None, "status": "unknown"}}, m, dem, -85.987, 2.08)
    assert c["checks"]["energy"].get("load_known") is False and c["index_weights"]["energy"] == 0


def test_long_night_without_heat_is_a_thermal_cap_not_just_a_note(dem):
    from computation_core.site_analysis import analyze_site
    from computation_core.vehicle_assessment import assess_vehicle, load_profiles
    profiles = load_profiles()
    for day in range(1, 29, 3):  # a start date where the Sun is up for a while before the first long night
        m = analyze_site(dem, -85.987, 2.08, f"2026-10-{day:02d}T00:00:00", days=30)
        if 0 < assess_vehicle({**profiles["im-1-odysseus"], "radioisotope_heat": {"value": False, "status": "unverified"}}, m, dem, -85.987, 2.08)["checks"]["thermal"]["alive_fraction"] < 1:
            break
    cold = assess_vehicle({**profiles["im-1-odysseus"], "radioisotope_heat": {"value": False, "status": "unverified"}}, m, dem, -85.987, 2.08)
    warm = assess_vehicle({**profiles["im-1-odysseus"], "radioisotope_heat": {"value": True, "status": "unverified"}}, m, dem, -85.987, 2.08)
    unknown = assess_vehicle({**profiles["im-1-odysseus"], "radioisotope_heat": {"value": None, "status": "unknown"}}, m, dem, -85.987, 2.08)
    th = cold["checks"]["thermal"]
    assert th["first_long_night_start_h"] is not None and 0 < th["alive_fraction"] < 1
    cap = lambda r: next((v["cap"] for v in r["vetoes"] if v["gate"] == "thermal"), None)
    assert cap(warm) is None and cap(cold) is not None and cap(unknown) is not None
    assert cap(cold) < cap(unknown) < 100  # known-cold is stricter than unknown
    assert cold["vehicle_index"] <= cap(cold) and warm["vehicle_index"] >= cold["vehicle_index"]


# ---------------------------------------------------------------- the garage and the landing analysis tell the same story ----
def _garage_and_profiles():
    import json
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    garage = {g["id"]: g for g in json.loads((root / "frontend/src/data/spacecrafts.json").read_text(encoding="utf-8"))}
    profiles = {p["id"]: p for p in json.loads((root / "data_engine/lander_profiles.json").read_text(encoding="utf-8"))}
    return garage, profiles


def test_garage_file_is_exactly_the_source_file_plus_the_recorded_corrections():
    """If someone edits the source txt or the corrections and forgets to rebuild (or edits the garage json by hand), this fails."""
    import json
    from data_engine import build_spacecrafts as b
    built = b.build_rows()
    current = json.loads(b.OUT.read_text(encoding="utf-8"))
    assert [json.loads(json.dumps(r, ensure_ascii=False)) for r in built] == current, "run: python -m data_engine.build_spacecrafts"


def test_every_profile_is_a_garage_spacecraft_and_differences_are_explained():
    import re
    garage, profiles = _garage_and_profiles()
    assert set(profiles) <= set(garage)
    num = lambda s: float(re.search(r"\d+(?:\.\d+)?", s.replace(",", ".")).group()) if re.search(r"\d", s) else None  # noqa: E731
    for pid, p in profiles.items():
        g = garage[pid]
        if p["name"] != g["name"]:
            assert g["name"].replace("'", "").split(" (")[0].split(" ")[0] in p["name"].replace("'", ""), (pid, p["name"], g["name"])
        # solar: where the garage cell is a plain number it must be the profile's value; where it lists parts, the profile must say which part it took
        cell = g["solarW"]
        if cell.strip().replace(".", "").isdigit():
            assert p["solar_w"]["value"] == float(cell), (pid, cell, p["solar_w"])
        elif p["solar_w"]["value"] is not None and num(cell) is not None and p["solar_w"]["value"] != num(cell):
            assert p["solar_w"].get("note"), f"{pid}: solar differs from the garage cell {cell!r} without a note"
        # battery: a profile value that is not the garage's own Wh figure must carry its working
        bc = g["batteryCapacity"]
        if p["battery_wh"]["value"] is not None and "Wh" not in bc:
            assert p["battery_wh"].get("note") or p["battery_wh"].get("source"), f"{pid}: battery value without a note or source"
        if "Wh" in bc and p["battery_wh"]["value"] is not None and p["battery_wh"]["value"] not in [float(x) for x in re.findall(r"(\d+(?:\.\d+)?)\s*Wh", bc)]:
            assert p["battery_wh"]["status"] == "verified" and "dataset lists" in p["battery_wh"].get("note", ""), (pid, bc, p["battery_wh"])


def test_blue_ghost_has_one_solar_figure_everywhere():
    garage, profiles = _garage_and_profiles()
    bg = next(i for i, g in garage.items() if g["name"].startswith("Blue Ghost"))
    assert float(garage[bg]["solarW"]) == profiles[bg]["solar_w"]["value"] == 400.0


# ---------------------------------------------------------------- one DEM for the whole Moon ----
def _site_dem():
    from computation_core.global_dem import load_global_dem
    from computation_core.site_dem import SiteDEM
    from data_engine import config
    from data_engine.lola_downloader import load_dem
    g = load_global_dem(config.GLOBAL_DEM_PATH)
    if g is None:
        import pytest
        pytest.skip("LDEM_64 not downloaded")
    polar = load_dem()
    return polar, SiteDEM(polar, g)


def test_site_dem_changes_nothing_where_the_polar_grid_exists():
    from computation_core.site_analysis import analyze_site
    from data_engine import spice_loader
    spice_loader.load_kernels()
    polar, site = _site_dem()
    a = analyze_site(polar, -85.987, 2.08, "2026-12-01T00:00:00", days=5)
    b = analyze_site(site, -85.987, 2.08, "2026-12-01T00:00:00", days=5)
    assert a == b and site.source(-85.987, 2.08) == "lola_40m"


def test_site_dem_uses_the_whole_moon_grid_elsewhere_and_agrees_at_the_seam():
    """Just outside the detailed grid the two sources must tell the same story, or the seam at 85S would show in results."""
    from computation_core.terrain_slope import site_hazard_stats
    polar, site = _site_dem()
    assert site.source(10.0, 20.0) == "ldem_470m" and site.source(-89.5, 0.0) == "lola_40m"
    h_polar, h_global = float(polar.height_at_latlon(-86.0, 30.0)), float(site.global_dem.height(-86.0, 30.0))
    assert abs(h_polar - h_global) < 400.0                           # the coarse grid is a smoothed version of the fine one
    s_polar = site_hazard_stats(site, -86.0, 30.0, 1500.0)["mean_slope_deg"]
    s_global = site_hazard_stats(type("OnlyGlobal", (), {"global_dem": site.global_dem, "covers": lambda *_: False})(), -86.0, 30.0, 1500.0)["mean_slope_deg"]
    assert abs(s_polar - s_global) < 6.0                              # regional slope, same order of size


def test_global_horizon_and_slope_have_the_right_shape():
    from computation_core.horizon import compute_horizon
    from computation_core.terrain_slope import site_hazard_stats
    _, site = _site_dem()
    hz = compute_horizon(site, 0.674, 23.473, 2.0, n_az=90)
    assert hz.el_deg.shape == (90,) and hz.el_deg.min() > -5 and hz.el_deg.max() < 30   # a mare: a low skyline
    st = site_hazard_stats(site, 0.674, 23.473, 500.0)
    assert st["terrain_source"] == "ldem_470m" and st["area_radius_m"] >= 1500 and 0 <= st["safe_area_fraction"] <= 1
    assert st["mean_slope_deg"] < 5                                    # Mare Tranquillitatis is flat
