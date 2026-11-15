"""Mission success probability and the call, on real NASA data (skipped if `python -m data_engine` was not run)."""
import pytest

from data_engine import config

pytestmark = pytest.mark.skipif(
    not (config.LOLA_DIR / f"{config.LOLA_PRODUCT['name']}.img").exists()
    or not (config.SPICE_DIR / "de440s.bsp").exists(),
    reason="NASA data not downloaded",
)

MALAPERT = (-85.987, 2.08)
START = "2027-01-20T00:00:00"


@pytest.fixture(scope="module")
def env():
    from computation_core.vehicle_assessment import load_profiles
    from data_engine import spice_loader
    from data_engine.lola_downloader import load_dem
    spice_loader.load_kernels()
    return load_dem(), load_profiles()


def _risk(env, profile, start=START, site=MALAPERT):
    from computation_core.mission_risk import mission_risk
    return mission_risk(profile, env[0], *site, start, 30, 60)


def _with(profile, **values):
    out = dict(profile)
    for k, v in values.items():
        out[k] = {"value": v, "status": "verified"} if not isinstance(v, dict) else v
    return out


def test_same_question_same_numbers_and_a_coherent_result(env):
    p = env[1]["chandrayaan-3-vikram-pragyan"]
    a, b = _risk(env, p), _risk(env, p)
    assert a == b  # a fixed seed: the page, the report and the assistant can never disagree
    assert 0 <= a["p_success"] <= a["p_landing"] <= 1
    assert sum(m["share"] for m in a["failure_modes"]) + a["p_success"] == pytest.approx(1.0)
    curve = a["survival_curve"]
    assert curve[0] == pytest.approx(a["p_landing"]) and all(x >= y - 1e-12 for x, y in zip(curve, curve[1:]))  # nobody comes back to life
    ms = [m["p"] for m in a["milestones"]]
    assert ms == sorted(ms, reverse=True) and a["operating_days"]["p10"] <= a["operating_days"]["p50"] <= a["operating_days"]["p90"]


def test_unknown_inputs_are_not_guessed_they_lower_the_confidence(env):
    p = env[1]["chandrayaan-3-vikram-pragyan"]
    r = _risk(env, _with(p, radioisotope_heat={"value": None, "status": "unknown"}))
    assert r["assessed"]["cold"] is False and r["confidence"]["level"] == "low"
    assert not any(m["mode"] == "cold" for m in r["failure_modes"])  # it cannot fail by a cause nobody can check
    r2 = _risk(env, _with(p, survival_power_w={"value": None, "status": "unknown"}))
    assert r2["assessed"]["power"] is False and "power" in r2["confidence"]["reasons"][0]


def test_physics_points_the_right_way(env):
    p = env[1]["chandrayaan-3-vikram-pragyan"]
    base = _risk(env, p)
    big = _risk(env, _with(p, battery_wh=p["battery_wh"]["value"] * 6))
    warm = _risk(env, _with(p, radioisotope_heat=True))
    assert big["operating_days"]["mean"] >= base["operating_days"]["mean"]  # more battery never shortens the mission
    assert warm["p_alive_end"] >= base["p_alive_end"] and not any(m["mode"] == "cold" for m in warm["failure_modes"])  # a heat source never freezes
    relay = _risk(env, _with(p, comms="relay"))
    assert relay["p_comms"] == 1.0


def test_a_battery_only_vehicle_ends_when_its_battery_does(env):
    p = env[1]["apollo-15"]
    r = _risk(env, p)
    bat, load = p["battery_wh"]["value"], p["survival_power_w"]["value"]
    assert r["operating_days"]["p90"] * 24 <= 1.6 * bat / load  # it cannot work longer than its battery lasts (+ spread)
    assert r["failure_modes"][0]["mode"] == "power"


def test_the_call_takes_the_more_cautious_reading_and_says_why(env):
    from computation_core.assistant import tools
    from computation_core.assistant.tools import Scenario
    from computation_core.verdict import ORDER
    dem, P = env
    for vid in ("chandrayaan-3-vikram-pragyan", "apollo-15", "lunokhod-1"):
        m, a = tools.assess(dem, P[vid], Scenario(START[:10], vid, *MALAPERT))
        v, r = a["verdict"], a["risk"]
        by_p = "go" if r["p_success"] >= 0.8 else "conditional" if r["p_success"] >= 0.5 else "weak" if r["p_success"] >= 0.2 else "stop"
        assert ORDER.index(v["kind"]) >= ORDER.index(by_p)  # never more optimistic than the simulation
        assert v["probability_line"] and f"{100 * r['p_success']:.0f}%" in v["probability_line"]
    m, a = tools.assess(dem, P["apollo-15"], Scenario(START[:10], "apollo-15", *MALAPERT))
    assert a["verdict"]["kind"] == "stop"  # a two-day battery lander cannot do a week at the pole


def test_back_test_on_real_landings():
    """The simulation, untouched, at real landings (real date and site) against what really happened (public mission records)."""
    from computation_core.global_dem import load_global_dem
    from computation_core.mission_risk import mission_risk
    from computation_core.site_dem import SiteDEM
    from computation_core.vehicle_assessment import load_profiles
    from data_engine import spice_loader
    from data_engine.lola_downloader import load_dem
    if not config.GLOBAL_DEM_PATH.exists():
        pytest.skip("whole-Moon LOLA grid not downloaded")
    spice_loader.load_kernels()
    dem, P = SiteDEM(load_dem(), load_global_dem(config.GLOBAL_DEM_PATH)), load_profiles()
    # Chandrayaan-3 Vikram: landed 23 Aug 2023, put to sleep at sunset ~11.5 days later, never woke in the night
    r = mission_risk(P["chandrayaan-3-vikram-pragyan"], dem, -69.373, 32.319, "2023-08-23T12:34:00", 30, 60)
    assert 9 <= r["operating_days"]["p50"] <= 14 and r["p_alive_end"] < 0.1
    # Luna 9: battery only, about 3 days
    r = mission_risk(P["luna-9"], dem, 7.08, 295.63, "1966-02-03T18:45:00", 30, 60)
    assert 2 <= r["operating_days"]["p50"] <= 5
    # Chang'e 4: radioisotope heater, worked for years: alive after the first month
    r = mission_risk(P["chang-e-4-yutu-2"], dem, -45.444, 177.588, "2019-01-03T02:26:00", 30, 60)
    assert r["p_alive_end"] > 0.9
