"""API tests against real data (skipped if data missing)."""
import pytest
from fastapi.testclient import TestClient

from data_engine import config

pytestmark = pytest.mark.skipif(
    not (config.LOLA_DIR / f"{config.LOLA_PRODUCT['name']}.img").exists()
    or not (config.SPICE_DIR / "de440s.bsp").exists(),
    reason="NASA data not downloaded",
)


@pytest.fixture(scope="module")
def client():
    from api.server import app
    with TestClient(app) as c:  # runs lifespan (loads kernels + DEM)
        yield c


def test_health_and_sites(client):
    assert client.get("/health").json()["status"] == "ok"
    names = {s["name"] for s in client.get("/sites").json()}
    assert {"shackleton", "haworth", "malapert"} <= names


def test_analyze_malapert(client):
    r = client.post("/analyze", json={"lat": -85.987, "lon": 2.08})
    assert r.status_code == 200
    j = r.json()
    assert 0.0 < j["illumination_fraction"] <= 1.0
    assert j["terrain"]["class"] == "summit_or_massif"


def test_terrain_and_horizon(client):
    assert client.get("/terrain", params={"lat": -89.625, "lon": 119.13}).json()["class"] == "crater_floor_or_basin"
    h = client.get("/horizon", params={"lat": -85.987, "lon": 2.08, "n_az": 72}).json()
    assert len(h["az_deg"]) == len(h["el_deg"]) == 72


def test_timeseries_lengths(client):
    j = client.post("/timeseries", json={"lat": -85.987, "lon": 2.08, "days": 2, "step_minutes": 60}).json()
    assert len(j["hours"]) == len(j["sun_elevation_deg"]) == len(j["earth_visible"]) == 48


def test_validation_rejects_bad_input(client):
    assert client.post("/analyze", json={"lat": -95.0, "lon": 0.0}).status_code == 422                  # not a latitude
    assert client.post("/map", json={"lat": -50.0, "lon": 0.0, "start_utc": "2026-12-01T00:00:00"}).status_code == 422  # the neural map is polar only
    assert client.get("/terrain", params={"lat": -85.5, "lon": 400.0}).status_code == 422  # bad longitude


def test_query_endpoint(client):
    j = client.post("/query", json={"lat": -85.987, "lon": 2.08, "utc": "2026-09-12T00:00:00", "days": 10}).json()
    assert j["now"]["sun"]["state"] in {"visible", "terrain_blocked", "below_local_horizon"}
    assert j["findings"] and {"severity", "message"} <= set(j["findings"][0])
    assert client.post("/query", json={"lat": -85.987, "lon": 2.08, "utc": "not a date"}).status_code == 422


def test_map_endpoint(client):
    if not client.get("/model").json()["loaded"]:
        pytest.skip("ensemble surrogate not trained")
    j = client.post("/map", json={"lat": -85.987, "lon": 2.08, "start_utc": "2026-09-12T00:00:00", "n": 12,
                                  "refine_budget_s": 0.2}).json()
    assert len(j["suitability_index"]) == 12 and len(j["suitability_index"][0]) == 12
    assert any(v == 1 for row in j["source"] for v in row if v is not None)  # decision layer refined some


def test_map_covers_the_whole_polar_cap(client):
    # the Landing Map page asks for the full 85-90 S cap centred on the pole
    if not client.get("/model").json()["loaded"]:
        pytest.skip("ensemble surrogate not trained")
    j = client.post("/map", json={"lat": -90.0, "lon": 0.0, "start_utc": "2026-09-12T00:00:00", "half_size_km": 155, "n": 24}).json()
    lat, s = j["lat"], j["suitability_index"]
    inside = [(la, v) for rl, rv in zip(lat, s) for la, v in zip(rl, rv) if la is not None and la <= -85]
    assert inside and all(v is not None and 0 <= v <= 100 for _, v in inside)
    assert min(la for rl in lat for la in rl if la is not None) < -89.5  # reaches the pole


def test_area_relief_is_a_real_height_grid_centred_on_the_site(client):
    j = client.get("/area-relief", params={"lat": -85.987, "lon": 2.08, "half_km": 2, "n": 32}).json()
    h = j["heights"]
    assert len(h) == 32 and len(h[0]) == 32 and j["half_m"] == 2000.0
    assert abs(h[15][15]) < 60 and abs(h[16][16]) < 60  # relative to the site: the centre is about 0
    far = client.get("/area-relief", params={"lat": 10, "lon": 20}).json()
    assert far["data_source"] == "ldem_470m" and far["half_m"] == 12000.0  # elsewhere the whole-Moon grid, over a wider square


def test_analyze_has_suitability_index(client):
    j = client.post("/analyze", json={"lat": -85.987, "lon": 2.08, "days": 5}).json()
    assert 0 <= j["suitability_index"] <= 100


def test_operations_lifecycle(client, tmp_path):
    from api.server import _state
    from api.storage import OperationStore
    _state["store"] = OperationStore(tmp_path / "ops.sqlite3")  # isolated from the real history
    r = client.post("/operations", json={"name": "Malapert check", "kind": "analysis",
                                         "params": {"lat": -85.987, "lon": 2.08, "days": 3}})
    assert r.status_code == 201
    op = r.json()
    assert 0 <= op["result"]["suitability_index"] <= 100
    listing = client.get("/operations").json()
    assert listing[0]["id"] == op["id"] and "result" not in listing[0]
    assert client.get(f"/operations/{op['id']}").json()["params"]["days"] == 3
    assert client.get("/operations", params={"q": "Malapert"}).json()[0]["id"] == op["id"]
    bad = {"name": "bad", "kind": "analysis", "params": {"lat": 110.0, "lon": 0.0}}
    assert client.post("/operations", json=bad).status_code == 422
    assert client.delete(f"/operations/{op['id']}").status_code == 204
    assert client.get(f"/operations/{op['id']}").status_code == 404


def test_compare_ranks_sites(client):
    j = client.post("/compare", json={"days": 5, "sites": [
        {"label": "Shackleton floor", "lat": -89.625, "lon": 119.13},
        {"label": "Malapert", "lat": -85.987, "lon": 2.08}]}).json()
    assert [r["label"] for r in j["ranking"]] == ["Malapert", "Shackleton floor"]
    assert j["ranking"][0]["rank"] == 1


def test_model_card_and_basemap(client):
    assert "loaded" in client.get("/model").json()
    b = client.get("/basemap").json()["lola_color_shade"]
    assert "{z}" in b["url_template"] and len(b["resolutions_m"]) == 10


def test_vehicle_assessment_endpoint(client):
    v = client.get("/vehicles").json()
    assert {p["id"] for p in v} >= {"apollo-11", "chang-e-4-yutu-2", "chandrayaan-3-vikram-pragyan"}
    j = client.post("/vehicle-assessment", json={"vehicle_id": "apollo-11", "lat": -85.987, "lon": 2.08, "days": 5})
    assert j.status_code == 200 and j.json()["vehicle"]["name"].startswith("Apollo 11")
    assert client.post("/vehicle-assessment", json={"vehicle_id": "lro", "lat": -85.9, "lon": 2.0}).status_code == 404


def test_space_time_endpoint(client):
    j = client.get("/space-time", params={"lat": -85.987, "lon": 2.08, "at": "2026-11-14T12:00:00"}).json()
    assert j["now_utc"] == "2026-11-14T12:00:00Z"
    assert 0 <= j["moon"]["illumination"] <= 1
    utcs = [e["utc"] for e in j["events"]]
    assert utcs == sorted(utcs) and all(u > j["now_utc"] for u in utcs)
    sun = j["site"]["sun"]["transitions"]
    assert sun and sun[0]["becomes_up"] is (not j["site"]["sun"]["up_now"])  # the next change flips the current state
    assert all(a["becomes_up"] != b["becomes_up"] for a, b in zip(sun, sun[1:]))  # rises and sets alternate
    only_moon = client.get("/space-time", params={"at": "2026-11-14T12:00:00"}).json()
    assert only_moon["site"] is None and len(only_moon["events"]) >= 7


def test_space_time_validation(client):
    assert client.get("/space-time", params={"lat": -85.9}).status_code == 422       # lat without lon
    assert client.get("/space-time", params={"lat": 95, "lon": 0}).status_code == 422  # not a latitude
    # anywhere on the Moon is allowed; the south pole has 40 m terrain, elsewhere the coarse global grid (or a smooth sphere)
    far = client.get("/space-time", params={"lat": -50, "lon": 0, "at": "2026-11-14T00:00:00"}).json()["site"]
    assert far["terrain"] in ("global", "smooth") and len(far["sun"]["transitions"]) >= 3
    pole = client.get("/space-time", params={"lat": -85.987, "lon": 2.08, "at": "2026-11-14T00:00:00"}).json()["site"]
    assert pole["terrain"] == "lola"
    assert client.get("/space-time", params={"at": "not a date"}).status_code == 422


def test_earth_sun_endpoint(client):
    j = client.get("/earth-sun", params={"lat": 41.01, "lon": 28.98, "at": "2026-11-14T00:00:00"}).json()
    assert j["sun_up_now"] is False and [t["becomes_up"] for t in j["transitions"][:2]] == [True, False]
    assert j["transitions"][0]["utc"] == "2026-11-14T04:50:30Z"
    assert -90 <= j["subsolar"]["lat"] <= 90 and -180 <= j["subsolar"]["lon"] <= 180
    assert client.get("/earth-sun", params={"lat": 120, "lon": 0}).status_code == 422
    assert client.get("/earth-sun", params={"lat": 10, "lon": 0, "at": "nope"}).status_code == 422


def test_dem_view_endpoints(client):
    meta = client.get("/dem-view").json()
    r = client.get("/dem-view.png")
    assert r.status_code == 200 and r.headers["content-type"] == "image/png" and "max-age" in r.headers["cache-control"]
    assert r.content[:8] == b"\x89PNG\r\n\x1a\n" and meta["extent_m"] == 151680.0 and meta["size_px"] > 1000


def test_client_visibility_rule_matches_the_backend_flags(client):
    """The Moon Map recomputes 'is it visible' in the browser from the horizon and the az/el series.
    It must agree with the exact engine's own flags, hour by hour (the rule is mirrored in frontend/src/lib/sky.ts)."""
    import numpy as np

    def horizon_at(h, az):
        n = len(h["az_deg"])
        pos = (np.asarray(az) % 360) / (360 / n)
        i = np.floor(pos).astype(int) % n
        f = pos - np.floor(pos)
        e = np.asarray(h["el_deg"])
        return e[i] * (1 - f) + e[(i + 1) % n] * f

    for lat, lon in [(-85.987, 2.08), (-89.54, 0.0), (-85.2, 53.5)]:
        h = client.get("/horizon", params={"lat": lat, "lon": lon, "n_az": 720}).json()
        ts = client.post("/timeseries", json={"lat": lat, "lon": lon, "start_utc": "2026-11-14T00:00:00", "days": 30}).json()
        sun = np.array(ts["sun_elevation_deg"]) - horizon_at(h, ts["sun_azimuth_deg"])
        earth = np.array(ts["earth_elevation_deg"]) - np.maximum(horizon_at(h, ts["earth_azimuth_deg"]), 0)
        assert ((sun >= 0) == (np.array(ts["sun_lit_fraction"]) >= 0.5)).all()
        assert ((earth >= 0) == np.array(ts["earth_visible"])).all()


def test_dem_view_layers(client):
    for layer in ("relief", "elevation", "slope"):
        r = client.get("/dem-view.webp", params={"layer": layer, "size": 512})
        assert r.status_code == 200 and r.headers["content-type"] == "image/webp" and r.content[:4] == b"RIFF"
        meta = client.get("/dem-view", params={"layer": layer, "size": 512}).json()
        assert meta["layer"] == layer and meta["size_px"] == 512 and meta["extent_m"] == 151680.0
        if layer != "relief":
            lg = meta["legend"]
            assert lg["min"] < lg["max"] and lg["stops"][0][0] == 0.0 and lg["stops"][-1][0] == 1.0
    assert client.get("/dem-view.webp", params={"layer": "optical"}).status_code == 422


def test_moon_sites_are_real_places(client):
    sites = client.get("/moon-sites").json()
    by = {s["id"]: s for s in sites}
    assert {"apollo-11", "apollo-17", "shackleton-rim"} <= set(by)
    assert abs(by["apollo-11"]["lat"] - 0.674) < 0.01 and abs(by["apollo-11"]["lon"] - 23.473) < 0.01  # NASA Tranquility Base
    assert all(-90 <= s["lat"] <= 90 and -180 <= s["lon"] <= 180 and s["source"] for s in sites)


def test_moon_point_report_is_consistent(client):
    r = client.get("/moon-point", params={"lat": -85.987, "lon": 2.08, "utc": "2026-11-20T12:00:00"}).json()
    assert r["terrain"] == "lola" and 0 <= r["surface"]["slope_deg"] < 90 and len(r["day"]["sun_lit"]) == 24
    assert abs(r["sun"]["incidence_flat_deg"] - (90 - r["sun"]["elevation_deg"])) < 1e-9
    assert r["earth"]["visible"] == (r["earth"]["margin_deg"] > 0)
    flat = client.get("/moon-point", params={"lat": 0.674, "lon": 23.473, "utc": "2026-11-20T12:00:00"}).json()
    assert flat["terrain"] in ("global", "smooth")
    assert (flat["surface"] is None) == (flat["sun"]["incidence_slope_deg"] is None) == (flat["terrain"] == "smooth")
    # two dates differ: the Sun has moved on by about 6 days of the lunar day
    later = client.get("/moon-point", params={"lat": 0.674, "lon": 23.473, "utc": "2026-11-26T12:00:00"}).json()
    assert abs(later["sun"]["elevation_deg"] - flat["sun"]["elevation_deg"]) > 10
    assert client.get("/moon-point", params={"lat": 0, "lon": 0, "utc": "nope"}).status_code == 422


def test_moon_point_uses_the_global_grid_away_from_the_pole(client):
    r = client.get("/moon-point", params={"lat": 0.674, "lon": 23.473, "utc": "2026-11-20T12:00:00"}).json()
    if r["terrain"] == "smooth":
        pytest.skip("LDEM_64 not downloaded")
    assert r["terrain"] == "global" and r["surface"]["baseline_m"] == 1000.0 and 0 <= r["surface"]["slope_deg"] < 45
    assert -2300 < r["surface"]["height_m"] < -1600


def test_vehicle_assessment_is_safe_under_concurrent_requests(client):
    """The assessment reads SPICE itself; running it next to other SPICE users must not crash the ephemeris."""
    from concurrent.futures import ThreadPoolExecutor
    body = {"lat": -85.987, "lon": 2.08, "start_utc": "2026-09-12T00:00:00", "days": 30}

    def call(i):
        if i % 2:
            return client.post("/vehicle-assessment", json={**body, "vehicle_id": "chandrayaan-3-vikram-pragyan"}).status_code
        return client.post("/query", json={"lat": body["lat"], "lon": body["lon"], "utc": body["start_utc"], "days": 30}).status_code

    with ThreadPoolExecutor(6) as pool:
        assert set(pool.map(call, range(12))) == {200}


def test_moon_tile_serves_real_lola_heights_for_the_3d_viewer(client):
    import numpy as np
    r = client.get("/moon-tile/3/5/7")
    assert r.status_code == 200 and len(r.content) == 5 * 91 * 91 * 2
    a = np.frombuffer(r.content, "<u2").reshape(5, 91, 91).astype(float) * 0.5 - 10000
    assert -9500 < a.min() and a.max() < 11000  # the Moon's real elevation range
    k = client.get("/moon-height", params={"lat": 0.67, "lon": 23.47}).json()  # Apollo 11 area
    assert -2100 < k["height"] < -1800
    assert client.get("/moon-tile/9/0/0").status_code == 422


def test_sub_solar_series_matches_the_point_in_time_answer(client):
    r = client.get("/sub-solar-series", params={"start": "2026-11-15T00:00:00", "hours": 48}).json()
    assert len(r["lat"]) == 49 and len(r["earth_lit"]) == 49
    one = client.get("/space-time", params={"at": "2026-11-16T12:00:00"}).json()["moon"]
    lon = r["lon"][36] % 360
    assert abs(lon - one["sub_solar"]["lon"]) < 0.05 and abs(r["lat"][36] - one["sub_solar"]["lat"]) < 0.05
    assert abs(r["earth_lit"][36] - one["illumination"]) < 0.01
    assert 11 < (r["lon"][0] - r["lon"][24]) % 360 < 13.5  # the Sun moves west about 12.2 degrees per day


def test_moon_patch_is_real_terrain_in_the_observers_frame(client):
    import numpy as np
    n, nn = 129, 101
    for lat, lon in [(-89.5, 10.0), (0.67, 23.47)]:  # inside the 40 m grid, and on the whole-Moon grid
        params = {"lat": lat, "lon": lon, "radius_km": 15, "n": n, "near_m": 1000, "near_n": nn}
        r = client.get("/moon-patch", params=params)
        assert r.status_code == 200 and len(r.content) == (2 * n * n + nn * nn) * 4
        a = np.frombuffer(r.content, "<f4")
        far, real, near = a[: n * n].reshape(n, n), a[n * n: 2 * n * n].reshape(n, n), a[2 * n * n:].reshape(nn, nn)
        assert near[nn // 2, nn // 2] == 0.0                  # the observer's own ground is the origin
        assert np.isfinite(a).all()
        assert real[0, 0] < 200 and real.max() - real.min() < 12000   # real relief; the Moon curves away ~125 m at 21 km
        # the real-only grid is exactly what the plain request returns: the invented detail never leaks into it
        plain = np.frombuffer(client.get("/moon-patch", params={**params, "detail": False}).content, "<f4")[: n * n].reshape(n, n)
        assert np.allclose(real - real[n // 2, n // 2], plain - plain[n // 2, n // 2], atol=1e-3)
        assert not np.allclose(far, real)                      # while the picture does get relief
    assert client.get("/moon-patch", params={"lat": 0, "lon": 0, "n": 10}).status_code == 422


def test_fine_relief_is_repeatable_and_not_wilder_than_the_moon(client):
    import numpy as np
    from computation_core import surface_detail as sd
    from computation_core.surface_patch import terrain_tiers
    from data_engine import config
    from data_engine.lola_downloader import load_dem
    from computation_core.global_dem import load_global_dem
    dem, g = load_dem(), load_global_dem(config.GLOBAL_DEM_PATH)
    a = terrain_tiers(dem, g, -86.5, 40.0, far_n=257, near_n=201)
    b = terrain_tiers(dem, g, -86.5, 40.0, far_n=257, near_n=201)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))       # the same place always looks the same
    m = terrain_tiers(dem, g, 30.0, -20.0, far_n=257, near_n=401)[2]   # a mare: ground should be gentle
    gy, gx = np.gradient(m, 5.0)
    assert np.median(np.degrees(np.arctan(np.hypot(gx, gy)))) < 9
    # the roughness estimate recovers the exponent of a surface that really has one
    rng = np.random.default_rng(1)
    k = np.fft.fftfreq(256)[:, None] ** 2 + np.fft.fftfreq(256)[None, :] ** 2
    k[0, 0] = 1
    z = np.real(np.fft.ifft2(np.fft.fft2(rng.normal(size=(256, 256))) * k ** (-(2 * 0.9 + 2) / 4)))
    assert 0.75 <= sd.roughness(z * 10, 1.0, True)[0] <= 0.95


def test_sky_series_agrees_with_the_moon_map_point_report(client):
    s = client.get("/sky-series", params={"lat": -89.5, "lon": 10, "start": "2026-11-15T00:00:00", "hours": 24}).json()
    p = client.get("/moon-point", params={"lat": -89.5, "lon": 10, "utc": "2026-11-15T12:00:00"}).json()
    assert s["data"] == "lola_40m" and len(s["sun"]["el"]) == 25
    for body in ("sun", "earth"):
        assert abs(s[body]["el"][12] - p[body]["elevation_deg"]) < 0.3, body
        assert abs(((s[body]["az"][12] - p[body]["azimuth_deg"] + 180) % 360) - 180) < 0.5, body


def test_sky_series_gives_earths_real_orientation(client):
    import numpy as np
    s = client.get("/sky-series", params={"lat": 0.67, "lon": 23.47, "start": "2026-11-15T00:00:00", "hours": 24}).json()
    rot = np.array(s["earth_rot"])
    assert rot.shape == (25, 9)
    for m in rot[[0, 12, 24]]:
        m = m.reshape(3, 3)
        assert np.allclose(m @ m.T, np.eye(3), atol=1e-4) and abs(np.linalg.det(m) - 1) < 1e-4   # a real rotation
    # Earth's pole, written in this Moon-fixed frame, drifts only by the Moon's own turn (~13 deg a day), not by Earth's spin
    assert np.degrees(np.arccos(np.clip(rot[0].reshape(3, 3)[:, 2] @ rot[24].reshape(3, 3)[:, 2], -1, 1))) < 14.0
    a, b = rot[0].reshape(3, 3)[:, 0], rot[1].reshape(3, 3)[:, 0]
    assert 10 < np.degrees(np.arccos(np.clip(a @ b, -1, 1))) < 20      # ~15 deg per hour (the Moon's sky also shifts a little)


def test_saved_operations_keep_their_inputs_and_their_conversation(client):
    state = {"a": {"date": "2026-12-01", "vehicleId": "im-1-odysseus", "lat": "-85.9", "lon": "2"}, "mode": "instant"}
    made = client.post("/sessions", json={"name": "IM-1 at the pole", "state": state, "chat": []})
    assert made.status_code == 201
    sid = made.json()["id"]
    assert any(s["id"] == sid and s["name"] == "IM-1 at the pole" for s in client.get("/sessions").json())
    chat = [{"from": "user", "text": "why is the score low"}, {"from": "bot", "blocks": [{"type": "text", "text": "x"}]}, {"from": "memory", "memory": {"focus": 1}}]
    assert client.put(f"/sessions/{sid}", json={"chat": chat}).json()["chat"] == chat
    got = client.get(f"/sessions/{sid}").json()
    assert got["state"] == state and got["chat"] == chat            # a partial update leaves the rest alone
    renamed = client.put(f"/sessions/{sid}", json={"name": "Renamed", "state": {**state, "mode": "full"}}).json()
    assert renamed["name"] == "Renamed" and renamed["state"]["mode"] == "full" and renamed["chat"] == chat
    assert client.delete(f"/sessions/{sid}").status_code == 204
    assert client.get(f"/sessions/{sid}").status_code == 404 and client.delete(f"/sessions/{sid}").status_code == 404
    assert client.post("/sessions", json={"name": "no inputs"}).status_code == 422


def test_assistant_endpoint_takes_two_scenarios_and_no_more(client):
    a = {"date": "2026-12-01", "vehicleId": "im-1-odysseus", "lat": -85.9, "lon": 2.0}
    b = {**a, "date": "2026-12-15"}
    ok = client.post("/assistant", json={"message": "compare the two results", "scenarios": [a, b], "memory": {}})
    assert ok.status_code == 200 and ok.json()["intent"] == "compare_results"
    assert client.post("/assistant", json={"message": "hi", "scenarios": [a, b, a], "memory": {}}).status_code == 422


def test_assistant_endpoint_takes_the_detail_level(client):
    a = {"date": "2026-12-01", "vehicleId": "im-1-odysseus", "lat": -85.9, "lon": 2.0}
    r = client.post("/assistant", json={"message": "summarize the result", "scenarios": [a], "memory": {}, "level": "simple"})
    assert r.status_code == 200 and not any(b.get("type") == "table" for b in r.json()["reply"]["blocks"])
    assert client.post("/assistant", json={"message": "hi", "scenarios": [a], "memory": {}, "level": "expert"}).status_code == 422


def test_analysis_works_anywhere_on_the_moon_and_says_which_terrain_data_it_used(client):
    polar = client.get("/terrain", params={"lat": -85.987, "lon": 2.08}).json()
    assert polar["data_source"] == "lola_40m"
    for lat, lon in ((0.674, 23.473), (-43.3, 348.6), (0.0, 180.0), (89.8, 0.0)):
        t = client.get("/terrain", params={"lat": lat, "lon": lon})
        assert t.status_code == 200 and t.json()["data_source"] == "ldem_470m", (lat, lon)
        q = client.post("/query", json={"lat": lat, "lon": lon, "utc": "2026-12-01T00:00:00", "days": 30})
        assert q.status_code == 200 and 0 <= q.json()["window"]["sunlit_fraction"] <= 1
        v = client.post("/vehicle-assessment", json={"vehicle_id": "apollo-15", "lat": lat, "lon": lon, "start_utc": "2026-12-01T00:00:00", "days": 30})
        assert v.status_code == 200 and 0 <= v.json()["vehicle_index"] <= 100


def test_whole_moon_results_are_physically_sensible(client):
    """Apollo 11's site is on the near side (Earth always in view, about half the month sunlit); a far-side equator point never sees Earth."""
    near = client.post("/query", json={"lat": 0.674, "lon": 23.473, "utc": "2026-12-01T00:00:00", "days": 30}).json()["window"]
    far = client.post("/query", json={"lat": 0.0, "lon": 180.0, "utc": "2026-12-01T00:00:00", "days": 30}).json()["window"]
    assert near["dte_fraction"] > 0.95 and 0.4 < near["sunlit_fraction"] < 0.6
    assert far["dte_fraction"] == 0.0 and 0.4 < far["sunlit_fraction"] < 0.6


# ---------------------------------------------------------------- garage-designed vehicles and the mission planner ----
_DESIGN = {"name": "Test Scout", "kind": "lander", "note": "pytest", "battery_wh": 1500, "solar_w": 400, "survival_power_w": 60,
           "payload_power_w": 150, "max_slope_deg": 15, "radioisotope_heat": False, "comms": "direct_to_earth"}


def test_custom_vehicle_lifecycle_and_it_works_in_an_assessment(client):
    made = client.post("/custom-vehicles", json=_DESIGN)
    assert made.status_code == 201
    v = made.json()
    try:
        assert v["custom"] is True and v["id"].startswith("custom-") and v["battery_wh"]["status"] == "unverified"  # the user's own figures are never "verified"
        assert v["id"] in {x["id"] for x in client.get("/vehicles").json()}
        a = client.post("/vehicle-assessment", json={"vehicle_id": v["id"], "lat": -85.987, "lon": 2.08, "start_utc": "2027-01-20T00:00:00"}).json()
        assert a["vehicle"]["name"] == "Test Scout" and "risk" in a and "verdict" in a
        upd = client.put(f"/custom-vehicles/{v['id']}", json={**_DESIGN, "name": "Test Scout 2", "radioisotope_heat": True})
        assert upd.status_code == 200 and upd.json()["radioisotope_heat"]["value"] is True
        assert client.put("/custom-vehicles/apollo-15", json=_DESIGN).status_code == 404   # the project's own vehicles cannot be edited
        assert client.delete("/custom-vehicles/apollo-15").status_code == 404
        assert client.post("/custom-vehicles", json={**_DESIGN, "max_slope_deg": 0}).status_code == 422
    finally:
        assert client.delete(f"/custom-vehicles/{v['id']}").status_code == 204
    assert v["id"] not in {x["id"] for x in client.get("/vehicles").json()}


def test_vehicle_preview_runs_without_saving(client):
    n = len(client.get("/vehicles").json())
    r = client.post("/vehicle-preview", json={**_DESIGN, "lat": -85.987, "lon": 2.08, "start_utc": "2027-01-20T00:00:00"})
    assert r.status_code == 200 and 0 <= r.json()["p_success"] <= 1 and r.json()["samples"] == 200
    assert len(client.get("/vehicles").json()) == n


def test_planner_paints_every_date_exactly_as_the_analysis_page_would(client):
    from datetime import date, timedelta
    vid, lat, lon = "chandrayaan-3-vikram-pragyan", -85.987, 2.08
    j = client.post("/planner", json={"vehicle_id": vid, "lat": lat, "lon": lon, "start_date": "2026-12-01", "span_days": 90}).json()
    days = j["days"]
    assert len(days) == 90 and all(0 <= d["p"] <= 1 and 0 <= d["score"] <= 100 for d in days)
    assert {d["verdict"] for d in days} >= {"stop", "conditional"}  # a heaterless lander: good dates and hopeless dates, by the lunar cycle
    for k in (0, 7, 14, 33, 60, 89):  # whatever the planner shows for a date, the analysis of that date shows too
        iso = (date(2026, 12, 1) + timedelta(days=k)).isoformat()
        a = client.post("/vehicle-assessment", json={"vehicle_id": vid, "lat": lat, "lon": lon, "start_utc": iso + "T00:00:00"}).json()
        assert days[k]["score"] == pytest.approx(a["vehicle_index"], abs=1e-6)
        assert days[k]["p"] == a["risk"]["p_success"] and days[k]["verdict"] == a["verdict"]["kind"] and days[k]["title"] == a["verdict"]["title"]
    rank = {"go": 3, "conditional": 2, "weak": 1, "stop": 0}
    top = max(rank[d["verdict"]] for d in days)
    w = j["windows"]
    assert w and all(x["from"] <= x["at"] <= x["to"] for x in w)
    assert all(rank[days[i]["verdict"]] == top for x in w for i in range(x["from"], x["to"] + 1))  # windows hold only the best call
    assert client.post("/planner", json={"vehicle_id": "nope", "lat": -85.9, "lon": 2.0, "start_date": "2026-12-01"}).status_code == 404
    assert client.post("/planner", json={"vehicle_id": "luna-9", "lat": -85.9, "lon": 2.0, "start_date": "31-12-2026"}).status_code == 422


# ---------------------------------------------------------------- libration (Earth's wobble over the Moon's sky) ----
def test_libration_is_the_reason_earth_is_seen_from_the_pole_and_changes_nothing_at_the_centre_of_the_near_side(client):
    pole = client.post("/libration", json={"lat": -89.54, "lon": 0.0, "start_utc": "2026-12-01T00:00:00"}).json()
    assert pole["dte_fraction_no_libration"] == 0.0 and pole["dte_fraction"] > 0.2 and pole["role"] == "enables" and pole["hours_gained"] > 0
    centre = client.post("/libration", json={"lat": 0.67, "lon": 23.47, "start_utc": "2026-12-01T00:00:00"}).json()
    assert centre["dte_fraction"] == centre["dte_fraction_no_libration"] == 1.0 and centre["role"] == "little"
    # the real optical + physical libration is about 7 deg in latitude and 8 deg in longitude (published ranges 6.7 / 7.9)
    assert 5.5 <= pole["swing"]["lat_deg"][1] <= 7.5 and 6.5 <= pole["swing"]["lon_deg"][1] <= 9.0
    assert len(pole["hours"]) == len(pole["earth_el_deg"]) == len(pole["earth_el_no_libration_deg"]) == len(pole["sub_earth_lat_deg"]) == 720
    assert max(pole["earth_el_no_libration_deg"]) - min(pole["earth_el_no_libration_deg"]) < 0.5  # without libration Earth hardly moves
    # the planner/analysis Earth-in-view number is the same physics as the libration report's "real" number
    q = client.post("/query", json={"lat": -89.54, "lon": 0.0, "utc": "2026-12-01T00:00:00"}).json()
    assert q["window"]["dte_fraction"] == pytest.approx(pole["dte_fraction"], abs=0.02)
