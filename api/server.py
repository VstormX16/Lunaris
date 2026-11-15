"""FastAPI service exposing the analysis core. Run: uvicorn api.server:app --reload"""
import json
import uuid
import threading
from pathlib import Path
from datetime import datetime, timezone
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from fastapi.middleware.cors import CORSMiddleware

from computation_core.coordinates import enu_basis
from computation_core.dem_view import LAYERS, render_layer
from computation_core.earth_sun import earth_sun
from computation_core.assistant.dialog import Assistant
from computation_core.assistant.nlu_model import IntentEnsemble
from computation_core.global_dem import load_global_dem
from computation_core.site_dem import SiteDEM, tangent_points
from computation_core.horizon import compute_horizon
from computation_core.libration import libration_report
from computation_core.mission_query import mission_query
from computation_core.mission_risk import mission_risk
from computation_core.planner import plan as plan_dates, windows as plan_windows
from computation_core.point_report import point_report
from computation_core.region_map import region_map
from computation_core.scratch_nn.landing_scorer import METRICS_PATH, MODEL_PATH, Ensemble, suitability_index
from computation_core.space_time import global_events, moon_state, site_events
from computation_core.site_analysis import analyze_site, site_timeseries
from computation_core import surface_patch, terrain_tiles
from computation_core.terrain_context import classify_terrain
from computation_core.vehicle_assessment import assess_vehicle, load_profiles
from computation_core.verdict import decide
from data_engine import config, spice_loader, trek_client
from data_engine.lola_downloader import load_dem

from .schemas import (AnalyzeRequest, AssistantRequest, ComparisonRequest, CustomVehicleIn, LibrationRequest, MapRequest, OperationCreate, PlannerRequest,
                      QueryRequest, SessionIn, TimeseriesRequest, VehicleAssessmentRequest, VehiclePreviewRequest)
from .storage import CustomVehicleStore, OperationStore, SessionStore

_state = {}
_spice_lock = threading.RLock()  # CSPICE is not thread-safe; re-entrant because a locked endpoint may call _iso (also a SPICE call)


@asynccontextmanager
async def lifespan(app):
    spice_loader.load_kernels()
    polar = load_dem()
    _state["gdem"] = load_global_dem(config.GLOBAL_DEM_PATH)  # None until LDEM_64 is downloaded
    _state["dem"] = SiteDEM(polar, _state["gdem"])  # the polar grid where it exists, the whole-Moon grid elsewhere
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    _state["store"] = OperationStore(config.DATA_DIR / "operations.sqlite3")
    _state["sessions"] = SessionStore(config.DATA_DIR / "operations.sqlite3")
    _state["vehicles"] = load_profiles()
    _state["custom"] = CustomVehicleStore(config.DATA_DIR / "operations.sqlite3")
    for cp in _state["custom"].all():  # the vehicles designed in the garage live next to the project's own
        _state["vehicles"][cp["id"]] = cp
    # the assistant's own language model (trained by `python -m computation_core.assistant.train_nlu`)
    _state["assistant"] = Assistant(_state["dem"], _state["vehicles"], IntentEnsemble.load()) if IntentEnsemble.paths() else None
    # trained by `python -m computation_core.scratch_nn.automl`; /map is disabled without it
    _state["model"], _state["model_error"] = None, "not trained yet"
    if MODEL_PATH.exists():
        try:
            _state["model"], _state["model_error"] = Ensemble.load(), None
        except (KeyError, ValueError) as e:  # e.g. a model file from an older feature set
            _state["model_error"] = f"incompatible model file: {e}"
    yield


app = FastAPI(title="Lunar South Pole Landing Site Analyzer", version="0.1", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",  # the dev server, whichever local port it got
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(spice_loader.sp.stypes.SpiceyError)
async def _bad_time(_: Request, e):
    """A date SPICE cannot parse, or one outside the span the kernels cover, is the caller's mistake (422), not a crash (500)."""
    return JSONResponse(status_code=422, content={"detail": f"invalid time: {e.short}"})



def _dem():
    return _state["dem"]


def _check_site(lat, lon):
    # NaN only when neither grid has the place (the whole-Moon grid has not been downloaded and the point is off the polar one)
    if np.isnan(float(_dem().height_at_latlon(lat, lon))):
        raise HTTPException(422, "location outside DEM coverage")


@app.get("/health")
def health():
    return {"status": "ok", "dem": config.LOLA_PRODUCT["name"]}


@app.get("/sites")
def sites():
    return [{"name": n, "lat": la, "lon": lo} for n, (la, lo) in config.CANDIDATE_SITES.items()]


@app.get("/terrain")
def terrain(lat: float = Query(..., ge=-90, le=90), lon: float = Query(..., ge=0, lt=360)):
    _check_site(lat, lon)
    return classify_terrain(_dem(), lat, lon)


@app.get("/area-relief")
def area_relief(lat: float = Query(..., ge=-90, le=90), lon: float = Query(..., ge=0, lt=360),
                half_km: float = Query(None, ge=0.5, le=30), n: int = Query(96, ge=16, le=160)):
    """Real heights (m, relative to the site) of the square around a site, north at the top: for a small terrain picture.

    Default span: 4 km each way on the 40 m polar grid, 12 km on the 470 m whole-Moon grid (4 km would be about 17 pixels)."""
    _check_site(lat, lon)
    if half_km is None:
        half_km = 4.0 if _dem().source(lat, lon) == "lola_40m" else 12.0
    offs = np.linspace(-half_km * 1000.0, half_km * 1000.0, n)
    ee, nn = np.meshgrid(offs, offs[::-1])  # row 0 = north
    la, lo = tangent_points(lat, lon, ee, nn)
    dem = _dem()
    h = np.asarray(dem.polar.height_at_latlon(la, lo), dtype=float)  # the 40 m grid where it reaches
    if dem.global_dem is not None and np.isnan(h).any():
        h = np.where(np.isnan(h), np.asarray(dem.global_dem.height(la, lo), dtype=float), h)  # the 470 m grid elsewhere
    h0 = float(dem.height_at_latlon(lat, lon))
    rel = np.where(np.isnan(h), None, np.round(h - h0, 1))
    return {"n": n, "half_m": half_km * 1000.0, "heights": rel.tolist(), "data_source": _dem().source(lat, lon)}


@app.get("/horizon")
def horizon(lat: float = Query(..., ge=-90, le=90), lon: float = Query(..., ge=0, lt=360),
            observer_height_m: float = Query(2.0, ge=0, le=50), n_az: int = Query(360, ge=36, le=1440)):
    _check_site(lat, lon)
    hp = compute_horizon(_dem(), lat, lon, observer_height_m, n_az=n_az)
    return {"az_deg": hp.az_deg.tolist(), "el_deg": hp.el_deg.tolist()}


@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    _check_site(req.lat, req.lon)
    with _spice_lock:
        metrics = analyze_site(_dem(), req.lat, req.lon, req.start_utc, req.days,
                               req.step_minutes, req.observer_height_m, req.area_radius_m)
    window_h = req.days * 24.0
    exact = np.array([[metrics["illumination_fraction"], metrics["dte_fraction"],
                       metrics["longest_darkness_hours"] / window_h, metrics["longest_outage_hours"] / window_h]])
    return {**metrics, "terrain": classify_terrain(_dem(), req.lat, req.lon),
            "suitability_index": float(suitability_index(exact, metrics["mean_slope_deg"])[0])}


@app.get("/model")
def model_card():
    """Status, architecture and honest test metrics of the neural surrogate."""
    m = _state.get("model")
    card = {"loaded": m is not None, "error": _state.get("model_error")}
    if m is not None:
        card.update(members=len(m.nets), weights=m.weights.tolist(), total_params=m.n_params,
                    network_used_per_target=m.use_nn.tolist(),
                    architectures=[{"head": c["head"], "hidden": c["hidden"]} for c in m.cfgs])
    if METRICS_PATH.exists():
        card["metrics"] = json.loads(METRICS_PATH.read_text())
    return card


@app.post("/map")
def map_(req: MapRequest):
    """Neural-surrogate maps (sunlight, DTE, darkness, suitability) for an n x n region."""
    if _state.get("model") is None:
        raise HTTPException(503, f"surrogate model unavailable: {_state.get('model_error')}")
    _check_site(req.lat, req.lon)
    try:
        m = region_map(_dem(), _state["model"], req.lat, req.lon, req.start_utc, req.half_size_km, req.n,
                       req.refine_budget_s, spice_lock=_spice_lock)
    except spice_loader.sp.stypes.SpiceyError as e:
        raise HTTPException(422, f"invalid time: {e.short}")
    # NaN (off-DEM pixels) is not valid JSON
    return {k: np.where(np.isnan(v), None, np.round(v, 4)).tolist() for k, v in m.items()}


@app.post("/query")
def query(req: QueryRequest):
    """State of Sun/Earth at `utc` with reasons, next events and rule-based findings."""
    _check_site(req.lat, req.lon)
    try:
        with _spice_lock:
            return mission_query(_dem(), req.lat, req.lon, req.utc, req.days,
                                 req.step_minutes, req.observer_height_m)
    except spice_loader.sp.stypes.SpiceyError as e:  # unparsable date or outside kernel coverage
        raise HTTPException(422, f"invalid time: {e.short}")


@app.post("/timeseries")
def timeseries(req: TimeseriesRequest):
    _check_site(req.lat, req.lon)
    with _spice_lock:
        ts = site_timeseries(_dem(), req.lat, req.lon, req.start_utc, req.days,
                             req.step_minutes, req.observer_height_m)
    hours = (np.arange(ts["sun_el"].size) * req.step_minutes / 60.0).tolist()
    return {
        "hours": hours,
        "sun_elevation_deg": ts["sun_el"].tolist(),
        "sun_azimuth_deg": ts["sun_az"].tolist(),
        "sun_lit_fraction": ts["sun_lit_fraction"].tolist(),
        "earth_elevation_deg": ts["earth_el"].tolist(),
        "earth_azimuth_deg": ts["earth_az"].tolist(),
        "earth_visible": ts["earth_visible"].tolist(),
    }


@app.post("/compare")
def compare(req: ComparisonRequest):
    """Exact analysis of several sites over the same window, ranked by suitability index."""
    rows = []
    for site in req.sites:
        r = analyze(AnalyzeRequest(lat=site.lat, lon=site.lon, start_utc=req.start_utc,
                                   days=req.days, step_minutes=req.step_minutes))
        rows.append({"label": site.label, **r})
    rows.sort(key=lambda r: r["suitability_index"], reverse=True)
    for rank, r in enumerate(rows, 1):
        r["rank"] = rank
    return {"window": {"start_utc": req.start_utc, "days": req.days}, "ranking": rows}


@app.post("/assistant")
def assistant(req: AssistantRequest):
    """Ask the assistant about the scenario on the page. Stateless: the client sends the scenario and sends back the memory it got."""
    bot = _state.get("assistant")
    if bot is None:
        raise HTTPException(503, "assistant model not trained yet (python -m computation_core.assistant.train_nlu)")
    for sc in req.scenarios:
        if sc.vehicleId not in _state["vehicles"]:
            raise HTTPException(404, f"unknown vehicle '{sc.vehicleId}'")
    with _spice_lock:  # the tools read SPICE
        out = bot.respond(req.message, {"scenarios": [s.model_dump() for s in req.scenarios], "free": req.free, "level": req.level}, req.memory)
    if out["intent"] in ("clarify", "unknown", "complaint", "doubt", "confused"):  # the moments the assistant let someone down  # kept locally so the next training round can learn what people really ask
        with open(config.DATA_DIR / "assistant_unrecognized.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({"message": req.message, "intent": out["intent"], "confidence": round(out["confidence"], 3)}, ensure_ascii=False) + "\n")
    return out


@app.get("/vehicles")
def vehicles():
    """Landers/rovers with per-field provenance (verified / unverified / unknown)."""
    return list(_state["vehicles"].values())


@app.post("/vehicle-assessment")
def vehicle_assessment(req: VehicleAssessmentRequest):
    """Would this vehicle cope at this site in this window? Exact physics + the vehicle's own limits."""
    profile = _state["vehicles"].get(req.vehicle_id)
    if profile is None:
        raise HTTPException(404, f"unknown vehicle '{req.vehicle_id}' (see GET /vehicles)")
    _check_site(req.lat, req.lon)
    with _spice_lock:  # the assessment re-reads the site's time series, which also uses SPICE
        metrics = analyze_site(_dem(), req.lat, req.lon, req.start_utc, req.days, req.step_minutes)
        assessment = assess_vehicle(profile, metrics, _dem(), req.lat, req.lon)
        risk = mission_risk(profile, _dem(), req.lat, req.lon, req.start_utc, req.days, req.step_minutes)
    return {"site_metrics": metrics, **assessment, "risk": risk, "verdict": decide(assessment, metrics, risk)}


def _custom_profile(body: CustomVehicleIn, vid: str):
    src = "User-defined vehicle (figures entered in the garage)"
    f = lambda v: {"value": v, "status": "unverified", "source": src, "note": ""}  # noqa: E731
    return {"id": vid, "name": body.name.strip(), "kind": body.kind, "custom": True, "note": body.note.strip(), "mass_kg": body.mass_kg,
            "battery_wh": f(body.battery_wh), "solar_w": f(body.solar_w), "radioisotope_heat": f(body.radioisotope_heat),
            "max_slope_deg": f(body.max_slope_deg), "comms": f(body.comms), "survival_power_w": f(body.survival_power_w),
            "payload_power_w": f(body.payload_power_w)}


@app.post("/custom-vehicles", status_code=201)
def create_custom_vehicle(body: CustomVehicleIn):
    vid = "custom-" + uuid.uuid4().hex[:8]
    prof = _custom_profile(body, vid)
    _state["custom"].save(prof)
    _state["vehicles"][vid] = prof
    return prof


@app.put("/custom-vehicles/{vid}")
def update_custom_vehicle(vid: str, body: CustomVehicleIn):
    if not (_state["vehicles"].get(vid) or {}).get("custom"):
        raise HTTPException(404, "not a custom vehicle")
    prof = _custom_profile(body, vid)
    _state["custom"].save(prof)
    _state["vehicles"][vid] = prof
    return prof


@app.delete("/custom-vehicles/{vid}", status_code=204)
def delete_custom_vehicle(vid: str):
    if not (_state["vehicles"].get(vid) or {}).get("custom") or not _state["custom"].delete(vid):
        raise HTTPException(404, "not a custom vehicle")
    _state["vehicles"].pop(vid, None)
    return Response(status_code=204)


@app.post("/libration")
def libration(req: LibrationRequest):
    """Earth's wobble over the Moon's sky and what it does to line-of-sight to Earth at this site (real vs a non-librating Moon)."""
    _check_site(req.lat, req.lon)
    try:
        with _spice_lock:
            return libration_report(_dem(), req.lat, req.lon, req.start_utc, req.days)
    except spice_loader.sp.stypes.SpiceyError as e:
        raise HTTPException(422, f"invalid time: {e.short}")


@app.post("/vehicle-preview")
def vehicle_preview(req: VehiclePreviewRequest):
    """The mission simulation for a vehicle that is not saved yet (the garage form's live read-out)."""
    _check_site(req.lat, req.lon)
    with _spice_lock:
        risk = mission_risk(_custom_profile(req, "preview"), _dem(), req.lat, req.lon, req.start_utc, 30, 60, n=200)
    return risk


@app.post("/planner")
def planner(req: PlannerRequest):
    """Success probability for every start date in the range, plus the best windows."""
    profile = _state["vehicles"].get(req.vehicle_id)
    if profile is None:
        raise HTTPException(404, f"unknown vehicle '{req.vehicle_id}'")
    _check_site(req.lat, req.lon)
    try:
        start = datetime.fromisoformat(req.start_date)
    except ValueError:
        raise HTTPException(422, "start_date must be yyyy-mm-dd")
    start_utc = start.strftime("%Y-%m-%dT00:00:00")
    days = plan_dates(profile, _dem(), req.lat, req.lon, start_utc, req.span_days, lock=_spice_lock)
    return {"vehicle": {"id": profile["id"], "name": profile["name"]}, "start_date": req.start_date, "span_days": req.span_days,
            "days": days, "windows": plan_windows(days)}


_RUNNERS = {"analysis": (AnalyzeRequest, analyze), "query": (QueryRequest, query),
            "comparison": (ComparisonRequest, compare),
            "vehicle_assessment": (VehicleAssessmentRequest, vehicle_assessment)}


@app.post("/sessions", status_code=201)
def create_session(body: SessionIn):
    if not body.name or body.state is None:
        raise HTTPException(422, "a session needs a name and its inputs")
    return _state["sessions"].create_session(body.name, body.state, body.chat or [])


@app.get("/sessions")
def list_sessions():
    return _state["sessions"].list_sessions()


@app.get("/sessions/{sid}")
def get_session(sid: str):
    s = _state["sessions"].get_session(sid)
    if s is None:
        raise HTTPException(404, "session not found")
    return s


@app.put("/sessions/{sid}")
def update_session(sid: str, body: SessionIn):
    s = _state["sessions"].update_session(sid, body.name, body.state, body.chat)
    if s is None:
        raise HTTPException(404, "session not found")
    return s


@app.delete("/sessions/{sid}", status_code=204)
def delete_session(sid: str):
    if not _state["sessions"].delete_session(sid):
        raise HTTPException(404, "session not found")
    return Response(status_code=204)


@app.post("/operations", status_code=201)
def create_operation(op: OperationCreate):
    schema, runner = _RUNNERS[op.kind]
    try:
        req = schema(**op.params)
    except ValidationError as e:
        raise HTTPException(422, json.loads(e.json()))
    return _state["store"].create(op.name, op.kind, req.model_dump(), runner(req))


@app.get("/operations")
def list_operations(kind: str | None = None, q: str | None = None, limit: int = Query(50, ge=1, le=200)):
    return _state["store"].list(kind, q, limit)


@app.get("/operations/{op_id}")
def get_operation(op_id: str):
    op = _state["store"].get(op_id)
    if op is None:
        raise HTTPException(404, "operation not found")
    return op


@app.delete("/operations/{op_id}", status_code=204)
def delete_operation(op_id: str):
    if not _state["store"].delete(op_id):
        raise HTTPException(404, "operation not found")
    return Response(status_code=204)


@app.get("/basemap")
def basemap():
    """NASA Moon Trek tile layers (CORS-enabled; the browser loads tiles directly)."""
    return trek_client.basemap_config()


def _iso(et):
    with _spice_lock:  # this is a SPICE call too: unlocked, two requests at once crashed the whole server (BADSUBSCRIPT in CSPICE)
        return spice_loader.et_to_utc_s(et)


@app.get("/space-time")
def space_time(lat: float | None = Query(None, ge=-90, le=90), lon: float | None = Query(None, ge=-180, lt=360),
               at: str | None = Query(None, description="ISO UTC instant to evaluate; defaults to the server clock")):
    """Moon state and upcoming events with exact UTC instants (NASA SPICE DE440).

    Give lat+lon to also get Sun and Earth rise/set times at that lunar site, from the real terrain.
    The response carries `now_utc` so a client can run its countdowns against the server's clock.
    """
    if (lat is None) != (lon is None):
        raise HTTPException(422, "give both lat and lon, or neither")
    # the 85 S LOLA grid gives real terrain for the pole; anywhere else the horizon is a smooth sphere
    dem = None
    if lat is not None and lat <= -85 and not np.isnan(float(_dem().height_at_latlon(lat, lon % 360))):
        dem = _dem()
    ghz = _state["gdem"].horizon(lat, lon) if dem is None and lat is not None and _state.get("gdem") else None
    now = at or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    try:
        with _spice_lock:
            et0 = spice_loader.utc_to_et(now)
            state = moon_state(et0)
            events = global_events(et0)
            site = (site_events(dem, lat, lon % 360 if dem is not None else lon, et0, horizon=ghz)
                    if lat is not None else None)
    except spice_loader.sp.stypes.SpiceyError as e:
        raise HTTPException(422, f"invalid time: {e.short}")

    out = {
        "now_utc": _iso(et0),
        "moon": state,
        "events": [{"id": e["id"], "label": e["label"], "note": e["note"], "utc": _iso(e["et"]),
                    "kind": e["kind"], "uncertainty_days": e["uncertainty_days"]} for e in events],
        "site": None,
    }
    if site is not None:
        fmt = lambda body: {"up_now": body["up_now"],  # noqa: E731
                            "transitions": [{"utc": _iso(t["et"]), "becomes_up": t["becomes"]} for t in body["transitions"]]}
        out["site"] = {"lat": lat, "lon": lon, "terrain": "lola" if dem is not None else ("global" if ghz is not None else "smooth"), "sun": fmt(site["sun"]), "earth": fmt(site["earth"]),
                       "searched_days": site["searched_days"]}
    return out


@app.get("/moon-sites")
def moon_sites():
    """Known places on the Moon: mission landing/crash sites and the south-pole candidates."""
    return json.loads((Path(__file__).resolve().parents[1] / "data_engine" / "lunar_sites.json").read_text(encoding="utf-8"))


@app.get("/sub-solar-series")
def sub_solar_series(start: str = Query(..., description="ISO UTC start"), hours: int = Query(744, ge=1, le=1500),
                     step_h: float = Query(1.0, ge=0.25, le=24)):
    """Where the Sun is overhead on the Moon, and how much of the Moon Earth sees lit, at regular steps (NASA SPICE DE440)."""
    try:
        with _spice_lock:
            et0 = spice_loader.utc_to_et(start)
            ets = et0 + np.arange(0, hours + 1e-9, step_h) * 3600.0
            sun = spice_loader.body_vectors_me("SUN", ets)
            earth = spice_loader.body_vectors_me("EARTH", ets)
    except spice_loader.sp.stypes.SpiceyError as e:
        raise HTTPException(422, f"invalid time: {e.short}")
    n = np.linalg.norm
    lat = np.degrees(np.arcsin(sun[:, 2] / n(sun, axis=1)))
    lon = np.degrees(np.arctan2(sun[:, 1], sun[:, 0]))
    # phase angle at the Moon between the Sun and Earth directions (Moon-centred approximation of the Sun-Moon-Earth angle)
    cos_i = np.einsum("ij,ij->i", sun, earth) / (n(sun, axis=1) * n(earth, axis=1))
    return {"start_utc": _iso(et0), "step_h": step_h, "lat": np.round(lat, 4).tolist(), "lon": np.round(lon, 4).tolist(),
            "earth_lit": np.round((1 + cos_i) / 2, 4).tolist()}


_patch_cache: dict = {}


@app.get("/moon-patch")
def moon_patch(lat: float = Query(..., ge=-90, le=90), lon: float = Query(..., ge=-180, le=360),
               radius_km: float = Query(15.0, ge=1, le=30), n: int = Query(769, ge=65, le=1025),
               near_m: float = Query(1000.0, ge=200, le=3000), near_n: int = Query(801, ge=65, le=1201),
               detail: bool = True):
    """Terrain around a point for the 360 view, three float32 grids back to back (north row first, metres above the local plane):
    the wide area with fine relief, the same area from real data only (for skyline numbers), and the nearest square kilometres.
    The fine relief is representative, scaled to the real data's roughness; see computation_core.surface_detail."""
    if surface_patch.resolution_note(_dem(), _state.get("gdem"), lat, lon) == "none":
        raise HTTPException(503, "no elevation data for this place")
    key = (round(lat, 4), round(lon, 4), radius_km, n, near_m, near_n, detail)
    if key not in _patch_cache:
        if len(_patch_cache) >= 6:
            _patch_cache.pop(next(iter(_patch_cache)))
        parts = surface_patch.terrain_tiers(_dem(), _state.get("gdem"), lat, lon, radius_km * 1000.0, n, near_m, near_n, detail)
        _patch_cache[key] = b"".join(p.tobytes() for p in parts)
    return Response(_patch_cache[key], media_type="application/octet-stream")


@app.get("/sky-series")
def sky_series(lat: float = Query(..., ge=-90, le=90), lon: float = Query(..., ge=-180, le=360),
               start: str = Query(..., description="ISO UTC start"), hours: int = Query(744, ge=1, le=1500)):
    """Sun and Earth azimuth/elevation seen from a point on the Moon, hourly (NASA SPICE DE440), plus the ground there."""
    dem, gdem = _dem(), _state.get("gdem")
    note = surface_patch.resolution_note(dem, gdem, lat, lon)
    ground = surface_patch.point_height(dem, gdem, lat, lon) if note != "none" else 0.0
    try:
        with _spice_lock:
            et0 = spice_loader.utc_to_et(start)
            ets = et0 + np.arange(0, hours + 1, 1.0) * 3600.0
            sun = spice_loader.topocentric("SUN", lat, lon, ets, elevation_m=ground)
            earth = spice_loader.topocentric("EARTH", lat, lon, ets, elevation_m=ground)
            # how Earth is turned as seen from here: Earth-fixed axes expressed in east/north/up of this point, hour by hour
            basis = np.stack(enu_basis(lat, lon))
            earth_rot = [(basis @ np.array(spice_loader.sp.pxform("IAU_EARTH", "MOON_ME", float(t)))).ravel() for t in ets]
    except spice_loader.sp.stypes.SpiceyError as e:
        raise HTTPException(422, f"invalid time: {e.short}")
    r = lambda a: np.round(a, 3).tolist()  # noqa: E731
    return {"start_utc": _iso(et0), "ground_m": ground, "data": note,
            "sun": {"az": r(sun[0]), "el": r(sun[1])}, "earth": {"az": r(earth[0]), "el": r(earth[1])},
            "earth_rot": np.round(np.array(earth_rot), 5).tolist()}


@app.get("/moon-tile/{z}/{x}/{y}")
def moon_tile(z: int, x: int, y: int):
    """Height tile (real LOLA LDEM_64) for the 3D Moon viewer; see computation_core.terrain_tiles."""
    gdem = _state.get("gdem")
    if gdem is None:
        raise HTTPException(503, "global elevation grid (LDEM_64) is not downloaded")
    try:
        data = terrain_tiles.tile_bytes(gdem, z, x, y)
    except ValueError as e:
        raise HTTPException(422, str(e))
    return Response(data, media_type="application/octet-stream", headers={"Cache-Control": "public, max-age=3600"})


@app.get("/moon-height")
def moon_height(lat: float = Query(..., ge=-90, le=90), lon: float = Query(..., ge=-180, le=360)):
    """Real LOLA height (m, relative to the 1737.4 km sphere) and slope at a point, for the 3D viewer's camera."""
    gdem = _state.get("gdem")
    if gdem is None:
        raise HTTPException(503, "global elevation grid (LDEM_64) is not downloaded")
    s = gdem.surface(lat, lon)
    return {"lat": lat, "lon": lon, "height": s["height_m"], "slope": s["slope_deg"]}


@app.get("/moon-point")
def moon_point(lat: float = Query(..., ge=-90, le=90), lon: float = Query(..., ge=-180, lt=360),
               utc: str = Query(None, description="ISO UTC instant; defaults to the server clock")):
    """Sun angle, Earth line of sight, slope and the chosen date's day for one point on the Moon."""
    when = utc or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    try:
        with _spice_lock:
            return point_report(_dem(), lat, lon, when, gdem=_state.get("gdem"))
    except spice_loader.sp.stypes.SpiceyError as e:
        raise HTTPException(422, f"invalid time: {e.short}")


@app.get("/earth-sun")
def earth_sun_events(lat: float = Query(..., ge=-90, le=90), lon: float = Query(..., ge=-180, le=360),
                     at: str | None = Query(None, description="ISO UTC instant; defaults to the server clock")):
    """Sunrise and sunset countdown data for a place on Earth (NASA SPICE; almanac definition)."""
    now = at or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    try:
        with _spice_lock:
            et0 = spice_loader.utc_to_et(now)
            r = earth_sun(lat, lon, et0)
    except spice_loader.sp.stypes.SpiceyError as e:
        raise HTTPException(422, f"invalid time: {e.short}")
    return {"now_utc": _iso(et0), "lat": lat, "lon": lon, "sun_up_now": r["sun_up_now"],
            "elevation_deg": r["elevation_deg"], "subsolar": r["subsolar"], "searched_days": r["searched_days"],
            "transitions": [{"utc": _iso(t["et"]), "becomes_up": t["becomes"]} for t in r["transitions"]]}


@app.get("/dem-view")
def dem_view_meta(layer: str = "relief", size: int = Query(1084, ge=256, le=4096)):
    """Geometry (and legend range) of /dem-view.png: pixel size and the half-width in metres of the square."""
    return _dem_view(layer, size)[1]


@app.get("/dem-view.png")
def dem_view_image(layer: str = "relief", size: int = Query(1084, ge=256, le=4096)):
    """South-pole picture from NASA LOLA: layer = relief | elevation | slope."""
    return Response(content=_dem_view(layer, size)[0], media_type="image/png", headers={"Cache-Control": "public, max-age=86400"})


@app.get("/dem-view.webp")
def dem_view_webp(layer: str = "relief", size: int = Query(2048, ge=256, le=4096)):
    """Same picture as lossy WebP (about a tenth of the bytes), for the interactive map."""
    return Response(content=_dem_view(layer, size, "webp")[0], media_type="image/webp", headers={"Cache-Control": "public, max-age=86400"})


def _dem_view(layer, size, fmt="png"):
    if layer not in LAYERS:
        raise HTTPException(422, f"layer must be one of {list(LAYERS)}")
    cache = _state.setdefault("dem_view", {})
    if (layer, size, fmt) not in cache:  # rendered once per layer, size and format, then served from memory
        cache[(layer, size, fmt)] = render_layer(_dem(), layer, size, fmt)
    return cache[(layer, size, fmt)]
