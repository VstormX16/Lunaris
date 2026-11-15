"""Pydantic request/response schemas."""
from typing import Literal

from pydantic import BaseModel, Field

# The detailed 40 m LOLA grid covers 85S-90S; the analysis also works anywhere else on the 470 m whole-Moon grid.
Lat = Field(..., ge=-90.0, le=90.0, description="Planetocentric latitude, deg (anywhere on the Moon; 85S-90S has the detailed terrain)")
PolarLat = Field(..., ge=-90.0, le=-85.0, description="Planetocentric latitude, deg (85S-90S): the neural map is trained on the polar grid only")
Lon = Field(..., ge=0.0, lt=360.0, description="Longitude east, deg")


class AnalyzeRequest(BaseModel):
    lat: float = Lat
    lon: float = Lon
    start_utc: str = Field("2026-11-14T00:00:00", description="ISO UTC start")
    days: int = Field(30, ge=1, le=365)
    step_minutes: int = Field(60, ge=5, le=360)
    observer_height_m: float = Field(2.0, ge=0.0, le=50.0)
    area_radius_m: float = Field(500.0, ge=40.0, le=5000.0)


class QueryRequest(BaseModel):
    lat: float = Lat
    lon: float = Lon
    utc: str = Field(..., description="ISO UTC moment to evaluate, e.g. 2026-09-12T00:00:00")
    days: int = Field(30, ge=1, le=365, description="Look-ahead window")
    step_minutes: int = Field(30, ge=5, le=360)
    observer_height_m: float = Field(2.0, ge=0.0, le=50.0)


class MapRequest(BaseModel):
    lat: float = PolarLat
    lon: float = Lon
    start_utc: str = Field(..., description="Start of the 30-day window the model was trained on")
    half_size_km: float = Field(20.0, ge=1.0, le=160.0)
    n: int = Field(64, ge=8, le=160, description="Grid is n x n pixels")
    refine_budget_s: float = Field(0.0, ge=0.0, le=30.0,
                                   description="Seconds of exact-physics recomputation for the most uncertain pixels")


class TimeseriesRequest(BaseModel):
    lat: float = Lat
    lon: float = Lon
    start_utc: str = "2026-11-14T00:00:00"
    days: int = Field(30, ge=1, le=365)
    step_minutes: int = Field(60, ge=5, le=360)
    observer_height_m: float = Field(2.0, ge=0.0, le=50.0)


class Site(BaseModel):
    label: str = Field(..., min_length=1, max_length=80)
    lat: float = Lat
    lon: float = Lon


class ComparisonRequest(BaseModel):
    sites: list[Site] = Field(..., min_length=2, max_length=12)
    start_utc: str = "2026-11-14T00:00:00"
    days: int = Field(30, ge=1, le=365)
    step_minutes: int = Field(60, ge=5, le=360)


class VehicleAssessmentRequest(BaseModel):
    vehicle_id: str = Field(..., description="Lander id from GET /vehicles, e.g. apollo-15")
    lat: float = Lat
    lon: float = Lon
    start_utc: str = "2026-11-14T00:00:00"
    days: int = Field(30, ge=1, le=365)
    step_minutes: int = Field(60, ge=5, le=360)


class OperationCreate(BaseModel):
    """A named operation; the server runs it (params validated per kind) and stores the result."""
    name: str = Field(..., min_length=1, max_length=120)
    kind: Literal["analysis", "query", "comparison", "vehicle_assessment"]
    params: dict


class AssistantScenario(BaseModel):
    date: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")
    vehicleId: str
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=360)


class AssistantRequest(BaseModel):
    message: str = Field(min_length=1, max_length=500)
    scenarios: list[AssistantScenario] = Field(default=[], max_length=2)  # A, and B when two results are on screen
    memory: dict = {}
    level: Literal["simple", "advanced"] = "advanced"  # how technical the answers are
    free: bool = False  # the free chat: no analysis on screen, the scenario is built from what the user says


class SessionIn(BaseModel):
    """A saved New Operation: its inputs (`state`) and the assistant conversation (`chat`), both opaque to the server."""
    name: str | None = Field(default=None, min_length=1, max_length=160)
    state: dict | None = None
    chat: list | None = Field(default=None, max_length=400)


class CustomVehicleIn(BaseModel):
    """A vehicle the user designs in the garage. Every figure is the user's own, so it is stored as unverified."""
    name: str = Field(..., min_length=1, max_length=60)
    kind: Literal["lander", "rover"] = "lander"
    note: str = Field("", max_length=300)
    mass_kg: float | None = Field(None, ge=0, le=1e6)
    battery_wh: float = Field(..., ge=0, le=1e6)
    solar_w: float = Field(..., ge=0, le=1e5)
    survival_power_w: float = Field(..., ge=0, le=1e5)
    payload_power_w: float = Field(0, ge=0, le=1e5)
    max_slope_deg: float = Field(..., gt=0, le=45)
    radioisotope_heat: bool = False
    comms: Literal["direct_to_earth", "relay"] = "direct_to_earth"


class VehiclePreviewRequest(CustomVehicleIn):
    lat: float = Lat
    lon: float = Lon
    start_utc: str = "2026-11-14T00:00:00"


class PlannerRequest(BaseModel):
    vehicle_id: str
    lat: float = Lat
    lon: float = Lon
    start_date: str = Field(..., description="first start date to try, yyyy-mm-dd")
    span_days: int = Field(90, ge=7, le=180)


class LibrationRequest(BaseModel):
    lat: float = Lat
    lon: float = Lon
    start_utc: str = "2026-11-14T00:00:00"
    days: int = Field(30, ge=1, le=120)
