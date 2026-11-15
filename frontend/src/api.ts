// Client for the FastAPI backend (uvicorn api.server:app).
import { parseLat, parseLon } from './lib/input'
import type { Scenario } from './types'

export const API_BASE = import.meta.env.VITE_API_URL ?? 'http://127.0.0.1:8000'
const BASE = API_BASE

export type Provenance = 'verified' | 'unverified' | 'unknown'
export type Field<T> = { value: T | null; status: Provenance; source?: string; note?: string }
export type VehicleProfile = {
  id: string
  name: string
  kind: string
  battery_wh: Field<number>
  solar_w: Field<number>
  radioisotope_heat: Field<boolean>
  max_slope_deg: Field<number>
  comms: Field<string>
  survival_power_w?: Field<number>
  payload_power_w?: Field<number>
  /** a vehicle designed in the garage (all its figures are the user's own) */
  custom?: boolean
  note?: string
  mass_kg?: number | null
}

/** The fields of the garage designer form. */
export type CustomVehicleInput = {
  name: string
  kind: 'lander' | 'rover'
  note: string
  mass_kg: number | null
  battery_wh: number
  solar_w: number
  survival_power_w: number
  payload_power_w: number
  max_slope_deg: number
  radioisotope_heat: boolean
  comms: 'direct_to_earth' | 'relay'
}

/** id -> name of every vehicle the server has told us about (the saved operations' titles use it). */
export const vehicleNames: Record<string, string> = {}

export type Severity = 'critical' | 'warning' | 'info'
export type Finding = { severity: Severity; message: string }

export type BodyNow = {
  state: 'visible' | 'terrain_blocked' | 'below_local_horizon'
  azimuth_deg: number
  elevation_deg: number
  blocker_distance_km?: number
}
export type Change = { utc: string; in_hours: number } | null
export type Longest = { hours: number; starts_utc: string | null }

export type QueryResult = {
  now: { sun: BodyNow; earth: BodyNow }
  next: { sun_change: Change; earth_change: Change }
  window: {
    sunlit_fraction: number
    dte_fraction: number
    longest_darkness: Longest
    longest_dte_outage: Longest
  }
  findings: Finding[]
}

export type TerrainInfo = {
  /** which elevation data the numbers come from: the 40 m south-pole grid or the 470 m whole-Moon grid */
  data_source?: 'lola_40m' | 'ldem_470m'
  class: string
  site_height_m: number
  relief_vs_surroundings_m: number
  mean_horizon_deg: number
}

export type VehicleAssessment = {
  site_metrics: {
    mean_slope_deg: number
    safe_area_fraction: number
    illumination_fraction: number
    dte_fraction: number
    longest_darkness_hours: number
    longest_outage_hours: number
  }
  vehicle: { id: string; name: string; kind: string }
  vehicle_index: number
  weighted_index: number
  limiting_factor: 'slope' | 'comms' | 'power' | 'thermal' | 'energy' | null
  vetoes: { gate: string; cap: number; reason: string }[]
  checks: { comms?: { mode?: string } } & Record<string, unknown>
  findings: Finding[]
  data_quality: { inputs_verified: number; inputs_unverified: number; inputs_unknown: number }
  /** the mission simulated many times under its real uncertainty; missing on results saved before it existed */
  risk?: MissionRisk
  /** the call, made on the server so the page, the report and the assistant agree; missing on older saved results */
  verdict?: Verdict
}

export type RiskMode = 'landing' | 'cold' | 'power' | 'comms'
export type MissionRisk = {
  samples: number
  goal_days: number
  p_success: number
  p_landing: number
  p_works_goal: number
  p_comms: number
  p_alive_end: number
  milestones: { days: number; p: number }[]
  assessed: Record<RiskMode, boolean>
  operating_days: { p10: number; p50: number; p90: number; mean: number }
  survival_curve: number[]
  failure_modes: { mode: RiskMode; share: number }[]
  drivers: { factor: string; p_low: number; p_high: number; effect: number; days_low: number; days_high: number; low_value: number; high_value: number }[]
  confidence: { level: 'high' | 'medium' | 'low'; reasons: string[] }
  assumptions: { slip_days: number; spread_verified: number; spread_unverified: number; cold_range_h: [number, number]; goal_days: number; min_contact: number; area_radius_m: number }
}
export type Verdict = {
  kind: 'go' | 'conditional' | 'weak' | 'stop'
  title: string
  summary: string
  probability_line: string | null
  good: string[]
  risks: string[]
  next: string[]
  rule: string
}

export type ScenarioResult = { assessment: VehicleAssessment; query: QueryResult; terrain: TerrainInfo }

export class ApiError extends Error {}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response | undefined
  // a read that fails to connect is retried a few times (the server may still be starting up)
  const tries = init?.method === 'POST' ? 1 : 4
  for (let i = 0; i < tries && !res; i++) {
    try {
      res = await fetch(`${BASE}${path}`, {
        ...init,
        headers: init?.body ? { 'Content-Type': 'application/json' } : undefined,
      })
    } catch {
      if (i < tries - 1) await new Promise((r) => setTimeout(r, 800 * (i + 1)))
    }
  }
  if (!res) throw new ApiError('The analysis server is not answering. Double-click start.bat in the project folder to start it.')
  if (!res.ok) {
    let detail = `Request failed (${res.status})`
    try {
      const body = await res.json()
      const d = body.detail
      detail = typeof d === 'string' ? d : (d?.[0]?.msg ?? detail)
    } catch {
      /* keep generic message */
    }
    throw new ApiError(detail)
  }
  return res.json() as Promise<T>
}

const post = <T>(path: string, body: unknown) => request<T>(path, { method: 'POST', body: JSON.stringify(body) })

export const getVehicles = () => request<VehicleProfile[]>('/vehicles').then((v) => { v.forEach((p) => { vehicleNames[p.id] = p.name }); return v })
export const createCustomVehicle = (v: CustomVehicleInput) => post<VehicleProfile>('/custom-vehicles', v).then((p) => { vehicleNames[p.id] = p.name; return p })
export const updateCustomVehicle = (id: string, v: CustomVehicleInput) => request<VehicleProfile>(`/custom-vehicles/${id}`, { method: 'PUT', body: JSON.stringify(v) }).then((p) => { vehicleNames[p.id] = p.name; return p })
export const deleteCustomVehicle = async (id: string) => {
  const res = await fetch(`${BASE}/custom-vehicles/${id}`, { method: 'DELETE' })
  if (!res.ok && res.status !== 404) throw new ApiError('The vehicle could not be deleted')
  delete vehicleNames[id]
}

export type VehiclePreview = { p_success: number; goal_days: number; operating_days: { p10: number; p50: number; p90: number }; milestones: { days: number; p: number }[]; failure_modes: { mode: string; share: number }[]; confidence: { level: 'high' | 'medium' | 'low' } }
export const previewVehicle = (v: CustomVehicleInput, lat: number, lon: number, start_utc: string) => post<VehiclePreview>('/vehicle-preview', { ...v, lat, lon, start_utc })

export type PlannerDay = { day: number; score: number; p: number; verdict: Verdict['kind']; title: string; days_p50: number; sun: number; earth: number; confidence: 'high' | 'medium' | 'low'; top_failure: string | null }
export type PlannerResult = { vehicle: { id: string; name: string }; start_date: string; span_days: number; days: PlannerDay[]; windows: { from: number; to: number; at: number; verdict: Verdict['kind']; score: number; p: number }[] }
export const getPlanner = (vehicle_id: string, lat: number, lon: number, start_date: string, span_days: number) => post<PlannerResult>('/planner', { vehicle_id, lat, lon, start_date, span_days })
export const getSites = () => request<{ name: string; lat: number; lon: number }[]>('/sites')

/** Everything the analysis page shows for one scenario (30-day window from the chosen date). */
export async function runScenario(s: Scenario): Promise<ScenarioResult> {
  const lat = parseLat(s.lat)
  const lon = parseLon(s.lon)
  if (lat === null || lon === null) throw new ApiError('Invalid coordinates')
  const start_utc = `${s.date}T00:00:00`
  const [assessment, query, terrain] = await Promise.all([
    post<VehicleAssessment>('/vehicle-assessment', { vehicle_id: s.vehicleId, lat, lon, start_utc, days: 30 }),
    post<QueryResult>('/query', { lat, lon, utc: start_utc, days: 30 }),
    request<TerrainInfo>(`/terrain?lat=${lat}&lon=${lon}`),
  ])
  return { assessment, query, terrain }
}

export type MoonState = {
  elongation_deg: number
  phase_angle_deg: number
  illumination: number
  phase_name: string
  age_days: number
  distance_km: number
  sub_earth: { lat: number; lon: number }
  sub_solar: { lat: number; lon: number }
}
export type SpaceEvent = { id: string; label: string; note: string; utc: string; kind: 'instant' | 'peak'; uncertainty_days: number }
export type SiteBody = { up_now: boolean; transitions: { utc: string; becomes_up: boolean }[] }
export type SpaceTime = {
  now_utc: string
  moon: MoonState
  events: SpaceEvent[]
  site: { lat: number; lon: number; terrain: 'lola' | 'global' | 'smooth'; sun: SiteBody; earth: SiteBody; searched_days: number } | null
}

/** Moon state and upcoming events from NASA SPICE; pass a site to get its Sun/Earth rise and set times. */
export const getSpaceTime = (site?: { lat: number; lon: number }, at?: string) => {
  const q = [site ? `lat=${site.lat}&lon=${site.lon}` : '', at ? `at=${encodeURIComponent(at)}` : ''].filter(Boolean).join('&')
  return request<SpaceTime>(`/space-time${q ? `?${q}` : ''}`)
}

export type EarthSun = {
  now_utc: string
  lat: number
  lon: number
  sun_up_now: boolean
  elevation_deg: number
  subsolar: { lat: number; lon: number }
  searched_days: number
  transitions: { utc: string; becomes_up: boolean }[]
}

/** Sunrise and sunset at a place on Earth (NASA SPICE, almanac definition). */
export const getEarthSun = (lat: number, lon: number) => request<EarthSun>(`/earth-sun?lat=${lat}&lon=${lon}`)

export type DemView = {
  size_px: number
  extent_m: number
  layer: string
  legend?: { min: number; max: number; unit: string; stops: [number, [number, number, number]][] }
}

/** Geometry (and colour legend) of the south-pole picture served at `${API_BASE}/dem-view.png`. */
export const getDemView = (layer = 'relief', size = 1084) => request<DemView>(`/dem-view?layer=${layer}&size=${size}`)

export const getTerrain = (lat: number, lon: number) => request<TerrainInfo>(`/terrain?lat=${lat}&lon=${lon}`)

/** Real terrain skyline of a site: obstruction elevation angle for every azimuth (NASA LOLA ray-trace). */
export const getHorizon = (lat: number, lon: number) =>
  request<{ az_deg: number[]; el_deg: number[] }>(`/horizon?lat=${lat}&lon=${lon}&n_az=720`)

export type TimeSeries = {
  hours: number[]
  sun_elevation_deg: number[]
  sun_azimuth_deg: number[]
  sun_lit_fraction: number[]
  earth_elevation_deg: number[]
  earth_azimuth_deg: number[]
  earth_visible: boolean[]
}

/** Hourly Sun and Earth az/el over `days` days from a start date (NASA SPICE). */
export const getTimeseries = (lat: number, lon: number, startIso: string, days = 30) =>
  post<TimeSeries>('/timeseries', { lat, lon, start_utc: `${startIso}T00:00:00`, days, step_minutes: 60 })

export type MoonSite = { id: string; name: string; kind: 'landing' | 'crash' | 'candidate'; lat: number; lon: number; source: string }
export const getMoonSites = () => request<MoonSite[]>('/moon-sites')

export type PointReport = {
  utc: string
  lat: number
  lon: number
  terrain: 'lola' | 'global' | 'smooth'
  surface: { height_m: number; slope_deg: number; downhill_azimuth_deg: number; baseline_m: number } | null
  sun: {
    azimuth_deg: number
    elevation_deg: number
    incidence_flat_deg: number
    incidence_slope_deg: number | null
    skyline_deg: number
    lit_fraction: number
    lit: boolean
  }
  earth: {
    azimuth_deg: number
    elevation_deg: number
    distance_km: number
    skyline_deg: number
    visible: boolean
    below_horizon: boolean
    blocked_by_terrain: boolean
    margin_deg: number
    blocker_distance_m: number | null
  }
  day: { hours_utc: number[]; sun_elevation_deg: number[]; sun_lit: boolean[]; earth_elevation_deg: number[]; earth_visible: boolean[]; lit_hours: number; link_hours: number }
}
/** Sun angle, Earth line of sight, slope and the chosen date's day for one point (NASA SPICE + LOLA). */
export const getMoonPoint = (lat: number, lon: number, utc: string) => request<PointReport>(`/moon-point?lat=${lat}&lon=${lon}&utc=${encodeURIComponent(utc)}`)

// ---- the assistant -------------------------------------------------------------------------------------------
export type AssistantAction = { type: 'apply_scenario'; scenario?: 'a' | 'b'; date?: string; vehicleId?: string; lat?: number; lon?: number }
export type AssistantBlock =
  | { type: 'text'; text: string }
  | { type: 'table'; title: string; columns: string[]; rows: string[][] }
  | { type: 'bullets'; items: string[] }
  | { type: 'chips'; items: string[] }
  | { type: 'actions'; buttons: { label: string; action: AssistantAction }[] }
export type AssistantReply = {
  reply: { blocks: AssistantBlock[]; actions?: AssistantAction[] }
  intent: string
  confidence: number
  memory: Record<string, unknown>
  language: 'tr' | 'en'
}
export type AssistantScenario = { date: string; vehicleId: string; lat: number; lon: number }
/** Ask the project's own assistant about the scenario(s) on the page: A, and B when two results are shown. `memory` is whatever the previous reply returned. */
export const askAssistant = (message: string, scenarios: AssistantScenario[], memory: Record<string, unknown>, free = false, level: 'simple' | 'advanced' = 'advanced') =>
  post<AssistantReply>('/assistant', { message, scenarios, memory, free, level })

// ---- saved operations (the History list) ----------------------------------------------------------------------
export type SessionSummary = { id: string; name: string; created_at: string; updated_at: string; kind?: string | null; vehicle?: string | null; mode?: string | null }
export type Session = SessionSummary & { state: unknown; chat: unknown[] }
export const listSessions = () => request<SessionSummary[]>('/sessions')
export const getSession = (id: string) => request<Session>(`/sessions/${id}`)
export const createSession = (name: string, state: unknown, chat: unknown[] = []) => post<Session>('/sessions', { name, state, chat })
export const updateSession = (id: string, body: { name?: string; state?: unknown; chat?: unknown[] }) =>
  request<Session>(`/sessions/${id}`, { method: 'PUT', body: JSON.stringify(body) })
export const deleteSession = async (id: string) => {
  const res = await fetch(`${BASE}/sessions/${id}`, { method: 'DELETE' })
  if (!res.ok && res.status !== 404) throw new ApiError('The operation could not be deleted')
}

export type SubSolarSeries = { start_utc: string; step_h: number; lat: number[]; lon: number[]; earth_lit: number[] }
/** The Sun's sub-solar point and the Earth-facing lit fraction, hour by hour, for one window (NASA SPICE). */
export const getSubSolarSeries = (startUtc: string, hours: number) =>
  request<SubSolarSeries>(`/sub-solar-series?start=${encodeURIComponent(startUtc)}&hours=${hours}&step_h=1`)

export type SkySeries = {
  start_utc: string
  ground_m: number
  data: 'lola_40m' | 'ldem_470m' | 'none'
  sun: { az: number[]; el: number[] }
  earth: { az: number[]; el: number[] }
  earth_rot: number[][] // per hour: Earth-fixed axes in east/north/up of the point, row-major 3x3
}
/** Sun and Earth azimuth/elevation from one point on the Moon, hourly (NASA SPICE). */
export const getSkySeries = (lat: number, lon: number, startUtc: string, hours: number) =>
  request<SkySeries>(`/sky-series?lat=${lat}&lon=${lon}&start=${encodeURIComponent(startUtc)}&hours=${hours}`)

export type Terrain = { far: Float32Array; farReal: Float32Array; near: Float32Array }
export const FAR_N = 769
export const NEAR_N = 801

/** Terrain around a point: the wide area with fine relief, the same area from real data only, and the nearest square km. */
export async function getMoonPatch(lat: number, lon: number, radiusKm: number, nearM: number): Promise<Terrain> {
  const res = await fetch(`${BASE}/moon-patch?lat=${lat}&lon=${lon}&radius_km=${radiusKm}&n=${FAR_N}&near_m=${nearM}&near_n=${NEAR_N}`)
  if (!res.ok) throw new ApiError(res.status === 503 ? 'No elevation data for this place' : `Terrain request failed (${res.status})`)
  const all = new Float32Array(await res.arrayBuffer())
  const f = FAR_N * FAR_N, nn = NEAR_N * NEAR_N
  if (all.length !== 2 * f + nn) throw new ApiError('The terrain came back incomplete')
  return { far: all.subarray(0, f), farReal: all.subarray(f, 2 * f), near: all.subarray(2 * f) }
}

export type SuitabilityMap = Record<'lat' | 'lon' | 'suitability_index' | 'sunlit_fraction' | 'dte_fraction' | 'longest_darkness_hours' | 'slope_deg' | 'uncertainty', (number | null)[][]>
/** The neural surrogate's map of the south-polar cap for the 30 days after `date`. One request per date and size: a page that is left and opened again, or a double render, reuses it instead of queueing the server's slow job twice. */
const mapCache = new Map<string, Promise<SuitabilityMap>>()
export const getSuitabilityMap = (date: string, halfSizeKm: number, n: number) => {
  const key = `${date}|${halfSizeKm}|${n}`
  let hit = mapCache.get(key)
  if (!hit) {
    hit = post<SuitabilityMap>('/map', { lat: -90, lon: 0, start_utc: `${date}T00:00:00`, half_size_km: halfSizeKm, n })
    hit.catch(() => mapCache.delete(key)) // a failed request is asked again next time
    mapCache.set(key, hit)
  }
  return hit
}

export type AreaRelief = { n: number; half_m: number; heights: (number | null)[][]; data_source: 'lola_40m' | 'ldem_470m' | 'none' }
/** Real heights (m, relative to the site) of the square around a site, north at the top. */
export const getAreaRelief = (lat: number, lon: number, n = 96) => request<AreaRelief>(`/area-relief?lat=${lat}&lon=${lon}&n=${n}`)


export type LibrationReport = {
  hours: number[]
  sub_earth_lat_deg: number[]
  sub_earth_lon_deg: number[]
  earth_el_deg: number[]
  earth_el_no_libration_deg: number[]
  terrain_toward_earth_deg: number[]
  dte_fraction: number
  dte_fraction_no_libration: number
  longest_outage_hours: number
  longest_outage_hours_no_libration: number
  hours_gained: number
  hours_lost: number
  role: 'enables' | 'adds' | 'hides' | 'little'
  swing: { lat_deg: [number, number]; lon_deg: [number, number] }
  method: string
}
export const getLibration = (lat: number, lon: number, start_utc: string) => post<LibrationReport>('/libration', { lat, lon, start_utc, days: 30 })
