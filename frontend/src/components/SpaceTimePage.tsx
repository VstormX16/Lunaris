import { useCallback, useEffect, useRef, useState } from 'react'
import { getEarthSun, getSites, getSpaceTime, type EarthSun, type SpaceEvent, type SpaceTime } from '../api'
import { formatApprox, formatCountdown, formatUncertainty, formatWhen } from '../lib/countdown'
import EarthMap from './EarthMap'
import MoonGlobalMap from './MoonGlobalMap'
import MoonPhase from './MoonPhase'
import RiseSet from './RiseSet'
import CoordinateFields from './wizard/CoordinateFields'
import { parseLat, parseLon } from '../lib/input'

const REFRESH_MS = 10 * 60 * 1000
const MALAPERT = { lat: -85.987, lon: 2.08 } // summit of Malapert: a candidate south-pole site
const ISTANBUL = { lat: 41.01, lon: 28.98 }

const formatLatLon = (lat: number, lon: number) => `${Math.abs(lat).toFixed(2)}°${lat < 0 ? 'S' : 'N'}, ${Math.abs(lon).toFixed(2)}°${lon < 0 ? 'W' : 'E'}`

function Fact({ value, label }: { value: string; label: string }) {
  return (
    <div>
      <div className="text-[26px] font-bold leading-none tabular-nums text-strong">{value}</div>
      <div className="mt-1.5 text-[12px] uppercase tracking-wide text-muted">{label}</div>
    </div>
  )
}

function When({ iso }: { iso: string }) {
  const w = formatWhen(iso)
  return (
    <div className="mt-1 text-[13px] leading-snug text-muted">
      <div>{w.utc}</div>
      <div>{w.local} your time</div>
    </div>
  )
}

function EventRow({ e, secondsLeft }: { e: SpaceEvent; secondsLeft: number }) {
  return (
    <li className="border-t border-line py-5">
      <div className="text-[13px] uppercase tracking-wide text-muted">{e.label}</div>
      <div className="mt-1 text-[34px] font-bold leading-none tabular-nums text-strong">
        {e.kind === 'instant' ? formatCountdown(secondsLeft) : formatApprox(secondsLeft)}
      </div>
      <div className="mt-2 text-[15px] text-ink">{e.note}</div>
      {e.kind === 'peak' && <div className="text-[13px] text-muted">Peak time {formatUncertainty(e.uncertainty_days)}</div>}
      <When iso={e.utc} />
    </li>
  )
}

export default function SpaceTimePage() {
  const [site, setSite] = useState(MALAPERT) // chosen spot on the Moon
  // the typed coordinates stay as text while being edited; a map click or a jump rewrites them
  const r3 = (v: number) => String(Math.round(v * 1000) / 1000)
  const [latText, setLatText] = useState(r3(MALAPERT.lat))
  const [lonText, setLonText] = useState(r3(MALAPERT.lon))
  const typed = (la: string, lo: string) => {
    const a = parseLat(la)
    const b = parseLon(lo)
    if (a !== null && b !== null) setSite({ lat: a, lon: b > 180 ? b - 360 : b })
  }
  useEffect(() => {
    const a = parseLat(latText)
    const b = parseLon(lonText)
    if (a !== null && b !== null && Math.abs(a - site.lat) < 1e-6 && Math.abs((b > 180 ? b - 360 : b) - site.lon) < 1e-6) return // it is what was typed
    setLatText(r3(site.lat))
    setLonText(r3(site.lon))
    // eslint-disable-next-line react-hooks/exhaustive-deps -- only a change of the chosen spot rewrites the fields
  }, [site])
  const [place, setPlace] = useState(ISTANBUL) // chosen place on Earth
  const [known, setKnown] = useState<{ name: string; lat: number; lon: number }[]>([])
  const [data, setData] = useState<SpaceTime | null>(null)
  const [earth, setEarth] = useState<EarthSun | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [world, setWorld] = useState<'moon' | 'earth'>('moon')
  const [geoNote, setGeoNote] = useState<string | null>(null)
  const [, setTick] = useState(0)
  const offset = useRef(0) // server clock minus this browser's clock, so countdowns follow the server
  const asked = useRef({ moon: 0, earth: 0 }) // only the newest answer may be shown; a slow older one is dropped

  const syncClock = (nowUtc: string) => {
    offset.current = Date.parse(nowUtc) - Date.now()
  }
  const loadMoon = useCallback(() => {
    const id = ++asked.current.moon
    getSpaceTime(site)
      .then((d) => {
        if (id !== asked.current.moon) return
        syncClock(d.now_utc)
        setData(d)
        setError(null)
      })
      .catch((e: Error) => id === asked.current.moon && setError(e.message))
  }, [site])
  const loadEarth = useCallback(() => {
    const id = ++asked.current.earth
    getEarthSun(place.lat, place.lon)
      .then((d) => {
        if (id !== asked.current.earth) return
        syncClock(d.now_utc)
        setEarth(d)
      })
      .catch(() => id === asked.current.earth && setEarth(null))
  }, [place])

  // refetch when a chosen spot changes (debounced for quick clicks), and every few minutes
  useEffect(() => {
    const t = setTimeout(loadMoon, 250)
    const every = setInterval(loadMoon, REFRESH_MS)
    return () => {
      clearTimeout(t)
      clearInterval(every)
    }
  }, [loadMoon])
  useEffect(() => {
    const t = setTimeout(loadEarth, 250)
    const every = setInterval(loadEarth, REFRESH_MS)
    return () => {
      clearTimeout(t)
      clearInterval(every)
    }
  }, [loadEarth])

  // the server may still be starting: keep trying until it answers
  useEffect(() => {
    if (!error) return
    const t = setTimeout(loadMoon, 4000)
    return () => clearTimeout(t)
  }, [error, loadMoon])

  useEffect(() => {
    getSites().then(setKnown).catch(() => setKnown([]))
    const id = setInterval(() => setTick((n) => n + 1), 1000)
    return () => clearInterval(id)
  }, [])

  const nowMs = Date.now() + offset.current
  // exact instants and tight peaks first; seasonal events (uncertain by days) go last so they never outrank a precise one
  const upcoming = (data?.events.filter((e) => Date.parse(e.utc) > nowMs) ?? []).sort(
    (x, y) => Number(x.uncertainty_days >= 1) - Number(y.uncertainty_days >= 1) || Date.parse(x.utc) - Date.parse(y.utc),
  )

  // something just came due: ask the server for what comes next
  const past = (t: { utc: string }) => Date.parse(t.utc) <= nowMs
  const moonDue = data !== null && (data.events.some(past) || (data.site?.sun.transitions.every(past) && data.site.sun.transitions.length > 0))
  const earthDue = earth !== null && earth.transitions.length > 0 && earth.transitions.every(past)
  useEffect(() => {
    if (moonDue) loadMoon()
  }, [moonDue, loadMoon])
  useEffect(() => {
    if (earthDue) loadEarth()
  }, [earthDue, loadEarth])

  const useMyLocation = () => {
    if (!navigator.geolocation) return setGeoNote('This browser cannot share a location.')
    setGeoNote('Waiting for permission…')
    navigator.geolocation.getCurrentPosition(
      (p) => {
        setPlace({ lat: Math.round(p.coords.latitude * 100) / 100, lon: Math.round(p.coords.longitude * 100) / 100 })
        setGeoNote(null)
      },
      () => setGeoNote('Location was not shared. Click the map instead.'),
      { timeout: 8000 },
    )
  }

  // results of an earlier click must never be shown as if they belonged to the spot now chosen
  const siteReady = data?.site?.lat === site.lat && data?.site?.lon === site.lon
  const earthReady = earth?.lat === place.lat && earth?.lon === place.lon
  const m = data?.moon
  const waxing = m ? m.elongation_deg < 180 : true
  const pole = m ? -m.sub_earth.lat : 0 // Earth's elevation seen from the south pole itself

  return (
    <main className="flex-1 overflow-y-auto">
      <div className="mx-auto max-w-[1280px] px-10 pb-20 pt-10">
        <h1 className="text-[34px] font-medium text-brand">Space Time</h1>
        <p className="mt-1 max-w-[680px] text-muted">
          Real events from NASA/JPL's DE440 ephemeris, found by solving the geometry, not from a table. Countdowns tick live; every time is shown in UTC and in your own time zone.
        </p>

        {error && <p className="mt-10 text-amber-700">{error}</p>}
        {!data && !error && <p className="mt-10 animate-pulse text-muted">Reading the ephemeris…</p>}

        {data && m && (
          <>
            <section className="mt-10 flex flex-wrap items-center gap-x-12 gap-y-6">
              <MoonPhase illumination={m.illumination} waxing={waxing} />
              <div>
                <div className="text-[13px] uppercase tracking-wide text-muted">The Moon now</div>
                <div className="mt-1 text-[40px] font-bold leading-tight text-strong">{m.phase_name}</div>
                <div className="mt-6 grid grid-cols-2 gap-x-12 gap-y-5 sm:grid-cols-4">
                  <Fact value={`${(100 * m.illumination).toFixed(1)}%`} label="Illuminated" />
                  <Fact value={`${m.age_days.toFixed(1)} d`} label="Since new moon" />
                  <Fact value={`${Math.round(m.distance_km).toLocaleString('en')} km`} label="From Earth" />
                  <Fact value={`${Math.abs(m.sub_solar.lat).toFixed(2)}° ${m.sub_solar.lat < 0 ? 'S' : 'N'}`} label="Sun overhead at" />
                </div>
                <p className="mt-5 text-[15px] text-ink">
                  From the south pole, Earth is {Math.abs(pole).toFixed(1)}° {pole > 0 ? 'above' : 'below'} the horizon.
                </p>
              </div>
            </section>

            <h2 className="mb-1 mt-14 text-[13px] uppercase tracking-wide text-muted">Day and night on the Moon</h2>
            <p className="mb-4 max-w-[760px] text-[15px] text-muted">
              Where the Sun and Earth stand overhead right now, with the real boundary between lunar day and night. The dashed line is the edge of the side Earth can see; slow wobbles (libration) let us see a little past it.
            </p>
            <MoonGlobalMap subSolar={m.sub_solar} subEarth={m.sub_earth} />

            <h2 className="mb-1 mt-14 text-[13px] uppercase tracking-wide text-muted">Coming up</h2>
            <ul className="grid grid-cols-1 gap-x-14 md:grid-cols-2">
              {upcoming.slice(0, 8).map((e) => (
                <EventRow key={e.id} e={e} secondsLeft={(Date.parse(e.utc) - nowMs) / 1000} />
              ))}
            </ul>
            <p className="mt-3 text-[13px] text-muted">
              The Moon is tidally locked, so its far side never faces Earth. At new moon it is the far side that the Sun fully lights.
            </p>

            <h2 className="mb-1 mt-14 text-[13px] uppercase tracking-wide text-muted">Sunrise and sunset</h2>
            <p className="mb-4 text-[15px] text-muted">Click the map to choose a place.</p>
            <div className="mb-4 flex gap-6 border-b border-line text-[17px]">
              {(['moon', 'earth'] as const).map((w) => (
                <button
                  key={w}
                  onClick={() => setWorld(w)}
                  className={`-mb-px border-b-2 pb-2 font-medium ${world === w ? 'border-brand text-strong' : 'border-transparent text-muted hover:text-ink'}`}
                >
                  On the {w === 'moon' ? 'Moon' : 'Earth'}
                </button>
              ))}
            </div>

            {world === 'moon' ? (
              <section>
                <MoonGlobalMap subSolar={m.sub_solar} subEarth={m.sub_earth} chosen={site} onPick={(lat, lon) => setSite({ lat, lon })} />
                <div className="mt-3 flex flex-wrap items-baseline gap-x-4 gap-y-1 text-[14px]">
                  <span className="tabular-nums text-muted">
                    Chosen: <span className="font-semibold text-strong">{formatLatLon(site.lat, site.lon)}</span>
                  </span>
                  {known.length > 0 && <span className="text-muted">Jump to</span>}
                  {known.map((k) => (
                    <button key={k.name} onClick={() => setSite({ lat: k.lat, lon: k.lon > 180 ? k.lon - 360 : k.lon })} className="capitalize text-ink hover:text-brand">
                      {k.name.replace('_', ' ')}
                    </button>
                  ))}
                </div>
                <div className="mt-4 max-w-[520px]">
                  <CoordinateFields lat={latText} lon={lonText} onLat={(v) => { setLatText(v); typed(v, lonText) }} onLon={(v) => { setLonText(v); typed(latText, v) }} />
                </div>
                <div className="mt-5">
                  {siteReady && data.site ? (
                    <>
                      <RiseSet upNow={data.site.sun.up_now} transitions={data.site.sun.transitions} nowMs={nowMs} searchedDays={data.site.searched_days} upWord="The Sun is up" downWord="No direct sunlight" />
                      <p className="mt-3 max-w-[760px] text-[13px] leading-relaxed text-muted">
                        {data.site.terrain === 'lola'
                          ? 'Terrain: NASA LOLA elevation at 40 m detail, seen from a 2 m mast, so crater rims and mountains are included.'
                          : data.site.terrain === 'global'
                            ? 'Terrain: whole-Moon NASA LOLA elevation (about 470 m per pixel), seen from a 2 m mast. Hills and crater walls smaller than that are smoothed away, so a time can shift a little.'
                            : 'No terrain data is loaded for this place. The horizon is treated as a smooth sphere, so nearby mountains or crater walls could move these times.'}
                      </p>
                    </>
                  ) : (
                    <p className="min-h-[200px] animate-pulse border-t border-line pt-4 text-muted">Calculating from the ephemeris…</p>
                  )}
                </div>
              </section>
            ) : (
              <section>
                <EarthMap lat={place.lat} lon={place.lon} subsolar={earth?.subsolar ?? null} onPick={(lat, lon) => setPlace({ lat, lon })} />
                <div className="mt-3 flex flex-wrap items-center gap-x-4 gap-y-1 text-[14px]">
                  <button onClick={useMyLocation} className="text-ink hover:text-brand">Use my location</button>
                  {geoNote && <span className="text-muted">{geoNote}</span>}
                </div>
                <div className="mt-5">
                  {earthReady && earth ? (
                    <>
                      <RiseSet upNow={earth.sun_up_now} transitions={earth.transitions} nowMs={nowMs} searchedDays={earth.searched_days} upWord="It is daytime" downWord="It is night" />
                      <p className="mt-3 text-[13px] tabular-nums text-muted">Sun is {Math.abs(earth.elevation_deg).toFixed(1)}° {earth.elevation_deg >= 0 ? 'above' : 'below'} the horizon right now.</p>
                    </>
                  ) : (
                    <p className="min-h-[200px] animate-pulse border-t border-line pt-4 text-muted">Reading the Sun's position…</p>
                  )}
                </div>
              </section>
            )}
            <p className="mt-10 max-w-[760px] text-[12px] leading-relaxed text-muted">
              Moon: sunrise is the instant half of the Sun's disk clears the skyline, seen from a 2 m mast. Earth: the almanac definition, Sun's upper edge at the horizon with standard refraction; real weather can shift it by about a minute.
            </p>
          </>
        )}
      </div>
    </main>
  )
}
