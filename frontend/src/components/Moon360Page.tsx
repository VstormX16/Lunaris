import { useEffect, useMemo, useRef, useState } from 'react'
import { FAR_N, getMoonPatch, getSkySeries, NEAR_N, type SkySeries } from '../api'
import { todayIso } from '../lib/input'
import { bodyStatus, type Patch } from '../lib/skyline'
import Viewer from './moon3d/Viewer'
import type { Point } from './moon3d/geo'
import WalkView, { type Grids, type ViewState } from './moon3d/WalkView'
import TimelineBar from './TimelineBar'
import { GlassButton, inputCls } from './wizard/ui'

const RADIUS_KM = 15
const NEAR_RADIUS_M = 1000
const WINDOW_H = 744
const MIN_LOADING_MS = 2000
const SPEEDS = [{ label: '6 h/s', hoursPerSecond: 6 }, { label: '1 d/s', hoursPerSecond: 24 }, { label: '3 d/s', hoursPerSecond: 72 }]
const STEPS = ['Reading the elevations around the point (NASA LOLA)', 'Adding fine relief, scaled to how rough the measured ground is', 'Building the terrain', 'Placing the Sun and Earth (NASA SPICE)']

const fmt = (lat: number, lon: number) => `${Math.abs(lat).toFixed(3)}° ${lat < 0 ? 'S' : 'N'}, ${Math.abs(lon).toFixed(3)}° ${lon < 0 ? 'W' : 'E'}`
const shiftDay = (iso: string, d: number) => {
  const t = new Date(`${iso}T00:00:00Z`)
  t.setUTCDate(t.getUTCDate() + d)
  return t.toISOString().slice(0, 10)
}
const nowPos = (windowStart: string) => Math.round((Date.now() - Date.parse(`${windowStart}T00:00:00Z`)) / 3_600_000)

const dataNote = (lat: number) =>
  lat <= -85
    ? 'Detailed 40 m NASA LOLA terrain: hills and crater walls are measured, and the small relief around you is representative.'
    : 'Whole-Moon NASA LOLA terrain (about 470 m per pixel): hills are measured, and the small relief around you is representative.'

type Loaded = { lat: number; lon: number; patch: Patch; grids: Grids; groundM: number; data: SkySeries['data'] }

/** A place and moment handed over by another screen (the analysis page): open straight there, at exactly that time. */
type Target = { lat: number; lon: number; date: string; hour: number }

export default function Moon360Page({ target, onBack }: { target?: Target | null; onBack?: () => void }) {
  const [phase, setPhase] = useState<'pick' | 'loading' | 'view' | 'error'>('pick')
  const [chosen, setChosen] = useState<(Point & { height?: number }) | null>(null)
  const [step, setStep] = useState(0)
  const [error, setError] = useState('')
  const [loaded, setLoaded] = useState<Loaded | null>(null)
  const [series, setSeries] = useState<SkySeries | null>(null)
  const [windowStart, setWindowStart] = useState(() => shiftDay(todayIso(), -15))
  const [pos, setPos] = useState(() => nowPos(shiftDay(todayIso(), -15)))
  const [latText, setLatText] = useState('')
  const [lonText, setLonText] = useState('')
  const [fly, setFly] = useState<{ lat: number; lon: number; n: number } | undefined>()
  const [moved, setMoved] = useState(false)
  const [playing, setPlaying] = useState(false)
  const [speed, setSpeed] = useState(24)
  const [view, setView] = useState<ViewState>({ yaw: 0, pitch: 0, fov: 60, aspect: 1.8 })
  const [webglError, setWebglError] = useState('')
  const asked = useRef(0)
  const lastSky = useRef<{ sun: { az: number; el: number } | null; earth: { az: number; el: number } | null; earthRot: number[] | null }>({ sun: null, earth: null, earthRot: null })

  // ---- confirm: fetch the terrain and the sky together, show the steps for at least a moment ----
  const confirm = (given?: Target) => {
    const at = given ?? null
    const place = at ? { lat: at.lat, lon: at.lon > 180 ? at.lon - 360 : at.lon } : chosen
    if (!place) return
    const id = ++asked.current
    const lat = place.lat
    const lon = place.lon
    if (at) setChosen({ lat, lon })
    const ws = shiftDay(at ? at.date : todayIso(), -15)
    setPhase('loading')
    setStep(0)
    setError('')
    setSeries(null)
    const started = Date.now()
    const ticker = setInterval(() => setStep((s) => s + 1), 700)
    Promise.all([getMoonPatch(lat, lon, RADIUS_KM, NEAR_RADIUS_M), getSkySeries(lat, lon, `${ws}T00:00:00`, WINDOW_H)])
      .then(async ([t, sky]) => {
        await new Promise((res) => setTimeout(res, Math.max(0, MIN_LOADING_MS - (Date.now() - started))))
        if (id !== asked.current) return
        setLoaded({ lat, lon, patch: { z: t.farReal, n: FAR_N, radiusM: RADIUS_KM * 1000 }, grids: { far: t.far, nFar: FAR_N, farRadiusM: RADIUS_KM * 1000, near: t.near, nNear: NEAR_N, nearRadiusM: NEAR_RADIUS_M }, groundM: sky.ground_m, data: sky.data })
        setSeries(sky)
        setWindowStart(ws)
        // open at the hour nearest to now when the Sun is up, so the first view is not a black night
        const now = at ? 15 * 24 + at.hour : nowPos(ws)
        let best = now
        for (let d = 0; !at && d <= WINDOW_H; d++) {
          const a = now + d, b = now - d
          if (a <= WINDOW_H && sky.sun.el[a] > 0.3) { best = a; break }
          if (b >= 0 && sky.sun.el[b] > 0.3) { best = b; break }
        }
        setMoved(!at && best !== now && sky.sun.el[Math.min(WINDOW_H, Math.max(0, now))] <= 0.3)
        setPos(best)
        setPlaying(false)
        setPhase('view')
      })
      .catch((e: Error) => {
        if (id !== asked.current) return
        setError(e.message || 'The terrain could not be loaded')
        setPhase('error')
      })
      .finally(() => clearInterval(ticker))
  }

  // opened from the analysis page: go straight to that place and moment
  useEffect(() => {
    if (target) confirm(target)
    // eslint-disable-next-line react-hooks/exhaustive-deps -- once, when the page opens with a target
  }, [])

  // ---- the sky for another month of the timeline bar ----
  useEffect(() => {
    if (!loaded || (series && series.start_utc.startsWith(windowStart))) return
    const id = ++asked.current
    getSkySeries(loaded.lat, loaded.lon, `${windowStart}T00:00:00`, WINDOW_H)
      .then((s) => id === asked.current && setSeries(s))
      .catch((e: Error) => id === asked.current && setError(e.message))
  }, [loaded, windowStart, series])

  // ---- playback ----
  const posRef = useRef(pos)
  posRef.current = pos
  useEffect(() => {
    if (!playing) return
    let acc = posRef.current
    const timer = setInterval(() => {
      acc += speed / 10
      if (acc >= WINDOW_H) {
        setPos(WINDOW_H)
        setPlaying(false)
        return
      }
      setPos(Math.round(acc))
    }, 100)
    return () => clearInterval(timer)
  }, [playing, speed])

  const i = Math.min(WINDOW_H, Math.max(0, pos))
  const sky = useMemo(() => {
    if (!series || i >= series.sun.az.length) return lastSky.current
    const cur = { sun: { az: series.sun.az[i], el: series.sun.el[i] }, earth: { az: series.earth.az[i], el: series.earth.el[i] }, earthRot: series.earth_rot[i] ?? null }
    lastSky.current = cur
    return cur
  }, [series, i])

  const status = useMemo(() => {
    if (!loaded || !sky.sun || !sky.earth) return null
    return { sun: bodyStatus(loaded.patch, sky.sun.az, sky.sun.el), earth: bodyStatus(loaded.patch, sky.earth.az, sky.earth.el) }
  }, [loaded, sky])

  // a typed coordinate: put the marker there and turn the Moon to it
  const goTo = () => {
    const lat = Number(latText.replace(',', '.')), lon = Number(lonText.replace(',', '.'))
    if (!latText.trim() || !lonText.trim() || !Number.isFinite(lat) || !Number.isFinite(lon) || Math.abs(lat) > 90 || lon < -180 || lon > 360) {
      setError('Latitude must be between -90 and 90, longitude between -180 and 360.')
      return
    }
    setError('')
    const l = lon > 180 ? lon - 360 : lon
    setChosen({ lat, lon: l })
    setFly((f) => ({ lat, lon: l, n: (f?.n ?? 0) + 1 }))
  }

  const reset = () => {
    asked.current++
    setPlaying(false)
    setPhase('pick')
    setLoaded(null)
    setSeries(null)
  }

  const shown = Math.min(step, STEPS.length - 1)

  return (
    <main className="flex-1 overflow-y-auto">
      <div className="mx-auto max-w-[1280px] px-10 pb-20 pt-10">
        <h1 className="text-[34px] font-medium text-brand">360 View</h1>
        <p className="mt-1 max-w-[720px] text-muted">
          Choose a place on the Moon and stand there. You look around at the real terrain, with the Sun and Earth in the sky exactly where they are on the day you pick.
        </p>

        {(phase === 'pick' || phase === 'error' || phase === 'loading') && (
          <div className="mt-10">
            <div className="h-[540px] overflow-hidden rounded-[14px] border border-line bg-[#05080c]">
              <Viewer initialLat={-80} initialLon={0} initialAlt={4800} mode="natural" sun={0} sunLat={-25} marker={chosen} focus={fly} showHint={false} onPick={(p) => phase !== 'loading' && setChosen(p)} onError={setWebglError} />
            </div>
            <p className="mt-3 text-[13px] text-muted">Drag to turn the Moon, scroll to come closer, click to choose where you will stand. The Sun here is only for looks.{webglError && <span className="ml-3 text-amber-700">{webglError}</span>}</p>

            <div className="mt-8 flex flex-wrap items-end gap-x-6 gap-y-3">
              <span className="pb-2.5 text-[15px] text-muted">Or type a place</span>
              <input aria-label="Latitude" value={latText} onChange={(e) => setLatText(e.target.value)} onKeyDown={(e) => e.key === 'Enter' && goTo()} placeholder="Latitude, e.g. -86.5" className={`${inputCls} w-[190px]`} />
              <input aria-label="Longitude" value={lonText} onChange={(e) => setLonText(e.target.value)} onKeyDown={(e) => e.key === 'Enter' && goTo()} placeholder="Longitude, e.g. 40" className={`${inputCls} w-[190px]`} />
              <button onClick={goTo} className="pb-2.5 text-[17px] text-ink hover:text-brand">Go</button>
            </div>

            <div className="mt-10 border-t border-line pt-8">
              {!chosen && <p className="text-[17px] text-muted">Click the Moon to choose a point.</p>}
              {chosen && phase !== 'loading' && (
                <div className="flex flex-wrap items-end justify-between gap-x-12 gap-y-6">
                  <div>
                    <div className="text-[12px] uppercase tracking-wide text-muted">Chosen point</div>
                    <div className="mt-1 text-[30px] font-bold leading-none tabular-nums text-strong">{fmt(chosen.lat, chosen.lon)}</div>
                    <p className="mt-3 max-w-[520px] text-[14px] leading-relaxed text-muted">{dataNote(chosen.lat)}</p>
                  </div>
                  <GlassButton onClick={() => confirm()}>Confirm point</GlassButton>
                </div>
              )}
              {error && phase === 'pick' && <p className="mt-4 text-amber-700">{error}</p>}
              {phase === 'error' && (
                <p className="mt-6 text-amber-700">
                  {error} <button onClick={() => confirm()} className="ml-2 underline hover:text-brand">Try again</button>
                </p>
              )}
              {phase === 'loading' && (
                <div role="status" aria-live="polite">
                  <div className="text-[30px] font-bold leading-none tabular-nums text-strong">{chosen && fmt(chosen.lat, chosen.lon)}</div>
                  <div className="mt-6 h-[3px] w-full overflow-hidden rounded bg-line">
                    <div className="h-full rounded bg-brand transition-[width] duration-500 ease-out" style={{ width: `${((shown + 1) / STEPS.length) * 100}%` }} />
                  </div>
                  <ul className="mt-4 space-y-1.5 text-[15px]">
                    {STEPS.map((s, k) => (
                      <li key={s} className={k < shown ? 'text-muted' : k === shown ? 'animate-pulse text-ink' : 'text-muted/40'}>{k < shown ? '✓ ' : ''}{s}</li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          </div>
        )}

        {phase === 'view' && loaded && (
          <>
            <div className="mt-8 flex flex-wrap items-end justify-between gap-x-12 gap-y-3">
              <div>
                <div className="text-[12px] uppercase tracking-wide text-muted">You are standing at</div>
                <div className="mt-1 text-[30px] font-bold leading-none tabular-nums text-strong">{fmt(loaded.lat, loaded.lon)}</div>
              </div>
              <div className="flex gap-8 text-[17px]">
                {onBack && <button onClick={onBack} className="text-brand hover:underline">← Back to the analysis</button>}
                <button onClick={reset} className="text-ink hover:text-brand">Choose another place</button>
              </div>
            </div>

            <div className="relative mt-6 h-[560px] overflow-hidden rounded-[14px] border border-line bg-black">
              <WalkView grids={loaded.grids} sun={sky.sun} earth={sky.earth} earthRot={sky.earthRot} onView={setView} onError={setWebglError} />
              <Compass view={view} sun={sky.sun?.az} earth={sky.earth?.az} />
            </div>
            <p className="mt-3 text-[13px] text-muted">Drag to look around, scroll to zoom, arrow keys turn the view. Eye height 2 m.{moved && ' Opened at the nearest time the Sun is up; the Sun is below the horizon at this moment.'} {webglError && <span className="text-amber-700">{webglError}</span>}</p>

            <div className="mt-12 border-t border-line pt-10">
              <TimelineBar
                windowStart={windowStart} hours={WINDOW_H} pos={i} earthLit={null}
                playing={playing} speed={speed} speeds={SPEEDS}
                onPos={(h) => { setPlaying(false); setPos(h) }} onPlay={() => setPlaying((p) => !p)} onSpeed={setSpeed}
                onShift={(d) => { setSeries(null); setWindowStart(shiftDay(windowStart, d * 30)) }}
              />
            </div>

            <div className="mt-12 grid grid-cols-1 gap-x-12 gap-y-8 border-t border-line pt-8 sm:grid-cols-2">
              <SkyRow name="Sun" dot="#ba1a1a" az={sky.sun?.az} el={sky.sun?.el} s={status?.sun} />
              <SkyRow name="Earth" dot="#4d756a" az={sky.earth?.az} el={sky.earth?.el} s={status?.earth} />
            </div>

            <p className="mt-10 max-w-[760px] text-[12px] leading-relaxed text-muted">
              {dataNote(loaded.lat)} Terrain: the 15 km around the point, from NASA LRO LOLA heights, with the Moon's curvature included. Sun and Earth directions: NASA SPICE (DE440); shadows are cast by that terrain from the real Sun direction (shadowed ground is lifted slightly so it stays readable). Nobody has measured the ground at the scale of metres, so the fine relief and small craters you see are representative: built to match how rough the measured data around this point is (a fractal roughness exponent estimated from the data, and a lunar-typical crater size distribution), the same every time for this place. The Sun, Earth and horizon figures above use only the measured heights. The ground grain and stars are looks only. Earth: NASA SVS 3615 map of the surface and clouds (a fixed picture of the clouds), turned the way Earth really is at the chosen moment and lit by the real Sun, so its phase is right.</p>
          </>
        )}
      </div>
    </main>
  )
}

const STATUS_TEXT = { above: 'Above the horizon', behind: 'Hidden behind terrain', below: 'Below the horizon' } as const

function SkyRow({ name, dot, az, el, s }: { name: string; dot: string; az?: number; el?: number; s?: { status: keyof typeof STATUS_TEXT; skyline: number } }) {
  return (
    <div>
      <div className="flex items-center gap-2 text-[12px] uppercase tracking-wide text-muted">
        <span className="inline-block h-2.5 w-2.5 rounded-full" style={{ background: dot }} /> {name}
      </div>
      <div className="mt-1 text-[24px] font-bold tabular-nums text-strong">
        {el !== undefined && az !== undefined ? `${el.toFixed(1)}° high, ${az.toFixed(0)}°` : '…'}
      </div>
      <div className="mt-1 text-[14px] text-muted">
        {s ? `${STATUS_TEXT[s.status]}${s.status !== 'below' ? ` (skyline there: ${s.skyline.toFixed(1)}°)` : ''}` : ' '}
      </div>
    </div>
  )
}

/** A thin heading strip over the top of the picture: N, E, S, W, and where the Sun and Earth lie. */
function Compass({ view, sun, earth }: { view: ViewState; sun?: number; earth?: number }) {
  const hfov = (2 * Math.atan(Math.tan((view.fov * Math.PI) / 360) * view.aspect) * 180) / Math.PI
  const x = (az: number) => {
    const d = ((az - view.yaw + 540) % 360) - 180
    return Math.abs(d) <= hfov / 2 + 4 ? 50 + (d / hfov) * 100 : null
  }
  const marks: { az: number; label: string; strong: boolean; color?: string }[] = [
    ...[0, 45, 90, 135, 180, 225, 270, 315].map((az) => ({ az, label: ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'][az / 45], strong: az % 90 === 0 })),
    ...(sun !== undefined ? [{ az: sun, label: 'Sun', strong: true, color: '#ba1a1a' }] : []),
    ...(earth !== undefined ? [{ az: earth, label: 'Earth', strong: true, color: '#4d756a' }] : []),
  ]
  return (
    <div className="pointer-events-none absolute inset-x-0 top-0 h-[48px] overflow-hidden" aria-hidden>
      <div className="absolute inset-x-0 top-0 h-full bg-gradient-to-b from-black/55 to-transparent" />
      <div className="absolute left-1/2 top-[5px] -translate-x-1/2 text-[12px] tabular-nums text-white/80">{view.yaw.toFixed(0)}°</div>
      {marks.map((m) => {
        const left = x(m.az)
        return left === null ? null : (
          <span key={m.label} className="absolute top-[22px] -translate-x-1/2 whitespace-nowrap text-[12px]" style={{ left: `${left}%`, color: m.color ? '#fff' : m.strong ? '#fff' : 'rgba(255,255,255,0.6)' }}>
            {m.color && <span className="mr-1 inline-block h-2 w-2 rounded-full align-middle" style={{ background: m.color, boxShadow: '0 0 0 1.5px #fff' }} />}
            {m.label}
          </span>
        )
      })}
    </div>
  )
}
