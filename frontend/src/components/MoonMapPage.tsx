import { useEffect, useRef, useState } from 'react'
import { getMoonPoint, getMoonSites, type MoonSite, type PointReport } from '../api'
import { todayIso } from '../lib/input'
import Viewer from './moon3d/Viewer'
import MoonSurfaceMap, { type Point } from './MoonSurfaceMap'
import PointLoupe from './PointLoupe'
import PointReportView from './PointReport'
import DateField from './wizard/DateField'
import TimeField from './wizard/TimeField'
import { GlassButton } from './wizard/ui'

const STEPS_LOLA = ['Reading the Sun and Earth positions (NASA SPICE)', 'Tracing the terrain skyline (NASA LOLA)', 'Measuring the slope', 'Building the day profile']
const STEPS_SMOOTH = ['Reading the Sun and Earth positions (NASA SPICE)', 'Checking the horizon', 'Building the day profile']
const MIN_LOADING_MS = 1800

const fmt = (lat: number, lon: number) => `${Math.abs(lat).toFixed(3)}° ${lat < 0 ? 'S' : 'N'}, ${Math.abs(lon).toFixed(3)}° ${lon < 0 ? 'W' : 'E'}`
const pad = (n: number) => String(n).padStart(2, '0')
type View = '3d' | '2d'
const VIEW_KEY = 'lunaris-moonmap-view'
const readView = (): View => {
  try {
    return localStorage.getItem(VIEW_KEY) === '3d' ? '3d' : '2d'
  } catch {
    return '2d'
  }
}
const r3 = (v: number) => Math.round(v * 1000) / 1000

export default function MoonMapPage() {
  const [sites, setSites] = useState<MoonSite[]>([])
  const [chosen, setChosen] = useState<Point | null>(null)
  const [named, setNamed] = useState<MoonSite | null>(null)
  const [focus, setFocus] = useState<Point | null>(null) // where the close-up is centred
  const [hover, setHover] = useState<Point | null>(null)
  const [date, setDate] = useState(todayIso())
  const [hour, setHour] = useState(new Date().getUTCHours())
  const [phase, setPhase] = useState<'idle' | 'loading' | 'done' | 'error'>('idle')
  const [report, setReport] = useState<PointReport | null>(null)
  const [step, setStep] = useState(0)
  const [error, setError] = useState('')
  const asked = useRef(0)
  const [view, setViewState] = useState<View>(readView)
  const [fly, setFly] = useState<{ lat: number; lon: number; n: number } | undefined>()
  const [webglError, setWebglError] = useState('')
  const setView = (v: View) => {
    setViewState(v)
    try {
      localStorage.setItem(VIEW_KEY, v)
    } catch {
      /* not remembered */
    }
  }

  useEffect(() => {
    getMoonSites().then(setSites).catch(() => setSites([]))
  }, [])

  const analyze = (p: Point, d: string, h: number) => {
    const id = ++asked.current
    setPhase('loading')
    setStep(0)
    setError('')
    const started = Date.now()
    const ticker = setInterval(() => setStep((s) => s + 1), 450)
    getMoonPoint(p.lat, p.lon, `${d}T${pad(h)}:00:00`)
      .then(async (r) => {
        await new Promise((res) => setTimeout(res, Math.max(0, MIN_LOADING_MS - (Date.now() - started))))
        if (id !== asked.current) return
        setReport(r)
        setPhase('done')
      })
      .catch((e: Error) => {
        if (id !== asked.current) return
        setError(e.message)
        setPhase('error')
      })
      .finally(() => clearInterval(ticker))
  }

  const pick = (p: Point, s?: MoonSite, fromCloseUp = false) => {
    asked.current++ // whatever was loading belongs to the old point
    setChosen(p)
    if (!fromCloseUp) setFocus(p) // a click inside the close-up must not make it jump
    setNamed(s ?? null)
    setReport(null)
    setPhase('idle')
  }
  const changeWhen = (d: string, h: number) => {
    setDate(d)
    setHour(h)
    if (chosen && phase !== 'idle') analyze(chosen, d, h) // results on screen: refresh them for the new time
  }

  const steps = chosen && chosen.lat <= -85 ? STEPS_LOLA : STEPS_SMOOTH
  const shown = Math.min(step, steps.length - 1)

  return (
    <main className="flex-1 overflow-y-auto">
      <div className="mx-auto max-w-[1280px] px-10 pb-24 pt-10">
        <h1 className="text-[34px] font-medium text-brand">Moon Map</h1>
        <p className="mt-1 max-w-[700px] text-muted">
          The whole Moon, {view === '3d' ? 'as a globe' : 'flattened'}. Choose any point, confirm it, and see how the Sun strikes it, whether Earth can be reached from there, and how steep the ground is, on the date you pick.
        </p>

        <div className="mt-8 flex flex-wrap items-start gap-x-12 gap-y-4">
          <div className="w-[400px] max-w-full">
            <DateField value={date} onChange={(d) => changeWhen(d, hour)} solid />
          </div>
          <div className="w-[190px]">
            <TimeField value={hour} onChange={(h) => changeWhen(date, h)} />
          </div>
          <button onClick={() => changeWhen(todayIso(), new Date().getUTCHours())} className="pt-3.5 text-[17px] text-ink hover:text-brand">Now</button>
        </div>

        <div className="mt-6">
          <div className="mb-3 flex items-center justify-between gap-6">
            {view === '3d' ? (
              <label className="flex items-center gap-3 text-[15px] text-muted">
                Go to a known place
                <select
                  value=""
                  onChange={(e) => {
                    const site = sites.find((x) => x.name === e.target.value)
                    if (!site) return
                    const p = { lat: site.lat, lon: site.lon > 180 ? site.lon - 360 : site.lon }
                    pick(p, site)
                    setFly((f) => ({ ...p, n: (f?.n ?? 0) + 1 }))
                  }}
                  className="max-w-[260px] rounded-[8px] border border-line bg-white/70 px-2 py-1 text-[15px] text-ink outline-none focus:border-brand"
                >
                  <option value="">Choose…</option>
                  {sites.map((x) => <option key={x.name} value={x.name}>{x.name}</option>)}
                </select>
              </label>
            ) : <span />}
            <div role="radiogroup" aria-label="Map view" className="grid grid-cols-2 rounded-[8px] border border-line p-0.5 text-[14px]">
              {(['3d', '2d'] as const).map((v) => (
                <button key={v} role="radio" aria-checked={view === v} onClick={() => setView(v)} className={`rounded-[6px] px-4 py-1 transition-colors ${view === v ? 'bg-active font-medium text-brand' : 'text-muted hover:text-ink'}`}>
                  {v.toUpperCase()}
                </button>
              ))}
            </div>
          </div>
          {view === '3d' ? (
            <>
              <div className="h-[480px] overflow-hidden rounded-[12px] border border-line bg-[#05080c]">
                <Viewer initialLat={-35} initialLon={0} initialAlt={4300} mode="natural" sun={0} sunLat={-25} marker={chosen} focus={fly} showHint={false} onPick={(p) => pick({ lat: r3(p.lat), lon: r3(p.lon) })} onError={setWebglError} />
              </div>
              <p className="mt-2 text-[13px] text-muted">Drag to turn the Moon, scroll to come closer, click to choose a point.{webglError && <span className="ml-3 text-amber-700">{webglError}</span>}</p>
            </>
          ) : (
          <>
          <MoonSurfaceMap sites={sites} chosen={chosen} onPick={pick} onHover={setHover} />
          <div className="mt-2 flex flex-wrap items-center gap-x-6 gap-y-1 text-[13px] text-muted">
            <span className="flex items-center gap-2"><span className="inline-block h-3 w-3 rounded-full border-2 border-[#24231f] bg-white" /> Landing site</span>
            <span className="flex items-center gap-2"><span className="inline-block h-3 w-3 rounded-full bg-[#24231f]" /> Crash site</span>
            <span className="flex items-center gap-2"><span className="inline-block h-3 w-3 rounded-full bg-[#3d7a68]" /> South-pole candidate</span>
            <span className="flex items-center gap-2"><span className="inline-block h-0 w-6 border-t border-dashed border-ink" /> Detailed 40 m terrain below this line (85°S)</span>
            <span className="ml-auto min-w-[150px] text-right tabular-nums">{hover ? fmt(hover.lat, hover.lon) : 'Move over the map'}</span>
          </div>
          <p className="mt-1 text-[12px] text-muted">Image: NASA / GSFC / Arizona State University, LRO WAC mosaic, via Moon Trek. Mission coordinates as published for each mission.</p>
          </>
          )}
        </div>

        <div className="mt-8 border-t border-line pt-6">
          {!chosen && <p className="text-[17px] text-muted">{view === '3d' ? 'Click the globe, or pick a known place from the list, to choose a point.' : 'Click the map, or one of the marked places, to choose a point.'}</p>}

          {chosen && focus && (
            <div className="grid grid-cols-1 gap-x-10 gap-y-6 md:grid-cols-2">
              <div className="flex flex-col justify-between gap-6">
                <div>
                  <div className="text-[12px] uppercase tracking-wide text-muted">{named ? named.name : 'Chosen point'}</div>
                  <div className="mt-1 text-[30px] font-bold leading-none tabular-nums text-strong">{fmt(chosen.lat, chosen.lon)}</div>
                  <div className="mt-2 text-[14px] text-muted">
                    {chosen.lat <= -85 ? 'Detailed 40 m NASA LOLA terrain.' : 'Whole-Moon NASA LOLA elevation (about 470 m per pixel): regional slope and skyline.'}
                  </div>
                  <p className="mt-4 max-w-[420px] text-[14px] leading-relaxed text-muted">
                    The numbers below are worked out for exactly this coordinate, not for the ring on the map. The close-up lets you place it more precisely: drag to look around, click to move the point.
                  </p>
                </div>
                {phase === 'idle' && (
                  <div>
                    <GlassButton onClick={() => analyze(chosen, date, hour)}>Confirm point</GlassButton>
                  </div>
                )}
              </div>
              <PointLoupe center={focus} chosen={chosen} sites={sites} report={phase === 'done' && report && report.lat === chosen.lat && report.lon === chosen.lon ? report : null} onPick={(p, s) => pick(p, s, true)} />
            </div>
          )}

          {phase === 'loading' && (
            <div className="mt-6" role="status" aria-live="polite">
              <div className="h-[3px] w-full overflow-hidden rounded bg-line">
                <div className="h-full rounded bg-brand transition-[width] duration-500 ease-out" style={{ width: `${((shown + 1) / steps.length) * 100}%` }} />
              </div>
              <ul className="mt-4 space-y-1.5 text-[15px]">
                {steps.map((s, i) => (
                  <li key={s} className={i < shown ? 'text-muted' : i === shown ? 'animate-pulse text-ink' : 'text-muted/40'}>
                    {i < shown ? '✓ ' : ''}{s}
                  </li>
                ))}
              </ul>
            </div>
          )}
          {phase === 'error' && (
            <p className="mt-6 text-amber-700">
              {error} <button onClick={() => chosen && analyze(chosen, date, hour)} className="ml-2 underline hover:text-brand">Try again</button>
            </p>
          )}
          {phase === 'done' && report && <PointReportView r={report} hourUtc={hour} />}
        </div>
      </div>
    </main>
  )
}
