import { useEffect, useMemo, useRef, useState } from 'react'
import { getSites, getSuitabilityMap, getVehicles, runScenario, type ScenarioResult, type SuitabilityMap, type VehicleProfile } from '../api'
import { todayIso } from '../lib/input'
import { downloadReport } from '../lib/report'
import { useSettings } from '../lib/settings'
import Decision from './Decision'
import DateField from './wizard/DateField'
import { GlassButton } from './wizard/ui'

const R = 1_737_400 // lunar radius, m
const HALF_KM = 155
const N = 128
const EDGE = -85 // the LOLA polar grid, and the neural network's training area, end here
// light grey-blue (poor) to deep slate (good): neutral like the Moon, one hue so the order reads at a glance
const STOPS: [number, number, number][] = [[236, 239, 241], [160, 178, 192], [78, 106, 130], [30, 52, 76]]
const ramp = (t: number): [number, number, number] => {
  const x = Math.min(1, Math.max(0, t)) * (STOPS.length - 1)
  const i = Math.min(STOPS.length - 2, Math.floor(x))
  const f = x - i
  return [0, 1, 2].map((k) => Math.round(STOPS[i][k] + (STOPS[i + 1][k] - STOPS[i][k]) * f)) as [number, number, number]
}
const GRADIENT = `linear-gradient(to right, ${[0, 1 / 3, 2 / 3, 1].map((t) => `rgb(${ramp(t)})`).join(', ')})`

/** [1 2 1] blur in both directions, ignoring empty pixels: takes the network's pixel-to-pixel noise out of the picture. */
const smooth = (g: (number | null)[][], ok: boolean[][]) => {
  const n = g.length
  const w = [1, 2, 1]
  const out: (number | null)[][] = g.map((row) => row.map(() => null))
  for (let r = 0; r < n; r++) {
    for (let c = 0; c < n; c++) {
      if (g[r][c] === null || !ok[r][c]) continue
      let sum = 0
      let wt = 0
      for (let dr = -1; dr <= 1; dr++) {
        for (let dc = -1; dc <= 1; dc++) {
          const v = g[r + dr]?.[c + dc]
          if (v === null || v === undefined || !ok[r + dr][c + dc]) continue
          const k = w[dr + 1] * w[dc + 1]
          sum += v * k
          wt += k
        }
      }
      out[r][c] = sum / wt
    }
  }
  return out
}

type Metric = { id: keyof Pick<SuitabilityMap, 'suitability_index' | 'sunlit_fraction' | 'dte_fraction' | 'longest_darkness_hours' | 'slope_deg'>; label: string; lo: number; hi: number; higherIsBetter: boolean; show: (v: number) => string }
const METRICS: Metric[] = [
  { id: 'suitability_index', label: 'Suitability', lo: 0, hi: 80, higherIsBetter: true, show: (v) => `${Math.round(v)} / 100` },
  { id: 'sunlit_fraction', label: 'Sunlight', lo: 0, hi: 1, higherIsBetter: true, show: (v) => `${Math.round(v * 100)}% of the month` },
  { id: 'dte_fraction', label: 'Earth contact', lo: 0, hi: 1, higherIsBetter: true, show: (v) => `${Math.round(v * 100)}% of the month` },
  { id: 'longest_darkness_hours', label: 'Longest night', lo: 0, hi: 720, higherIsBetter: false, show: (v) => `${Math.round(v)} h` },
  { id: 'slope_deg', label: 'Slope', lo: 0, hi: 20, higherIsBetter: false, show: (v) => `${v.toFixed(1)}°` },
]

const colour = (m: Metric, v: number, lo: number, hi: number) => {
  const t = Math.min(1, Math.max(0, (v - lo) / (hi - lo || 1)))
  return ramp(m.higherIsBetter ? t : 1 - t)
}

const project = (lat: number, lon: number) => {
  const rho = 2 * R * Math.tan(((90 + lat) * Math.PI) / 360)
  const x = rho * Math.sin((lon * Math.PI) / 180)
  const y = rho * Math.cos((lon * Math.PI) / 180)
  return { fx: (x + HALF_KM * 1000) / (2 * HALF_KM * 1000), fy: (HALF_KM * 1000 - y) / (2 * HALF_KM * 1000) }
}

type Cell = { row: number; col: number }
const signedLon = (l: number) => (l > 180 ? l - 360 : l)

const Step = ({ n, title, children, off }: { n: number; title: string; children?: React.ReactNode; off?: boolean }) => (
  <section className={`border-t border-line py-6 first:border-t-0 first:pt-0 ${off ? 'opacity-50' : ''}`}>
    <h2 className="flex items-baseline gap-3 text-[13px] uppercase tracking-wide text-muted">
      <span className="text-brand">{n}</span>{title}
    </h2>
    {children}
  </section>
)

const LOAD_STEPS = ['Reading the NASA LOLA terrain of the south pole', 'Placing the Sun and Earth for the next 30 days (NASA SPICE)', 'The neural network scores every spot', 'Painting the map']

/** Full-page waiting screen for the first map: what is happening, step by step, and roughly how long. */
function MapLoading() {
  const [step, setStep] = useState(0)
  useEffect(() => {
    const t = setInterval(() => setStep((s) => Math.min(s + 1, LOAD_STEPS.length - 1)), 2600)
    return () => clearInterval(t)
  }, [])
  return (
    <main className="flex flex-1 items-center justify-center overflow-y-auto px-10" aria-busy="true">
      <div className="w-full max-w-[460px]">
        <div className="relative mx-auto h-[150px] w-[150px]">
          <div className="map-orbit absolute inset-0 rounded-full border border-line" />
          <div className="absolute inset-[18px] overflow-hidden rounded-full bg-[#dfe6e2]">
            <div className="skeleton absolute inset-0 rounded-none" />
          </div>
          <span className="map-orbit-dot absolute left-1/2 top-0 h-2.5 w-2.5 -translate-x-1/2 -translate-y-1/2 rounded-full bg-brand" />
        </div>
        <h1 className="gem-text mt-9 text-center text-[26px] font-medium">Preparing the South Area map</h1>
        <p className="mt-2 text-center text-[15px] text-muted">{(N * N).toLocaleString('en-US')} spots on the south-polar cap, about 10 seconds</p>
        <div className="mt-6 h-[3px] overflow-hidden rounded-full bg-line">
          <div className="map-progress h-full rounded-full bg-brand" />
        </div>
        <ol className="mt-6 space-y-2.5">
          {LOAD_STEPS.map((t, i) => (
            <li key={t} className={`flex items-center gap-3 text-[14.5px] transition-colors duration-500 ${i < step ? 'text-muted' : i === step ? 'text-ink' : 'text-muted/40'}`}>
              <span className={`grid h-4 w-4 shrink-0 place-items-center rounded-full border transition-colors duration-500 ${i < step ? 'border-brand bg-brand' : i === step ? 'border-brand' : 'border-line'}`}>
                {i < step && <span className="h-1.5 w-1.5 rounded-full bg-white" />}
                {i === step && <span className="h-1.5 w-1.5 animate-ping rounded-full bg-brand" />}
              </span>
              {t}
            </li>
          ))}
        </ol>
      </div>
    </main>
  )
}

/** A map of the whole south-polar cap where every pixel is the trained-from-scratch network's guess at how good a landing there would be. */
export default function LandingMapPage() {
  const { level } = useSettings()
  const [vehicles, setVehicles] = useState<VehicleProfile[]>([])
  const [vehicleId, setVehicleId] = useState('')
  const [check, setCheck] = useState<{ status: 'idle' } | { status: 'loading' } | { status: 'error'; message: string } | { status: 'done'; result: ScenarioResult }>({ status: 'idle' })
  const checked = useRef(0)
  const [date, setDate] = useState(todayIso())
  const [metric, setMetric] = useState<Metric>(METRICS[0])
  const [data, setData] = useState<SuitabilityMap | null>(null)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)
  const [picked, setPicked] = useState<Cell | null>(null)
  const [sites, setSites] = useState<{ name: string; lat: number; lon: number }[]>([])
  const cache = useRef(new Map<string, SuitabilityMap>())
  const asked = useRef(0)
  const canvas = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    getSites().then(setSites).catch(() => undefined)
    getVehicles().then((v) => { setVehicles(v); setVehicleId((id) => id || v[0]?.id || '') }).catch(() => undefined)
  }, [])

  useEffect(() => {
    if (!/^\d{4}-\d{2}-\d{2}$/.test(date)) return
    const hit = cache.current.get(date)
    if (hit) {
      setData(hit)
      setError('')
      return
    }
    const id = ++asked.current
    setLoading(true)
    setError('')
    setPicked(null)
    getSuitabilityMap(date, HALF_KM, N)
      .then((m) => {
        if (id !== asked.current) return
        cache.current.set(date, m)
        setData(m)
      })
      .catch((e: Error) => id === asked.current && setError(e.message))
      .finally(() => id === asked.current && setLoading(false))
  }, [date])

  const valid = useMemo(() => {
    if (!data) return null
    return data.lat.map((row) => row.map((l) => l !== null && l <= EDGE))
  }, [data])

  const shownGrid = useMemo(() => (data && valid ? smooth(data[metric.id], valid) : null), [data, valid, metric])

  // the colour scale spans what this map actually contains (2nd to 98th percentile), so the differences are easy to see
  const range = useMemo(() => {
    if (!shownGrid || !valid) return { lo: metric.lo, hi: metric.hi }
    const xs: number[] = []
    shownGrid.forEach((row, r) => row.forEach((v, c) => { if (v !== null && valid[r][c]) xs.push(v) }))
    xs.sort((a, b) => a - b)
    if (xs.length < 10) return { lo: metric.lo, hi: metric.hi }
    return { lo: xs[Math.floor(xs.length * 0.02)], hi: xs[Math.floor(xs.length * 0.98)] }
  }, [shownGrid, valid, metric])

  // paint the pixels
  useEffect(() => {
    const c = canvas.current
    if (!c || !shownGrid || !valid) return
    const small = document.createElement('canvas')
    small.width = N
    small.height = N
    const sctx = small.getContext('2d')
    c.width = 960
    c.height = 960
    const ctx = c.getContext('2d')
    if (!ctx || !sctx) return
    const img = sctx.createImageData(N, N)
    const grid = shownGrid
    for (let r = 0; r < N; r++) {
      for (let k = 0; k < N; k++) {
        const v = grid[r][k]
        const o = (r * N + k) * 4
        if (v === null || !valid[r][k]) continue
        const [R2, G, B] = colour(metric, v, range.lo, range.hi)
        img.data[o] = R2
        img.data[o + 1] = G
        img.data[o + 2] = B
        img.data[o + 3] = 255
      }
    }
    sctx.putImageData(img, 0, 0)
    ctx.imageSmoothingEnabled = true
    ctx.imageSmoothingQuality = 'high'
    ctx.drawImage(small, 0, 0, 960, 960)
  }, [shownGrid, valid, metric, range])

  const onClick = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!data || !valid) return
    const b = e.currentTarget.getBoundingClientRect()
    const col = Math.round(((e.clientX - b.left) / b.width) * (N - 1))
    const row = Math.round(((e.clientY - b.top) / b.height) * (N - 1))
    if (valid[row]?.[col] && data.suitability_index[row][col] !== null) setPicked({ row, col })
  }

  const at = (g: (number | null)[][]) => (picked ? g[picked.row][picked.col] : null)
  const pLat = picked && data ? data.lat[picked.row][picked.col] : null
  const pLon = picked && data ? data.lon[picked.row][picked.col] : null

  const spot = pLat !== null && pLon !== null ? { lat: String(Math.round(pLat * 1000) / 1000), lon: String(Math.round(signedLon(pLon) * 1000) / 1000) } : null
  const vehicleName = vehicles.find((v) => v.id === vehicleId)?.name ?? vehicleId

  // analyse the chosen spot with the chosen vehicle, here on the page
  const analyse = (vid: string) => {
    if (!spot || !vid) return
    const id = ++checked.current
    setCheck({ status: 'loading' })
    runScenario({ date, vehicleId: vid, lat: spot.lat, lon: spot.lon })
      .then((result) => id === checked.current && setCheck({ status: 'done', result }))
      .catch((e: Error) => id === checked.current && setCheck({ status: 'error', message: e.message }))
  }
  // a new spot or date makes the earlier answer stale
  useEffect(() => {
    checked.current++
    setCheck({ status: 'idle' })
  }, [picked, date])

  // best pixel of the map, so the page can point at it
  const best = useMemo(() => {
    if (!data || !valid) return null
    let top: Cell | null = null
    let v = -1
    for (let r = 0; r < N; r++) {
      for (let c = 0; c < N; c++) {
        const s = data.suitability_index[r][c]
        if (s !== null && valid[r][c] && s > v) { v = s; top = { row: r, col: c } }
      }
    }
    return top
  }, [data, valid])

  const pos = (p: Cell) => ({ left: `${(p.col / (N - 1)) * 100}%`, top: `${(p.row / (N - 1)) * 100}%` })

  const uncertain = picked && data ? (at(data.uncertainty) ?? 0) > 0.15 : false
  const score = picked && data ? at(data.suitability_index) : null

  // the first map of a visit: the page appears whole once it is ready, not piece by piece
  if (!data && !error) return <MapLoading />

  return (
    <main className="flex-1 overflow-y-auto">
      <div className="rise-in mx-auto max-w-[1280px] px-10 pb-24 pt-10">
        <h1 className="text-[34px] font-medium text-brand">South Area</h1>
        <p className="mt-1 max-w-[720px] text-muted">
          The whole south-polar cap, 85° to 90° S, scored for the 30 days after the date you pick. A neural network written from scratch for this project fills in every spot at once.
        </p>

        <div className="mt-8 flex flex-wrap items-end justify-between gap-x-12 gap-y-4 border-b border-line pb-4">
          <div className="w-[300px] max-w-full"><DateField value={date} onChange={setDate} solid /></div>
          <div role="radiogroup" aria-label="Map layer" className="flex flex-wrap gap-x-7 gap-y-2 pb-1 text-[16px]">
            {METRICS.map((m) => (
              <button key={m.id} role="radio" aria-checked={metric.id === m.id} onClick={() => setMetric(m)} className={`border-b pb-0.5 transition-colors ${metric.id === m.id ? 'border-brand font-medium text-brand' : 'border-transparent text-muted hover:text-ink'}`}>
                {m.label}
              </button>
            ))}
          </div>
        </div>

        <div className="mt-8 grid grid-cols-[minmax(0,620px)_minmax(300px,1fr)] items-start gap-x-14 gap-y-10 max-[1279px]:grid-cols-1">
          <div className="min-[1280px]:sticky min-[1280px]:top-6">
            <div className="relative aspect-square w-full cursor-crosshair select-none" onClick={onClick} aria-label="Suitability map of the lunar south pole">
              <canvas ref={canvas} className="absolute inset-0 h-full w-full" style={{ imageRendering: 'auto' }} />
              <svg viewBox="0 0 100 100" className="pointer-events-none absolute inset-0 h-full w-full" fill="none">
                <circle cx="50" cy="50" r={(project(EDGE, 90).fx - 0.5) * 100} stroke="#cfd6d1" strokeWidth="0.3" />
                <circle cx="50" cy="50" r="0.7" fill="#2d2f2c" />
              </svg>
              {sites.map((s) => {
                const { fx, fy } = project(s.lat, signedLon(s.lon))
                if (fx < 0 || fx > 1 || fy < 0 || fy > 1) return null
                return <span key={s.name} title={s.name} className="pointer-events-none absolute h-2.5 w-2.5 -translate-x-1/2 -translate-y-1/2 rounded-full border border-ink/70" style={{ left: `${fx * 100}%`, top: `${fy * 100}%` }} />
              })}
              {best && !loading && <span title="Highest score on this map" className="pointer-events-none absolute h-2 w-2 -translate-x-1/2 -translate-y-1/2 rotate-45 bg-ink" style={pos(best)} />}
              {picked && <span className="pointer-events-none absolute h-4 w-4 -translate-x-1/2 -translate-y-1/2 rounded-full border-2 border-ink" style={pos(picked)} />}
              {loading && (
                <div className="absolute inset-0 grid place-items-center text-center">
                  <div>
                    <div className="mx-auto h-5 w-5 animate-spin rounded-full border-2 border-brand/30 border-t-brand" />
                    <p className="gem-text mt-3 text-[15px]">The network is scoring {(N * N).toLocaleString('en-US')} spots…<br />about 10 seconds</p>
                  </div>
                </div>
              )}
            </div>
            <div className="mt-4 flex flex-wrap items-center gap-x-3 gap-y-2 text-[13px] text-muted">
              <span>{metric.higherIsBetter ? 'worse' : 'better'}</span>
              <span className="h-2 w-[180px] rounded-full" style={{ background: GRADIENT }} />
              <span>{metric.higherIsBetter ? 'better' : 'worse'}</span>
              <span className="ml-auto flex items-center gap-4">
                <span className="flex items-center gap-1.5"><span className="h-2.5 w-2.5 rounded-full border border-ink/70" />known site</span>
                <span className="flex items-center gap-1.5"><span className="h-2 w-2 rotate-45 bg-ink" />best here</span>
              </span>
            </div>
            <p className="mt-3 text-[13px] leading-relaxed text-muted">The dot in the middle is the pole. The thin ring is 85° S, where the detailed NASA terrain data ends.</p>
          </div>

          <div className="max-w-[500px]">
            {error && <p className="mb-6 text-[15px] text-[#ba1a1a]">{error}</p>}

            <Step n={1} title="Choose a spot">
              {!picked || pLat === null || pLon === null ? (
                <p className="mt-3 text-[16px] leading-relaxed text-muted">Press anywhere on the map to read what the network expects there.</p>
              ) : (
                <div className="mt-3">
                  <div className="flex items-end justify-between gap-6">
                    <p className="text-[24px] font-medium leading-tight text-ink">{Math.abs(pLat).toFixed(2)}° S, {Math.abs(signedLon(pLon)).toFixed(2)}° {signedLon(pLon) < 0 ? 'W' : 'E'}</p>
                    {score !== null && <p className="shrink-0 text-ink"><span className="text-[34px] font-medium leading-none text-brand">{Math.round(score)}</span><span className="ml-1 text-[14px] text-muted">/ 100</span></p>}
                  </div>
                  <dl className="mt-5 grid grid-cols-2 gap-x-8 gap-y-4">
                    {METRICS.filter((m) => m.id !== 'suitability_index').map((m) => {
                      const v = at(data![m.id])
                      return (
                        <div key={m.id}>
                          <dt className={`text-[13px] ${m.id === metric.id ? 'font-medium text-brand' : 'text-muted'}`}>{m.label}</dt>
                          <dd className="mt-0.5 text-[17px] text-ink">{v === null ? '–' : m.show(v)}</dd>
                        </div>
                      )
                    })}
                  </dl>
                  <p className="mt-4 text-[13px] leading-relaxed text-muted">The network's estimate{uncertain ? ', and it is less sure about this spot' : ''}. Step 2 recomputes it exactly.</p>
                </div>
              )}
            </Step>

            <Step n={2} title="Pick a vehicle" off={!picked}>
              <div className="mt-3 flex flex-wrap items-center gap-4">
                <select
                  value={vehicleId}
                  disabled={!picked}
                  onChange={(e) => { setVehicleId(e.target.value); if (check.status !== 'idle') analyse(e.target.value) }}
                  aria-label="Vehicle"
                  className="w-full rounded-[8px] border border-line bg-white/70 px-3 py-2.5 text-[16px] text-ink outline-none focus:border-brand"
                >
                  {vehicles.map((v) => <option key={v.id} value={v.id}>{v.name}</option>)}
                </select>
                <GlassButton onClick={() => analyse(vehicleId)} disabled={!picked || !vehicleId || check.status === 'loading'}>Check this spot</GlassButton>
              </div>
            </Step>

            <Step n={3} title="Result" off={check.status === 'idle'}>
              {check.status === 'idle' && <p className="mt-3 text-[16px] leading-relaxed text-muted">The call, its risks and what to do next appear here.</p>}
              {check.status === 'loading' && <p className="mt-3 animate-pulse text-[15px] text-muted">Running exact physics for 30 days…</p>}
              {check.status === 'error' && <p className="mt-3 text-[15px] text-[#ba1a1a]">{check.message}</p>}
              {check.status === 'done' && spot && (
                <div className="mt-3">
                  <p className="mb-1 text-[15px] text-muted">{vehicleName} · {check.result.assessment.vehicle_index.toFixed(0)} / 100</p>
                  <Decision a={check.result.assessment} q={check.result.query} level={level} />
                  <button
                    onClick={() => downloadReport([{ label: 'A', scenario: { date, vehicleId, ...spot }, vehicle: vehicleName, result: check.result }])}
                    className="mt-5 text-[15px] text-brand hover:underline hover:underline-offset-4"
                  >
                    Download report
                  </button>
                </div>
              )}
            </Step>
          </div>
        </div>
      </div>
    </main>
  )
}
