import { useEffect, useMemo, useRef, useState } from 'react'
import { getAreaRelief, getHorizon, getLibration, getTimeseries, getVehicles, type AreaRelief, type LibrationReport, type ScenarioResult, type TimeSeries } from '../api'
import { parseLat, parseLon } from '../lib/input'
import { EARTH, SUN } from '../lib/palette'
import type { Scenario } from '../types'

// ---- one request per place and date, however many times the page is drawn --------------------------------------------
const memo = <T,>(f: (key: string) => Promise<T>) => {
  const cache = new Map<string, Promise<T>>()
  return (key: string) => {
    let hit = cache.get(key)
    if (!hit) {
      hit = f(key)
      hit.catch(() => cache.delete(key))
      cache.set(key, hit)
    }
    return hit
  }
}
const seriesOf = memo((k) => { const [la, lo, d] = k.split('|'); return getTimeseries(Number(la), Number(lo), d) })
const horizonOf = memo((k) => { const [la, lo] = k.split('|'); return getHorizon(Number(la), Number(lo)) })
const reliefOf = memo((k) => { const [la, lo] = k.split('|'); return getAreaRelief(Number(la), Number(lo)) })
const vehiclesOf = memo(() => getVehicles())
const libOf = memo((k) => { const [la, lo, d] = k.split('|'); return getLibration(Number(la), Number(lo), `${d}T00:00:00`) })

/** false on the first frame, true right after: lets markers glide in from zero instead of appearing in place. */
function useArrived() {
  const [on, setOn] = useState(false)
  useEffect(() => {
    const id = requestAnimationFrame(() => setOn(true))
    return () => cancelAnimationFrame(id)
  }, [])
  return on
}

const INK = '#2d2f2c'
const MUTED = '#6e736e'
const LINE = '#d7dcd7'

const pad = (n: number) => String(n).padStart(2, '0')
const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
const dayLabel = (startIso: string, dayOffset: number) => {
  const d = new Date(`${startIso}T00:00:00Z`)
  d.setUTCDate(d.getUTCDate() + dayOffset)
  return `${d.getUTCDate()} ${MONTHS[d.getUTCMonth()]}`
}
const span = (h: number) => (h < 48 ? `${Math.round(h)} h` : `${(h / 24).toFixed(1)} days`)

/** Longest run of `flags[i] === value`: where it starts and how many samples it lasts. */
function longestRun(flags: boolean[], value: boolean) {
  let best = { start: 0, len: 0 }
  let start = -1
  for (let i = 0; i <= flags.length; i++) {
    const on = i < flags.length && flags[i] === value
    if (on && start < 0) start = i
    if (!on && start >= 0) {
      if (i - start > best.len) best = { start, len: i - start }
      start = -1
    }
  }
  return best
}
/** Runs of equal values, for drawing a strip as a few rectangles instead of 720. */
function runs(flags: boolean[]) {
  const out: { from: number; to: number; on: boolean }[] = []
  flags.forEach((f, i) => {
    const last = out[out.length - 1]
    if (last && last.on === f) last.to = i + 1
    else out.push({ from: i, to: i + 1, on: f })
  })
  return out
}

function Title({ children, note }: { children: string; note?: string }) {
  return (
    <div className="flex items-baseline justify-between gap-4">
      <h4 className="text-[13px] uppercase tracking-wide text-muted">{children}</h4>
      {note && <span className="text-[12px] text-muted">{note}</span>}
    </div>
  )
}

// ---- 1. the score on its scale --------------------------------------------------------------------------------------
const BANDS = [
  { from: 0, to: 20, label: 'Do not proceed' },
  { from: 20, to: 40, label: 'Not recommended' },
  { from: 40, to: 60, label: 'Possible' },
  { from: 60, to: 100, label: 'Go' },
]
const BAND_FILL = ['#c9d0cc', '#b4c4bc', '#91ab9f', '#5f8a7a']

export function ScoreScale({ score, uncapped }: { score: number; uncapped: number | null }) {
  const on = useArrived()
  const at = (v: number) => `${on ? Math.min(100, Math.max(0, v)) : 0}%`
  return (
    <div>
      <Title>Where the score falls</Title>
      <div className="relative mt-3 pt-4">
        <div className="flex h-2.5 gap-[3px]">
          {BANDS.map((b, i) => (
            <div key={b.label} style={{ width: `${b.to - b.from}%`, background: BAND_FILL[i] }} className={i === 0 ? 'rounded-l-full' : i === BANDS.length - 1 ? 'rounded-r-full' : ''} />
          ))}
        </div>
        {uncapped !== null && uncapped - score >= 3 && (
          <span title={`Without its limiting factor: ${uncapped.toFixed(0)}`} className="glide absolute top-[22px] h-[14px] w-[14px] -translate-x-1/2 -translate-y-1/2 rounded-full border-[1.5px] border-dashed border-muted bg-transparent" style={{ left: at(uncapped) }} />
        )}
        <span className="glide absolute top-0 -translate-x-1/2 text-[12px] font-medium tabular-nums text-ink" style={{ left: at(score) }}>{score.toFixed(0)}</span>
        <span className="glide absolute top-[14px] h-[18px] w-[2px] -translate-x-1/2 bg-ink" style={{ left: at(score) }} />
      </div>
      <div className="mt-2 flex text-[12px] text-muted">
        {BANDS.map((b) => (
          <span key={b.label} style={{ width: `${b.to - b.from}%` }} className={`pr-2 leading-tight ${score >= b.from && (score < b.to || b.to === 100) ? 'font-medium text-ink' : ''}`}>{b.label}</span>
        ))}
      </div>
      {uncapped !== null && uncapped - score >= 3 && <p className="mt-2 text-[12px] text-muted">The dashed circle is where it would sit without the limiting factor ({uncapped.toFixed(0)}).</p>}
    </div>
  )
}

// ---- 2. the month as two strips -------------------------------------------------------------------------------------
export function MissionStrips({ ts, date, hasRelay }: { ts: TimeSeries; date: string; hasRelay: boolean }) {
  const n = ts.hours.length
  const sunLit = useMemo(() => ts.sun_lit_fraction.map((v) => v >= 0.5), [ts])
  const earthOn = ts.earth_visible
  const dark = longestRun(sunLit, false)
  const out = longestRun(earthOn, false)
  const step = n > 1 ? ts.hours[1] - ts.hours[0] : 1
  const [hover, setHover] = useState<number | null>(null)
  const ref = useRef<SVGSVGElement>(null)
  const W = 460
  const LX = 40 // the strips start after the row names
  const x = (i: number) => LX + (i / n) * (W - LX)
  const move = (e: React.PointerEvent) => {
    const b = ref.current?.getBoundingClientRect()
    if (!b) return
    setHover(Math.min(n - 1, Math.max(0, Math.floor((((e.clientX - b.left) / b.width) * W - LX) / (W - LX) * n))))
  }
  const days = Math.round((n * step) / 24)
  const ticks = [0, 7, 14, 21, days - 1].filter((d, i, a) => d >= 0 && a.indexOf(d) === i)
  const rows = [
    { key: 'sun', label: 'Sun', color: SUN, flags: sunLit, gap: dark, gapWord: 'night' },
    ...(hasRelay ? [] : [{ key: 'earth', label: 'Earth', color: EARTH, flags: earthOn, gap: out, gapWord: 'no contact' }]),
  ]
  const H = 22
  const total = rows.length * (H + 16) + 28
  return (
    <div data-viz="strips">
      <Title note="hover to read a moment">The month at this spot</Title>
      <svg ref={ref} viewBox={`0 0 ${W} ${total}`} className="mt-2 w-full" onPointerMove={move} onPointerLeave={() => setHover(null)} role="img" aria-label="30-day strips of sunlight and Earth contact">
        {rows.map((r, ri) => {
          const y = ri * (H + 16) + 24
          return (
            <g key={r.key}>
              <rect x={LX} y={y} width={W - LX} height={H} rx={3} fill="#e7eae7" />
              <g className="grow-x" style={{ '--i': ri } as React.CSSProperties}>
              {runs(r.flags).filter((u) => u.on).map((u) => (
                <rect key={u.from} x={x(u.from)} y={y} width={Math.max(1, x(u.to) - x(u.from))} height={H} fill={r.color} fillOpacity={0.75} />
              ))}
              </g>
              {r.gap.len > 0 && (
                <g>
                  <path d={`M${x(r.gap.start)} ${y - 4} v-3 H${x(r.gap.start + r.gap.len)} v3`} fill="none" stroke={MUTED} strokeWidth={0.8} />
                  <text x={x(r.gap.start + r.gap.len / 2)} y={y - 9} textAnchor="middle" fontSize={10} fill={INK}>
                    {`longest ${r.gapWord}: ${span(r.gap.len * step)}`}
                  </text>
                </g>
              )}
              <text x={0} y={y + H / 2 + 3.5} fontSize={10.5} fill={INK}>{r.label}</text>
            </g>
          )
        })}
        {ticks.map((d) => (
          <g key={d}>
            <line x1={x((d * 24) / step)} x2={x((d * 24) / step)} y1={total - 16} y2={total - 11} stroke={LINE} />
            <text x={Math.min(W - 16, Math.max(LX + 14, x((d * 24) / step)))} y={total - 1} textAnchor="middle" fontSize={10} fill={MUTED}>{dayLabel(date, d)}</text>
          </g>
        ))}
        {hover !== null && <line x1={x(hover + 0.5)} x2={x(hover + 0.5)} y1={16} y2={total - 18} stroke={INK} strokeWidth={0.8} />}
      </svg>
      <p className="mt-1 min-h-[18px] text-[12px] tabular-nums text-muted">
        {hover === null
          ? `${hasRelay ? 'Sunlight' : 'Sunlight and Earth contact'} every hour for ${days} days from ${dayLabel(date, 0)}.`
          : `${dayLabel(date, Math.floor((ts.hours[hover] ?? 0) / 24))}, ${pad(Math.floor((ts.hours[hover] ?? 0) % 24))}:00 UTC · Sun ${sunLit[hover] ? 'up' : 'down'}${hasRelay ? '' : ` · Earth ${earthOn[hover] ? 'in view' : 'out of view'}`}`}
      </p>
    </div>
  )
}

// ---- 3. slope against the vehicle's limit ---------------------------------------------------------------------------
export function SlopeScale({ mean, limit, safe }: { mean: number; limit: number | null; safe: number }) {
  const max = Math.max(15, mean * 1.25, (limit ?? 0) * 1.6)
  const on = useArrived()
  const p = (v: number) => `${Math.min(100, (v / max) * 100)}%`
  const pm = (v: number) => (on ? p(v) : '0%')
  const over = limit !== null && mean > limit
  return (
    <div>
      <Title>Slope</Title>
      <div className="relative mt-3 pt-4">
        <div className="relative h-2.5 overflow-hidden rounded-full bg-[#e3e7e4]">
          {limit !== null && <div className="grow-x absolute inset-y-0 left-0 bg-[#a9bdb4]" style={{ width: p(limit) }} />}
        </div>
        {limit !== null && <span className="absolute top-[12px] h-[18px] w-[1.5px] bg-muted" style={{ left: p(limit) }} />}
        <span className="glide absolute top-0 -translate-x-1/2 text-[12px] font-medium tabular-nums text-ink" style={{ left: pm(mean) }}>{mean.toFixed(1)}°</span>
        <span className={`glide absolute top-[14px] h-[18px] w-[2px] -translate-x-1/2 ${over ? 'bg-[#ba1a1a]' : 'bg-ink'}`} style={{ left: pm(mean) }} />
      </div>
      <div className="relative mt-2 h-4 text-[12px] text-muted">
        <span className="absolute left-0">0°</span>
        {limit !== null && <span className="absolute -translate-x-1/2 whitespace-nowrap" style={{ left: p(limit) }}>limit {limit}°</span>}
      </div>
      <p className="mt-1 text-[12px] leading-snug text-muted">
        {limit === null ? 'This vehicle has no published slope limit. ' : over ? 'The average ground is steeper than the vehicle can take. ' : ''}
        {(safe * 100).toFixed(0)}% of the landing area is within the limit.
      </p>
    </div>
  )
}

// ---- 4. the ground around the site ----------------------------------------------------------------------------------
// grey-blue by height (low = dark, high = pale), the same family as the Landing Map
const TERRAIN_STOPS: [number, number, number][] = [[52, 72, 92], [104, 132, 152], [170, 188, 198], [232, 236, 238]]
const terrainColour = (t: number): [number, number, number] => {
  const x = Math.min(1, Math.max(0, t)) * (TERRAIN_STOPS.length - 1)
  const i = Math.min(TERRAIN_STOPS.length - 2, Math.floor(x))
  const f = x - i
  return [0, 1, 2].map((k) => TERRAIN_STOPS[i][k] + (TERRAIN_STOPS[i + 1][k] - TERRAIN_STOPS[i][k]) * f) as [number, number, number]
}

/** Height colours stretched over what this square actually contains, lit from the north-west with the relief exaggerated so gentle ground still reads. */
function hillshade(r: AreaRelief) {
  const { n, heights, half_m } = r
  const px = (2 * half_m) / (n - 1)
  const h = (i: number, j: number) => heights[Math.min(n - 1, Math.max(0, i))][Math.min(n - 1, Math.max(0, j))] ?? 0
  const all: number[] = []
  heights.forEach((row) => row.forEach((v) => { if (v !== null) all.push(v) }))
  all.sort((a, b) => a - b)
  const lo = all[Math.floor(all.length * 0.02)] ?? 0
  const hi = all[Math.floor(all.length * 0.98)] ?? 1
  // exaggeration: aim for a typical slope of about 25 degrees on screen
  let g2 = 0
  for (let i = 1; i < n - 1; i++) for (let j = 1; j < n - 1; j++) g2 += ((h(i, j + 1) - h(i, j - 1)) / (2 * px)) ** 2 + ((h(i - 1, j) - h(i + 1, j)) / (2 * px)) ** 2
  const rms = Math.sqrt(g2 / ((n - 2) * (n - 2))) || 1e-6
  const ex = Math.min(12, Math.max(1, 0.45 / rms))
  const img = new ImageData(n, n)
  const L = [-0.55, 0.55, 0.62] // light from the north-west
  const ln = Math.hypot(...L)
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      const dzdx = (ex * (h(i, j + 1) - h(i, j - 1))) / (2 * px)
      const dzdy = (ex * (h(i - 1, j) - h(i + 1, j))) / (2 * px) // north is up
      const nl = Math.hypot(dzdx, dzdy, 1)
      const shade = Math.max(0, (-dzdx * L[0] - dzdy * L[1] + L[2]) / (nl * ln))
      const t = hi > lo ? ((heights[i][j] ?? lo) - lo) / (hi - lo) : 0.5
      const [cr, cg, cb] = terrainColour(t)
      const k = 0.55 + 0.7 * shade
      const o = (i * n + j) * 4
      img.data[o] = Math.min(255, cr * k)
      img.data[o + 1] = Math.min(255, cg * k)
      img.data[o + 2] = Math.min(255, cb * k)
      img.data[o + 3] = 255
    }
  }
  return { img, lo, hi, ex }
}

export function AreaMap({ relief, radiusM = 500 }: { relief: AreaRelief; radiusM?: number }) {
  const cv = useRef<HTMLCanvasElement>(null)
  const range = useMemo(() => hillshade(relief), [relief])
  useEffect(() => {
    const c = cv.current
    if (!c) return
    c.width = relief.n
    c.height = relief.n
    c.getContext('2d')?.putImageData(range.img, 0, 0)
  }, [range, relief.n])
  const span2 = 2 * relief.half_m
  const scaleM = span2 >= 16000 ? 4000 : span2 >= 8000 ? 2000 : 1000
  return (
    <div data-viz="ground">
      <Title>The ground around it</Title>
      <div className="relative mt-3 aspect-square w-full max-w-[220px] overflow-hidden rounded-[6px]">
        <canvas ref={cv} className="absolute inset-0 h-full w-full" style={{ imageRendering: 'auto' }} />
        <svg viewBox="0 0 100 100" className="absolute inset-0 h-full w-full" fill="none">
          <circle cx="50" cy="50" r={(radiusM / span2) * 100} stroke="#ffffff" strokeOpacity="0.9" strokeWidth="0.6" strokeDasharray="2 1.6" />
          <circle cx="50" cy="50" r="1" fill="#ffffff" />
          <line x1="6" x2={6 + (scaleM / span2) * 100} y1="94" y2="94" stroke="#ffffff" strokeWidth="1" />
          <text x="6" y="90" fontSize="5.5" fill="#ffffff">{scaleM / 1000} km</text>
          <text x="94" y="9" fontSize="6" fill="#ffffff" textAnchor="end">N</text>
        </svg>
      </div>
      <p className="mt-2 max-w-[220px] text-[12px] leading-snug text-muted">
        {relief.data_source === 'ldem_470m' ? 'NASA LOLA at about 470 m per pixel, so only broad shapes show. ' : 'NASA LOLA at 40 m per pixel. '}
        Darker is lower, paler is higher ({Math.round(range.hi - range.lo)} m between them{range.ex > 1.5 ? `, relief stretched ${range.ex.toFixed(0)}×` : ''}). Dashed circle: the {radiusM} m landing area. The square is {Math.round((2 * relief.half_m) / 1000)} km wide.
      </p>
    </div>
  )
}

// ---- 5. the sky from the site ---------------------------------------------------------------------------------------
const SKY_TOP = '#1b2220'
const SKY_LOW = '#2c3834'
const RIDGE = '#9db3a8'
const DISC_EDGE = '#f3f1ea'
const STARS = (() => { // a fixed scatter (decoration only: the real star positions are not part of this analysis)
  let s = 7
  const rnd = () => { s = (s * 1664525 + 1013904223) % 4294967296; return s / 4294967296 }
  return Array.from({ length: 110 }, () => ({ x: rnd(), y: rnd(), r: 0.35 + rnd() * 0.8, o: 0.3 + rnd() * 0.6 }))
})()
const COMPASS = [{ a: 0, t: 'N' }, { a: 45, t: 'NE' }, { a: 90, t: 'E' }, { a: 135, t: 'SE' }, { a: 180, t: 'S' }, { a: 225, t: 'SW' }, { a: 270, t: 'W' }, { a: 315, t: 'NW' }]

/** What you would see turning around on the spot: the terrain skyline against the sky, and where the Sun and Earth travel over 30 days. */
export function SkyDiagram({ ts, horizon, hasRelay }: { ts: TimeSeries; horizon: { az_deg: number[]; el_deg: number[] }; hasRelay: boolean }) {
  const W = 460
  const H = 210
  const skyMax = Math.max(...horizon.el_deg)
  const bodiesMax = Math.max(...ts.sun_elevation_deg, ...(hasRelay ? [] : ts.earth_elevation_deg))
  const top = Math.min(90, Math.max(10, Math.ceil((Math.max(skyMax, bodiesMax) + 2) / 2) * 2))
  const bottom = -2
  const Y = (el: number) => ((top - el) / (top - bottom)) * H
  const norm = (az: number) => ((az % 360) + 360) % 360

  // the picture is cut where the sky is emptiest, so no path is torn in two at the edge
  const cut = useMemo(() => {
    const bins = new Array(72).fill(0)
    const mark = (az: number[], el: number[]) => az.forEach((a, i) => { if (el[i] >= bottom) bins[Math.floor(norm(a) / 5) % 72]++ })
    mark(ts.sun_azimuth_deg, ts.sun_elevation_deg)
    if (!hasRelay) mark(ts.earth_azimuth_deg, ts.earth_elevation_deg)
    let best = 0, bestAt = 0
    for (let s = 0; s < 72; s++) {
      let run = 0
      while (run < 72 && bins[(s + run) % 72] === 0) run++
      if (run > best) { best = run; bestAt = s }
    }
    return best >= 4 ? norm((bestAt + best / 2) * 5) : 0
  }, [ts, hasRelay])
  const X = (az: number) => (norm(az - cut) / 360) * W

  const skyline = (az: number) => {
    const i = Math.round((norm(az) / 360) * horizon.az_deg.length) % horizon.el_deg.length
    return horizon.el_deg[i]
  }
  const ground = useMemo(() => {
    const pts = horizon.az_deg.map((a, i) => ({ x: X(a), y: Y(horizon.el_deg[i]) })).sort((p, q) => p.x - q.x)
    const ridge = `M${pts.map((p) => `${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(' L')}`
    return { ridge, fill: `M0,${H} L${pts.map((p) => `${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(' L')} L${W},${H} Z` }
    // eslint-disable-next-line react-hooks/exhaustive-deps -- X and Y are derived from cut and top
  }, [horizon, top, cut])

  const track = (az: number[], el: number[], color: string, r: number) =>
    az.map((a, i) => {
      if (el[i] < bottom) return null
      const up = el[i] >= skyline(a)
      return <circle key={i} cx={X(a)} cy={Y(el[i])} r={up ? r : r * 0.7} fill={color} fillOpacity={up ? 0.95 : 0.22} />
    })
  const first = (az: number[], el: number[], color: string, rad: number, label: string) => {
    const x = X(az[0]), y = Y(el[0])
    const left = x > W - 70
    return (
      <g>
        <circle cx={x} cy={y} r={rad} fill={color} stroke={DISC_EDGE} strokeWidth={1.4} />
        <text x={left ? x - rad - 4 : x + rad + 4} y={y - rad - 2} textAnchor={left ? 'end' : 'start'} fontSize={10} fill={DISC_EDGE}>{label}</text>
      </g>
    )
  }
  const step = top <= 12 ? 2 : top <= 30 ? 5 : 10
  const grid = Array.from({ length: Math.floor(top / step) }, (_, i) => (i + 1) * step)

  return (
    <div data-viz="sky">
      <Title note="30 days, one dot per hour">The sky from the site</Title>
      <p className="mt-1 max-w-[460px] text-[12.5px] leading-snug text-muted">Standing on the spot and turning all the way round: left to right is the compass direction, up is the height above the horizon. The dark band is the terrain; the Sun and Earth can only be seen above it.</p>
      <svg viewBox={`0 0 ${W} ${H + 20}`} className="mt-2 w-full" role="img" aria-label="Terrain skyline with the paths of the Sun and Earth across the sky">
        <defs>
          <linearGradient id="sky-bg" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stopColor={SKY_TOP} /><stop offset="1" stopColor={SKY_LOW} /></linearGradient>
          <linearGradient id="sky-ground" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stopColor="#56635d" /><stop offset="1" stopColor="#343e3a" /></linearGradient>
          <clipPath id="sky-clip"><rect x={0} y={0} width={W} height={H} rx={4} /></clipPath>
        </defs>
        <g clipPath="url(#sky-clip)">
          <rect x={0} y={0} width={W} height={H} fill="url(#sky-bg)" />
          <g aria-hidden="true">{STARS.map((s, i) => <circle key={i} cx={s.x * W} cy={s.y * H * 0.9} r={s.r} fill="#f3f1ea" fillOpacity={s.o} />)}</g>
          {grid.map((e) => (
            <g key={e}>
              <line x1={0} x2={W} y1={Y(e)} y2={Y(e)} stroke="#f3f1ea" strokeOpacity={0.14} strokeWidth={0.6} strokeDasharray="2 4" />
              <text x={5} y={Y(e) - 3} fontSize={9} fill="#f3f1ea" fillOpacity={0.6}>{e}°</text>
            </g>
          ))}
          {track(ts.sun_azimuth_deg, ts.sun_elevation_deg, SUN, 2.1)}
          {hasRelay ? null : track(ts.earth_azimuth_deg, ts.earth_elevation_deg, EARTH, 2.1)}
          <path d={ground.fill} fill="url(#sky-ground)" />
          <path d={ground.ridge} fill="none" stroke={RIDGE} strokeWidth={0.9} />
          <line x1={0} x2={W} y1={Y(0)} y2={Y(0)} stroke="#f3f1ea" strokeOpacity={0.35} strokeWidth={0.6} strokeDasharray="3 3" />
          {first(ts.sun_azimuth_deg, ts.sun_elevation_deg, SUN, 6.5, 'Sun, day 1')}
          {!hasRelay && first(ts.earth_azimuth_deg, ts.earth_elevation_deg, EARTH, 4.5, 'Earth, day 1')}
        </g>
        {COMPASS.map((c) => {
          const x = X(c.a)
          if (x < 8 || x > W - 8) return null
          return (
            <g key={c.t}>
              <line x1={x} x2={x} y1={H} y2={H + 4} stroke={MUTED} strokeWidth={0.8} />
              <text x={x} y={H + 15} textAnchor="middle" fontSize={c.t.length === 1 ? 11 : 10} fontWeight={c.t.length === 1 ? 600 : 400} fill={c.t.length === 1 ? INK : MUTED}>{c.t}</text>
            </g>
          )
        })}
        <text x={W - 5} y={Y(0) + 11} textAnchor="end" fontSize={9} fill="#f3f1ea" fillOpacity={0.7}>horizon</text>
      </svg>
      <div className="mt-1 flex flex-wrap items-center gap-x-5 gap-y-1 text-[12px] text-muted">
        <span className="flex items-center gap-1.5"><span className="h-2 w-2 rounded-full" style={{ background: SUN }} />Sun</span>
        {!hasRelay && <span className="flex items-center gap-1.5"><span className="h-2 w-2 rounded-full" style={{ background: EARTH }} />Earth</span>}
        <span>faint dots are hidden behind the terrain · stars are only decoration</span>
      </div>
    </div>
  )
}

// ---- 6. Earth's wobble: libration ------------------------------------------------------------------------------------
const pc = (x: number) => Math.round(x * 100)
const LIB_TEXT = {
  enables: (r: LibrationReport) => `Without libration Earth would never rise above the horizon here. Because the Moon rocks, Earth climbs into view for ${pc(r.dte_fraction)}% of this month.`,
  adds: (r: LibrationReport) => `Libration lifts Earth into view for ${pc(r.dte_fraction - r.dte_fraction_no_libration)} more points of the month: ${pc(r.dte_fraction)}% with it, ${pc(r.dte_fraction_no_libration)}% on a Moon that did not rock.`,
  hides: (r: LibrationReport) => `On a Moon that did not rock, Earth would stay in view ${pc(r.dte_fraction_no_libration)}% of the time; the rocking swings it behind the horizon or the terrain for part of the month, leaving ${pc(r.dte_fraction)}%.`,
  little: (r: LibrationReport) => (r.dte_fraction > 0.95 ? 'This site faces Earth squarely, so the rocking never takes Earth out of view.' : r.dte_fraction < 0.05 ? 'This site is too far from the side that faces Earth: even at the extreme of the rocking Earth stays hidden.' : `Libration changes Earth's visibility little here (${pc(r.dte_fraction)}% against ${pc(r.dte_fraction_no_libration)}% on a Moon that did not rock).`),
} as const

/** Does the analysis account for libration? Yes: this shows the wobble, and what it changes at this very spot. */
export function LibrationPanel({ r }: { r: LibrationReport }) {
  const [W, H] = [300, 120]
  const n = r.hours.length
  const days = r.hours[n - 1] / 24
  const lo = Math.min(-4, ...r.earth_el_deg, ...r.earth_el_no_libration_deg)
  const hi = Math.max(4, ...r.earth_el_deg, ...r.earth_el_no_libration_deg, ...r.terrain_toward_earth_deg)
  const X = (h: number) => (h / 24 / days) * W
  const Y = (e: number) => ((hi - e) / (hi - lo)) * H
  const line = (v: number[]) => v.map((e, i) => `${i ? 'L' : 'M'}${X(r.hours[i]).toFixed(1)},${Y(e).toFixed(1)}`).join('')
  const terrain = `M0,${Y(0)} ${r.terrain_toward_earth_deg.map((e, i) => `L${X(r.hours[i]).toFixed(1)},${Y(e).toFixed(1)}`).join(' ')} L${W},${Y(0)} Z`
  // the wobble itself: where Earth sits around the Moon's mean Earth-facing point, one dot per hour
  const S = 120
  const R = 9.5
  const wx = (lon: number) => S / 2 + (lon / R) * (S / 2 - 8)
  const wy = (lat: number) => S / 2 - (lat / R) * (S / 2 - 8)
  const path = r.sub_earth_lon_deg.map((x, i) => `${i ? 'L' : 'M'}${wx(x).toFixed(1)},${wy(r.sub_earth_lat_deg[i]).toFixed(1)}`).join('')
  const swingLat = Math.max(Math.abs(r.swing.lat_deg[0]), r.swing.lat_deg[1])
  const swingLon = Math.max(Math.abs(r.swing.lon_deg[0]), r.swing.lon_deg[1])
  return (
    <div data-viz="libration">
      <Title note="30 days, from NASA SPICE">Earth's wobble: libration</Title>
      <p className="mt-1 max-w-[460px] text-[12.5px] leading-snug text-muted">The Moon keeps one face to Earth, but it rocks, so Earth never stands still in its sky. The analysis uses NASA's real orientation of the Moon, so this wobble is already inside every number above.</p>
      <div className="mt-3 grid grid-cols-[120px_minmax(0,1fr)] items-end gap-5">
        <div>
          <svg viewBox={`0 0 ${S} ${S}`} className="w-full" role="img" aria-label="Path of Earth around the Moon's mean Earth-facing point">
            <rect x={0} y={0} width={S} height={S} rx={4} fill="#f0f2f0" />
            <line x1={S / 2} x2={S / 2} y1={4} y2={S - 4} stroke={LINE} strokeWidth={0.7} />
            <line x1={4} x2={S - 4} y1={S / 2} y2={S / 2} stroke={LINE} strokeWidth={0.7} />
            <path d={path} fill="none" stroke={EARTH} strokeWidth={1.4} />
            <circle cx={wx(r.sub_earth_lon_deg[0])} cy={wy(r.sub_earth_lat_deg[0])} r={3.2} fill={EARTH} stroke="#fff" strokeWidth={1} />
            <text x={S / 2 + 3} y={11} fontSize={8} fill={MUTED}>N</text>
            <text x={S - 9} y={S / 2 - 3} fontSize={8} fill={MUTED}>E</text>
          </svg>
          <div className="mt-1 text-[11.5px] leading-snug text-muted">Earth's wobble: ±{swingLat.toFixed(1)}° north–south, ±{swingLon.toFixed(1)}° east–west</div>
        </div>
        <div>
          <svg viewBox={`0 0 ${W} ${H + 14}`} className="w-full" role="img" aria-label="Earth's height above the horizon at the site, with and without libration">
            <rect x={0} y={0} width={W} height={H} rx={3} fill="#f0f2f0" />
            <path d={terrain} fill="#4f5b57" opacity={0.28} />
            <line x1={0} x2={W} y1={Y(0)} y2={Y(0)} stroke={MUTED} strokeWidth={0.7} strokeDasharray="3 3" />
            <path d={line(r.earth_el_no_libration_deg)} fill="none" stroke={MUTED} strokeWidth={1.3} strokeDasharray="5 3" />
            <path d={line(r.earth_el_deg)} fill="none" stroke={EARTH} strokeWidth={1.9} />
            <text x={4} y={Y(0) - 3} fontSize={9} fill={MUTED}>horizon</text>
            <text x={0} y={H + 11} fontSize={10} fill={MUTED}>day 1</text>
            <text x={W} y={H + 11} fontSize={10} textAnchor="end" fill={MUTED}>day {Math.round(days)}</text>
          </svg>
          <div className="mt-1 flex flex-wrap gap-x-4 gap-y-0.5 text-[11.5px] text-muted">
            <span className="flex items-center gap-1.5"><span className="h-[2px] w-5" style={{ background: EARTH }} />Earth's height, real</span>
            <span className="flex items-center gap-1.5"><span className="h-0 w-5 border-t-2 border-dashed border-[#6e736e]" />on a Moon that did not rock</span>
            <span className="flex items-center gap-1.5"><span className="h-2.5 w-4 rounded-[2px] bg-[#4f5b57]/30" />terrain in the way</span>
          </div>
        </div>
      </div>
      <p className="mt-3 text-[14px] leading-snug text-ink">{LIB_TEXT[r.role](r)}</p>
      <p className="mt-1 text-[12px] leading-snug text-muted">Line of sight to Earth here: {pc(r.dte_fraction)}% of the month (a Moon without libration: {pc(r.dte_fraction_no_libration)}%). Method: {r.method}.</p>
    </div>
  )
}

/** How the PDF report finds the pictures drawn for a scenario. */
export const visualsKey = (s: Scenario) => `${s.lat}|${s.lon}|${s.date}|${s.vehicleId}`

// ---- the block the result page shows --------------------------------------------------------------------------------
export default function ResultVisuals({ scenario, result, side = false }: { scenario: Scenario; result: ScenarioResult; side?: boolean }) {
  const lat = parseLat(scenario.lat)
  const lon = parseLon(scenario.lon)
  const [ts, setTs] = useState<TimeSeries | null>(null)
  const [horizon, setHorizon] = useState<{ az_deg: number[]; el_deg: number[] } | null>(null)
  const [relief, setRelief] = useState<AreaRelief | null>(null)
  const [lib, setLib] = useState<LibrationReport | null>(null)
  const [limit, setLimit] = useState<number | null | undefined>(undefined)
  const [failed, setFailed] = useState(false)
  const a = result.assessment
  const relay = a.checks.comms?.mode === 'relay'

  useEffect(() => {
    if (lat === null || lon === null) return
    let live = true
    setTs(null); setHorizon(null); setRelief(null); setLib(null); setFailed(false)
    const key = `${lat}|${lon}`
    seriesOf(`${key}|${scenario.date}`).then((v) => live && setTs(v)).catch(() => live && setFailed(true))
    horizonOf(key).then((v) => live && setHorizon(v)).catch(() => undefined)
    reliefOf(key).then((v) => live && setRelief(v)).catch(() => undefined)
    libOf(`${key}|${scenario.date}`).then((v) => live && setLib(v)).catch(() => undefined)
    return () => { live = false }
  }, [lat, lon, scenario.date])

  useEffect(() => {
    let live = true
    vehiclesOf('all').then((vs) => {
      const v = vs.find((x) => x.id === scenario.vehicleId)
      if (live) setLimit(v?.max_slope_deg.value ?? null)
    }).catch(() => live && setLimit(null))
    return () => { live = false }
  }, [scenario.vehicleId])

  if (failed) return null
  return (
    <div data-report-visuals={visualsKey(scenario)} className={`space-y-9 py-6 ${side ? '' : 'border-t border-line'}`}>
      <ScoreScale score={a.vehicle_index} uncapped={a.limiting_factor ? a.weighted_index : null} />
      {ts ? <MissionStrips ts={ts} date={scenario.date} hasRelay={relay} /> : <div aria-busy="true"><div className="skeleton h-3 w-40" /><div className="skeleton mt-4 h-[22px] w-full" /><div className="skeleton mt-3 h-[22px] w-full" /></div>}
      <div className="grid grid-cols-1 gap-x-8 gap-y-9 min-[560px]:grid-cols-[minmax(0,1fr)_minmax(0,220px)]">
        {limit !== undefined ? <SlopeScale mean={a.site_metrics.mean_slope_deg} limit={limit} safe={a.site_metrics.safe_area_fraction} /> : <span />}
        {relief && <AreaMap relief={relief} />}
      </div>
      {ts && horizon && <SkyDiagram ts={ts} horizon={horizon} hasRelay={relay} />}
      {lib && <LibrationPanel r={lib} />}
    </div>
  )
}
