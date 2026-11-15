import { useEffect, useMemo, useRef, useState } from 'react'
import { getPlanner, getSites, getVehicles, type PlannerDay, type PlannerResult, type VehicleProfile } from '../api'
import { parseLat, parseLon } from '../lib/input'
import type { Scenario } from '../types'
import CoordinateFields from './wizard/CoordinateFields'
import { inputCls } from './wizard/ui'

const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
const SPANS = [60, 90, 120, 180]
const FALLBACK = [{ name: 'Malapert', lat: -85.987, lon: 2.08 }, { name: 'Shackleton', lat: -89.54, lon: 0 }, { name: 'Haworth', lat: -87.45, lon: -4.5 }, { name: 'Faustini', lat: -87.3, lon: 84 }, { name: 'Nobile', lat: -85.2, lon: 53.5 }]
const FAIL: Record<string, string> = { landing: 'unsafe touchdown', cold: 'freezing in a night', power: 'running out of power', comms: 'too little contact with Earth' }
const CONF = { high: 'high confidence', medium: 'medium confidence', low: 'low confidence: some vehicle data is not published' } as const

const addDays = (iso: string, n: number) => { const d = new Date(`${iso}T00:00:00Z`); d.setUTCDate(d.getUTCDate() + n); return d.toISOString().slice(0, 10) }
const pretty = (iso: string) => { const d = new Date(`${iso}T00:00:00Z`); return `${d.getUTCDate()} ${MONTHS[d.getUTCMonth()]} ${d.getUTCFullYear()}` }
const short = (iso: string) => { const d = new Date(`${iso}T00:00:00Z`); return `${d.getUTCDate()} ${MONTHS[d.getUTCMonth()]}` }
const pct = (p: number) => `${Math.round(p * 100)}%`
const today = () => new Date().toISOString().slice(0, 10)

/** One shade per call, darker = better: the same four calls the analysis page ends with. */
const KIND = {
  go: { color: '#3a6857', label: 'Go' },
  conditional: { color: '#7fa593', label: 'Possible, with risks' },
  weak: { color: '#c3d4cb', label: 'Not recommended' },
  stop: { color: '#e6e9e7', label: 'Do not proceed' },
} as const

export default function PlannerPage({ onAnalyze }: { onAnalyze: (s: Scenario) => void }) {
  const [vehicles, setVehicles] = useState<VehicleProfile[]>([])
  const [sites, setSites] = useState(FALLBACK)
  const [vehicleId, setVehicleId] = useState('')
  const [place, setPlace] = useState('Malapert')
  const [lat, setLat] = useState('-85.987')
  const [lon, setLon] = useState('2.08')
  const [start, setStart] = useState(today())
  const [span, setSpan] = useState(120)
  const [res, setRes] = useState<PlannerResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [err, setErr] = useState<string | null>(null)
  const [pick, setPick] = useState<number | null>(null)
  const [hover, setHover] = useState<number | null>(null)
  const seq = useRef(0)

  useEffect(() => {
    getVehicles().then((v) => {
      setVehicles(v)
      setVehicleId((id) => id || (v.find((x) => x.id === 'chandrayaan-3-vikram-pragyan') ?? v[0])?.id || '')
    }).catch(() => setErr('The analysis server is not answering. Double-click start.bat in the project folder to start it.'))
    getSites().then((s) => s.length && setSites(s.map((x) => ({ ...x, name: x.name.charAt(0).toUpperCase() + x.name.slice(1) })))).catch(() => undefined)
  }, [])

  const site = sites.find((s) => s.name === place)
  const la = site ? site.lat : parseLat(lat)
  const loRaw = site ? site.lon : parseLon(lon)
  const lo = loRaw === null ? null : loRaw < 0 ? loRaw + 360 : loRaw
  const okDate = /^\d{4}-\d\d-\d\d$/.test(start)

  // every control re-plans, a moment after the last change (the server needs about a second)
  useEffect(() => {
    if (!vehicleId || la === null || lo === null || !okDate) return
    const mine = ++seq.current
    setLoading(true); setErr(null)
    const t = setTimeout(() => {
      getPlanner(vehicleId, la, lo, start, span)
        .then((r) => { if (seq.current === mine) { setRes(r); setPick(null); setLoading(false) } })
        .catch((e) => { if (seq.current === mine) { setErr(e instanceof Error ? e.message : 'Could not plan'); setLoading(false) } })
    }, 350)
    return () => clearTimeout(t)
  }, [vehicleId, la, lo, start, span, okDate])

  const mine = vehicles.filter((v) => v.custom)
  const others = vehicles.filter((v) => !v.custom).sort((a, b) => a.name.localeCompare(b.name))

  // the calendar: one row per month, one cell per start date
  const rows = useMemo(() => {
    if (!res) return []
    const out = new Map<string, { label: string; cells: (number | null)[] }>()
    res.days.forEach((d) => {
      const iso = addDays(res.start_date, d.day)
      const key = iso.slice(0, 7)
      if (!out.has(key)) out.set(key, { label: `${MONTHS[Number(key.slice(5)) - 1]} ${key.slice(0, 4)}`, cells: new Array(31).fill(null) })
      out.get(key)!.cells[Number(iso.slice(8)) - 1] = d.day
    })
    return [...out.values()]
  }, [res])

  const inBest = useMemo(() => new Set(res ? res.windows.flatMap((w) => Array.from({ length: w.to - w.from + 1 }, (_, i) => w.from + i)) : []), [res])
  const level = (k: string) => ['stop', 'weak', 'conditional', 'go'].indexOf(k)
  const bestKind = res ? res.days.reduce((k, d) => (level(d.verdict) > level(k) ? d.verdict : k), 'stop' as PlannerDay['verdict']) : 'stop'
  const flat = !!res && res.days.every((d) => d.verdict === res.days[0].verdict)
  const shownDay = hover ?? pick ?? (res && res.windows[0] ? res.windows[0].at : null)
  const d = res && shownDay !== null ? res.days[shownDay] : null
  const scenario = (day: number): Scenario => ({ date: addDays(res!.start_date, day), vehicleId, lat: String(la), lon: String(loRaw) })

  return (
    <main className="h-full flex-1 overflow-y-auto">
      <div className="mx-auto max-w-[1180px] px-10 pb-24 pt-10">
        <h1 className="text-[34px] font-medium text-brand">Mission Planner</h1>
        <p className="mt-1 max-w-[680px] text-muted">When should it go? Every possible start date gets the full analysis (score, 400 simulated missions on the real Sun, Earth and terrain, and the call), painted on a calendar. Open a date and the analysis page shows exactly what the calendar showed.</p>

        <div className="mt-7 grid gap-x-6 gap-y-4 md:grid-cols-[minmax(0,1.3fr)_minmax(0,1fr)_minmax(0,0.8fr)_minmax(0,0.7fr)]">
          <label className="block"><span className="text-[13px] uppercase tracking-wide text-muted">Vehicle</span>
            <select value={vehicleId} onChange={(e) => setVehicleId(e.target.value)} className={`${inputCls} mt-1.5`}>
              {mine.length > 0 && <optgroup label="My vehicles">{mine.map((v) => <option key={v.id} value={v.id}>{v.name}</option>)}</optgroup>}
              <optgroup label="Lunar missions">{others.map((v) => <option key={v.id} value={v.id}>{v.name}</option>)}</optgroup>
            </select></label>
          <label className="block"><span className="text-[13px] uppercase tracking-wide text-muted">Place</span>
            <select value={place} onChange={(e) => setPlace(e.target.value)} className={`${inputCls} mt-1.5`}>
              {sites.map((s) => <option key={s.name}>{s.name}</option>)}<option value="">Coordinates…</option>
            </select></label>
          <label className="block"><span className="text-[13px] uppercase tracking-wide text-muted">First start date</span>
            <input type="date" value={start} onChange={(e) => setStart(e.target.value)} className={`${inputCls} mt-1.5`} /></label>
          <label className="block"><span className="text-[13px] uppercase tracking-wide text-muted">Look ahead</span>
            <select value={span} onChange={(e) => setSpan(Number(e.target.value))} className={`${inputCls} mt-1.5`}>{SPANS.map((n) => <option key={n} value={n}>{n} days</option>)}</select></label>
        </div>
        {!site && <div className="mt-4 max-w-[560px]"><CoordinateFields lat={lat} lon={lon} onLat={setLat} onLon={setLon} /></div>}

        {err && <p className="mt-8 text-amber-700" role="alert">{err}</p>}

        <div className={`mt-9 transition-opacity duration-300 ${loading && res ? 'opacity-50' : 'opacity-100'}`} aria-busy={loading}>
          {!res && loading && <div><div className="skeleton h-9 w-2/3" /><div className="skeleton mt-6 h-[220px] w-full" /></div>}
          {res && (
            <>
              <section aria-label="Best start dates" className="rise-in">
                {bestKind === 'stop' ? (
                  <h2 className="text-[24px] font-medium leading-snug text-ink">No start date in these {res.span_days} days works for {res.vehicle.name} here: every one ends in "Do not proceed".</h2>
                ) : flat ? (
                  <h2 className="text-[24px] font-medium leading-snug text-ink">The start date hardly matters here: every day gets the same call, "{KIND[bestKind].label}". The place and the vehicle decide.</h2>
                ) : (
                  <h2 className="text-[24px] font-medium leading-snug text-ink">Best start: <span className="text-brand">{short(addDays(res.start_date, res.windows[0].from))} – {short(addDays(res.start_date, res.windows[0].to))}</span>, call "{KIND[bestKind].label}", score up to {Math.round(res.windows[0].score)}.</h2>
                )}
                {!flat && bestKind !== 'stop' && res.windows.length > 1 && (
                  <p className="mt-2 text-[15px] text-muted">Other good windows: {res.windows.slice(1).map((w) => `${short(addDays(res.start_date, w.from))} – ${short(addDays(res.start_date, w.to))} (score ${Math.round(w.score)})`).join(' · ')}. The Moon repeats its day and night about every 29.5 days, so good dates come in waves.</p>
                )}
              </section>

              <section className="mt-8" aria-label="Calendar of start dates">
                <div className="overflow-x-auto pb-2">
                  <div className="min-w-[760px]">
                    <div className="grid grid-cols-[84px_repeat(31,minmax(0,1fr))] gap-[3px] text-[11px] text-muted">
                      <span />
                      {Array.from({ length: 31 }, (_, i) => <span key={i} className="text-center tabular-nums">{(i + 1) % 2 === 1 ? i + 1 : ''}</span>)}
                    </div>
                    {rows.map((r) => (
                      <div key={r.label} className="mt-[3px] grid grid-cols-[84px_repeat(31,minmax(0,1fr))] gap-[3px]">
                        <span className="self-center text-[13px] text-ink">{r.label}</span>
                        {r.cells.map((day, i) => day === null ? <span key={i} className="h-7 rounded-[3px] border border-dashed border-line/60" /> : (
                          <button key={i} onMouseEnter={() => setHover(day)} onMouseLeave={() => setHover(null)} onFocus={() => setHover(day)} onBlur={() => setHover(null)} onClick={() => setPick(day)}
                            aria-label={`${pretty(addDays(res.start_date, day))}: ${res.days[day].title}, score ${Math.round(res.days[day].score)}, ${pct(res.days[day].p)} chance`} aria-pressed={pick === day}
                            className={`h-7 rounded-[3px] transition-transform hover:scale-110 ${pick === day ? 'ring-2 ring-ink' : inBest.has(day) && !flat ? 'ring-1 ring-ink/55' : ''}`}
                            style={{ background: KIND[res.days[day].verdict].color }} />
                        ))}
                      </div>
                    ))}
                  </div>
                </div>
                <div className="mt-3 flex flex-wrap items-center gap-x-5 gap-y-2 text-[12.5px] text-muted">
                  {(['go', 'conditional', 'weak', 'stop'] as const).map((k) => <span key={k} className="flex items-center gap-1.5"><span className="h-3 w-3 rounded-[3px]" style={{ background: KIND[k].color }} />{KIND[k].label}</span>)}
                  {!flat && <span className="flex items-center gap-1.5"><span className="h-3 w-3 rounded-[3px] ring-1 ring-ink/55" style={{ background: KIND[bestKind].color }} />best window</span>}
                  <span>every day is the full analysis for a mission starting that day (30 days simulated)</span>
                </div>
              </section>

              {d && shownDay !== null && (
                <section className="mt-7 grid gap-6 border-t border-line pt-6 md:grid-cols-[minmax(0,1fr)_auto] md:items-center" aria-label="Chosen start date" aria-live="polite">
                  <div>
                    <div className="text-[13px] uppercase tracking-wide text-muted">{hover !== null ? 'Start date' : pick !== null ? 'Chosen start date' : 'Best start date'}</div>
                    <div className="mt-1 flex flex-wrap items-baseline gap-x-4 gap-y-1">
                      <span className="text-[26px] font-medium text-brand">{pretty(addDays(res.start_date, shownDay))}</span>
                      <span className="text-[22px] font-medium text-ink">{d.title}</span>
                    </div>
                    <div className="mt-1 flex flex-wrap items-baseline gap-x-5 gap-y-1 text-[15px] text-ink">
                      <span>Score <b className="tabular-nums">{Math.round(d.score)}</b>/100</span>
                      <span><b className="tabular-nums">{pct(d.p)}</b> chance of success</span>
                      <span className="text-muted">typically works {d.days_p50.toFixed(0)} days</span>
                    </div>
                    <p className="mt-1.5 max-w-[640px] text-[14px] leading-snug text-muted">
                      In those 30 days the Sun is up {pct(d.sun)} of the time and Earth is in view {pct(d.earth)}.
                      {d.top_failure && d.p < 0.95 ? ` The main way it fails: ${FAIL[d.top_failure] ?? d.top_failure}.` : ''} ({CONF[d.confidence]}) These are the same numbers the analysis page shows for this date.
                    </p>
                  </div>
                  <button onClick={() => onAnalyze(scenario(shownDay))} className="grain rounded-[8px] bg-brand px-5 py-2.5 text-[16px] text-white transition hover:opacity-90">Analyze this date</button>
                </section>
              )}
            </>
          )}
        </div>
      </div>
    </main>
  )
}
