import { useEffect, useRef, useState } from 'react'
import { API_BASE, getSubSolarSeries, type SubSolarSeries } from '../api'
import { todayIso } from '../lib/input'
import Viewer from './moon3d/Viewer'
import type { Point } from './moon3d/geo'
import TimelineBar from './TimelineBar'
import DateField from './wizard/DateField'
import TimeField from './wizard/TimeField'

type Mode = 'natural' | 'relief' | 'elevation'
const MODES: { id: Mode; label: string }[] = [
  { id: 'natural', label: 'Natural colour' },
  { id: 'relief', label: 'Relief' },
  { id: 'elevation', label: 'Height colours' },
]

const pad = (n: number) => String(n).padStart(2, '0')
const fmt = (lat: number, lon: number) => `${Math.abs(lat).toFixed(3)}° ${lat < 0 ? 'S' : 'N'}, ${Math.abs(lon).toFixed(3)}° ${lon < 0 ? 'W' : 'E'}`
const WINDOW_H = 744 // a 31-day bar, one point per hour
const SPEEDS = [{ label: '6 h/s', hoursPerSecond: 6 }, { label: '1 d/s', hoursPerSecond: 24 }, { label: '3 d/s', hoursPerSecond: 72 }]
const shiftDay = (iso: string, d: number) => {
  const t = new Date(`${iso}T00:00:00Z`)
  t.setUTCDate(t.getUTCDate() + d)
  return t.toISOString().slice(0, 10)
}

/** The Moon in 3D from real LOLA heights, lit by the Sun where NASA SPICE says it is on the chosen date. */
export default function MoonTimelinePage() {
  const [date, setDate] = useState(todayIso())
  const [hour, setHour] = useState(new Date().getUTCHours())
  const [mode, setMode] = useState<Mode>('natural')
  const [wire, setWire] = useState(false)
  const [grid, setGrid] = useState(false)
  const [windowStart, setWindowStart] = useState(() => shiftDay(todayIso(), -15))
  const [series, setSeries] = useState<SubSolarSeries | null>(null)
  const [sunError, setSunError] = useState('')
  const [follow, setFollow] = useState(true)
  const [look, setLook] = useState(0) // bumps to point the camera at the Sun again
  const [playing, setPlaying] = useState(false)
  const [speed, setSpeed] = useState(24)
  const last = useRef<{ lat: number; lon: number } | null>(null) // keeps the light steady while a new month loads
  const [picked, setPicked] = useState<(Point & { slope?: number }) | null>(null)
  const [error, setError] = useState('')
  const asked = useRef(0)

  const hoursFromStart = (d: string, h: number) => Math.round((Date.parse(`${d}T00:00:00Z`) - Date.parse(`${windowStart}T00:00:00Z`)) / 3_600_000) + h
  const pos = hoursFromStart(date, hour)

  // the Sun's position for the whole bar in one request
  useEffect(() => {
    const id = ++asked.current
    setSunError('')
    getSubSolarSeries(`${windowStart}T00:00:00`, WINDOW_H)
      .then((r) => id === asked.current && setSeries(r))
      .catch((e: Error) => id === asked.current && setSunError(e.message))
  }, [windowStart])

  // a date chosen in the fields that lies outside the bar: bring the bar to it
  useEffect(() => {
    if (pos < 0 || pos > WINDOW_H) setWindowStart(shiftDay(date, -15))
  }, [pos, date])

  const move = (h: number) => {
    const t = new Date(Date.parse(`${windowStart}T00:00:00Z`) + Math.round(h) * 3_600_000)
    setDate(t.toISOString().slice(0, 10))
    setHour(t.getUTCHours())
  }
  const shiftWindow = (dir: -1 | 1) => {
    setSeries(null)
    setWindowStart(shiftDay(windowStart, dir * 30))
    setDate(shiftDay(date, dir * 30))
  }

  // playback: advance by speed x elapsed time, stop at the end of the bar
  const posRef = useRef(pos)
  posRef.current = pos
  useEffect(() => {
    if (!playing) return
    let acc = posRef.current
    const timer = setInterval(() => {
      acc += speed / 10
      if (acc >= WINDOW_H) {
        move(WINDOW_H)
        setPlaying(false)
        return
      }
      move(acc)
    }, 100)
    return () => clearInterval(timer)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [playing, speed, windowStart])

  const i = Math.min(WINDOW_H, Math.max(0, pos))
  const here = series && i < series.lat.length ? { lat: series.lat[i], lon: series.lon[i] } : null
  if (here) last.current = here
  const sun = here ?? last.current

  const pick = (p: Point) => {
    setPicked(p)
    fetch(`${API_BASE}/moon-height?lat=${p.lat}&lon=${p.lon}`)
      .then((r) => (r.ok ? r.json() : Promise.reject()))
      .then((d: { height: number; slope: number }) => setPicked((cur) => (cur && cur.lat === p.lat && cur.lon === p.lon ? { ...cur, height: d.height, slope: d.slope } : cur)))
      .catch(() => undefined)
  }

  const subLon0 = sun ? (sun.lon > 180 ? sun.lon - 360 : sun.lon) : undefined
  const sunKey = follow ? i : 0 // while following, every step of the bar turns the camera to the lit point
  const focus = sun && subLon0 !== undefined ? { lat: sun.lat, lon: subLon0, n: sunKey + look * 100000 } : undefined
  const subLon = sun ? (sun.lon > 180 ? sun.lon - 360 : sun.lon) : undefined
  const toggle = 'border-b pb-0.5 transition-colors'

  return (
    <main className="flex-1 overflow-y-auto">
      <div className="mx-auto max-w-[1280px] px-10 pb-16 pt-10">
        <h1 className="text-[34px] font-medium text-brand">Moon Timeline</h1>
        <p className="mt-1 max-w-[720px] text-muted">
          The Moon as real terrain, lit by the Sun where it stands on the date you pick. Drag to turn it, scroll to come closer, right-drag to tilt, click to read a point.
        </p>

        <div className="mt-8 flex flex-wrap items-start gap-x-12 gap-y-4">
          <div className="w-[400px] max-w-full">
            <DateField value={date} onChange={setDate} solid />
          </div>
          <div className="w-[190px]">
            <TimeField value={hour} onChange={setHour} />
          </div>
          <button onClick={() => { setPlaying(false); setDate(todayIso()); setHour(new Date().getUTCHours()) }} className="pt-3.5 text-[17px] text-ink hover:text-brand">Now</button>
          <div className="flex gap-4 pt-3.5 text-[17px] text-ink">
            <button onClick={() => setDate(shiftDay(date, -1))} className="hover:text-brand">← Day</button>
            <button onClick={() => setDate(shiftDay(date, 1))} className="hover:text-brand">Day →</button>
          </div>
        </div>

        <div className="mt-12 border-t border-line pt-10">
          <TimelineBar
            windowStart={windowStart} hours={WINDOW_H} pos={Math.min(WINDOW_H, Math.max(0, pos))} earthLit={series?.earth_lit ?? null}
            playing={playing} speed={speed} speeds={SPEEDS}
            onPos={(h) => { setPlaying(false); move(h) }} onPlay={() => setPlaying((p) => !p)} onSpeed={setSpeed} onShift={shiftWindow}
          />
        </div>

        <div className="mt-12 flex flex-wrap items-center gap-x-7 gap-y-3 text-[16px]">
          {MODES.map((m) => (
            <button key={m.id} onClick={() => setMode(m.id)} aria-pressed={mode === m.id}
              className={`${toggle} ${mode === m.id ? 'border-brand text-brand' : 'border-transparent text-ink hover:text-brand'}`}>{m.label}</button>
          ))}
          <span className="h-4 border-l border-line" />
          <button onClick={() => setWire((w) => !w)} aria-pressed={wire} className={`${toggle} ${wire ? 'border-brand text-brand' : 'border-transparent text-ink hover:text-brand'}`}>Mesh</button>
          <button onClick={() => setFollow((f) => !f)} aria-pressed={follow} className={`${toggle} ${follow ? 'border-brand text-brand' : 'border-transparent text-ink hover:text-brand'}`}>Follow the Sun</button>
          <button onClick={() => setLook((n) => n + 1)} className={`${toggle} border-transparent text-ink hover:text-brand`}>Face the Sun</button>
          <button onClick={() => setGrid((g) => !g)} aria-pressed={grid} className={`${toggle} ${grid ? 'border-brand text-brand' : 'border-transparent text-ink hover:text-brand'}`}>Grid</button>
        </div>

        <div className="mt-5 h-[560px] overflow-hidden rounded-[14px] border border-line bg-[#05080c]">
          {sun ? (
            // the camera starts over the point where the Sun is overhead, so the first view is the lit side
            <Viewer initialLat={sun.lat} initialLon={subLon} initialAlt={4800} focus={focus} mode={mode} wire={wire} grid={grid} sun={subLon} sunLat={sun.lat} showHint={false} onPick={pick} onError={setError} />
          ) : (
            <div className="flex h-full items-center justify-center text-[14px] text-white/60">{sunError || 'Reading the Sun position…'}</div>
          )}
        </div>

        <div className="mt-3 flex flex-wrap items-center gap-x-6 gap-y-1 text-[13px] text-muted">
          <span>
            {sun ? `Sun overhead at ${fmt(sun.lat, subLon!)} on ${date} ${pad(hour)}:00 UTC (NASA SPICE DE440)` : sunError ? `Sun position unavailable: ${sunError}` : 'Reading the Sun position…'}
          </span>
          {error && <span className="text-amber-700">{error}</span>}
        </div>

        <div className="mt-12 border-t border-line pt-8">
          {picked ? (
            <div className="grid grid-cols-1 gap-6 sm:grid-cols-3">
              <div>
                <div className="text-[12px] uppercase tracking-wide text-muted">Point</div>
                <div className="mt-1 text-[24px] font-bold tabular-nums text-strong">{fmt(picked.lat, picked.lon)}</div>
              </div>
              <div>
                <div className="text-[12px] uppercase tracking-wide text-muted">Height (LOLA)</div>
                <div className="mt-1 text-[24px] font-bold tabular-nums text-strong">{picked.height !== undefined ? `${Math.round(picked.height).toLocaleString('en-US')} m` : '…'}</div>
              </div>
              <div>
                <div className="text-[12px] uppercase tracking-wide text-muted">Slope (regional)</div>
                <div className="mt-1 text-[24px] font-bold tabular-nums text-strong">{picked.slope !== undefined ? `${picked.slope.toFixed(1)}°` : '…'}</div>
              </div>
            </div>
          ) : (
            <p className="text-[17px] text-muted">Click the surface to read its coordinates, height and slope.</p>
          )}
          <p className="mt-4 max-w-[760px] text-[12px] leading-relaxed text-muted">
            Heights: NASA LRO LOLA, 64 pixels per degree (about 470 m at the equator); closer views do not add detail. Colour: NASA SVS 4720 LROC mosaic, for looks only. The slope is averaged over about 2 km, and the lighting direction comes from the Sun's position, not from a shadow calculation.
          </p>
        </div>
      </div>
    </main>
  )
}
