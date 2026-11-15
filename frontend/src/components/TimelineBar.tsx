import { ChevronLeft, ChevronRight, Pause, Play } from 'lucide-react'
import { useRef } from 'react'

const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
const DAY_MS = 86_400_000

type Props = {
  windowStart: string // ISO date (UTC midnight) where the bar begins
  hours: number // length of the bar in hours
  pos: number // current position, hours from windowStart
  earthLit: number[] | null // fraction of the Moon lit as seen from Earth, hourly; drawn as the soft shape behind the track
  playing: boolean
  speed: number // hours of Moon time per second of playback
  speeds: { label: string; hoursPerSecond: number }[]
  onPos: (hours: number) => void
  onPlay: () => void
  onSpeed: (hoursPerSecond: number) => void
  onShift: (direction: -1 | 1) => void
}

/** A horizontal time scrubber: drag along a month of days, the date follows. Moon phase shows as a quiet shape behind the track. */
export default function TimelineBar({ windowStart, hours, pos, earthLit, playing, speed, speeds, onPos, onPlay, onSpeed, onShift }: Props) {
  const track = useRef<HTMLDivElement>(null)
  const dragging = useRef(false)
  const t0 = Date.parse(`${windowStart}T00:00:00Z`)
  const pct = (h: number) => `${(h / hours) * 100}%`

  const set = (clientX: number) => {
    const r = track.current!.getBoundingClientRect()
    onPos(Math.round(Math.min(1, Math.max(0, (clientX - r.left) / r.width)) * hours))
  }

  // day ticks; the label shows the day of month, and the month name on the 1st and at the left edge
  const days = Math.floor(hours / 24)
  const ticks = Array.from({ length: days + 1 }, (_, i) => {
    const d = new Date(t0 + i * DAY_MS)
    return { i, day: d.getUTCDate(), month: MONTHS[d.getUTCMonth()], first: d.getUTCDate() === 1 }
  })

  const labelled = ticks.filter((t) => t.first || (t.i % 5 === 0 && !ticks.some((o) => o.first && Math.abs(o.i - t.i) < 3)))

  // phase landmarks: the lowest and highest points of the lit fraction, where it turns around within the window
  const marks: { h: number; label: string }[] = []
  if (earthLit) {
    for (let i = 24; i < earthLit.length - 24; i++) {
      const w = earthLit.slice(i - 24, i + 25)
      const label = earthLit[i] === Math.max(...w) && earthLit[i] > 0.97 ? 'Full' : earthLit[i] === Math.min(...w) && earthLit[i] < 0.03 ? 'New' : null
      const prev = marks.filter((m) => m.label === label).pop()
      if (label && (!prev || i - prev.h > 96)) marks.push({ h: i, label }) // one mark per phase, not one per tied hour
    }
  }
  const nowDate = new Date(t0 + pos * 3_600_000)
  const label = `${nowDate.getUTCDate()} ${MONTHS[nowDate.getUTCMonth()]} ${nowDate.getUTCFullYear()}, ${String(nowDate.getUTCHours()).padStart(2, '0')}:00 UTC`

  return (
    <div>
      <div className="flex flex-wrap items-end justify-between gap-x-10 gap-y-4">
        <div className="flex items-center gap-5">
          <button onClick={onPlay} aria-label={playing ? 'Pause' : 'Play'} className="flex w-[72px] items-center gap-2 text-[17px] text-ink transition-colors hover:text-brand">
            {playing ? <Pause size={16} className="text-brand" /> : <Play size={16} className="text-brand" />}
            {playing ? 'Pause' : 'Play'}
          </button>
          <div className="text-[30px] font-medium leading-none tabular-nums text-strong" aria-live="off">{label}</div>
        </div>
        <div className="flex items-center gap-6 pb-0.5 text-[15px] text-muted">
          {speeds.map((s) => (
            <button
              key={s.label}
              onClick={() => onSpeed(s.hoursPerSecond)}
              aria-pressed={speed === s.hoursPerSecond}
              className={speed === s.hoursPerSecond ? 'text-brand underline underline-offset-[6px]' : 'hover:text-brand'}
            >
              {s.label}
            </button>
          ))}
          <span className="h-4 border-l border-line" />
          <button onClick={() => onShift(-1)} aria-label="Earlier month" className="hover:text-brand"><ChevronLeft size={20} /></button>
          <button onClick={() => onShift(1)} aria-label="Later month" className="hover:text-brand"><ChevronRight size={20} /></button>
        </div>
      </div>

      <div
        ref={track}
        role="slider"
        tabIndex={0}
        aria-label="Date on the timeline"
        aria-valuemin={0}
        aria-valuemax={hours}
        aria-valuenow={Math.round(pos)}
        aria-valuetext={label}
        className="relative mt-10 h-[84px] cursor-pointer touch-none select-none outline-none focus-visible:ring-2 focus-visible:ring-brand/30"
        onPointerDown={(e) => { dragging.current = true; e.currentTarget.setPointerCapture(e.pointerId); set(e.clientX) }}
        onPointerMove={(e) => dragging.current && set(e.clientX)}
        onPointerUp={() => { dragging.current = false }}
        onPointerCancel={() => { dragging.current = false }}
        onKeyDown={(e) => {
          const step = e.shiftKey ? 24 : 1
          if (e.key === 'ArrowRight') { e.preventDefault(); onPos(Math.min(hours, pos + step)) }
          if (e.key === 'ArrowLeft') { e.preventDefault(); onPos(Math.max(0, pos - step)) }
        }}
      >
        {/* phase shape and the one baseline: no frame around the track */}
        <div className="absolute inset-x-0 top-0 h-[44px]">
          {marks.map((m) => (
            <span key={m.label + m.h} className="absolute bottom-3 -translate-x-1/2 text-[12px] text-muted" style={{ left: pct(m.h) }}>{m.label}</span>
          ))}
          <div className="absolute inset-x-0 bottom-0 border-b border-ink/25" />
          {labelled.map((t) => (
            <span key={t.i} className="absolute bottom-0 w-px bg-ink/25" style={{ left: pct(t.i * 24), height: 7 }} />
          ))}
        </div>
        {labelled.map((t) => (
          <span key={t.i} className={`absolute top-[58px] whitespace-nowrap text-[13px] tabular-nums text-muted ${t.i * 24 > hours * 0.97 ? '-translate-x-full' : t.i === 0 ? '' : '-translate-x-1/2'}`} style={{ left: pct(t.i * 24) }}>
            {t.first ? `${t.day} ${t.month}` : t.day}
          </span>
        ))}
        <div className="pointer-events-none absolute top-0 -translate-x-1/2" style={{ left: pct(pos), height: 44 }}>
          <div className="mx-auto h-full w-[2px] bg-brand" />
          <div className="absolute -bottom-[5px] left-1/2 h-[11px] w-[11px] -translate-x-1/2 rounded-full border-2 border-white bg-brand" />
        </div>
      </div>
      <p className="mt-3 text-[13px] text-muted">{earthLit ? "New and Full mark the Moon's phase as Earth sees it. " : ''}Marks are days at 00:00 UTC.</p>
    </div>
  )
}
