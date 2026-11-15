import { ChevronLeft, ChevronRight, ChevronsLeft, ChevronsRight } from 'lucide-react'
import { useEffect, useState } from 'react'
import { isoFromParts, MAX_YEAR, MIN_YEAR, todayIso } from '../../lib/input'

const MONTHS = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
const DAYS = ['Mo', 'Tu', 'We', 'Th', 'Fr', 'Sa', 'Su']

const split = (iso: string) => {
  const [y, m] = iso.split('-').map(Number)
  return { y, m: m - 1 }
}

/** `anchor` decides which month opens when nothing is selected yet. */
export default function Calendar({ value, anchor, onPick, big = false }: { value: string; anchor?: string; onPick: (iso: string) => void; big?: boolean }) {
  const today = todayIso()
  const [view, setView] = useState(() => split(value || anchor || today))
  useEffect(() => {
    if (value) setView(split(value))
  }, [value])

  const shift = (months: number) =>
    setView(({ y, m }) => {
      const total = Math.min(MAX_YEAR * 12 + 11, Math.max(MIN_YEAR * 12, y * 12 + m + months))
      return { y: Math.floor(total / 12), m: total % 12 }
    })

  const lead = (new Date(Date.UTC(view.y, view.m, 1)).getUTCDay() + 6) % 7 // Monday first
  const count = new Date(Date.UTC(view.y, view.m + 1, 0)).getUTCDate()
  const cells = [...Array<null>(lead).fill(null), ...Array.from({ length: count }, (_, i) => i + 1)]

  const nav = 'rounded-[8px] p-1.5 text-muted hover:bg-active hover:text-brand'
  return (
    <div>
      <div className="flex items-center justify-between">
        <div className="flex">
          <button aria-label="Previous year" onClick={() => shift(-12)} className={nav}><ChevronsLeft size={16} /></button>
          <button aria-label="Previous month" onClick={() => shift(-1)} className={nav}><ChevronLeft size={16} /></button>
        </div>
        <div className="text-[16px] font-medium text-ink">{MONTHS[view.m]} {view.y}</div>
        <div className="flex">
          <button aria-label="Next month" onClick={() => shift(1)} className={nav}><ChevronRight size={16} /></button>
          <button aria-label="Next year" onClick={() => shift(12)} className={nav}><ChevronsRight size={16} /></button>
        </div>
      </div>

      <div className="mt-3 grid grid-cols-7 gap-y-1 text-center">
        {DAYS.map((d) => (
          <div key={d} className="pb-1 text-[12px] uppercase tracking-wide text-muted">{d}</div>
        ))}
        {cells.map((d, i) => {
          if (d === null) return <span key={`b${i}`} />
          const iso = isoFromParts(view.y, view.m + 1, d)!
          const selected = iso === value
          return (
            <button
              key={iso}
              onClick={() => onPick(iso)}
              className={`mx-auto rounded-[8px] tabular-nums transition-colors ${big ? 'h-11 w-11 text-[17px]' : 'h-9 w-9 text-[15px]'} ${
                selected
                  ? 'bg-brand text-white'
                  : iso === today
                    ? 'border border-brand text-ink hover:bg-active'
                    : 'text-ink hover:bg-active'
              }`}
            >
              {d}
            </button>
          )
        })}
      </div>
    </div>
  )
}
