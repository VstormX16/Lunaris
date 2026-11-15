import { ChevronDown } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'

const pad = (n: number) => String(n).padStart(2, '0')

/** Hour of the day (UTC). Same look as the date field: big underlined value, a white panel that pours out below it. */
export default function TimeField({ value, onChange }: { value: number; onChange: (hour: number) => void }) {
  const [open, setOpen] = useState(false)
  const root = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!open) return
    const away = (e: MouseEvent) => !root.current?.contains(e.target as Node) && setOpen(false)
    const esc = (e: KeyboardEvent) => e.key === 'Escape' && setOpen(false)
    document.addEventListener('mousedown', away)
    document.addEventListener('keydown', esc)
    return () => {
      document.removeEventListener('mousedown', away)
      document.removeEventListener('keydown', esc)
    }
  }, [open])

  return (
    <div ref={root} className="relative">
      <button
        aria-label="Time (UTC)"
        aria-expanded={open}
        onClick={() => setOpen((o) => !o)}
        className={`flex w-full items-center gap-2 border-b pb-2 text-left transition-colors ${open ? 'border-brand' : 'border-line hover:border-brand'}`}
      >
        <span className="flex-1 text-[26px] tabular-nums tracking-wide text-ink">{pad(value)}:00</span>
        <ChevronDown size={20} className={`mr-1 transition-transform ${open ? 'rotate-180 text-brand' : 'text-muted'}`} />
      </button>
      <p className="mt-1.5 h-5 text-[13px] text-muted">UTC time</p>

      <div className="genie glass-shell solid absolute left-0 right-0 top-full z-20 mt-1 min-w-[260px] rounded-[14px] p-[5px]" data-open={open} aria-hidden={!open} inert={!open}>
        <div className="glass-pane rounded-[10px] p-3">
          <div className="grid grid-cols-4 gap-1">
            {Array.from({ length: 24 }, (_, h) => (
              <button
                key={h}
                onClick={() => {
                  onChange(h)
                  setOpen(false)
                }}
                className={`h-9 rounded-[8px] text-[15px] tabular-nums transition-colors ${h === value ? 'bg-brand text-white' : 'text-ink hover:bg-active'}`}
              >
                {pad(h)}:00
              </button>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}
