import { CalendarDays } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'
import { formatDate, MAX_YEAR, MIN_YEAR, parseDate } from '../../lib/input'
import Calendar from './Calendar'

/** "Enter date" input + calendar button. The calendar opens below, exactly as wide as the row. */
type Props = {
  value: string
  onChange: (iso: string) => void
  small?: boolean // a quieter size for toolbars
  anchor?: string // month to open on while empty (e.g. scenario A's date for scenario B)
  placement?: 'bottom' | 'top' // top keeps the calendar clear of the footer
  solid?: boolean // plain white panel instead of glass, for use over a picture
  inline?: boolean // no calendar button or popup: the page shows its own calendar beside the field
}

export default function DateField({ value, onChange, anchor, placement = 'bottom', small = false, solid = false, inline = false }: Props) {
  const [text, setText] = useState(value ? formatDate(value) : '')
  const [open, setOpen] = useState(false)
  const [touched, setTouched] = useState(false)
  const root = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (value) setText(formatDate(value)) // keep the text in sync when the calendar picks
  }, [value])

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

  const edit = (t: string) => {
    setText(t)
    const iso = parseDate(t)
    if (iso) onChange(iso)
  }
  const invalid = touched && text.trim() !== '' && parseDate(text) === null

  return (
    <div ref={root} className="relative">
      <div className={`flex items-center gap-2 border-b pb-2 transition-colors ${invalid ? 'border-amber-600' : 'border-line focus-within:border-brand'}`}>
        <input
          value={text}
          onChange={(e) => edit(e.target.value)}
          onBlur={() => setTouched(true)}
          onKeyDown={(e) => e.key === 'Enter' && setOpen(false)}
          placeholder="Enter date"
          inputMode="numeric"
          aria-label="Date"
          className={`min-w-0 flex-1 bg-transparent tracking-wide text-ink outline-none placeholder:text-muted/60 ${small ? 'text-[18px]' : 'text-[26px]'}`}
        />
        {!inline && <button
          aria-label="Open calendar"
          aria-expanded={open}
          onClick={() => setOpen((o) => !o)}
          className={`rounded-[8px] p-2 transition-colors ${open ? 'bg-active text-brand' : 'text-muted hover:text-brand'}`}
        >
          <CalendarDays size={22} />
        </button>}
      </div>
      <p className={`mt-1.5 h-5 text-[13px] ${invalid ? 'text-amber-700' : 'text-muted'}`}>
        {invalid ? `Use DD.MM.YYYY between ${MIN_YEAR} and ${MAX_YEAR}` : 'DD.MM.YYYY'}
      </p>

      {/* genie: grows out of the calendar button at the right end of the row */}
      {!inline && <div
        className={`genie glass-shell ${solid ? 'solid' : ''} absolute left-0 right-0 z-20 rounded-[14px] p-[5px] ${placement === 'top' ? 'bottom-full mb-1' : 'top-full mt-1'}`}
        data-open={open}
        data-up={placement === 'top'}
        aria-hidden={!open}
        inert={!open}
      >
        <div className="glass-pane rounded-[10px] p-4">
          <Calendar
            value={value}
            anchor={anchor}
            onPick={(iso) => {
              onChange(iso)
              setOpen(false)
            }}
          />
        </div>
      </div>}
    </div>
  )
}
