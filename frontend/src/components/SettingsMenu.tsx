import { Settings } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'
import { useSettings, type Level } from '../lib/settings'

const OPTIONS: { id: Level; title: string; text: string }[] = [
  { id: 'simple', title: 'Simple', text: 'Plain words and a short summary, no technical terms.' },
  { id: 'advanced', title: 'Advanced', text: 'Full detail: every measure, table and warning.' },
]

/** The quiet settings entry at the bottom of the sidebar, with the detail level switch. */
export default function SettingsMenu({ className = '' }: { className?: string }) {
  const { level, setLevel } = useSettings()
  const [open, setOpen] = useState(false)
  const box = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!open) return
    const down = (e: MouseEvent) => {
      if (box.current && !box.current.contains(e.target as Node)) setOpen(false)
    }
    const key = (e: KeyboardEvent) => e.key === 'Escape' && setOpen(false)
    document.addEventListener('mousedown', down)
    document.addEventListener('keydown', key)
    return () => {
      document.removeEventListener('mousedown', down)
      document.removeEventListener('keydown', key)
    }
  }, [open])

  return (
    <div ref={box} className={`relative ${className}`}>
      {open && (
        <div role="dialog" aria-label="Settings" className="step-in absolute bottom-full left-0 right-0 mb-2 rounded-[10px] border border-line bg-[#fbfcfb] p-4 shadow-[0_12px_32px_-14px_rgba(38,70,60,0.35)]">
          <div className="text-[13px] uppercase tracking-wide text-muted">Detail level</div>
          <div role="radiogroup" aria-label="Detail level" className="mt-2 grid grid-cols-2 rounded-[8px] border border-line p-0.5">
            {OPTIONS.map((o) => (
              <button
                key={o.id}
                role="radio"
                aria-checked={level === o.id}
                onClick={() => setLevel(o.id)}
                className={`rounded-[6px] py-1.5 text-[15px] transition-colors ${level === o.id ? 'bg-active font-medium text-brand' : 'text-muted hover:text-ink'}`}
              >
                {o.title}
              </button>
            ))}
          </div>
          <p className="mt-3 text-[13px] leading-snug text-muted">{OPTIONS.find((o) => o.id === level)?.text}</p>
          <p className="mt-2 text-[12px] leading-snug text-muted/80">Applies to results and to the assistant's answers.</p>
        </div>
      )}
      <button
        onClick={() => setOpen((o) => !o)}
        aria-expanded={open}
        className={`flex w-full items-center gap-2.5 rounded-[8px] px-2 py-2 text-[15px] transition-colors hover:text-brand ${open ? 'text-brand' : 'text-muted'}`}
      >
        <Settings size={18} /> Settings
        <span className="ml-auto text-[13px] text-muted">{level === 'simple' ? 'Simple' : 'Advanced'}</span>
      </button>
    </div>
  )
}
