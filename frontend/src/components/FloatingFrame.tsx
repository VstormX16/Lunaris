import { useEffect, useRef, useState, type ReactNode } from 'react'

type Box = { x: number; y: number; w: number; h: number }
const MIN_W = 340
const MIN_H = 320
const KEY = 'assistant-window'

const clamp = (b: Box): Box => {
  const w = Math.max(MIN_W, Math.min(b.w, window.innerWidth - 16))
  const h = Math.max(MIN_H, Math.min(b.h, window.innerHeight - 16))
  return { w, h, x: Math.max(8, Math.min(b.x, window.innerWidth - w - 8)), y: Math.max(8, Math.min(b.y, window.innerHeight - h - 8)) }
}

function initial(): Box {
  try {
    const saved = JSON.parse(localStorage.getItem(KEY) ?? 'null') as Box | null
    if (saved && [saved.x, saved.y, saved.w, saved.h].every(Number.isFinite)) return clamp(saved)
  } catch {
    /* no storage: use the default */
  }
  return clamp({ w: 440, h: 580, x: window.innerWidth - 470, y: window.innerHeight - 610 })
}

const EDGES = [
  { id: 'n', style: { left: 12, right: 12, top: 0, height: 8 }, cursor: 'ns-resize' },
  { id: 's', style: { left: 12, right: 12, bottom: 0, height: 8 }, cursor: 'ns-resize' },
  { id: 'w', style: { top: 12, bottom: 12, left: 0, width: 8 }, cursor: 'ew-resize' },
  { id: 'e', style: { top: 12, bottom: 12, right: 0, width: 8 }, cursor: 'ew-resize' },
  { id: 'nw', style: { left: 0, top: 0, width: 14, height: 14 }, cursor: 'nwse-resize' },
  { id: 'ne', style: { right: 0, top: 0, width: 14, height: 14 }, cursor: 'nesw-resize' },
  { id: 'sw', style: { left: 0, bottom: 0, width: 14, height: 14 }, cursor: 'nesw-resize' },
  { id: 'se', style: { right: 0, bottom: 0, width: 14, height: 14 }, cursor: 'nwse-resize' },
] as const

/**
 * A window that floats over the page: drag it by its header, resize it from any edge or corner. It remembers where you left it.
 * `header` receives the drag handlers to put on whatever should act as the title bar.
 */
export default function FloatingFrame({ header, children }: { header: (drag: { onPointerDown: (e: React.PointerEvent) => void; className: string }) => ReactNode; children: ReactNode }) {
  const [box, setBox] = useState<Box>(initial)
  const live = useRef(box)
  live.current = box

  useEffect(() => {
    const fit = () => setBox((b) => clamp(b))
    window.addEventListener('resize', fit)
    return () => window.removeEventListener('resize', fit)
  }, [])

  const save = () => {
    try {
      localStorage.setItem(KEY, JSON.stringify(live.current))
    } catch {
      /* not remembered */
    }
  }

  const begin = (mode: string) => (e: React.PointerEvent) => {
    if ((e.target as HTMLElement).closest('button, input, a')) return // the header's own buttons stay clickable
    e.preventDefault()
    const start = { px: e.clientX, py: e.clientY, box: { ...live.current } }
    const move = (ev: PointerEvent) => {
      const dx = ev.clientX - start.px, dy = ev.clientY - start.py
      const b = { ...start.box }
      if (mode === 'move') {
        b.x += dx
        b.y += dy
      } else {
        if (mode.includes('e')) b.w = Math.max(MIN_W, start.box.w + dx)
        if (mode.includes('s')) b.h = Math.max(MIN_H, start.box.h + dy)
        if (mode.includes('w')) { b.w = Math.max(MIN_W, start.box.w - dx); b.x = start.box.x + start.box.w - b.w }
        if (mode.includes('n')) { b.h = Math.max(MIN_H, start.box.h - dy); b.y = start.box.y + start.box.h - b.h }
      }
      setBox(clamp(b))
    }
    const up = () => {
      window.removeEventListener('pointermove', move)
      window.removeEventListener('pointerup', up)
      save()
    }
    window.addEventListener('pointermove', move)
    window.addEventListener('pointerup', up)
  }

  return (
    <div
      role="dialog"
      aria-label="Assistant window"
      className="fixed z-40 flex flex-col overflow-hidden rounded-[12px] border border-line bg-[#fbfcfb] shadow-[0_12px_40px_-12px_rgba(38,70,60,0.35)]"
      style={{ left: box.x, top: box.y, width: box.w, height: box.h }}
    >
      {header({ onPointerDown: begin('move'), className: 'cursor-move touch-none select-none' })}
      <div className="flex min-h-0 flex-1 flex-col">{children}</div>
      {EDGES.map((e) => (
        <div key={e.id} onPointerDown={begin(e.id)} className="absolute touch-none" style={{ ...e.style, cursor: e.cursor }} aria-hidden />
      ))}
    </div>
  )
}
