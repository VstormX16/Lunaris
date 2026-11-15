import { ArrowDown, ArrowUp, Pencil, Plus, Trash2 } from 'lucide-react'
import { useEffect, useMemo, useRef, useState } from 'react'
import { deleteCustomVehicle, getVehicles, type VehicleProfile } from '../api'
import { spacecrafts, type Spacecraft } from '../data/spacecrafts'
import { nation } from '../lib/facts'
import Photo from './Photo'
import SpacecraftDetail from './SpacecraftDetail'
import VehicleDesigner from './VehicleDesigner'

type SortKey = 'year' | 'name' | 'country'

const SORTS: { key: SortKey; label: string }[] = [
  { key: 'year', label: 'Year' },
  { key: 'name', label: 'Name' },
  { key: 'country', label: 'Country' },
]

const EASE = 'ease-[cubic-bezier(0.32,0.72,0,1)]'
const LIST_WIDTH = 440
const FADE_MS = 170 // gallery fades out at full width first, then the screen splits

function compare(a: Spacecraft, b: Spacecraft, key: SortKey) {
  const primary = key === 'year' ? a.year - b.year : a[key].localeCompare(b[key], 'tr')
  return primary || a.name.localeCompare(b.name, 'tr') // stable tie-break
}

function Stat({ value, label }: { value: string | number; label: string }) {
  return (
    <div>
      <div className="text-[38px] font-bold leading-none tabular-nums text-strong">{value}</div>
      <div className="mt-1.5 text-[13px] uppercase tracking-wide text-muted">{label}</div>
    </div>
  )
}

/** Photo tile: the picture stays untouched; the caption sits below it on the canvas. */
function Tile({ craft, onOpen }: { craft: Spacecraft; onOpen: () => void }) {
  return (
    <button onClick={onOpen} data-tile className="group block text-left">
      <div className="relative aspect-square overflow-hidden rounded-[10px] border border-line">
        <div className="absolute inset-0 transition-transform duration-700 ease-out group-hover:scale-[1.05]">
          <Photo id={craft.id} name={craft.name} iconSize={34} className="h-full w-full" />
        </div>
      </div>
      <div className="mt-2.5 flex items-baseline justify-between gap-3">
        <span className="truncate text-[17px] font-semibold text-strong group-hover:text-brand">{craft.name}</span>
        <span className="shrink-0 text-[14px] tabular-nums text-muted">{craft.year}</span>
      </div>
      <div className="truncate text-[13px] text-muted">{craft.country} · {craft.type}</div>
    </button>
  )
}

/** Gallery of tiles until a craft is chosen, then master/detail: compact list left, full specification right. */
export default function SpacecraftsPage() {
  const [query, setQuery] = useState('')
  const [sortKey, setSortKey] = useState<SortKey>('year')
  const [asc, setAsc] = useState(true)
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [split, setSplit] = useState(false)
  const [leaving, setLeaving] = useState(false)
  const [profiles, setProfiles] = useState<Record<string, VehicleProfile> | null>(null) // the landing analysis' own values, to show beside the spec sheet
  const reload = () => getVehicles().then((v) => setProfiles(Object.fromEntries(v.map((p) => [p.id, p])))).catch(() => setProfiles(null))
  useEffect(() => { void reload() }, [])
  const [designing, setDesigning] = useState<'new' | VehicleProfile | null>(null) // the vehicle designer replaces the page while it is open
  const [confirmDelete, setConfirmDelete] = useState<string | null>(null)
  const mine = Object.values(profiles ?? {}).filter((p) => p.custom)
  const detailRef = useRef<HTMLElement>(null)

  const q = query.trim().toLocaleLowerCase('tr')
  const shown = spacecrafts
    .filter((c) => [c.name, c.country, c.type, String(c.year)].some((v) => v.toLocaleLowerCase('tr').includes(q)))
    .sort((a, b) => (asc ? 1 : -1) * compare(a, b, sortKey))
  const selected = spacecrafts.find((c) => c.id === selectedId) ?? null

  // neighbours in the current (filtered, sorted) order, and the missions closest in time
  const at = shown.findIndex((c) => c.id === selectedId)
  const prev = at > 0 ? shown[at - 1] : null
  const next = at >= 0 && at < shown.length - 1 ? shown[at + 1] : null
  const related = selected
    ? spacecrafts
        .filter((c) => c.id !== selected.id && nation(c) === nation(selected))
        .sort((a, b) => Math.abs(a.year - selected.year) - Math.abs(b.year - selected.year))
        .slice(0, 4)
    : []

  const stats = useMemo(() => {
    const years = spacecrafts.map((c) => c.year)
    return { missions: spacecrafts.length, nations: new Set(spacecrafts.map(nation)).size, span: `${Math.min(...years)}–${Math.max(...years)}` }
  }, [])

  const open = (id: string) => {
    if (split) return setSelectedId(id)
    setLeaving(true) // fade the gallery at its full width first: no reflow while it disappears
    setTimeout(() => {
      setSelectedId(id)
      setSplit(true)
      setLeaving(false)
    }, FADE_MS)
  }
  const close = () => {
    setSelectedId(null)
    setSplit(false)
  }

  // arrow keys browse, Escape closes (ignored while typing in the search box)
  useEffect(() => {
    if (!split || !selected) return
    const onKey = (e: KeyboardEvent) => {
      if ((e.target as HTMLElement | null)?.closest('input, textarea, select')) return
      if (e.key === 'ArrowRight' && next) setSelectedId(next.id)
      else if (e.key === 'ArrowLeft' && prev) setSelectedId(prev.id)
      else if (e.key === 'Escape') close()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  })

  // a new vehicle starts at the top of the detail pane
  useEffect(() => {
    detailRef.current?.scrollTo({ top: 0 })
  }, [selectedId])

  // keep the chosen row in view when the list narrows beside the detail pane
  useEffect(() => {
    document.querySelector('[data-craft][aria-pressed="true"]')?.scrollIntoView({ block: 'nearest', behavior: 'smooth' })
  }, [selectedId])

  // Only `width` and `padding` animate (both interpolate cleanly); the detail content keeps a
  // fixed width and is merely revealed, so nothing reflows while the panes move.
  if (designing) {
    return <VehicleDesigner initial={designing === 'new' ? null : designing} onCancel={() => setDesigning(null)} onDone={() => { setDesigning(null); void reload() }} />
  }

  const pad = { paddingInline: split ? 32 : 'max(32px, calc((100% - 1100px) / 2))' }
  const galleryOn = !split && !leaving

  return (
    <main className="flex h-full min-w-0 flex-1">
      <section className={`flex min-h-0 flex-none flex-col transition-[width] duration-500 ${EASE}`} style={{ width: split ? LIST_WIDTH : '100%' }}>
        <div className={`pb-4 pt-10 transition-[padding] duration-500 ${EASE}`} style={pad}>
          <h1 className="text-[34px] font-medium text-brand">Spacecrafts</h1>
          {split && <p className="mt-1 text-muted">{shown.length} lunar missions, 1959 – 2026</p>}

          <div className="mt-4 flex flex-wrap items-center gap-3 text-sm">
            <span className="text-muted">Sort by</span>
            {SORTS.map(({ key, label }) => (
              <button
                key={key}
                onClick={() => setSortKey(key)}
                aria-pressed={sortKey === key}
                className={sortKey === key ? 'font-medium text-brand underline underline-offset-4' : 'text-muted hover:text-ink'}
              >
                {label}
              </button>
            ))}
            <button
              onClick={() => setAsc((a) => !a)}
              aria-label={asc ? 'Ascending order, switch to descending' : 'Descending order, switch to ascending'}
              className="rounded-[8px] p-1 text-brand hover:bg-active"
            >
              {asc ? <ArrowUp size={16} /> : <ArrowDown size={16} />}
            </button>
            <input
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Search name, country, type, year"
              className={`rounded-[8px] border border-line bg-white/60 px-3 py-1.5 outline-none focus:border-brand placeholder:text-muted/70 ${split ? 'mt-2 w-full' : 'ml-auto w-72'}`}
            />
          </div>
        </div>

        <div className="relative min-h-0 flex-1">
          {/* gallery: shown until a craft is chosen; fades in only after the width has settled again */}
          <div
            className={`absolute inset-0 overflow-y-auto pb-12 transition-opacity ${galleryOn ? 'opacity-100 delay-[450ms] duration-300' : 'pointer-events-none opacity-0 duration-150'}`}
            style={pad}
            inert={!galleryOn}
            aria-hidden={!galleryOn}
          >
            <div className="mb-6 flex flex-wrap gap-x-14 gap-y-4 border-b border-line pb-6">
              <Stat value={stats.missions} label="Lunar missions" />
              <Stat value={stats.nations} label="Nations & agencies" />
              <Stat value={stats.span} label="Years of exploration" />
            </div>
            <section className="mb-8" aria-label="My vehicles">
              <div className="flex items-center justify-between gap-4">
                <h2 className="text-[20px] font-medium text-strong">My vehicles</h2>
                <button onClick={() => setDesigning('new')} className="grain flex items-center gap-1.5 rounded-[8px] border border-brand px-3.5 py-1.5 text-[15px] text-brand transition hover:bg-brand hover:text-white"><Plus size={16} /> Design a vehicle</button>
              </div>
              {mine.length === 0 ? (
                <p className="mt-2 max-w-[560px] text-[15px] text-muted">Design your own lander or rover: name it, set its battery, solar array, loads and slope limit, and see how it would do at a real place. It then shows up in New Operation and in the Mission Planner.</p>
              ) : (
                <div className="mt-3 grid grid-cols-[repeat(auto-fill,minmax(280px,1fr))] gap-4">
                  {mine.map((p) => (
                    <div key={p.id} className="rounded-[10px] border border-line bg-white/55 p-4">
                      <div className="flex items-start justify-between gap-2">
                        <div className="min-w-0">
                          <div className="truncate text-[17px] font-semibold text-strong">{p.name}</div>
                          <div className="text-[13px] text-muted">{p.kind === 'rover' ? 'Rover' : 'Lander'} · custom</div>
                        </div>
                        <div className="flex shrink-0 items-center gap-0.5">
                          <button onClick={() => setDesigning(p)} aria-label={`Edit ${p.name}`} className="rounded-[8px] p-1.5 text-muted transition hover:bg-active hover:text-brand"><Pencil size={15} /></button>
                          <button onClick={() => setConfirmDelete(p.id)} aria-label={`Delete ${p.name}`} className="rounded-[8px] p-1.5 text-muted transition hover:bg-active hover:text-red-700"><Trash2 size={15} /></button>
                        </div>
                      </div>
                      <div className="mt-2 text-[13.5px] leading-snug text-ink">
                        {p.battery_wh.value?.toLocaleString('en')} Wh · {p.solar_w.value === 0 ? 'no array' : `${p.solar_w.value} W array`} · {p.max_slope_deg.value}° slope · {p.radioisotope_heat.value ? 'heater' : 'no heater'} · {p.comms.value === 'relay' ? 'relay' : 'direct link'}
                      </div>
                      {p.note && <p className="mt-1.5 truncate text-[13px] text-muted">{p.note}</p>}
                      {confirmDelete === p.id && (
                        <div className="mt-3 flex items-center gap-3 border-t border-line pt-3 text-[14px]">
                          <span className="text-ink">Delete it?</span>
                          <button onClick={() => { void deleteCustomVehicle(p.id).then(() => { setConfirmDelete(null); return reload() }) }} className="text-red-700 hover:underline">Delete</button>
                          <button onClick={() => setConfirmDelete(null)} className="text-muted hover:text-ink">Keep</button>
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </section>
            <h2 className="mb-3 text-[20px] font-medium text-strong">Lunar missions</h2>
            <div className="grid grid-cols-[repeat(auto-fill,minmax(240px,1fr))] gap-4">
              {shown.map((c) => (
                <Tile key={c.id} craft={c} onOpen={() => open(c.id)} />
              ))}
            </div>
            {shown.length === 0 && <p className="text-muted">No matches.</p>}
          </div>

          {/* compact list: takes over once the screen is split */}
          <ul
            className={`absolute inset-0 space-y-1.5 overflow-y-auto pb-10 transition-opacity ${split ? 'opacity-100 delay-[250ms] duration-300' : 'pointer-events-none opacity-0 duration-100'}`}
            style={pad}
            inert={!split}
            aria-hidden={!split}
          >
            {shown.map((c) => {
              const active = c.id === selectedId
              return (
                <li key={c.id}>
                  <button
                    onClick={() => (active ? close() : open(c.id))}
                    aria-pressed={active}
                    data-craft
                    className={`flex w-full items-center gap-4 rounded-[10px] border p-2.5 text-left transition-colors ${
                      active ? 'border-brand bg-white/85' : 'border-transparent hover:bg-active'
                    }`}
                  >
                    <Photo id={c.id} name={c.name} className="h-[56px] w-[56px] shrink-0 rounded-[8px]" iconSize={20} />
                    <span className="min-w-0">
                      <span className="block truncate text-[17px] text-ink">{c.name}</span>
                      <span className="block truncate text-[13px] text-muted">{c.year} · {c.country} · {c.type}</span>
                    </span>
                  </button>
                </li>
              )
            })}
            {shown.length === 0 && <li className="text-muted">No matches.</li>}
          </ul>
        </div>
      </section>

      <section
        ref={detailRef}
        className={`min-h-0 min-w-0 flex-1 overflow-y-auto overflow-x-hidden border-l border-line transition-opacity duration-500 ${split ? 'opacity-100' : 'pointer-events-none opacity-0'}`}
        aria-hidden={!split}
      >
        {/* fixed minimum width: the text keeps its line breaks while this pane grows */}
        <div className="min-w-[640px] px-10 pt-10">
          {selected && (
            <SpacecraftDetail
              profile={profiles?.[selected.id] ?? null}
              profilesReady={profiles !== null}
              craft={selected}
              position={at >= 0 ? { index: at + 1, total: shown.length } : null}
              prev={prev}
              next={next}
              related={related}
              onSelect={setSelectedId}
              onClose={close}
            />
          )}
        </div>
      </section>
    </main>
  )
}
