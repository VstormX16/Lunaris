import { Bot, CalendarRange, ChevronDown, Home, MessageSquare, Clock, Filter, Map as MapIcon, PanelLeft, Rotate3d, Rocket, Satellite, Search, Orbit, Target, Trash2 } from 'lucide-react'
import { useRef, useState } from 'react'
import type { SessionSummary } from '../api'
import SettingsMenu from './SettingsMenu'
import { kindOf, SESSION_TABS, type SessionKind } from '../lib/sessionKind'

export type View = 'home' | 'spacecrafts' | 'wizard' | 'analysis' | 'spacetime' | 'moonmap' | 'landmap' | 'planner' | 'timeline' | 'view360' | 'assistant'

const NAV: { label: string; icon: typeof Rocket; view?: View }[] = [
  { label: 'New Operation', icon: Rocket, view: 'wizard' },
  { label: 'Space Time', icon: Orbit, view: 'spacetime' },
  { label: 'Moon Map', icon: MapIcon, view: 'moonmap' },
  { label: 'South Area', icon: Target, view: 'landmap' },
  { label: 'Mission Planner', icon: CalendarRange, view: 'planner' },
  { label: 'Moon Timeline', icon: Clock, view: 'timeline' },
  { label: '360 View', icon: Rotate3d, view: 'view360' },
  { label: 'Spacecrafts', icon: Satellite, view: 'spacecrafts' },
  { label: 'AI Assistant', icon: Bot, view: 'assistant' },
]

type Props = {
  collapsed: boolean
  onToggle: () => void
  view: View
  onNavigate: (v: View) => void
  sessions: SessionSummary[]
  activeSession: string | null
  onOpenSession: (id: string) => void
  onDeleteSession: (id: string) => void
}

export default function Sidebar({ collapsed, onToggle, view, onNavigate, sessions, activeSession, onOpenSession, onDeleteSession }: Props) {
  const [confirming, setConfirming] = useState<string | null>(null)
  const [historyOpen, setHistoryOpen] = useState(true)
  const [searching, setSearching] = useState(false)
  const [query, setQuery] = useState('')
  const inputRef = useRef<HTMLInputElement>(null)
  const q = query.trim().toLowerCase()
  const [filterOpen, setFilterOpen] = useState(false)
  const [filter, setFilter] = useState<SessionKind | null>(null)
  const shown = sessions.filter((s) => (!q || s.name.toLowerCase().includes(q)) && (!filter || kindOf(s) === filter))
  const toggleSearch = () => {
    setHistoryOpen(true)
    if (!searching) setTimeout(() => inputRef.current?.focus(), 300) // after the field has opened
    setSearching((o) => !o)
    setQuery('')
  }

  // One panel for both states: the shell animates its width and clips the fixed-width
  // inner column, so nothing reflows or snaps while resizing.
  const fade = `transition-opacity duration-200 ${collapsed ? 'pointer-events-none opacity-0' : 'opacity-100 delay-100'}`

  return (
    // detached frosted shell: 5px glass rim around a crisp inner panel
    <aside
      className={`glass-shell m-[14px] h-[calc(100vh-28px)] shrink-0 rounded-[14px] p-[5px] transition-[width] duration-300 ease-in-out ${
        collapsed ? 'w-[74px]' : 'w-[300px]'
      }`}
    >
      <div className="glass-pane h-full overflow-hidden rounded-[10px]">
        <div className="flex h-full w-[290px] flex-col px-[17px] py-5">
          <div className="relative h-8">
            {/* the mark turns over on hover and shows its back: a home button */}
            <button
              onClick={() => onNavigate('home')}
              aria-label="Lunaris: go to the start page"
              title="Start page"
              className={`group absolute left-0 top-0 flex items-center gap-2.5 whitespace-nowrap ${searching ? 'pointer-events-none opacity-0 transition-opacity duration-300' : fade}`}
            >
              <span className="relative h-8 w-8 shrink-0 [perspective:500px]">
                <span className="absolute inset-0 transition-transform duration-500 ease-[cubic-bezier(0.3,1.4,0.5,1)] [transform-style:preserve-3d] group-hover:[transform:rotateY(180deg)] group-focus-visible:[transform:rotateY(180deg)]">
                  <span className="logo-mark absolute inset-0 bg-brand [backface-visibility:hidden]" />
                  <span className="absolute inset-0 grid place-items-center rounded-[8px] bg-brand text-white [backface-visibility:hidden] [transform:rotateY(180deg)]">
                    <Home size={17} strokeWidth={2} />
                  </span>
                </span>
              </span>
              <span className="text-2xl font-medium text-brand">Lunaris</span>
            </button>
            {/* anchored to the right: opening grows it to the left, so the icon slides across while the field opens */}
            <div
              className={`absolute right-[30px] top-0 flex h-8 items-center overflow-hidden border-b ${searching ? 'border-brand' : 'border-transparent'} ${collapsed ? 'pointer-events-none opacity-0' : ''}`}
              // width is set here, not by a class: the shared `fade` class would replace the width transition with an opacity one
              style={{ width: searching ? 226 : 24, transition: 'width 650ms cubic-bezier(0.45, 0, 0.2, 1), border-color 300ms ease' }}
            >
              <button aria-label="Search history" aria-pressed={searching} onClick={toggleSearch} className={`flex h-8 w-6 shrink-0 items-center justify-center hover:text-brand ${searching ? 'text-brand' : 'text-muted'}`}>
                <Search size={20} />
              </button>
              <input
                ref={inputRef}
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                onKeyDown={(e) => e.key === 'Escape' && toggleSearch()}
                tabIndex={searching ? 0 : -1}
                placeholder="Search history…"
                aria-label="Search history text"
                className={`min-w-0 flex-1 bg-transparent px-2 text-[15px] text-ink outline-none transition-opacity duration-500 placeholder:text-muted/60 ${searching ? "opacity-100 delay-200" : "opacity-0"}`}
              />
            </div>
            <button
              aria-label={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
              onClick={onToggle}
              className={`absolute top-1 text-muted transition-[left] duration-300 ease-in-out hover:text-brand ${collapsed ? 'left-[4px]' : 'left-[234px]'}`}
            >
              <PanelLeft size={22} />
            </button>
          </div>

          <nav className="mt-8 flex flex-col gap-1.5">
            {NAV.map(({ label, icon: Icon, view: target }) => {
              // the analysis page is the end of the New Operation flow
              const isActive = target !== undefined && (target === view || (target === 'wizard' && view === 'analysis'))
              return (
                <button
                  key={label}
                  aria-label={label}
                  aria-current={isActive ? 'page' : undefined}
                  onClick={() => target && onNavigate(target)}
                  // collapsed: the button shrinks to a small rounded square around the icon instead of being cut off by the panel edge
                  className={`grain -mx-2 flex items-center gap-3 overflow-hidden whitespace-nowrap rounded-[8px] border px-2 py-2.5 text-left text-[17px] transition-[width,translate,background-color,border-color,box-shadow] duration-300 ease-in-out ${collapsed ? 'w-[38px] translate-x-[4px]' : 'w-[272px]'} ${
                    isActive
                      ? 'border-brand bg-white/85 text-ink shadow-[0_1px_2px_rgba(38,70,60,0.08)]'
                      : 'border-transparent text-ink hover:bg-active'
                  }`}
                >
                  {/* only the active icon carries the accent color; the rest stay quiet */}
                  <Icon size={20} className={`shrink-0 ${isActive ? 'text-brand' : 'text-muted'}`} />{' '}
                  <span className={fade}>{label}</span>
                </button>
              )
            })}
          </nav>

          <div inert={collapsed} className={`flex min-h-0 flex-1 flex-col justify-between ${fade}`}>
            <div className="min-h-0">
            <hr className="mb-5 mt-6 border-line" />

            <div className="flex items-center justify-between px-1 text-sm text-muted">
              <button
                onClick={() => setHistoryOpen((o) => !o)}
                aria-expanded={historyOpen}
                className="flex items-center gap-1.5 hover:text-brand"
              >
                <ChevronDown size={16} className={`transition-transform duration-300 ${historyOpen ? '' : '-rotate-90'}`} />
                History
              </button>
              <button
                aria-label="Filter history"
                aria-expanded={filterOpen}
                onClick={() => { setHistoryOpen(true); setFilterOpen((o) => !o) }}
                className={`relative rounded-[6px] p-1 transition-colors hover:bg-active hover:text-brand ${filterOpen || filter ? 'text-brand' : ''}`}
              >
                <Filter size={15} />
                {filter && <span className="absolute right-0.5 top-0.5 h-1.5 w-1.5 rounded-full bg-brand" />}
              </button>
            </div>

            {/* the filter row opens under the header the same way the list does */}
            <div className={`grid transition-all duration-300 ease-in-out ${filterOpen ? 'mt-2 grid-rows-[1fr] opacity-100' : 'grid-rows-[0fr] opacity-0'}`}>
              <div role="radiogroup" aria-label="Show in history" className="flex min-h-0 flex-wrap gap-x-2.5 gap-y-1 overflow-hidden px-1 text-[13px]">
                {SESSION_TABS.map(({ label, kind }) => (
                  <button
                    key={label}
                    role="radio"
                    aria-checked={filter === kind}
                    tabIndex={filterOpen ? 0 : -1}
                    onClick={() => setFilter(kind)}
                    className={`border-b pb-0.5 transition-colors ${filter === kind ? 'border-brand font-medium text-brand' : 'border-transparent text-muted hover:text-ink'}`}
                  >
                    {label}
                  </button>
                ))}
              </div>
            </div>

            {/* grid-rows 1fr -> 0fr animates height without measuring the content */}
            <div
              className={`grid min-h-0 transition-all duration-300 ease-in-out ${
                historyOpen ? 'mt-2 grid-rows-[1fr] opacity-100' : 'grid-rows-[0fr] opacity-0'
              }`}
            >
              <ul className="min-h-0 max-h-[220px] overflow-y-auto">
                {sessions.length === 0 && (
                  <li className="px-3 py-3 text-[14px] text-muted">
                    <Rocket size={18} strokeWidth={1.5} className="mb-1.5 text-brand/60" />
                    Your operations and conversations will appear here.
                    <button onClick={() => onNavigate('wizard')} tabIndex={historyOpen ? 0 : -1} className="mt-1 block text-brand hover:underline hover:underline-offset-4">Start one</button>
                  </li>
                )}
                {sessions.length > 0 && shown.length === 0 && (
                  <li className="px-3 py-[7px] text-[14px] text-muted">
                    {q ? <>Nothing matches “{query.trim()}”.</> : 'Nothing of this kind yet.'}
                    <button onClick={() => { setQuery(''); setFilter(null) }} tabIndex={historyOpen ? 0 : -1} className="ml-2 text-brand hover:underline hover:underline-offset-4">Show all</button>
                  </li>
                )}
                {shown.map((it) => {
                  const isActive = it.id === activeSession && (view === 'analysis' || view === 'wizard' || view === 'assistant')
                  return (
                    <li key={it.id} className="group relative">
                      {confirming === it.id ? (
                        <div className="flex items-center justify-between gap-2 rounded-[8px] bg-active px-3 py-[7px] text-[14px]">
                          <span className="text-ink">Delete this operation?</span>
                          <span className="flex gap-3">
                            <button onClick={() => { setConfirming(null); onDeleteSession(it.id) }} className="text-red-700 hover:underline">Delete</button>
                            <button onClick={() => setConfirming(null)} className="text-muted hover:text-brand">Keep</button>
                          </span>
                        </div>
                      ) : (
                        <>
                          <button
                            onClick={() => onOpenSession(it.id)}
                            tabIndex={historyOpen ? 0 : -1}
                            title={it.name}
                            className={`grain w-full truncate rounded-[8px] py-[7px] pl-3 pr-9 text-left text-[16px] ${isActive ? 'bg-active font-medium text-ink' : 'text-ink/80 hover:bg-active'}`}
                          >
                            {it.kind === 'chat' && <MessageSquare size={13} className="mr-1.5 inline-block -translate-y-px text-muted" aria-label="AI conversation" />}{it.name}
                          </button>
                          <button
                            onClick={() => setConfirming(it.id)}
                            aria-label={`Delete ${it.name}`}
                            tabIndex={historyOpen ? 0 : -1}
                            className="absolute right-2 top-1/2 -translate-y-1/2 rounded p-1 text-muted opacity-0 transition hover:text-red-700 focus:opacity-100 group-hover:opacity-100"
                          >
                            <Trash2 size={15} />
                          </button>
                        </>
                      )}
                    </li>
                  )
                })}
              </ul>
            </div>
            </div>
            <div>
              <hr className="mb-2 mt-3 border-line" />
              <SettingsMenu />
            </div>
          </div>
        </div>
      </div>
    </aside>
  )
}
