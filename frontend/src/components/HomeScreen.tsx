import { ArrowRight, Bot, Map as MapIcon, MessageSquare, Orbit, Rocket, Search, Target } from 'lucide-react'
import { useState } from 'react'
import type { SessionSummary } from '../api'
import type { View } from './Sidebar'
import Photo from './Photo'
import { kindOf, SESSION_TABS, singular, type SessionKind } from '../lib/sessionKind'

const day = (iso: string) => {
  const d = new Date(iso.endsWith('Z') || iso.includes('+') ? iso : `${iso}Z`)
  return Number.isNaN(d.getTime()) ? '' : `${String(d.getDate()).padStart(2, '0')}.${String(d.getMonth() + 1).padStart(2, '0')}.${d.getFullYear()}`
}

const EXPLORE: { view: View; icon: typeof Rocket; title: string; text: string }[] = [
  { view: 'landmap', icon: Target, title: 'South Area', text: 'The whole south pole, scored by our neural network' },
  { view: 'moonmap', icon: MapIcon, title: 'Moon Map', text: 'Any point on the Moon, on any date' },
  { view: 'spacetime', icon: Orbit, title: 'Space Time', text: 'Sunrise, sunset and what is coming up' },
  { view: 'assistant', icon: Bot, title: 'AI Assistant', text: 'Ask about dates, places and vehicles' },
]

export default function HomeScreen({ onStart, sessions, onOpen, onNavigate }: { onStart: () => void; sessions: SessionSummary[]; onOpen: (id: string) => void; onNavigate: (v: View) => void }) {
  const [tab, setTab] = useState<SessionKind | null>(null)
  const [query, setQuery] = useState('')

  const shown = sessions.filter((c) => (!tab || kindOf(c) === tab) && c.name.toLowerCase().includes(query.toLowerCase()))

  return (
    <main className="flex-1 overflow-y-auto">
      <div className="mx-auto flex min-h-full max-w-[960px] flex-col px-6 pb-12 pt-[6vh]">
        {/* hero: unboxed, generous breathing room */}
        <section className="pb-12 pt-[56px] text-center">
          <h1 className="hero-sweep text-[56px] font-medium leading-tight tracking-[-0.01em] text-brand">What’s the move today?</h1>

          {/* the only double-halo element: 4px glass rim around a crisp white trigger */}
          <div className="mt-10 flex justify-center">
            <span className="glass-rim inline-flex rounded-[10px] p-[4px]">
              <button onClick={onStart} className="gem-border grain group flex items-center gap-3 rounded-[8px] border border-brand bg-white px-8 py-3.5 text-[22px] text-ink transition duration-200 hover:text-brand hover:shadow-[0_8px_22px_-10px_rgba(38,70,60,0.45)] active:scale-[0.98]">
                Start New Operation <ArrowRight size={20} className="text-brand transition-transform duration-200 group-hover:translate-x-1" />
              </button>
            </span>
          </div>
        </section>

        <section className="mb-14 mt-2 grid grid-cols-4 gap-4 max-[900px]:grid-cols-2" aria-label="Explore">
          {EXPLORE.map(({ view, icon: Icon, title, text }, i) => (
            <button
              key={view}
              onClick={() => onNavigate(view)}
              className="rise-in glass-preview grain group flex flex-col items-start gap-2 px-4 py-4 text-left transition duration-300 hover:-translate-y-0.5 hover:border-brand/40 hover:shadow-[0_10px_24px_-14px_rgba(38,70,60,0.5)]"
              style={{ '--i': i + 2 } as React.CSSProperties}
            >
              <Icon size={20} strokeWidth={1.6} className="text-brand transition-transform duration-300 group-hover:scale-110" />
              <span className="text-[16px] font-medium text-ink transition-colors group-hover:text-brand">{title}</span>
              <span className="text-[13px] leading-snug text-muted">{text}</span>
            </button>
          ))}
        </section>

        <section>
          {/* header, filters and search sit on the raw canvas, split only by hairlines */}
          <div className="flex flex-wrap items-center gap-x-5 gap-y-2 border-b border-line pb-3">
            <h2 className="text-[22px] font-medium text-ink">Let’s pick up where you left off</h2>
            <div className="flex items-center gap-4 border-l border-line pl-5 text-[15px]">
              {SESSION_TABS.map(({ label, kind }) => (
                <button
                  key={label}
                  onClick={() => setTab(kind)}
                  aria-pressed={tab === kind}
                  className={tab === kind ? 'font-medium text-brand' : 'text-muted hover:text-ink'}
                >
                  {label}
                </button>
              ))}
            </div>
            <label className="ml-auto flex items-center gap-2 border-l border-line pl-5 text-muted">
              <Search size={15} />
              <input
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Search"
                className="w-32 bg-transparent text-[15px] text-ink outline-none placeholder:text-muted/70"
              />
            </label>
          </div>

          <div className="mt-6 grid grid-cols-3 gap-x-8 gap-y-7">
            {shown.map((c, i) => (
              <button key={c.id} onClick={() => onOpen(c.id)} className="rise-in group text-left" style={{ '--i': Math.min(i, 8) } as React.CSSProperties}>
                <div className="glass-preview grid h-[132px] place-items-center overflow-hidden text-brand/60 transition duration-300 group-hover:-translate-y-0.5 group-hover:border-brand/40 group-hover:text-brand group-hover:shadow-[0_10px_24px_-14px_rgba(38,70,60,0.5)] [&_img]:transition-transform [&_img]:duration-500 group-hover:[&_img]:scale-[1.04]">
                  {c.vehicle && c.kind !== 'chat' ? (
                    <Photo id={c.vehicle} name={c.vehicle} shape="wide" className="h-full w-full" />
                  ) : kindOf(c) === 'Conversations' ? <MessageSquare size={26} strokeWidth={1.5} /> : <Rocket size={26} strokeWidth={1.5} />}
                </div>
                <div className="mt-2.5 truncate text-[17px] leading-tight text-ink transition-colors group-hover:text-brand">{c.name}</div>
                <div className="mt-0.5 text-[12px] text-muted">{singular[kindOf(c)]} · {day(c.updated_at)}</div>
              </button>
            ))}
            {shown.length === 0 && (
              <div className="col-span-3 flex flex-col items-center py-10 text-center">
                <Rocket size={26} strokeWidth={1.4} className="text-brand/60" />
                <p className="mt-3 text-[16px] text-ink">{sessions.length === 0 ? 'Nothing here yet' : 'No matches'}</p>
                <p className="mt-1 text-[14px] text-muted">{sessions.length === 0 ? 'Your analyses, comparisons and conversations will appear here.' : 'Try another word or tab.'}</p>
                {sessions.length === 0 ? (
                  <button onClick={onStart} className="mt-4 text-[15px] text-brand hover:underline hover:underline-offset-4">Start your first operation</button>
                ) : (
                  <button onClick={() => { setQuery(''); setTab(null) }} className="mt-4 text-[15px] text-brand hover:underline hover:underline-offset-4">Show everything</button>
                )}
              </div>
            )}
          </div>
        </section>
      </div>
    </main>
  )
}
