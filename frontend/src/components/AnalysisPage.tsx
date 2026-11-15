import { useEffect, useLayoutEffect, useRef, useState, type ReactNode } from 'react'
import { runScenario, type AssistantAction, type AssistantScenario, type Change, type Finding, type ScenarioResult, type Severity } from '../api'
import type { Chat } from '../lib/chat'
import { formatDate } from '../lib/input'
import { useSettings, type Level } from '../lib/settings'
import { useAssistant } from '../lib/useAssistant'
import { scenarioB, type Scenario, type WizardState } from '../types'
import AssistantPanel from './AssistantPanel'
import { ChevronDown } from 'lucide-react'
import Decision from './Decision'
import ResultVisuals from './ResultVisuals'
import { downloadReport } from '../lib/report'
import TimeField from './wizard/TimeField'

const LIMIT_WORDS = { slope: 'ground too steep', comms: 'no line of sight to Earth', power: 'no sunlight', thermal: 'a long night with no heat source', energy: 'a flat battery' } as const
const TERRAIN: Record<string, string> = {
  crater_floor_or_basin: 'Crater floor / basin',
  summit_or_massif: 'Summit / massif',
  slope_or_rim: 'Slope or rim',
  plain_or_gentle_terrain: 'Plain / gentle terrain',
}
const DOT: Record<Severity, string> = { critical: 'bg-red-600', warning: 'bg-amber-500', info: 'bg-muted/50' }

const fmtUtc = (iso: string | null) => {
  if (!iso) return '—'
  const [d, t] = iso.replace('Z', '').split('T')
  return `${formatDate(d)} ${t} UTC`
}
const fmtHours = (h: number) => (h < 48 ? `${h.toFixed(0)} h` : `${(h / 24).toFixed(1)} days`)
const pct = (f: number) => `${(100 * f).toFixed(0)}%`

/** A titled block. With `glance`, it starts folded: the title row shows the one number that matters and opens the rest. */
function Section({ title, children, glance }: { title: string; children: ReactNode; glance?: ReactNode }) {
  const [open, setOpen] = useState(false)
  if (glance === undefined) {
    return (
      <section className="border-t border-line py-5">
        <h3 className="mb-2 text-[13px] uppercase tracking-wide text-muted">{title}</h3>
        {children}
      </section>
    )
  }
  return (
    <section className="border-t border-line">
      <button onClick={() => setOpen((o) => !o)} aria-expanded={open} className="grain group -mx-2 flex w-[calc(100%+16px)] items-center gap-4 rounded-[8px] px-2 py-4 text-left">
        <h3 className="text-[13px] uppercase tracking-wide text-muted transition-colors group-hover:text-brand">{title}</h3>
        <span className={`ml-auto truncate text-[15px] text-ink transition-opacity duration-200 ${open ? 'opacity-0' : 'opacity-100'}`}>{glance}</span>
        <ChevronDown size={16} className={`shrink-0 text-muted transition-transform duration-300 group-hover:text-brand ${open ? 'rotate-180' : ''}`} />
      </button>
      <div className={`grid transition-all duration-300 ease-in-out ${open ? 'grid-rows-[1fr] pb-5 opacity-100' : 'grid-rows-[0fr] opacity-0'}`}>
        <div className="min-h-0 overflow-hidden">{children}</div>
      </div>
    </section>
  )
}

function Row({ k, v, warn }: { k: string; v: ReactNode; warn?: boolean }) {
  return (
    <div className="flex items-baseline justify-between gap-6 py-1">
      <span className="text-[15px] text-muted">{k}</span>
      <span className={`text-right text-[16px] ${warn ? 'text-amber-700' : 'text-ink'}`}>{v}</span>
    </div>
  )
}

function Findings({ items }: { items: Finding[] }) {
  return (
    <ul className="space-y-2">
      {items.map((f) => (
        <li key={f.message} className="flex gap-3 text-[15px] leading-snug text-ink">
          <span className={`mt-[7px] h-2 w-2 shrink-0 rounded-full ${DOT[f.severity]}`} aria-label={f.severity} />
          {f.message}
        </li>
      ))}
    </ul>
  )
}

function nextText(change: Change, nowVisible: boolean, sun: boolean) {
  if (!change) return 'No change in 30 days'
  const what = sun ? (nowVisible ? 'Sunset' : 'Sunrise') : nowVisible ? 'Loss of signal' : 'Contact'
  return `${what} in ${fmtHours(change.in_hours)} · ${fmtUtc(change.utc)}`
}

type Parts = Record<'label' | 'verdict' | 'decision' | 'visuals' | 'plain' | 'terrain' | 'sun' | 'earth' | 'coping' | 'warnings', ReactNode>
type BaseParts = Omit<Parts, 'visuals'>
const ORDER: (keyof Parts)[] = ['label', 'verdict', 'decision', 'visuals', 'plain', 'terrain', 'sun', 'earth', 'coping', 'warnings']

const PLAIN_LIMIT = {
  slope: 'the ground is too steep for this vehicle',
  comms: 'Earth can never be seen from here, so there is no radio link',
  power: 'there is no sunlight to recharge it',
  thermal: 'the nights are long and cold, and it has no known heater',
  energy: 'its battery would run out too early',
} as const
const plainVerdict = (i: number) => (i >= 60 ? 'Good' : i >= 40 ? 'Fair' : i >= 20 ? 'Weak' : 'Very weak')
const plainDays = (h: number) => (h < 36 ? `${h.toFixed(0)} hours` : `${(h / 24).toFixed(1)} days`)

/** The same result for someone who does not want the technical picture: one verdict, a few plain sentences, nothing to decode. */
function buildSimple(r: ScenarioResult, vehicleName: string, label?: string): BaseParts {
  const { assessment: a, query: q, terrain: t } = r
  const w = q.window
  const idx = a.vehicle_index
  const relay = a.checks.comms?.mode === 'relay'
  const tilt = a.site_metrics.mean_slope_deg
  const sunUp = q.now.sun.state === 'visible'
  const earthUp = q.now.earth.state === 'visible'
  const lim = a.limiting_factor
  const darkDays = w.longest_darkness.hours / 24
  const nextSun = q.next.sun_change
  const P = ({ children }: { children: ReactNode }) => <p className="text-[16px] leading-relaxed text-ink">{children}</p>
  const Sec = ({ title, children }: { title: string; children: ReactNode }) => (
    <Section title={title}>
      <div className="space-y-2">{children}</div>
    </Section>
  )

  const overall =
    idx >= 60
      ? `A good match: ${vehicleName} should do well here.`
      : idx >= 40
        ? `A fair match: ${vehicleName} can work here, but with some difficulties.`
        : idx >= 20
          ? `A weak match: ${vehicleName} would struggle here.`
          : `A very hard place for ${vehicleName}.`
  const terrain: Record<string, string> = {
    crater_floor_or_basin: 'The spot is on the floor of a crater or basin. Low ground like this is often shaded by the rim around it, so it can be dark and very cold.',
    summit_or_massif: 'The spot is on a high summit or mountain. High ground tends to catch the Sun for longer, which is good for power.',
    slope_or_rim: 'The spot is on a slope or a crater rim. Rims often get a lot of sunlight, but the ground can be tilted.',
    plain_or_gentle_terrain: 'The spot is on fairly open, gentle ground.',
  }
  return {
    label: label ? <div className="truncate text-[13px] uppercase tracking-wide text-brand" title={label}>{label}</div> : null,
    verdict: (
      <div className="pb-4">
        <div className="truncate text-[13px] uppercase tracking-wide text-muted" title={vehicleName}>How well suited is {vehicleName}?</div>
        <div className="flex items-baseline gap-4">
          <span className="text-[56px] font-medium leading-none text-brand tabular-nums">{idx.toFixed(0)}<span className="text-[22px] text-muted"> / 100</span></span>
          <span className="text-[24px] text-ink">{plainVerdict(idx)}</span>
        </div>
        <p className="mt-2 text-[17px] leading-snug text-ink">{overall}</p>
        {lim && <p className="mt-1.5 text-[15px] leading-snug text-muted">The biggest problem: {PLAIN_LIMIT[lim]}.</p>}
      </div>
    ),
    decision: <Decision a={a} q={q} level="simple" />,
    plain: (
      <Sec title="What this score means">
        <P>The score runs from 0 to 100 and tells you how comfortably this vehicle could live at this place for 30 days from the start date. Above 60 is good, 40 to 60 is fair, and below 20 means it would probably not make it.</P>
        <P>We looked at three things: how much sunlight there is for power, whether the vehicle can talk to Earth, and whether the ground is safe to land on.</P>
      </Sec>
    ),
    sun: (
      <Sec title="Sunlight">
        <P>Over the 30 days, the Sun shines here about {pct(w.sunlit_fraction)} of the time. Sunlight is the vehicle’s power and warmth, so more is better.</P>
        <P>
          The longest stretch without sunlight lasts {plainDays(w.longest_darkness.hours)}.{' '}
          {darkDays >= 8 ? 'That is a very long night. A vehicle has to survive it on its battery and keep itself warm, which is the hardest part of being on the Moon.' : darkDays >= 2 ? 'That is a long night, so the battery has to carry the vehicle through it.' : 'That is short, so the battery only has to last a little while.'}
        </P>
        <P>
          On the start date the Sun is {sunUp ? 'up' : 'below the horizon'}
          {nextSun ? `, and it ${sunUp ? 'sets' : 'rises'} in about ${plainDays(nextSun.in_hours)}` : ''}.
        </P>
      </Sec>
    ),
    earth: (
      <Sec title="Contact with Earth">
        {relay ? (
          <P>This vehicle talks to Earth through a relay satellite, so Earth being out of sight from the ground does not matter.</P>
        ) : w.dte_fraction === 0 ? (
          <P>Earth can never be seen from this spot, so the vehicle cannot talk to Earth directly. That is a serious problem for a mission.</P>
        ) : (
          <>
            <P>Earth can be seen about {pct(w.dte_fraction)} of the time, and the vehicle can only send and receive directly while Earth is above the horizon.</P>
            <P>The longest time without contact is {plainDays(w.longest_dte_outage.hours)}{w.longest_dte_outage.hours >= 72 ? ', so the vehicle would have to work on its own for a long time.' : '.'} On the start date Earth is {earthUp ? 'in view' : 'out of view'}.</P>
          </>
        )}
      </Sec>
    ),
    terrain: (
      <Sec title="The ground">
        <P>{terrain[t.class] ?? 'The spot is on ordinary lunar ground.'}</P>
        <P>
          The ground around the landing point is {tilt < 3 ? 'fairly flat' : tilt < 8 ? 'gently sloped' : 'steep'} (an average tilt of {tilt.toFixed(0)}°), and {pct(a.site_metrics.safe_area_fraction)} of the landing area is within what this vehicle can handle.
        </P>
      </Sec>
    ),
    coping: (
      <Sec title={`Can ${vehicleName} last 30 days?`}>
        <P>
          {lim
            ? `${idx < 40 ? 'Probably not without help' : 'It can, but there is a risk'}: ${PLAIN_LIMIT[lim]}. This is what holds the score down; without it the score would be ${a.weighted_index.toFixed(0)}.`
            : 'Nothing in our checks stops it from lasting the whole 30 days.'}
        </P>
        {a.data_quality.inputs_unknown > 0 && (
          <p className="text-[13px] leading-snug text-muted">Some details about this vehicle are not published. We never guess them, so the real result could be a little better or worse.</p>
        )}
      </Sec>
    ),
    warnings: (
      <Sec title="Want to dig deeper?">
        <p className="text-[15px] leading-relaxed text-muted">Ask the assistant below, for example “How can we shift the date?” or “Which vehicle works best here?”</p>
      </Sec>
    ),
  }
}

/** One scenario split into blocks, so a comparison can lay A and B out row by row and keep them aligned. */
function buildParts(r: ScenarioResult, vehicleName: string, s: Scenario, label?: string, level: Level = 'advanced'): Parts {
  const base = level === 'simple' ? buildSimple(r, vehicleName, label) : buildAdvanced(r, vehicleName, label)
  return { ...base, visuals: <ResultVisuals scenario={s} result={r} /> }
}

function buildAdvanced(r: ScenarioResult, vehicleName: string, label?: string): BaseParts {
  const { assessment: a, query: q, terrain: t } = r
  const sun = q.now.sun
  const earth = q.now.earth
  const relay = a.checks.comms?.mode === 'relay'
  const warnings = q.findings.filter((f) => f.severity !== 'info')
  const headline = a.findings[0]

  const sunNow =
    sun.state === 'visible'
      ? `Up, ${sun.elevation_deg.toFixed(1)}° above the horizon`
      : sun.state === 'terrain_blocked'
        ? `Hidden behind terrain ${sun.blocker_distance_km?.toFixed(1)} km away`
        : `Below the local horizon (${sun.elevation_deg.toFixed(1)}°)`
  const earthNow =
    earth.state === 'visible'
      ? `In view, ${earth.elevation_deg.toFixed(1)}° elevation`
      : earth.state === 'terrain_blocked'
        ? `Hidden behind terrain ${earth.blocker_distance_km?.toFixed(1)} km away`
        : `Below the local horizon (${earth.elevation_deg.toFixed(1)}°)`
  const w = q.window

  return {
    plain: null,
    decision: <Decision a={a} q={q} level="advanced" />,
    label: label ? <div className="truncate text-[13px] uppercase tracking-wide text-brand" title={label}>{label}</div> : null,
    verdict: (
      <div>
        <div className="flex items-end justify-between gap-4 pb-4">
          <div className="min-w-0">
            <div className="truncate text-[13px] uppercase tracking-wide text-muted" title={vehicleName}>Suitability for {vehicleName}</div>
            <div className="text-[56px] font-medium leading-none text-brand tabular-nums">
              {a.vehicle_index.toFixed(0)}<span className="text-[22px] text-muted"> / 100</span>
            </div>
            {a.limiting_factor && (
              <div className="mt-1.5 text-[13px] text-muted">
                Held down by {LIMIT_WORDS[a.limiting_factor]}; without it {a.weighted_index.toFixed(0)}
              </div>
            )}
          </div>
          <div className="shrink-0 whitespace-nowrap text-right text-[22px] text-ink">{TERRAIN[t.class] ?? t.class}</div>
        </div>
        {headline && <div className="pb-4"><Findings items={[headline]} /></div>}
      </div>
    ),
    terrain: (
      <Section title="Terrain" glance={`${a.site_metrics.mean_slope_deg.toFixed(1)}° slope · ${t.site_height_m.toFixed(0)} m`}>
        <Row k="Elevation" v={`${t.site_height_m.toFixed(0)} m`} />
        <Row k="Relative to surroundings" v={`${t.relief_vs_surroundings_m > 0 ? '+' : ''}${t.relief_vs_surroundings_m.toFixed(0)} m`} />
        <Row k="Average terrain skyline" v={`${t.mean_horizon_deg.toFixed(1)}° above the horizon`} />
        <Row k="Landing area slope" v={`${a.site_metrics.mean_slope_deg.toFixed(1)}° mean, ${pct(a.site_metrics.safe_area_fraction)} within the vehicle limit`} />
      </Section>
    ),
    sun: (
      <Section title="Sun" glance={`${pct(w.sunlit_fraction)} sunlit · longest night ${fmtHours(w.longest_darkness.hours)}`}>
        <Row k="At start" v={sunNow} warn={sun.state !== 'visible'} />
        <Row k="Next change" v={nextText(q.next.sun_change, sun.state === 'visible', true)} />
        <Row k="Sunlight over 30 days" v={pct(w.sunlit_fraction)} warn={w.sunlit_fraction < 0.2} />
        <Row k="Longest darkness" v={`${fmtHours(w.longest_darkness.hours)}${w.longest_darkness.starts_utc ? ` · from ${fmtUtc(w.longest_darkness.starts_utc)}` : ''}`} warn={w.longest_darkness.hours >= 72} />
      </Section>
    ),
    earth: (
      <Section title="Earth link" glance={relay ? 'Via relay satellite' : `${pct(w.dte_fraction)} in view · longest gap ${fmtHours(w.longest_dte_outage.hours)}`}>
        {relay ? (
          <Row k="Mode" v="Via relay satellite; no direct link needed" />
        ) : (
          <>
            <Row k="At start" v={earthNow} warn={earth.state !== 'visible'} />
            <Row k="Next change" v={nextText(q.next.earth_change, earth.state === 'visible', false)} />
            <Row k="Direct link over 30 days" v={pct(w.dte_fraction)} warn={w.dte_fraction === 0} />
            <Row k="Longest outage" v={`${fmtHours(w.longest_dte_outage.hours)}${w.longest_dte_outage.starts_utc ? ` · from ${fmtUtc(w.longest_dte_outage.starts_utc)}` : ''}`} warn={w.longest_dte_outage.hours >= 24} />
          </>
        )}
      </Section>
    ),
    coping: (
      <Section title={`Can ${vehicleName} cope for 30 days?`} glance={`${a.findings.length} ${a.findings.length === 1 ? 'note' : 'notes'}`}>
        <Findings items={a.findings} />
        <p className="mt-3 text-[13px] text-muted">
          Vehicle data: {a.data_quality.inputs_verified} verified · {a.data_quality.inputs_unverified} from the project dataset · {a.data_quality.inputs_unknown} unknown.
          Unknown values are never guessed.
        </p>
      </Section>
    ),
    warnings:
      warnings.length > 0 ? (
        <Section title="Site warnings" glance={`${warnings.length} ${warnings.length === 1 ? 'warning' : 'warnings'}`}>
          <Findings items={warnings} />
        </Section>
      ) : null,
  }
}

type Load = { status: 'loading' } | { status: 'error'; message: string } | { status: 'done'; a: ScenarioResult; b: ScenarioResult | null }

const COMPARED = { date: 'dates', vehicleId: 'vehicles', lat: 'locations', lon: 'locations' } as const

/** Names of what differs between two scenarios, e.g. ['dates']. */
function differing(a: Scenario, b: Scenario): string[] {
  const keys = (Object.keys(COMPARED) as (keyof typeof COMPARED)[]).filter((k) => a[k] !== b[k])
  return [...new Set(keys.map((k) => COMPARED[k]))]
}

/** Rows are shared by A and B, so a longer text on one side never pushes the other out of line. */
/** One sentence on who is ahead and the one or two things that put it there. */
function compareLine(a: ScenarioResult, b: ScenarioResult) {
  const ia = a.assessment.vehicle_index
  const ib = b.assessment.vehicle_index
  if (Math.abs(ia - ib) < 3) return { lead: `A and B are about even (${ia.toFixed(0)} and ${ib.toFixed(0)}).`, why: '' }
  const [win, lose, W, L] = ia > ib ? (['A', 'B', a, b] as const) : (['B', 'A', b, a] as const)
  const w = W.query.window
  const l = L.query.window
  const sw = W.assessment.site_metrics.safe_area_fraction
  const sl = L.assessment.site_metrics.safe_area_fraction
  const reasons: { weight: number; text: string }[] = [
    { weight: (w.sunlit_fraction - l.sunlit_fraction) / 0.3, text: `more sunlight (${pct(w.sunlit_fraction)} against ${pct(l.sunlit_fraction)})` },
    { weight: (w.dte_fraction - l.dte_fraction) / 0.3, text: `Earth in view more often (${pct(w.dte_fraction)} against ${pct(l.dte_fraction)})` },
    { weight: (l.longest_darkness.hours - w.longest_darkness.hours) / 168, text: `a shorter longest night (${fmtHours(w.longest_darkness.hours)} against ${fmtHours(l.longest_darkness.hours)})` },
    { weight: (sw - sl) / 0.3, text: `flatter ground (${pct(sw)} of the landing area within the slope limit against ${pct(sl)})` },
    {
      weight: !W.assessment.limiting_factor && L.assessment.limiting_factor ? 2 : 0,
      text: `no hard limit holding it back (${lose} is held down by ${L.assessment.limiting_factor ? LIMIT_WORDS[L.assessment.limiting_factor] : ''})`,
    },
  ]
  const top = reasons.filter((r) => r.weight > 0.15).sort((x, y) => y.weight - x.weight).slice(0, 2).map((r) => r.text)
  return { lead: `${win} is ahead by ${Math.abs(ia - ib).toFixed(0)} points.`, why: top.length ? `Mainly ${top.join(', and ')}.` : '' }
}

const rise = (i: number) => ({ '--i': i }) as React.CSSProperties

/** Sticky offset for a side column: pinned to the top when it fits the window, otherwise pinned by its bottom edge so none of it is ever out of reach. */
function useStickyTop<T extends HTMLElement>() {
  const ref = useRef<T>(null)
  const [top, setTop] = useState(24)
  useLayoutEffect(() => {
    const el = ref.current
    if (!el) return
    const fit = () => setTop(Math.min(24, window.innerHeight - el.offsetHeight - 24))
    fit()
    const ro = new ResizeObserver(fit)
    ro.observe(el)
    window.addEventListener('resize', fit)
    return () => { ro.disconnect(); window.removeEventListener('resize', fit) }
  }, [])
  return { ref, top }
}

function Results({ a, b, sa, sb, nameA, nameB, labelA, labelB, level }: { a: ScenarioResult; b: ScenarioResult | null; sa: Scenario; sb: Scenario | null; nameA: string; nameB: string; labelA?: string; labelB?: string; level: Level }) {
  const pa = buildParts(a, nameA, sa, labelA, level)
  const pb = b && sb ? buildParts(b, nameB, sb, labelB, level) : null
  const side = useStickyTop<HTMLDivElement>()
  if (!pb || !b) {
    // one scenario: the text on the left, the pictures in the space on the right
    return (
      <div className="mt-6 grid grid-cols-[minmax(0,600px)_minmax(0,1fr)] gap-x-14 max-[1279px]:grid-cols-1">
        <div className="min-w-0">{ORDER.filter((id) => id !== 'visuals' && pa[id]).map((id, i) => <div key={id} className="rise-in" style={rise(i)}>{pa[id]}</div>)}</div>
        <div ref={side.ref} className="rise-in min-w-0 self-start min-[1280px]:sticky" style={{ ...rise(2), top: side.top }}><ResultVisuals scenario={sa} result={a} side /></div>
      </div>
    )
  }
  const line = compareLine(a, b)
  return (
    <div className="mt-6">
      <div className="rise-in mb-2 border-b border-line pb-5">
        <p className="text-[22px] font-medium text-strong">{line.lead}</p>
        {line.why && <p className="mt-1 max-w-[860px] text-[16px] leading-relaxed text-muted">{line.why}</p>}
      </div>
      {ORDER.filter((id) => pa[id] || pb[id]).map((id, i) => (
        <div key={id} className="rise-in grid grid-cols-2 gap-12" style={rise(i + 1)}>
          <div className="min-w-0">{pa[id]}</div>
          <div className="min-w-0">{pb[id]}</div>
        </div>
      ))}
    </div>
  )
}

export type Open360 = { lat: number; lon: number; date: string; hour: number }

const asAssistant = (s: Scenario): AssistantScenario => ({ date: s.date, vehicleId: s.vehicleId, lat: Number(s.lat), lon: Number(s.lon) })
const FLOAT_KEY = 'assistant-floating'
const wasFloating = () => {
  try {
    return localStorage.getItem(FLOAT_KEY) === '1'
  } catch {
    return false
  }
}

type PageProps = {
  state: WizardState
  vehicleNames: Record<string, string>
  initialChat: Chat
  onChat: (c: Chat) => void
  onEdit: () => void
  onApply: (a: AssistantAction) => void
  onOpen360: (t: Open360) => void
}

export default function AnalysisPage({ state, vehicleNames, initialChat, onChat, onEdit, onApply, onOpen360 }: PageProps) {
  const [load, setLoad] = useState<Load>({ status: 'loading' })
  const [floating, setFloating] = useState(wasFloating)
  const [hour, setHour] = useState(0)
  const { level } = useSettings()
  const sb = scenarioB(state)

  const assistant = useAssistant({
    scenarios: () => [asAssistant(state.a), ...(sb ? [asAssistant(sb)] : [])],
    initial: initialChat,
    onChange: onChat,
    onApply,
    level,
  })

  useEffect(() => {
    let live = true
    setLoad({ status: 'loading' })
    Promise.all([runScenario(state.a), sb ? runScenario(sb) : Promise.resolve(null)])
      .then(([a, b]) => live && setLoad({ status: 'done', a, b }))
      .catch((e: Error) => live && setLoad({ status: 'error', message: e.message }))
    return () => {
      live = false
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps -- rerun only when the wizard answers change
  }, [JSON.stringify(state.a), JSON.stringify(sb)])

  const name = (s: Scenario) => vehicleNames[s.vehicleId] ?? s.vehicleId
  const describe = (s: Scenario) => `${name(s)} · ${formatDate(s.date)} · ${s.lat}, ${s.lon}`
  const changed = sb ? differing(state.a, sb) : []
  const scenarios: { label: 'A' | 'B'; s: Scenario }[] = [{ label: 'A', s: state.a }, ...(sb ? [{ label: 'B' as const, s: sb }] : [])]

  const toggleFloat = () => {
    setFloating((f) => {
      try {
        localStorage.setItem(FLOAT_KEY, f ? '0' : '1')
      } catch {
        /* not remembered */
      }
      return !f
    })
  }
  const focusAssistant = () => {
    document.getElementById('assistant-section')?.scrollIntoView({ behavior: 'smooth', block: 'center' })
    setTimeout(() => document.getElementById('assistant-input')?.focus(), 350)
  }

  return (
    <main className="min-w-0 flex-1 overflow-y-auto">
      <div className="mx-auto max-w-[1280px] px-10 pb-16 pt-10">
        <div className="flex items-start justify-between gap-6 border-b border-line pb-4">
          <div>
            <h1 className="text-[32px] font-medium text-brand">Analysis</h1>
            <p className="mt-1 text-[15px] text-muted">
              {sb ? (
                <>Comparing {changed.join(', ')} · 30 days from the start date</>
              ) : (
                <>{describe(state.a)} · 30 days from the start date</>
              )}
            </p>
          </div>
          <div className="flex shrink-0 items-center gap-6">
            {load.status === 'done' && (
              <button
                onClick={() => downloadReport(scenarios.map(({ label, s }, i) => ({ label, scenario: s, vehicle: name(s), result: i === 0 ? load.a : (load.b as ScenarioResult) })))}
                className="text-[15px] text-brand hover:underline hover:underline-offset-4"
              >
                Download report
              </button>
            )}
            <button onClick={focusAssistant} className="text-[15px] text-brand hover:underline hover:underline-offset-4">Ask the assistant</button>
            <button onClick={onEdit} className="text-[15px] text-muted hover:text-brand">Edit inputs</button>
          </div>
        </div>

        {load.status === 'loading' && (
          <div className="mt-6 grid grid-cols-[minmax(0,600px)_minmax(0,1fr)] gap-x-14 max-[1279px]:grid-cols-1" aria-busy="true">
            <div>
              <p className="text-[14px] text-muted">Running exact physics for 30 days of Sun and Earth positions…</p>
              <div className="skeleton mt-5 h-3 w-40" />
              <div className="skeleton mt-3 h-14 w-48" />
              <div className="skeleton mt-8 h-3 w-24" />
              <div className="skeleton mt-3 h-7 w-72" />
              <div className="skeleton mt-3 h-3 w-full" />
              <div className="skeleton mt-2 h-3 w-5/6" />
              {[0, 1, 2].map((k) => <div key={k} className="skeleton mt-8 h-3 w-full" />)}
            </div>
            <div className="max-[1279px]:hidden">
              <div className="skeleton mt-10 h-3 w-36" />
              <div className="skeleton mt-4 h-2.5 w-full" />
              <div className="skeleton mt-10 h-3 w-40" />
              <div className="skeleton mt-4 h-5 w-full" />
              <div className="skeleton mt-2 h-5 w-full" />
              <div className="skeleton ml-auto mt-10 h-[200px] w-[200px]" />
            </div>
          </div>
        )}
        {load.status === 'error' && (
          <div className="py-16 text-center">
            <p className="text-amber-700">{load.message}</p>
            <button onClick={onEdit} className="mt-3 text-[15px] text-brand underline underline-offset-4">Back to the inputs</button>
          </div>
        )}
        {load.status === 'done' && <Results level={level} a={load.a} b={load.b} sa={state.a} sb={sb} nameA={name(state.a)} nameB={sb ? name(sb) : ''} labelA={load.b ? `A · ${describe(state.a)}` : undefined} labelB={sb ? `B · ${describe(sb)}` : undefined} />}

        {load.status === 'done' && (
          <section className="mt-12 border-t border-line pt-8" aria-label="360 degree view">
            <h3 className="text-[22px] font-medium text-brand">See the place in 360°</h3>
            <p className="mt-1 max-w-[640px] text-[15px] text-muted">Stand at the site and look around: the real terrain, with the Sun and Earth where they are at the moment you choose.</p>
            <div className="mt-5 flex flex-wrap items-end gap-x-6 gap-y-2">
              <div className="w-[190px]"><TimeField value={hour} onChange={setHour} /></div>
              <span className="pb-2 text-[13px] text-muted">UTC hour, on the start date of each scenario</span>
            </div>
            <ul className="mt-5 divide-y divide-line border-y border-line">
              {scenarios.map(({ label, s }) => (
                <li key={label} className="flex flex-wrap items-center justify-between gap-x-8 gap-y-2 py-4">
                  <div className="min-w-0">
                    {scenarios.length > 1 && <span className="mr-2 text-[13px] uppercase tracking-wide text-brand">{label}</span>}
                    <span className="text-[17px] text-ink">{s.lat}, {s.lon}</span>
                    <span className="ml-3 text-[14px] text-muted">{formatDate(s.date)} · {String(hour).padStart(2, '0')}:00 UTC</span>
                  </div>
                  <button onClick={() => onOpen360({ lat: Number(s.lat), lon: Number(s.lon), date: s.date, hour })} className="text-[16px] text-brand hover:underline hover:underline-offset-4">Open 360 View →</button>
                </li>
              ))}
            </ul>
          </section>
        )}

        <div id="assistant-section">
          <AssistantPanel
            lines={scenarios.map(({ label, s }) => ({ label, text: describe(s) }))}
            chat={assistant.chat}
            busy={assistant.busy}
            onSend={(q) => void assistant.send(q)}
            onApply={onApply}
            floating={floating}
            onToggleFloat={toggleFloat}
          />
        </div>
      </div>
    </main>
  )
}
