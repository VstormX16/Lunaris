import type { QueryResult, RiskMode, VehicleAssessment } from '../api'
import type { Level } from '../lib/settings'

type Kind = 'go' | 'conditional' | 'weak' | 'stop'

const LIMIT = {
  slope: 'the ground is too steep for this vehicle',
  comms: 'Earth can never be seen from here, so there is no radio link',
  power: 'there is no sunlight to recharge it',
  thermal: 'the nights are long and cold, and it has no known heater',
  energy: 'its battery would run out too early',
} as const
const FIX = {
  slope: 'Pick a flatter landing place.',
  comms: 'Pick a place with a view of Earth, or a vehicle that uses a relay satellite.',
  power: 'Try another start date, or a different place with more sunlight.',
  thermal: 'Try another start date with a shorter night, or a vehicle with a heater.',
  energy: 'Try another start date, or a vehicle with a bigger battery.',
} as const

const pct = (f: number) => `${(100 * f).toFixed(0)}%`
const span = (h: number) => (h < 48 ? `${h.toFixed(0)} hours` : `${(h / 24).toFixed(1)} days`)

/** The call for a result: the server's (score + simulated success probability), or, for results saved before the server
 *  made it, the older score-only reading below. */
export function verdictOf(a: VehicleAssessment, q: QueryResult) {
  return a.verdict ?? { ...decideFromScore(a, q), probability_line: null, rule: '' }
}

/** Score-only call, kept for results saved before the server made the call. The thresholds match the assistant's wording (60 good, 40 fair, 20 weak). */
function decideFromScore(a: VehicleAssessment, q: QueryResult) {
  const idx = a.vehicle_index
  const lim = a.limiting_factor
  const w = q.window
  const relay = a.checks.comms?.mode === 'relay'
  const safe = a.site_metrics.safe_area_fraction

  const good: string[] = []
  if (w.sunlit_fraction >= 0.6) good.push(`Plenty of sunlight (${pct(w.sunlit_fraction)} of the month)`)
  if (relay) good.push('Talks to Earth through a relay satellite, so Earth being out of view does not matter')
  else if (w.dte_fraction >= 0.7) good.push(`Earth is in view most of the time (${pct(w.dte_fraction)})`)
  if (safe >= 0.9) good.push(`The landing area is within the vehicle's slope limit (${pct(safe)})`)
  if (w.longest_darkness.hours < 72) good.push(`Short nights (longest ${span(w.longest_darkness.hours)})`)

  const risks: string[] = []
  if (lim) risks.push(LIMIT[lim][0].toUpperCase() + LIMIT[lim].slice(1))
  if (w.sunlit_fraction < 0.3) risks.push(`Little sunlight (${pct(w.sunlit_fraction)} of the month)`)
  if (w.longest_darkness.hours >= 168) risks.push(`A very long night: ${span(w.longest_darkness.hours)} without sunlight`)
  if (!relay && w.dte_fraction < 0.4) risks.push(`Earth is often out of view (visible ${pct(w.dte_fraction)} of the time)`)
  if (!relay && w.longest_dte_outage.hours >= 72) risks.push(`Up to ${span(w.longest_dte_outage.hours)} without contact with Earth`)
  if (safe < 0.9) risks.push(`Only ${pct(safe)} of the landing area is within the vehicle's slope limit`)
  if (a.data_quality.inputs_unknown >= 3) risks.push('Several details about this vehicle are not published, so the score is less certain')

  let kind: Kind = idx >= 60 ? (lim ? 'conditional' : 'go') : idx >= 40 ? 'conditional' : idx >= 20 ? 'weak' : 'stop'
  if (kind === 'go' && risks.length >= 2) kind = 'conditional' // a high score with several real warnings is not a clean go
  const title = { go: 'Go: recommended', conditional: 'Possible, with risks', weak: 'Not recommended as it is', stop: 'Do not proceed' }[kind]
  const summary = {
    go: risks.length ? 'This vehicle should do well at this place and date. One thing to keep an eye on is listed below.' : 'This vehicle should do well at this place and date. Nothing in the checks stands in the way.',
    conditional: lim ? `It can work, but one thing needs attention: ${LIMIT[lim]}.` : 'It can work here, but it will be a hard month. Read the risks below before you commit.',
    weak: lim ? `As it stands this is a poor choice: ${LIMIT[lim]}.` : 'As it stands this is a poor choice. A different date, place or vehicle would be better.',
    stop: lim ? `This would very likely fail: ${LIMIT[lim]}.` : 'This would very likely fail. Choose a different date, place or vehicle.',
  }[kind]

  const next = kind === 'go' ? ['Compare it with another date, place or vehicle to be sure.'] : lim ? [FIX[lim]] : ['Try another start date or another place and compare the scores.']
  return { kind, title, summary, good: good.slice(0, 4), risks: risks.slice(0, 4), next }
}

const TONE: Record<Kind, string> = { go: 'text-brand', conditional: 'text-amber-700', weak: 'text-amber-700', stop: 'text-red-700' }

export const MODE_LABEL: Record<RiskMode, string> = { landing: 'Unsafe touchdown', cold: 'Froze in a night', power: 'Too little power', comms: 'Too little contact' }

export default function Decision({ a, q, level }: { a: VehicleAssessment; q: QueryResult; level: Level }) {
  const d = verdictOf(a, q)
  const List = ({ title, items, dot }: { title: string; items: string[]; dot: string }) =>
    items.length === 0 ? null : (
      <div>
        <div className="text-[13px] uppercase tracking-wide text-muted">{title}</div>
        <ul className="mt-1.5 space-y-1.5">
          {items.map((t) => (
            <li key={t} className="flex gap-3 text-[15px] leading-snug text-ink"><span className={`mt-[7px] h-2 w-2 shrink-0 rounded-full ${dot}`} />{t}</li>
          ))}
        </ul>
      </div>
    )
  return (
    <section className="border-t border-line py-5" aria-label="Verdict">
      <h3 className="mb-2 text-[13px] uppercase tracking-wide text-muted">Verdict</h3>
      <div className={`text-[26px] font-medium leading-tight ${TONE[d.kind]}`}>{d.title}</div>
      <p className="mt-1.5 text-[16px] leading-relaxed text-ink">{d.summary}</p>
      {d.probability_line && <p className="mt-1.5 text-[15px] leading-relaxed text-muted">{d.probability_line}</p>}
      <div className="mt-4 space-y-4">
        <List title="In its favour" items={d.good} dot="bg-brand" />
        <List title="Watch out for" items={d.risks} dot="bg-amber-500" />
        <List title="What you can do" items={d.next} dot="bg-muted/50" />
      </div>
      {level === 'advanced' && d.rule && <p className="mt-4 text-[12px] leading-snug text-muted">How the call is made: {d.rule}</p>}
      {level === 'advanced' && !d.rule && <p className="mt-4 text-[12px] leading-snug text-muted">How the call is made: 60 or more is recommended, 40 to 60 possible with risks, 20 to 40 not recommended, below 20 do not proceed. A hard limit (slope, Earth link, power, cold or battery) caps the score and shows up under Watch out for.</p>}
    </section>
  )
}
