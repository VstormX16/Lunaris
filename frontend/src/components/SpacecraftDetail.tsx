import { ChevronLeft, ChevronRight, X } from 'lucide-react'
import type { Spacecraft } from '../data/spacecrafts'
import { keyFacts } from '../lib/facts'
import { splitItems } from '../lib/text'
import type { VehicleProfile } from '../api'
import Photo from './Photo'
import VehicleInputs from './VehicleInputs'

type Row = [label: string, value: string]
type Section = { title: string; rows: Row[] }

const NOT_PUBLISHED = 'Yayımlanmamış'
const CHECK_TITLE = { nasa: 'Checked against NASA', wikipedia: 'Checked against a second source', unchecked: 'Not independently checked' } as const
const FIELD_LABEL: Record<string, string> = { massKg: 'Mass (kg)', massNote: 'Mass note', height: 'Height (m)', width: 'Width (m)', solarW: 'Solar array (W)', serviceThrustN: 'Service engine thrust (N)' }
/** A cell worth showing: not empty, not "Uygulanmaz" (not applicable) and not a bare zero. */
const shown = (v: string) => v !== '' && v !== '0' && v !== 'Uygulanmaz'

/** The dataset's columns, grouped the way a spec sheet reads. Rows with nothing to say are left out. */
function sectionsOf(c: Spacecraft): Section[] {
  const mass = c.massKg ? `${Number(c.massKg).toLocaleString('en')} kg` : ''
  const all: Section[] = [
    { title: 'Mass & Size', rows: [['Mass', mass], ['Mass includes', c.massNote], ['Dimensions', c.dimensions], ['Height (m)', c.height], ['Width / diameter (m)', c.width], ['Leg span (m)', c.legSpan], ['Other dimensions', c.otherDims]] },
    { title: 'Power System', rows: [['Power source', c.powerSource], ['Battery type', c.batteryType], ['Battery count', c.batteryCount], ['Battery capacity', c.batteryCapacity], ['Solar array / power generation', c.solarW], ['RTG (W)', c.rtgW], ['Heating', c.heatSource], ['Survival load', c.survivalW], ['Science payload load', c.payloadW], ['Voltage (V)', c.voltage]] },
    { title: 'Wheels & Landing Gear', rows: [['Wheels', c.wheels], ['Wheel diameter (m)', c.wheelDiameter], ['Landing legs', c.legs], ['Landing mechanism', c.landingMechanism], ['Slope limit', c.slopeLimit], ['Top driving speed', c.speed]] },
    { title: 'Computer & Software', rows: [['Computer', c.computer], ['Clock (MHz)', c.clockMhz], ['Flight software', c.software], ['Radar & sensors', c.navSensors]] },
    { title: 'Scientific Payload', rows: [['Cameras', c.cameras], ['Instruments', c.instruments], ['Measurements / images', c.results]] },
    { title: 'Propulsion', rows: [['Main propulsion', c.propulsion], ['Landing engine thrust (N)', c.landerThrustN], ['Ascent engine thrust (N)', c.ascentThrustN], ['Service engine thrust (N)', c.serviceThrustN], ['RCS thrusters', c.rcs], ['Other', c.otherPropulsion]] },
    { title: 'Communications', rows: [['Antennas & radio', c.antenna], ['Band', c.band], ['Data rate / frequency', c.dataRate], ['Details', c.commDetails]] },
  ]
  return all.map((s) => ({ ...s, rows: s.rows.filter(([, v]) => shown(v)) })).filter((s) => s.rows.length > 0)
}

function Value({ text }: { text: string }) {
  if (text === NOT_PUBLISHED) return <span className="text-muted">Not published</span>
  const items = splitItems(text)
  if (items.length > 2 || text.length > 90) {
    return (
      <ul className="list-disc space-y-1 pl-5 marker:text-brand/60">
        {items.map((item, i) => (
          <li key={`${i}-${item}`}>{item}</li>
        ))}
      </ul>
    )
  }
  return <>{text.charAt(0).toLocaleUpperCase('tr') + text.slice(1)}</>
}

type Props = {
  craft: Spacecraft
  position: { index: number; total: number } | null
  prev: Spacecraft | null
  next: Spacecraft | null
  related: Spacecraft[]
  /** the landing analysis' own profile of this spacecraft (null: it has none) and whether the profiles have loaded */
  profile: VehicleProfile | null
  profilesReady: boolean
  onSelect: (id: string) => void
  onClose: () => void
}

/** Right-hand pane: a profile of one spacecraft, not a data dump. */
export default function SpacecraftDetail({ craft, position, prev, next, related, profile, profilesReady, onSelect, onClose }: Props) {
  const facts = keyFacts(craft)
  const sections = sectionsOf(craft)
  const host = (() => { try { return new URL(craft.sourceUrl).hostname } catch { return '' } })()

  return (
    <article key={craft.id} className="step-in pb-16">
      {/* hero: the photo is left alone; title and facts are set below it */}
      <div className="relative aspect-video overflow-hidden rounded-[12px] border border-line">
        <Photo id={craft.id} name={craft.name} shape="wide" iconSize={44} className="absolute inset-0 h-full w-full" />
        <button onClick={onClose} aria-label="Close details" className="absolute right-3 top-3 rounded-[8px] bg-white/85 p-2 text-ink transition hover:bg-white hover:text-brand">
          <X size={18} />
        </button>
      </div>
      <div className="mt-5">
        <div className="text-[14px] tabular-nums text-muted">{craft.year} · {craft.country}</div>
        <h2 className="mt-0.5 text-[40px] font-bold leading-tight text-strong">{craft.name}</h2>
        <div className="text-[15px] text-muted">{craft.type}</div>
      </div>

      {/* browse neighbours without going back to the list */}
      <nav className="mt-4 flex items-center justify-between text-[14px]" aria-label="Browse spacecraft">
        <button disabled={!prev} onClick={() => prev && onSelect(prev.id)} className="flex min-w-0 items-center gap-1 text-muted transition hover:text-brand disabled:invisible">
          <ChevronLeft size={16} /> <span className="truncate">{prev?.name}</span>
        </button>
        {position && <span className="shrink-0 tabular-nums text-muted">{position.index} of {position.total}</span>}
        <button disabled={!next} onClick={() => next && onSelect(next.id)} className="flex min-w-0 items-center gap-1 text-muted transition hover:text-brand disabled:invisible">
          <span className="truncate">{next?.name}</span> <ChevronRight size={16} />
        </button>
      </nav>

      {facts.length >= 2 && (
        <dl className="mt-6 grid grid-cols-2 gap-x-8 gap-y-5 border-y border-line py-6 sm:grid-cols-4">
          {facts.map((f) => (
            <div key={f.label} className="min-w-0">
              <dd className="truncate text-[28px] font-bold leading-none tabular-nums text-strong" title={f.value}>{f.value}</dd>
              <dt className="mt-1.5 text-[12px] uppercase tracking-wide text-muted">{f.label}</dt>
            </div>
          ))}
        </dl>
      )}

      {craft.outcome && (
        <blockquote className="mt-7 border-l-2 border-brand pl-5 text-[19px] leading-relaxed text-ink">{craft.outcome}</blockquote>
      )}

      <VehicleInputs profile={profile} craft={craft} ready={profilesReady} />

      <h3 className="mb-5 mt-10 text-[13px] uppercase tracking-wide text-muted">Specifications</h3>
      <div className="grid grid-cols-1 gap-x-10 gap-y-8 2xl:grid-cols-2">
        {sections.map((sec, i) => (
          <section key={sec.title} className="border-t border-line pt-4">
            {/* numbered like a printed spec sheet: type does the work an icon would */}
            <h4 className="mb-3 flex items-baseline gap-3 text-[16px] font-medium text-ink">
              <span className="text-[13px] font-normal tabular-nums text-brand">{String(i + 1).padStart(2, '0')}</span>
              {sec.title}
            </h4>
            <dl className="space-y-3">
              {sec.rows.map(([label, value]) => (
                <div key={label}>
                  <dt className="text-[12px] uppercase tracking-wide text-muted">{label}</dt>
                  <dd className="mt-0.5 text-[15px] leading-snug text-ink"><Value text={value} /></dd>
                </div>
              ))}
            </dl>
          </section>
        ))}
      </div>

      <section className="mt-10 border-t border-line pt-4">
        <h4 className="mb-2 text-[13px] uppercase tracking-wide text-muted">Source and check</h4>
        <p className="flex items-center gap-2 text-[15px] font-medium text-strong">
          <span className={`h-2.5 w-2.5 rounded-full ${craft.checkLevel === 'unchecked' ? 'bg-[#c9a227]' : 'bg-brand'}`} />
          {CHECK_TITLE[craft.checkLevel]}
        </p>
        <p className="mt-1.5 max-w-[760px] text-[14px] leading-relaxed text-muted">{craft.checkNote}</p>
        {craft.corrections.length > 0 && (
          <ul className="mt-3 max-w-[760px] space-y-2 text-[14px] leading-snug text-ink">
            {craft.corrections.map((c) => (
              <li key={c.field}>
                <span className="font-medium">{FIELD_LABEL[c.field] ?? c.field}:</span> <span className="text-muted line-through">{c.from}</span> → {c.to}
                <div className="text-[13px] text-muted">{c.why}</div>
              </li>
            ))}
          </ul>
        )}
        <p className="mt-3 text-[13px] text-muted">The other columns (computer, clock speed, software, antenna details) are not on these pages and stay as written in the dataset.</p>
        {craft.verification && <p className="mt-1 text-[13px] text-muted">Dataset note: {craft.verification}</p>}
        {craft.sourceUrl && (
          <a href={craft.sourceUrl} target="_blank" rel="noreferrer noopener" className="mt-2 inline-block break-all text-[14px] text-brand underline-offset-2 hover:underline">
            {host || craft.sourceUrl}
          </a>
        )}
      </section>

      {related.length > 0 && (
        <>
          <h3 className="mb-4 mt-12 text-[13px] uppercase tracking-wide text-muted">From the same era</h3>
          <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
            {related.map((c) => (
              <button key={c.id} onClick={() => onSelect(c.id)} className="group text-left">
                <Photo id={c.id} name={c.name} shape="wide" iconSize={22} className="aspect-video w-full rounded-[10px] border border-white/60 transition group-hover:brightness-110" />
                <div className="mt-2 truncate text-[15px] text-ink group-hover:text-brand">{c.name}</div>
                <div className="text-[12px] tabular-nums text-muted">{c.year}</div>
              </button>
            ))}
          </div>
        </>
      )}
    </article>
  )
}
