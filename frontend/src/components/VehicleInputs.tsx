import type { Field, VehicleProfile } from '../api'
import type { Spacecraft } from '../data/spacecrafts'
import { missingNote, type FieldKey } from '../lib/profileNotes'

const DOT = { verified: 'bg-brand', unverified: 'bg-amber-500', unknown: 'bg-muted/40' } as const
const DOT_LABEL = { verified: 'Verified from an agency source', unverified: 'From the project dataset, not verified', unknown: 'Not published' } as const

function Row({ label, field, fmt, craft, k }: { label: string; field: Field<unknown> | undefined; fmt: (v: never) => string; craft: Spacecraft; k: FieldKey }) {
  if (!field) return null
  const missing = field.value === null ? missingNote(k, craft) : null
  return (
    <div className="py-2">
      <div className="flex items-baseline justify-between gap-4">
        <dt className="text-[14px] text-muted">{label}</dt>
        <dd className="flex items-center gap-2 text-[15px] text-ink">
          {missing ? <span className="text-muted">{missing.label}</span> : fmt(field.value as never)}
          <span title={DOT_LABEL[field.status]} className={`h-2 w-2 rounded-full ${DOT[field.status]}`} />
        </dd>
      </div>
      {field.note && field.value !== null && <p className="mt-0.5 text-[12px] leading-snug text-muted">{field.note.slice(0, 260)}</p>}
      {missing?.note && <p className="mt-0.5 text-[12px] leading-snug text-muted">{missing.note}</p>}
    </div>
  )
}

/**
 * What the landing analysis actually uses for this spacecraft. The spec sheet below lists the dataset as published; the analysis
 * counts only what matters on the surface (for example the landing stage's batteries) and never fills a gap with a guess.
 */
export default function VehicleInputs({ profile, craft, ready }: { profile: VehicleProfile | null; craft: Spacecraft; ready: boolean }) {
  if (!ready) return null
  if (!profile)
    return (
      <p className="mt-7 border-t border-line pt-5 text-[14px] leading-relaxed text-muted">
        Not part of the landing analysis: this spacecraft does not land on the Moon (orbiter, fly-by, impactor or crewed mission without a landing), so New Operation does not offer it.
      </p>
    )
  return (
    <section className="mt-10 border-t border-line pt-5" aria-label="Values used in the landing analysis">
      <h3 className="text-[13px] uppercase tracking-wide text-muted">Used in the landing analysis</h3>
      <p className="mt-1 max-w-[640px] text-[13px] leading-relaxed text-muted">
        These can differ from the specifications below: the analysis counts only what matters on the surface, such as the landing stage's batteries, and an unknown value stays unknown.
      </p>
      <dl className="mt-3 divide-y divide-line">
        <Row craft={craft} k="battery_wh" label="Battery" field={profile.battery_wh} fmt={(v: number) => `${v.toLocaleString('en')} Wh`} />
        <Row craft={craft} k="solar_w" label="Solar array" field={profile.solar_w} fmt={(v: number) => (v === 0 ? 'None (battery only)' : `${v} W`)} />
        <Row craft={craft} k="survival_power_w" label="Survival load" field={profile.survival_power_w} fmt={(v: number) => `${v} W`} />
        <Row craft={craft} k="payload_power_w" label="Payload load" field={profile.payload_power_w} fmt={(v: number) => `${v} W`} />
        <Row craft={craft} k="max_slope_deg" label="Slope limit" field={profile.max_slope_deg} fmt={(v: number) => `${v}°`} />
        <Row craft={craft} k="radioisotope_heat" label="Radioisotope heat" field={profile.radioisotope_heat} fmt={(v: boolean) => (v ? 'Yes' : 'No')} />
        <Row craft={craft} k="comms" label="Communications" field={profile.comms} fmt={(v: string) => (v === 'relay' ? 'Via relay satellite' : 'Direct to Earth')} />
      </dl>
    </section>
  )
}
