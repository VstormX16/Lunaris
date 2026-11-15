import { useMemo } from 'react'
import type { Field, VehicleProfile } from '../../api'
import { missingNote, type FieldKey } from '../../lib/profileNotes'
import Photo from '../Photo'
import { spacecrafts, type Spacecraft } from '../../data/spacecrafts'
import { customCraft } from '../../lib/customVehicle'
import { GlassButton, StepFooter } from './ui'

type Entry = { profile: VehicleProfile; craft: Spacecraft }

export function joinVehicles(vehicles: VehicleProfile[]): Entry[] {
  return vehicles
    .map((profile) => ({ profile, craft: spacecrafts.find((c) => c.id === profile.id) ?? (profile.custom ? customCraft(profile) : undefined) }))
    .filter((e): e is Entry => e.craft !== undefined)
    .sort((a, b) => a.craft.year - b.craft.year)
}

const DOT = { verified: 'bg-brand', unverified: 'bg-amber-500', unknown: 'bg-muted/40' } as const
const DOT_LABEL = { verified: 'Verified from an agency source', unverified: 'From the project dataset, not verified', unknown: 'Not published' } as const

function Spec({ label, field, fmt, craft, k }: { label: string; field: Field<unknown>; fmt: (v: never) => string; craft: Spacecraft; k: FieldKey }) {
  const missing = field.value === null ? missingNote(k, craft) : null
  return (
    <div className="py-1.5">
      <div className="flex items-baseline justify-between gap-4">
        <dt className="text-[14px] text-muted">{label}</dt>
        <dd className="flex items-center gap-2 text-[15px] text-ink">
          {missing ? <span className="text-muted">{missing.label}</span> : fmt(field.value as never)}
          <span title={DOT_LABEL[field.status]} className={`h-2 w-2 rounded-full ${DOT[field.status]}`} />
        </dd>
      </div>
      {missing?.note && <p className="mt-0.5 text-[12px] leading-snug text-muted">{missing.note}</p>}
    </div>
  )
}

function Preview({ entry }: { entry: Entry }) {
  const { profile: p, craft: c } = entry
  return (
    <div className="step-in" key={p.id}>
      <Photo id={p.id} name={c.name} shape="wide" className="aspect-[16/8] max-h-[34vh] w-full rounded-[10px] border border-white/60" />
      <h3 className="mt-4 text-[26px] font-medium leading-tight text-brand">{c.name}</h3>
      <div className="text-[14px] text-muted">{c.year} · {c.country} · {c.type}</div>
      <dl className="mt-3 divide-y divide-line border-t border-line">
        <div className="flex justify-between gap-4 py-1.5"><dt className="text-[14px] text-muted">Mass</dt><dd className="text-right text-[15px]">{c.massKg ? `${Number(c.massKg).toLocaleString('en')} kg` : '—'}</dd></div>
        <Spec craft={c} k="battery_wh" label="Battery" field={p.battery_wh} fmt={(v: number) => `${v.toLocaleString('en')} Wh`} />
        <Spec craft={c} k="solar_w" label="Solar array" field={p.solar_w} fmt={(v: number) => (v === 0 ? 'None (battery only)' : `${v} W`)} />
        {p.survival_power_w && <Spec craft={c} k="survival_power_w" label="Survival load" field={p.survival_power_w} fmt={(v: number) => `${v} W`} />}
        {p.payload_power_w && <Spec craft={c} k="payload_power_w" label="Payload load" field={p.payload_power_w} fmt={(v: number) => `${v} W`} />}
        <Spec craft={c} k="max_slope_deg" label="Slope limit" field={p.max_slope_deg} fmt={(v: number) => `${v}°`} />
        <Spec craft={c} k="radioisotope_heat" label="Radioisotope heat" field={p.radioisotope_heat} fmt={(v: boolean) => (v ? 'Yes' : 'No')} />
        <Spec craft={c} k="comms" label="Communications" field={p.comms} fmt={(v: string) => (v === 'relay' ? 'Via relay satellite' : 'Direct to Earth')} />
      </dl>
    </div>
  )
}

type Props = {
  vehicles: VehicleProfile[] | null
  error: string | null
  value: string
  onChange: (id: string) => void
  onBack: () => void
  onNext: () => void
}

export default function VehicleStep({ vehicles, error, value, onChange, onBack, onNext }: Props) {
  const entries = useMemo(() => (vehicles ? joinVehicles(vehicles) : []), [vehicles])
  const selected = entries.find((e) => e.profile.id === value)

  return (
    <>
      <div className="grid min-h-0 flex-1 grid-cols-[minmax(260px,340px)_1fr] gap-10 py-8">
        <div className="max-h-[calc(100vh-235px)] space-y-2 overflow-y-auto pb-2 pr-2">
          {error && <p className="text-[14px] text-amber-700">{error}</p>}
          {!vehicles && !error && <p className="text-muted">Loading vehicles…</p>}
          {entries.map(({ profile: p, craft: c }) => (
            <button
              key={p.id}
              onClick={() => onChange(p.id)}
              aria-pressed={p.id === value}
              className={`block w-full rounded-[10px] border px-4 py-3 text-left transition-colors ${
                p.id === value ? 'border-brand bg-white/80' : 'border-line bg-white/40 hover:bg-white/70'
              }`}
            >
              <div className="text-[17px] text-ink">{c.name}</div>
              <div className="text-[13px] text-muted">{c.year} · {c.country}</div>
            </button>
          ))}
          {vehicles && entries.length > 0 && spacecrafts.length > entries.length && (
            <p className="px-1 pt-3 text-[12px] leading-relaxed text-muted">
              {spacecrafts.length - entries.length} more spacecraft in the garage are not listed: they do not land on the Moon (orbiters, fly-bys, impactors), so there is no landing to analyse.
            </p>
          )}
        </div>

        <div className="flex min-h-0 flex-col">
          <div className="max-h-[calc(100vh-235px)] flex-1 overflow-y-auto pr-2">
            {selected ? <Preview entry={selected} /> : <p className="pt-2 text-muted">Select a vehicle to preview it here.</p>}
          </div>
        </div>
      </div>
      <StepFooter onBack={onBack}>
        <GlassButton onClick={onNext} disabled={!value}>Next</GlassButton>
      </StepFooter>
    </>
  )
}
