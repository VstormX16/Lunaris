import { ArrowLeft } from 'lucide-react'
import { useEffect, useMemo, useState } from 'react'
import { createCustomVehicle, getSites, previewVehicle, updateCustomVehicle, type CustomVehicleInput, type VehiclePreview, type VehicleProfile } from '../api'
import { inputCls } from './wizard/ui'

const FALLBACK_SITES = [{ name: 'Malapert', lat: -85.987, lon: 2.08 }, { name: 'Shackleton', lat: -89.54, lon: 0 }, { name: 'Nobile', lat: -85.2, lon: 53.5 }]
const today = () => new Date().toISOString().slice(0, 10)
const MODE_LABEL: Record<string, string> = { landing: 'unsafe touchdown', cold: 'freezing in a night', power: 'running out of power', comms: 'too little contact with Earth' }

const fromProfile = (p: VehicleProfile): CustomVehicleInput => ({
  name: p.name, kind: p.kind === 'rover' ? 'rover' : 'lander', note: p.note ?? '', mass_kg: p.mass_kg ?? null,
  battery_wh: p.battery_wh.value ?? 0, solar_w: p.solar_w.value ?? 0, survival_power_w: p.survival_power_w?.value ?? 0, payload_power_w: p.payload_power_w?.value ?? 0,
  max_slope_deg: p.max_slope_deg.value ?? 15, radioisotope_heat: p.radioisotope_heat.value === true, comms: p.comms.value === 'relay' ? 'relay' : 'direct_to_earth',
})
const BLANK: CustomVehicleInput = { name: '', kind: 'lander', note: '', mass_kg: null, battery_wh: 1500, solar_w: 400, survival_power_w: 60, payload_power_w: 150, max_slope_deg: 15, radioisotope_heat: false, comms: 'direct_to_earth' }

/** A number with its slider: type it, or drag it and watch the read-out on the right change. */
function Num({ label, unit, value, onChange, min, max, step, hint }: { label: string; unit: string; value: number; onChange: (v: number) => void; min: number; max: number; step: number; hint: string }) {
  return (
    <div>
      <div className="flex items-baseline justify-between gap-3">
        <span className="text-[13px] uppercase tracking-wide text-muted">{label}</span>
        <span className="flex items-baseline gap-1.5">
          <input type="number" value={Number.isFinite(value) ? value : ''} min={0} step={step} onChange={(e) => onChange(e.target.value === '' ? NaN : Number(e.target.value))} aria-label={label}
            className="w-24 rounded-[8px] border border-line bg-white/70 px-2 py-1 text-right text-[16px] tabular-nums text-ink outline-none focus:border-brand" />
          <span className="w-8 text-[14px] text-muted">{unit}</span>
        </span>
      </div>
      <input type="range" min={min} max={max} step={step} value={Number.isFinite(value) ? Math.min(max, Math.max(min, value)) : min} onChange={(e) => onChange(Number(e.target.value))} aria-label={`${label} slider`} className="mt-1.5 w-full accent-[#4a7566]" />
      <p className="mt-0.5 text-[12px] leading-snug text-muted">{hint}</p>
    </div>
  )
}

function Seg<T extends string | boolean>({ value, options, onChange, label }: { value: T; options: { v: T; t: string }[]; onChange: (v: T) => void; label: string }) {
  return (
    <div>
      <div className="text-[13px] uppercase tracking-wide text-muted">{label}</div>
      <div role="radiogroup" aria-label={label} className="mt-1.5 inline-flex overflow-hidden rounded-[8px] border border-line bg-white/60">
        {options.map((o) => (
          <button key={String(o.v)} type="button" role="radio" aria-checked={value === o.v} onClick={() => onChange(o.v)}
            className={`px-4 py-1.5 text-[15px] transition ${value === o.v ? 'bg-brand text-white' : 'text-ink hover:bg-active'}`}>{o.t}</button>
        ))}
      </div>
    </div>
  )
}

const valid = (v: CustomVehicleInput) => v.name.trim() !== '' && [v.battery_wh, v.solar_w, v.survival_power_w, v.payload_power_w, v.max_slope_deg].every((x) => Number.isFinite(x) && x >= 0) && v.max_slope_deg > 0

/** Design a vehicle: name it, set its figures, see how it would do at a real place and date, save it to the garage. */
export default function VehicleDesigner({ initial, onDone, onCancel }: { initial: VehicleProfile | null; onDone: (p: VehicleProfile) => void; onCancel: () => void }) {
  const [v, setV] = useState<CustomVehicleInput>(initial ? fromProfile(initial) : BLANK)
  const set = <K extends keyof CustomVehicleInput>(k: K, x: CustomVehicleInput[K]) => setV((o) => ({ ...o, [k]: x }))
  const [sites, setSites] = useState(FALLBACK_SITES)
  const [site, setSite] = useState('Malapert')
  const [date, setDate] = useState(today())
  const [pv, setPv] = useState<VehiclePreview | null>(null)
  const [pvErr, setPvErr] = useState(false)
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState<string | null>(null)
  useEffect(() => { getSites().then((s) => s.length && setSites(s.map((x) => ({ ...x, name: x.name.charAt(0).toUpperCase() + x.name.slice(1) })))).catch(() => undefined) }, [])

  const where = useMemo(() => sites.find((s) => s.name === site) ?? sites[0], [sites, site])
  const ok = valid(v)
  // the read-out follows the form, a moment after the last change
  useEffect(() => {
    if (!ok || !/^\d{4}-\d\d-\d\d$/.test(date)) return
    let live = true
    setPvErr(false)
    const t = setTimeout(() => {
      previewVehicle(v, where.lat, where.lon < 0 ? where.lon + 360 : where.lon, `${date}T00:00:00`).then((r) => live && setPv(r)).catch(() => live && setPvErr(true))
    }, 450)
    return () => { live = false; clearTimeout(t) }
  }, [v, where, date, ok])

  const save = async () => {
    setBusy(true); setErr(null)
    try { onDone(initial ? await updateCustomVehicle(initial.id, v) : await createCustomVehicle(v)) }
    catch (e) { setErr(e instanceof Error ? e.message : 'Could not save'); setBusy(false) }
  }
  const top = pv?.failure_modes[0]

  return (
    <main className="h-full flex-1 overflow-y-auto">
      <div className="mx-auto max-w-[1180px] px-10 pb-20 pt-10">
        <button onClick={onCancel} className="mb-4 flex items-center gap-1.5 text-[15px] text-muted transition hover:text-brand"><ArrowLeft size={16} /> Back to the garage</button>
        <h1 className="text-[34px] font-medium text-brand">{initial ? `Edit ${initial.name}` : 'Design a vehicle'}</h1>
        <p className="mt-1 max-w-[640px] text-muted">Give it a name and its figures. The read-out on the right runs the same mission simulation as an analysis, so you see at once what each change does. Saved vehicles can be picked in New Operation and in the Mission Planner.</p>

        <div className="mt-8 grid gap-12 lg:grid-cols-[minmax(0,1fr)_380px]">
          <div className="space-y-6">
            <div className="grid gap-5 sm:grid-cols-[1fr_auto]">
              <label className="block">
                <span className="text-[13px] uppercase tracking-wide text-muted">Name</span>
                <input value={v.name} maxLength={60} onChange={(e) => set('name', e.target.value)} placeholder="e.g. Polar Scout 1" className={`${inputCls} mt-1.5`} />
              </label>
              <Seg label="Type" value={v.kind} onChange={(x) => set('kind', x)} options={[{ v: 'lander', t: 'Lander' }, { v: 'rover', t: 'Rover' }]} />
            </div>
            <label className="block">
              <span className="text-[13px] uppercase tracking-wide text-muted">Note (optional)</span>
              <input value={v.note} maxLength={300} onChange={(e) => set('note', e.target.value)} placeholder="What is it for?" className={`${inputCls} mt-1.5`} />
            </label>
            <div className="grid gap-x-10 gap-y-6 sm:grid-cols-2">
              <Num label="Battery" unit="Wh" value={v.battery_wh} onChange={(x) => set('battery_wh', x)} min={0} max={20000} step={50} hint="Energy stored for the night. 1,000 Wh runs 50 W for 20 hours." />
              <Num label="Solar array" unit="W" value={v.solar_w} onChange={(x) => set('solar_w', x)} min={0} max={2000} step={10} hint="Peak power in full sunlight. 0 = battery only, it never recharges." />
              <Num label="Survival load" unit="W" value={v.survival_power_w} onChange={(x) => set('survival_power_w', x)} min={0} max={1000} step={5} hint="What it draws all the time to stay alive: avionics, heaters." />
              <Num label="Payload load" unit="W" value={v.payload_power_w} onChange={(x) => set('payload_power_w', x)} min={0} max={1000} step={5} hint="Extra draw while the instruments work, in daylight." />
              <Num label="Slope limit" unit="°" value={v.max_slope_deg} onChange={(x) => set('max_slope_deg', x)} min={1} max={40} step={1} hint="Steepest ground it can land on without tipping over." />
              <Num label="Mass (optional)" unit="kg" value={v.mass_kg ?? NaN} onChange={(x) => set('mass_kg', Number.isFinite(x) && x > 0 ? x : null)} min={0} max={5000} step={10} hint="For your own record; the simulation does not use it." />
            </div>
            <div className="flex flex-wrap gap-x-10 gap-y-5">
              <Seg label="Heat source" value={v.radioisotope_heat} onChange={(x) => set('radioisotope_heat', x)} options={[{ v: false, t: 'None' }, { v: true, t: 'Radioisotope heater' }]} />
              <Seg label="Communications" value={v.comms} onChange={(x) => set('comms', x)} options={[{ v: 'direct_to_earth', t: 'Direct to Earth' }, { v: 'relay', t: 'Relay satellite' }]} />
            </div>
          </div>

          <aside className="lg:sticky lg:top-8 lg:self-start" aria-label="How it would do">
            <div className="rounded-[10px] border border-line bg-white/55 p-5">
              <div className="text-[13px] uppercase tracking-wide text-muted">How it would do</div>
              <div className="mt-3 grid grid-cols-2 gap-3">
                <label className="block"><span className="text-[12px] text-muted">Place</span>
                  <select value={site} onChange={(e) => setSite(e.target.value)} className={`${inputCls} mt-1 !py-1.5 !text-[15px]`}>{sites.map((s) => <option key={s.name}>{s.name}</option>)}</select></label>
                <label className="block"><span className="text-[12px] text-muted">Start date</span>
                  <input type="date" value={date} onChange={(e) => setDate(e.target.value)} className={`${inputCls} mt-1 !py-1.5 !text-[15px]`} /></label>
              </div>
              {!ok ? <p className="mt-5 text-[14px] text-muted">Fill in the name and the figures to see the read-out.</p> : pvErr ? <p className="mt-5 text-[14px] text-amber-700">The read-out could not be computed. Check that the server is running.</p> : !pv ? (
                <div className="mt-5" aria-busy="true"><div className="skeleton h-10 w-28" /><div className="skeleton mt-4 h-3 w-full" /><div className="skeleton mt-2 h-3 w-4/5" /></div>
              ) : (
                <div className="mt-5">
                  <div className="flex items-baseline gap-2"><span className="text-[42px] font-medium leading-none tabular-nums text-brand">{Math.round(pv.p_success * 100)}%</span><span className="text-[15px] text-ink">chance it succeeds</span></div>
                  <p className="mt-1.5 text-[13px] leading-snug text-muted">It lands safely, works at least {pv.goal_days.toFixed(0)} days in daylight and gets its data home. Typically it works {pv.operating_days.p50.toFixed(0)} days.</p>
                  <ul className="mt-4 space-y-1.5">
                    {pv.milestones.map((m) => (
                      <li key={m.days} className="flex items-center gap-3 text-[13px]">
                        <span className="w-[54px] shrink-0 text-ink">{m.days} {m.days === 1 ? 'day' : 'days'}</span>
                        <span className="h-[6px] flex-1 overflow-hidden rounded-full bg-line"><span className="block h-full rounded-full bg-brand transition-[width] duration-500" style={{ width: `${m.p * 100}%` }} /></span>
                        <span className="w-[36px] shrink-0 text-right tabular-nums text-strong">{Math.round(m.p * 100)}%</span>
                      </li>
                    ))}
                  </ul>
                  {top && top.share >= 0.05 && <p className="mt-4 text-[13px] leading-snug text-ink">Most common way it fails: {MODE_LABEL[top.mode] ?? top.mode} ({Math.round(top.share * 100)}% of runs).</p>}
                  <p className="mt-3 text-[12px] leading-snug text-muted">Your figures are counted as unverified, so each is varied by ±20% in the simulation. 200 runs, real NASA Sun, Earth and terrain data.</p>
                </div>
              )}
            </div>
            {err && <p className="mt-3 text-[14px] text-amber-700" role="alert">{err}</p>}
            <div className="mt-4 flex gap-3">
              <button onClick={() => void save()} disabled={!ok || busy} className="grain rounded-[8px] bg-brand px-5 py-2 text-[16px] text-white transition hover:opacity-90 disabled:opacity-40">{busy ? 'Saving…' : initial ? 'Save changes' : 'Add to garage'}</button>
              <button onClick={onCancel} className="rounded-[8px] px-4 py-2 text-[16px] text-muted transition hover:bg-active hover:text-ink">Cancel</button>
            </div>
          </aside>
        </div>
      </div>
    </main>
  )
}
