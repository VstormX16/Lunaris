import { ChevronDown } from 'lucide-react'
import { useEffect, useMemo, useState } from 'react'
import type { VehicleProfile } from '../../api'
import { formatDate, parseLat, parseLon } from '../../lib/input'
import { isComplete, scenarioB, type CompareKind, type Scenario, type WizardState } from '../../types'
import CoordinateFields from './CoordinateFields'
import DateField from './DateField'
import { GlassButton, inputCls, StepFooter } from './ui'
import { joinVehicles } from './VehicleStep'

const KINDS: { kind: CompareKind; title: string; hint: string }[] = [
  { kind: 'date', title: 'Different date', hint: 'Same vehicle and site, another start date' },
  { kind: 'location', title: 'Different location', hint: 'Same vehicle and date, another site' },
  { kind: 'vehicle', title: 'Different vehicle', hint: 'Same site and date, another vehicle' },
]

type Props = {
  state: WizardState
  vehicles: VehicleProfile[] | null
  patch: (p: Partial<WizardState>) => void
  patchB: (p: Partial<Scenario>) => void
  onBack: () => void
  onAnalyze: () => void
  onSetUpB: () => void // full comparison: go through the steps again for the second operation
}

function ModeCard({ title, text, selected, open, onClick }: { title: string; text: string; selected: boolean; open?: boolean; onClick: () => void }) {
  return (
    <button
      onClick={onClick}
      aria-pressed={selected}
      className={`flex w-full items-start justify-between rounded-[10px] border px-5 py-4 text-left transition-colors ${
        selected ? 'border-brand bg-white/85' : 'border-line bg-white/40 hover:bg-white/70'
      }`}
    >
      <span>
        <span className="block text-[20px] text-ink">{title}</span>
        <span className="block text-[14px] text-muted">{text}</span>
      </span>
      {open !== undefined && <ChevronDown size={20} className={`mt-1 text-muted transition-transform duration-300 ${open ? 'rotate-180' : ''}`} />}
    </button>
  )
}

export default function ModeStep({ state, vehicles, patch, patchB, onBack, onAnalyze, onSetUpB }: Props) {
  const { a, b, mode, compareKind } = state
  const entries = useMemo(() => (vehicles ? joinVehicles(vehicles) : []), [vehicles])
  const nameOf = (id: string) => entries.find((e) => e.profile.id === id)?.craft.name ?? id

  const sb = scenarioB(state)
  const differs =
    sb !== null &&
    (compareKind === 'date'
      ? sb.date !== '' && sb.date !== a.date
      : compareKind === 'vehicle'
        ? sb.vehicleId !== '' && sb.vehicleId !== a.vehicleId
        : parseLat(sb.lat) !== null &&
          parseLon(sb.lon) !== null &&
          (parseLat(sb.lat) !== parseLat(a.lat) || parseLon(sb.lon) !== parseLon(a.lon)))
  const full = mode === 'full'
  const canAnalyze = mode === 'instant' || (mode === 'comparison' && differs) || full

  const comparing = mode === 'comparison'
  // The list clips while it animates open; afterwards it must not clip, or the calendar
  // popover of a date comparison would be cut off at the list's edge.
  const [settled, setSettled] = useState(false)
  useEffect(() => {
    if (!comparing) return setSettled(false)
    const t = setTimeout(() => setSettled(true), 320)
    return () => clearTimeout(t)
  }, [comparing])

  return (
    <>
      <div className="flex flex-1 flex-col items-center py-10">
        <div className="w-[640px] max-w-full space-y-3">
          <ModeCard title="Instant analysis" text="Analyze this vehicle, date and site on its own" selected={mode === 'instant'} onClick={() => patch({ mode: 'instant' })} />
          <ModeCard title="Comparison" text="Run two scenarios side by side, changing one thing" selected={comparing} open={comparing} onClick={() => patch({ mode: 'comparison' })} />

          {/* grid-rows 0fr -> 1fr animates the list open without measuring it */}
          <div className={`grid transition-all duration-300 ease-in-out ${comparing ? 'grid-rows-[1fr] opacity-100' : 'grid-rows-[0fr] opacity-0'}`}>
            <div className={`min-h-0 ${settled ? '' : 'overflow-hidden'}`} inert={!comparing}>
              <ul className="ml-4 space-y-1 border-l border-line pl-4 pt-1">
                {KINDS.map(({ kind, title, hint }) => (
                  <li key={kind}>
                    <button
                      onClick={() => patch({ compareKind: kind })}
                      aria-pressed={compareKind === kind}
                      className={`w-full rounded-[8px] px-3 py-2 text-left transition-colors ${compareKind === kind ? 'bg-active' : 'hover:bg-active'}`}
                    >
                      <span className={`block text-[17px] ${compareKind === kind ? 'text-brand' : 'text-ink'}`}>{title}</span>
                      <span className="block text-[13px] text-muted">{hint}</span>
                    </button>
                  </li>
                ))}
              </ul>

              {compareKind && (
                <div key={compareKind} className="step-in mt-5 ml-4 border-t border-line pl-4 pt-5">
                  <div className="mb-3 text-[13px] uppercase tracking-wide text-muted">
                    Scenario A <span className="normal-case tracking-normal text-ink">
                      {compareKind === 'date' && formatDate(a.date)}
                      {compareKind === 'vehicle' && nameOf(a.vehicleId)}
                      {compareKind === 'location' && `${a.lat}, ${a.lon}`}
                    </span>
                  </div>
                  <div className="mb-2 text-[13px] uppercase tracking-wide text-muted">Scenario B</div>
                  {compareKind === 'date' && <DateField value={b.date} anchor={a.date} placement="top" onChange={(date) => patchB({ date })} />}
                  {compareKind === 'vehicle' && (
                    <select value={b.vehicleId} onChange={(e) => patchB({ vehicleId: e.target.value })} className={inputCls} aria-label="Second vehicle">
                      <option value="">Select a vehicle</option>
                      {entries.filter((e) => e.profile.id !== a.vehicleId).map(({ profile: p, craft: c }) => (
                        <option key={p.id} value={p.id}>{c.name} ({c.year})</option>
                      ))}
                    </select>
                  )}
                  {compareKind === 'location' && (
                    <CoordinateFields lat={b.lat} lon={b.lon} onLat={(lat) => patchB({ lat })} onLon={(lon) => patchB({ lon })} />
                  )}
                  {sb !== null && !differs && (b.date || b.vehicleId || b.lat || b.lon) && (
                    <p className="mt-2 text-[13px] text-amber-700">Scenario B must differ from scenario A.</p>
                  )}
                </div>
              )}
            </div>
          </div>

          <ModeCard title="Full comparison" text="Set up a second operation from the start, with its own date, vehicle and place; the two are compared at the end" selected={full} onClick={() => patch({ mode: 'full' })} />
          {full && isComplete(state.b) && (
            <div className="step-in ml-4 border-l border-line pl-4 pt-1 text-[14px] text-muted">
              Scenario B: <span className="text-ink">{nameOf(state.b.vehicleId)} · {formatDate(state.b.date)} · {state.b.lat}, {state.b.lon}</span>{' '}
              <button onClick={onSetUpB} className="ml-2 text-brand hover:underline">Change</button>
            </div>
          )}
        </div>
      </div>
      <StepFooter onBack={onBack}>
        <GlassButton onClick={full && !isComplete(state.b) ? onSetUpB : onAnalyze} disabled={!canAnalyze}>{full && !isComplete(state.b) ? 'Set up scenario B' : 'Analyze'}</GlassButton>
      </StepFooter>
    </>
  )
}
