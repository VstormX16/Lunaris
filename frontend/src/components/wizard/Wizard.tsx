import { useEffect, useState, type Dispatch, type SetStateAction } from 'react'
import { getVehicles, type VehicleProfile } from '../../api'
import type { Scenario, Step, WizardState } from '../../types'
import DateStep from './DateStep'
import LocationStep from './LocationStep'
import ModeStep from './ModeStep'
import VehicleStep from './VehicleStep'

const STEPS = ['Date', 'Vehicle', 'Location', 'Mode']
const STEPS_B = ['Date', 'Vehicle', 'Location']

type Props = { state: WizardState; setState: Dispatch<SetStateAction<WizardState>>; onAnalyze: () => void }

export default function Wizard({ state, setState, onAnalyze }: Props) {
  const [vehicles, setVehicles] = useState<VehicleProfile[] | null>(null)
  const [error, setError] = useState<string | null>(null)
  useEffect(() => {
    getVehicles().then(setVehicles).catch((e: Error) => setError(e.message))
  }, [])

  const go = (step: Step) => setState((s) => ({ ...s, step }))
  const patch = (p: Partial<WizardState>) => setState((s) => ({ ...s, ...p }))
  const patchA = (p: Partial<Scenario>) => setState((s) => ({ ...s, a: { ...s.a, ...p } }))
  const patchB = (p: Partial<Scenario>) => setState((s) => ({ ...s, b: { ...s.b, ...p } }))
  const { step } = state
  const second = state.editing === 'b' // filling in the second operation of a full comparison
  const a = second ? state.b : state.a
  const patchCur = second ? patchB : patchA
  const labels = second ? STEPS_B : STEPS

  return (
    <main className="flex flex-1 flex-col overflow-y-auto">
      {/* progress: the bar fills one quarter per step */}
      <div className="mx-auto w-full max-w-[1280px] px-10 pt-9">
        {second && (
          <div className="mb-4 flex items-baseline justify-between border-b border-line pb-3">
            <div><span className="text-[13px] uppercase tracking-wide text-brand">Scenario B</span> <span className="text-[15px] text-muted">a second operation, set up from the start; the two are compared at the end</span></div>
            <button onClick={() => patch({ editing: 'a', step: 4 })} className="text-[15px] text-muted hover:text-brand">← Back to scenario A</button>
          </div>
        )}
        <ol className={`grid text-[13px] uppercase tracking-wide ${second ? 'grid-cols-3' : 'grid-cols-4'}`}>
          {labels.map((label, i) => (
            <li key={label} className={i + 1 <= step ? 'text-brand' : 'text-muted'}>
              <span className="tabular-nums">0{i + 1}</span> {label}
            </li>
          ))}
        </ol>
        <div className="relative mt-3 h-px bg-line" role="progressbar" aria-valuemin={1} aria-valuemax={labels.length} aria-valuenow={step}>
          <div className="absolute left-0 top-1/2 h-[3px] -translate-y-1/2 rounded-full bg-brand transition-[width] duration-500 ease-out" style={{ width: `${(step * 100) / labels.length}%` }} />
        </div>
      </div>

      <div key={step} className="step-in mx-auto flex min-h-0 w-full max-w-[1280px] flex-1 flex-col px-10 pb-8">
        {step === 1 && <DateStep value={a.date} onChange={(date) => patchCur({ date })} onNext={() => go(2)} />}
        {step === 2 && (
          <VehicleStep vehicles={vehicles} error={error} value={a.vehicleId} onChange={(vehicleId) => patchCur({ vehicleId })} onBack={() => go(1)} onNext={() => go(3)} />
        )}
        {step === 3 && <LocationStep lat={a.lat} lon={a.lon} onChange={patchCur} onBack={() => go(2)} onNext={() => (second ? onAnalyze() : go(4))} nextLabel={second ? 'Analyze' : 'Next'} />}
        {step === 4 && <ModeStep state={state} vehicles={vehicles} patch={patch} patchB={patchB} onBack={() => go(3)} onAnalyze={onAnalyze} onSetUpB={() => setState((s) => ({ ...s, editing: 'b', step: 1 }))} />}
      </div>
    </main>
  )
}
