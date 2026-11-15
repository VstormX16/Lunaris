export type Step = 1 | 2 | 3 | 4
export type CompareKind = 'date' | 'location' | 'vehicle'
/** instant: one operation. comparison: the same operation with one thing changed. full: a second operation set up from scratch. */
export type Mode = 'instant' | 'comparison' | 'full'

/** One operation's inputs. lat/lon stay strings while the user is typing. */
export type Scenario = { date: string; vehicleId: string; lat: string; lon: string }

export type WizardState = {
  step: Step
  a: Scenario
  mode: Mode | null
  compareKind: CompareKind | null
  b: Scenario // comparison: only the field(s) of compareKind are used; full: all of it
  editing: 'a' | 'b' // which scenario the steps are filling in (b only in a full comparison)
}

export const emptyScenario: Scenario = { date: '', vehicleId: '', lat: '', lon: '' }

export const initialWizard: WizardState = {
  step: 1,
  a: emptyScenario,
  mode: null,
  compareKind: null,
  b: emptyScenario,
  editing: 'a',
}

export const isComplete = (s: Scenario) => s.date !== '' && s.vehicleId !== '' && s.lat !== '' && s.lon !== ''

/** Scenario B: a full comparison's own scenario, or scenario A with only the compared attribute replaced. */
export function scenarioB(w: WizardState): Scenario | null {
  if (w.mode === 'full') return isComplete(w.b) ? w.b : null
  if (w.mode !== 'comparison' || !w.compareKind) return null
  if (w.compareKind === 'date') return { ...w.a, date: w.b.date }
  if (w.compareKind === 'vehicle') return { ...w.a, vehicleId: w.b.vehicleId }
  return { ...w.a, lat: w.b.lat, lon: w.b.lon }
}
