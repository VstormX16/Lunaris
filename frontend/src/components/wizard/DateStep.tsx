import { todayIso } from '../../lib/input'
import Calendar from './Calendar'
import DateField from './DateField'
import { GlassButton, StepFooter } from './ui'

const addDays = (iso: string, n: number) => {
  const d = new Date(`${iso}T00:00:00Z`)
  d.setUTCDate(d.getUTCDate() + n)
  return d.toISOString().slice(0, 10)
}
const QUICK = [
  { label: 'Today', days: 0 },
  { label: 'In a week', days: 7 },
  { label: 'In a month', days: 30 },
]

export default function DateStep({ value, onChange, onNext }: { value: string; onChange: (iso: string) => void; onNext: () => void }) {
  return (
    <>
      <div className="grid flex-1 grid-cols-[minmax(300px,380px)_1fr] items-center gap-x-20 gap-y-10 py-10">
        <div>
          <h2 className="text-[30px] font-medium leading-tight text-brand">When does the mission start?</h2>
          <p className="mt-2 max-w-[380px] text-[15px] leading-relaxed text-muted">Pick the first day. The analysis looks at the 30 days after it.</p>
          <div className="mt-8">
            <DateField value={value} onChange={onChange} inline />
          </div>
          <div className="mt-4 flex items-center gap-6 text-[15px]">
            {QUICK.map((q) => (
              <button key={q.label} onClick={() => onChange(addDays(todayIso(), q.days))} className="text-muted hover:text-brand">{q.label}</button>
            ))}
          </div>
        </div>
        <div className="w-[360px] max-w-full justify-self-center">
          <Calendar value={value} onPick={onChange} big />
        </div>
      </div>
      <StepFooter>
        <GlassButton onClick={onNext} disabled={!value}>Next</GlassButton>
      </StepFooter>
    </>
  )
}
