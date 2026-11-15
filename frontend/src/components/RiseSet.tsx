import { formatCountdown, formatWhen } from '../lib/countdown'

type Transition = { utc: string; becomes_up: boolean }

function Block({ label, t, nowMs, none }: { label: string; t: Transition | undefined; nowMs: number; none: string }) {
  return (
    <div>
      <div className="text-[13px] uppercase tracking-wide text-muted">{label} in</div>
      {t ? (
        <>
          <div className="mt-1 text-[32px] font-bold leading-none tabular-nums text-strong">{formatCountdown((Date.parse(t.utc) - nowMs) / 1000)}</div>
          <div className="mt-1.5 text-[13px] leading-snug text-muted">
            <div>{formatWhen(t.utc).utc}</div>
            <div>{formatWhen(t.utc).local} your time</div>
          </div>
        </>
      ) : (
        <div className="mt-2 text-[15px] text-ink">{none}</div>
      )}
    </div>
  )
}

/** Time to the next sunrise AND the next sunset, soonest first. Used for both worlds. */
export default function RiseSet({ upNow, transitions, nowMs, searchedDays, upWord, downWord, rise = 'Sunrise', set = 'Sunset' }: {
  upNow: boolean
  transitions: Transition[]
  nowMs: number
  searchedDays: number
  upWord: string
  downWord: string
  rise?: string
  set?: string
}) {
  const future = transitions.filter((t) => Date.parse(t.utc) > nowMs)
  const nextRise = future.find((t) => t.becomes_up)
  const nextSet = future.find((t) => !t.becomes_up)
  // when neither change comes, the current state holds for the whole window
  const stays = (word: string) => `${word} for all of the next ${searchedDays} days`
  const blocks = [
    { label: rise, t: nextRise, none: !upNow && !nextSet ? stays('Stays down') : `No ${rise.toLowerCase()} in the next ${searchedDays} days` },
    { label: set, t: nextSet, none: upNow && !nextRise ? stays('Stays up') : `No ${set.toLowerCase()} in the next ${searchedDays} days` },
  ].sort((a, b) => (a.t ? Date.parse(a.t.utc) : Infinity) - (b.t ? Date.parse(b.t.utc) : Infinity))

  return (
    // min-h: its height must not change between states ("no sunrise" vs a full countdown), or the page height
    // changes and the map above jumps under the cursor after each click
    <div className="min-h-[200px] border-t border-line pt-4">
      <div className="text-[22px] font-bold text-strong">{upNow ? upWord : downWord}</div>
      <div className="mt-4 grid grid-cols-2 gap-x-8 gap-y-5">
        {blocks.map((b) => (
          <Block key={b.label} label={b.label} t={b.t} nowMs={nowMs} none={b.none} />
        ))}
      </div>
    </div>
  )
}
