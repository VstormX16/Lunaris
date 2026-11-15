const pad = (n: number) => String(n).padStart(2, '0')

/** Exact countdown, e.g. "3d 04:12:55" or "00:07:31". */
export function formatCountdown(seconds: number): string {
  const s = Math.max(0, Math.floor(seconds))
  const d = Math.floor(s / 86400)
  const rest = `${pad(Math.floor((s % 86400) / 3600))}:${pad(Math.floor((s % 3600) / 60))}:${pad(s % 60)}`
  return d > 0 ? `${d}d ${rest}` : rest
}

/** For peaks, whose instant is only defined to hours or days: no seconds, e.g. "≈ 3d 12h". */
export function formatApprox(seconds: number): string {
  const h = Math.max(0, Math.round(seconds / 3600))
  return h >= 24 ? `≈ ${Math.floor(h / 24)}d ${h % 24}h` : `≈ ${h}h`
}

export function formatUncertainty(days: number): string {
  return days < 1 ? `±${Math.max(1, Math.round(days * 24))} h` : `±${days} days`
}

const utcFmt = new Intl.DateTimeFormat('en-GB', { timeZone: 'UTC', weekday: 'short', day: 'numeric', month: 'short', year: 'numeric', hour: '2-digit', minute: '2-digit', hour12: false })
const localFmt = new Intl.DateTimeFormat('en-GB', { weekday: 'short', day: 'numeric', month: 'short', hour: '2-digit', minute: '2-digit', hour12: false, timeZoneName: 'short' })

/** The same instant in UTC and in the viewer's own time zone. */
export function formatWhen(iso: string): { utc: string; local: string } {
  const d = new Date(iso)
  return { utc: `${utcFmt.format(d).replace(',', '').replace(',', ' ·')} UTC`, local: localFmt.format(d).replace(',', '') }
}
