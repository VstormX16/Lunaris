// Parsing/validation of user-typed dates and coordinates.
// Ranges mirror the backend: the analysis works anywhere on the Moon (detailed 40 m terrain below 85S, 470 m elsewhere), SPICE kernels cover well beyond 1950-2100.

export const MIN_YEAR = 1950
export const MAX_YEAR = 2100

const pad = (n: number) => String(n).padStart(2, '0')

export function isoFromParts(y: number, m: number, d: number): string | null {
  if (y < MIN_YEAR || y > MAX_YEAR) return null
  const dt = new Date(Date.UTC(y, m - 1, d))
  const real = dt.getUTCFullYear() === y && dt.getUTCMonth() === m - 1 && dt.getUTCDate() === d
  return real ? `${y}-${pad(m)}-${pad(d)}` : null
}

/** DD.MM.YYYY (also / or -) or YYYY-MM-DD -> ISO date, or null when it is not a real date in range. */
export function parseDate(text: string): string | null {
  const t = text.trim()
  let m = /^(\d{1,2})[./-](\d{1,2})[./-](\d{4})$/.exec(t)
  if (m) return isoFromParts(+m[3], +m[2], +m[1])
  m = /^(\d{4})-(\d{1,2})-(\d{1,2})$/.exec(t)
  if (m) return isoFromParts(+m[1], +m[2], +m[3])
  return null
}

export function formatDate(iso: string): string {
  const [y, m, d] = iso.split('-')
  return `${d}.${m}.${y}`
}

export function todayIso(): string {
  const n = new Date()
  return `${n.getFullYear()}-${pad(n.getMonth() + 1)}-${pad(n.getDate())}`
}

const num = (s: string): number | null => {
  const t = s.trim().replace(',', '.')
  if (t === '') return null
  const v = Number(t)
  return Number.isFinite(v) ? v : null
}

export function parseLat(s: string): number | null {
  const v = num(s)
  return v !== null && v >= -90 && v <= 90 ? v : null
}

/** Accepts -180..360, returns the 0..360 east longitude the API expects. */
export function parseLon(s: string): number | null {
  const v = num(s)
  return v !== null && v >= -180 && v < 360 ? ((v % 360) + 360) % 360 : null
}
