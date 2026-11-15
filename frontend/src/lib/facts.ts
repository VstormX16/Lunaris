import type { Spacecraft } from '../data/spacecrafts'

/** "ABD (NASA)" -> "ABD": the nation or agency without the parenthesised detail. */
export const nation = (c: Spacecraft) => c.country.split('(')[0].trim()

const grouped = (n: number) => n.toLocaleString('en', { maximumFractionDigits: 3 })
/** The leading number of a source cell ("360", "0 (Güneş paneli yok)", "13 km/sa"), or null. */
export const leadingNumber = (text: string): number | null => {
  const m = text.trim().match(/^~?\d[\d.]*/)
  return m ? Number(m[0]) : null
}

/**
 * Headline numbers taken verbatim from the dataset's numeric columns. A fact is left out when the
 * dataset has no such figure (or says it is zero / not applicable).
 */
export function keyFacts(c: Spacecraft): { label: string; value: string }[] {
  const mass = leadingNumber(c.massKg)
  const solar = leadingNumber(c.solarW)
  const thrust = Math.max(0, ...[c.landerThrustN, c.ascentThrustN, c.serviceThrustN].map((t) => leadingNumber(t) ?? 0))
  const facts = [
    { label: 'Launched', value: String(c.year) },
    mass ? { label: 'Mass', value: `${grouped(mass)} kg` } : null,
    solar ? { label: 'Solar power', value: `${grouped(solar)} W` } : null,
    thrust ? { label: 'Main thrust', value: `${grouped(thrust)} N` } : null,
  ]
  return facts.filter((f): f is { label: string; value: string } => f !== null)
}
