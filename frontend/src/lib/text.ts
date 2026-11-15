// Split on ; , and " + " that sit outside parentheses, so "(49 kN, LOX)" stays intact.
export function splitItems(text: string): string[] {
  const items: string[] = []
  let depth = 0
  let cur = ''
  for (let i = 0; i < text.length; i++) {
    const ch = text[i]
    if (ch === '(') depth++
    if (ch === ')') depth = Math.max(0, depth - 1)
    const atPlus = depth === 0 && text.startsWith(' + ', i)
    if (depth === 0 && (ch === ';' || ch === ',' || atPlus)) {
      items.push(cur)
      cur = ''
      if (atPlus) i += 2
    } else cur += ch
  }
  items.push(cur)
  return items
    .map((s) => s.trim())
    .filter(Boolean)
    .map((s) => s.charAt(0).toLocaleUpperCase('tr') + s.slice(1)) // Turkish-aware: i -> İ
}
