import { jsPDF } from 'jspdf'
import { getVehicles, type MissionRisk, type RiskMode, type ScenarioResult } from '../api'
import { visualsKey } from '../components/ResultVisuals'
import { MODE_LABEL, verdictOf } from '../components/Decision'
import type { Scenario } from '../types'
import { formatDate } from './input'

export type ReportItem = { label: string; scenario: Scenario; vehicle: string; result: ScenarioResult }

const BRAND: [number, number, number] = [74, 117, 102]
const INK: [number, number, number] = [45, 47, 44]
const MUTED: [number, number, number] = [110, 115, 110]
const LINE: [number, number, number] = [210, 215, 210]
const W = 210
const M = 20 // page margin, mm
const PDF_MODE: Record<RiskMode, [number, number, number]> = { landing: [154, 107, 79], cold: [93, 125, 150], power: [192, 138, 43], comms: [139, 143, 147] }
const CONF_PDF = { high: 'high confidence', medium: 'medium confidence', low: 'low confidence: some vehicle data is not published' } as const
const pct = (f: number) => `${Math.round(f * 100)}%`
const span = (h: number) => (h < 48 ? `${Math.round(h)} hours` : `${(h / 24).toFixed(1)} days`)

type Picture = { data: string; w: number; h: number }

/** An on-screen SVG chart as a PNG (3x for print): serialised on its own, so it uses a plain sans-serif font. */
async function svgPicture(svg: SVGSVGElement | null | undefined): Promise<Picture | null> {
  if (!svg) return null
  const vb = svg.viewBox.baseVal
  if (!vb || !vb.width) return null
  const clone = svg.cloneNode(true) as SVGSVGElement
  clone.setAttribute('xmlns', 'http://www.w3.org/2000/svg')
  clone.setAttribute('width', String(vb.width))
  clone.setAttribute('height', String(vb.height))
  clone.setAttribute('font-family', 'Helvetica, Arial, sans-serif')
  clone.querySelectorAll('line[stroke="#2d2f2c"][stroke-width="0.8"]').forEach((l) => l.remove()) // the hover cursor, if any
  const img = new Image()
  const url = `data:image/svg+xml;charset=utf-8,${encodeURIComponent(new XMLSerializer().serializeToString(clone))}`
  try {
    await new Promise<void>((res, rej) => { img.onload = () => res(); img.onerror = () => rej(new Error('svg')); img.src = url })
  } catch {
    return null
  }
  const k = 3
  const c = document.createElement('canvas')
  c.width = vb.width * k
  c.height = vb.height * k
  const ctx = c.getContext('2d')
  if (!ctx) return null
  ctx.fillStyle = '#ffffff'
  ctx.fillRect(0, 0, c.width, c.height)
  ctx.drawImage(img, 0, 0, c.width, c.height)
  return { data: c.toDataURL('image/jpeg', 0.9), w: vb.width, h: vb.height }
}

/** The terrain picture is a small canvas (one pixel per height sample): enlarge it smoothly before printing. */
function canvasPicture(src: HTMLCanvasElement | null | undefined): Picture | null {
  if (!src || !src.width) return null
  const size = 600
  const c = document.createElement('canvas')
  c.width = size
  c.height = size
  const ctx = c.getContext('2d')
  if (!ctx) return null
  ctx.imageSmoothingEnabled = true
  ctx.imageSmoothingQuality = 'high'
  ctx.drawImage(src, 0, 0, size, size)
  // the landing-area circle and the centre dot, as on screen
  const svgCircle = src.parentElement?.querySelector('svg circle')
  const r = svgCircle ? (Number(svgCircle.getAttribute('r')) / 100) * size : 0
  ctx.strokeStyle = 'rgba(255,255,255,0.9)'
  ctx.setLineDash([12, 9])
  ctx.lineWidth = 3
  if (r > 0) { ctx.beginPath(); ctx.arc(size / 2, size / 2, r, 0, 2 * Math.PI); ctx.stroke() }
  ctx.setLineDash([])
  ctx.fillStyle = '#ffffff'
  ctx.beginPath(); ctx.arc(size / 2, size / 2, 5, 0, 2 * Math.PI); ctx.fill()
  return { data: c.toDataURL('image/jpeg', 0.9), w: size, h: size }
}

/** One-or-two scenario PDF: the call, the score, the numbers and the warnings, in the same words as the results page. Drawn as text, so it stays small and selectable. */
export async function downloadReport(items: ReportItem[]) {
  const doc = new jsPDF({ unit: 'mm', format: 'a4', compress: true })
  let y = M
  const vehicles = await getVehicles().catch(() => [])

  const ensure = (h: number) => {
    if (y + h > 297 - M) {
      doc.addPage()
      y = M
    }
  }
  const text = (s: string, size: number, color: [number, number, number], bold = false, x = M, width = W - 2 * M) => {
    doc.setFont('helvetica', bold ? 'bold' : 'normal').setFontSize(size).setTextColor(...color)
    const lines = doc.splitTextToSize(s, width) as string[]
    const lh = size * 0.46
    ensure(lines.length * lh)
    doc.text(lines, x, y + size * 0.3)
    y += lines.length * lh
  }
  const rule = () => {
    doc.setDrawColor(...LINE).setLineWidth(0.2).line(M, y, W - M, y)
    y += 4
  }
  const row = (k: string, v: string) => {
    ensure(7)
    doc.setFont('helvetica', 'normal').setFontSize(10.5).setTextColor(...MUTED).text(k, M, y + 4)
    doc.setTextColor(...INK).text(v, W - M, y + 4, { align: 'right' })
    y += 7
    doc.setDrawColor(...LINE).setLineWidth(0.1).line(M, y, W - M, y)
  }
  // the simulated missions: chance of success, how long it works, how the runs ended (vector, like the page)
  const odds = (r: MissionRisk) => {
    ensure(46)
    text('MISSION SUCCESS PROBABILITY', 8.5, MUTED, true)
    const top = y + 2
    doc.setFont('helvetica', 'bold').setFontSize(26).setTextColor(...BRAND).text(`${Math.round(r.p_success * 100)}%`, M, top + 9)
    doc.setFont('helvetica', 'normal').setFontSize(9).setTextColor(...MUTED)
    doc.text(doc.splitTextToSize(`chance it lands safely, works at least ${r.goal_days.toFixed(0)} days and gets its data home (${r.samples} simulated missions, ${CONF_PDF[r.confidence.level]})`, 52) as string[], M, top + 15)
    const bx = M + 66, bw = W - M - bx - 12
    doc.setFontSize(8.5).setTextColor(...MUTED).text('WORKS AT LEAST', bx, top + 1)
    r.milestones.forEach((m, i) => {
      const yy = top + 6 + i * 5
      doc.setFontSize(9).setTextColor(...INK).text(`${m.days} ${m.days === 1 ? 'day' : 'days'}`, bx, yy + 1.3)
      doc.setFillColor(...LINE).roundedRect(bx + 16, yy - 0.8, bw - 16, 2.2, 1.1, 1.1, 'F')
      if (m.p > 0) doc.setFillColor(...BRAND).roundedRect(bx + 16, yy - 0.8, Math.max(2.2, (bw - 16) * m.p), 2.2, 1.1, 1.1, 'F')
      doc.setTextColor(...INK).text(`${Math.round(m.p * 100)}%`, W - M, yy + 1.3, { align: 'right' })
    })
    y = top + 8 + r.milestones.length * 5 + 3
    // how the runs ended, as one stacked bar
    const parts = [...r.failure_modes.map((f) => ({ share: f.share, color: PDF_MODE[f.mode], label: MODE_LABEL[f.mode] })), { share: r.p_success, color: BRAND, label: 'Success' }].filter((p) => p.share > 0)
    let x = M
    parts.forEach((p) => {
      doc.setFillColor(...p.color).rect(x, y, (W - 2 * M) * p.share, 2.6, 'F')
      x += (W - 2 * M) * p.share
    })
    y += 6
    text(parts.filter((p) => p.share >= 0.005).map((p) => `${p.label} ${Math.round(p.share * 100)}%`).join('   ·   '), 9, MUTED)
    y += 2
    rule()
  }
  const list = (title: string, entries: string[]) => {
    if (!entries.length) return
    y += 3
    text(title.toUpperCase(), 8.5, MUTED, true)
    y += 1
    entries.forEach((e) => {
      ensure(6)
      doc.setFillColor(...BRAND).circle(M + 1, y + 2.6, 0.7, 'F')
      text(e, 10.5, INK, false, M + 5, W - 2 * M - 5)
      y += 1
    })
  }

  const BANDS = [
    { from: 0, to: 20, label: 'Do not proceed', fill: [201, 208, 204] },
    { from: 20, to: 40, label: 'Not recommended', fill: [180, 196, 188] },
    { from: 40, to: 60, label: 'Possible', fill: [145, 171, 159] },
    { from: 60, to: 100, label: 'Go', fill: [95, 138, 122] },
  ] as const
  const scoreScale = (score: number, uncapped: number | null) => {
    ensure(20)
    text('WHERE THE SCORE FALLS', 8.5, MUTED, true)
    const w = W - 2 * M
    const top = y + 6
    const X = (v: number) => M + (Math.min(100, Math.max(0, v)) / 100) * w
    BANDS.forEach((b) => {
      doc.setFillColor(b.fill[0], b.fill[1], b.fill[2]).rect(X(b.from) + 0.4, top, X(b.to) - X(b.from) - 0.8, 2.6, 'F')
      doc.setFont('helvetica', score >= b.from && (score < b.to || b.to === 100) ? 'bold' : 'normal').setFontSize(8).setTextColor(...(score >= b.from && (score < b.to || b.to === 100) ? INK : MUTED))
      doc.text(b.label, X(b.from) + 0.4, top + 7)
    })
    if (uncapped !== null && uncapped - score >= 3) {
      doc.setDrawColor(...MUTED).setLineWidth(0.3).setLineDashPattern([0.8, 0.6], 0).circle(X(uncapped), top + 1.3, 1.9, 'S').setLineDashPattern([], 0)
    }
    doc.setDrawColor(...INK).setLineWidth(0.6).line(X(score), top - 1.6, X(score), top + 4.2)
    doc.setFont('helvetica', 'bold').setFontSize(8.5).setTextColor(...INK).text(String(Math.round(score)), X(score), top - 2.4, { align: 'center' })
    y = top + 10
    if (uncapped !== null && uncapped - score >= 3) text(`The dashed circle is where it would sit without its limiting factor (${Math.round(uncapped)}).`, 8.5, MUTED)
    y += 4
  }
  const slopeScale = (mean: number, limit: number | null, safe: number, x0: number, w: number, y0: number) => {
    const max = Math.max(15, mean * 1.25, (limit ?? 0) * 1.6)
    const X = (v: number) => x0 + (Math.min(max, v) / max) * w
    doc.setFont('helvetica', 'bold').setFontSize(8.5).setTextColor(...MUTED).text('SLOPE', x0, y0 + 3)
    const top = y0 + 11
    doc.setFillColor(227, 231, 228).rect(x0, top, w, 2.6, 'F')
    if (limit !== null) {
      doc.setFillColor(169, 189, 180).rect(x0, top, X(limit) - x0, 2.6, 'F')
      doc.setDrawColor(...MUTED).setLineWidth(0.3).line(X(limit), top - 1, X(limit), top + 4)
      doc.setFont('helvetica', 'normal').setFontSize(8).setTextColor(...MUTED).text(`limit ${limit}°`, X(limit), top + 7.5, { align: 'center' })
    }
    const over = limit !== null && mean > limit
    if (over) doc.setDrawColor(186, 26, 26); else doc.setDrawColor(...INK)
    doc.setLineWidth(0.6).line(X(mean), top - 1.6, X(mean), top + 4.2)
    doc.setFont('helvetica', 'bold').setFontSize(8.5).setTextColor(...INK).text(`${mean.toFixed(1)}°`, X(mean), top - 2.4, { align: 'center' })
    doc.setFont('helvetica', 'normal').setFontSize(8).setTextColor(...MUTED).text('0°', x0, top + 7.5)
    const note = `${limit === null ? 'No published slope limit for this vehicle. ' : over ? 'The average ground is steeper than the vehicle can take. ' : ''}${Math.round(safe * 100)}% of the landing area is within the limit.`
    doc.setFontSize(9).text(doc.splitTextToSize(note, w) as string[], x0, top + 13)
  }
  const picture = (pic: Picture, x0: number, w: number) => {
    const h = (pic.h / pic.w) * w
    ensure(h + 2)
    doc.addImage(pic.data, 'JPEG', x0, y, w, h)
    y += h
  }

  text('Lunaris', 11, BRAND, true)
  y += 1
  text('Landing site report', 24, BRAND)
  y += 2
  text(`Generated ${new Date().toISOString().slice(0, 10)} · all numbers computed from NASA LOLA terrain and NASA SPICE positions of the Sun and Earth (DE440 ephemeris with the Moon's real orientation, so its libration is included)`, 9, MUTED)
  y += 3
  rule()

  for (const [idx, it] of items.entries()) {
    const { assessment: a, query: q, terrain: t } = it.result
    const d = verdictOf(a, q)
    if (idx > 0) {
      doc.addPage()
      y = M
    }
    if (items.length > 1) text(`SCENARIO ${it.label}`, 9, BRAND, true)
    text(it.vehicle, 18, INK, true)
    text(`${formatDate(it.scenario.date)} · ${it.scenario.lat}, ${it.scenario.lon} · 30 days from the start date`, 10.5, MUTED)
    y += 5

    const score = Math.round(a.vehicle_index)
    doc.setFont('helvetica', 'bold').setFontSize(40).setTextColor(...BRAND).text(String(score), M, y + 12)
    doc.setFontSize(12).setTextColor(...MUTED).text('/ 100', M + (score >= 100 ? 31 : 22), y + 12)
    doc.setFontSize(15).setTextColor(...INK).text(d.title, 70, y + 6)
    doc.setFont('helvetica', 'normal')
    const sum = doc.splitTextToSize(d.summary, W - M - 70) as string[]
    doc.setFontSize(10.5).setTextColor(...MUTED).text(sum, 70, y + 12)
    y += 22
    if (d.probability_line) {
      text(d.probability_line, 10.5, INK)
      y += 2
    }
    rule()
    if (a.risk) odds(a.risk)

    list('Working in its favour', d.good)
    list('Risks', d.risks)
    list('What to do next', d.next)
    y += 6

    // ---- the pictures from the result page ----
    const el = document.querySelector<HTMLElement>(`[data-report-visuals="${CSS.escape(visualsKey(it.scenario))}"]`)
    const strips = await svgPicture(el?.querySelector<SVGSVGElement>('[data-viz="strips"] svg'))
    const sky = await svgPicture(el?.querySelector<SVGSVGElement>('[data-viz="sky"] svg'))
    const libEl = el?.querySelector<HTMLElement>('[data-viz="libration"]')
    const lib = await svgPicture(libEl?.querySelectorAll<SVGSVGElement>('svg')[1])
    const libText = libEl?.querySelector('p.text-ink')?.textContent ?? ''
    const ground = canvasPicture(el?.querySelector<HTMLCanvasElement>('[data-viz="ground"] canvas'))
    const limit = vehicles.find((v) => v.id === it.scenario.vehicleId)?.max_slope_deg.value ?? null
    ensure(60)
    rule()
    text('AT A GLANCE', 9, BRAND, true)
    y += 4
    scoreScale(a.vehicle_index, a.limiting_factor ? a.weighted_index : null)
    if (strips) {
      text('THE MONTH AT THIS SPOT', 8.5, MUTED, true)
      y += 1
      picture(strips, M, W - 2 * M)
      y += 5
    }
    const gw = 48
    ensure(gw + 22) // title, picture and caption stay on one page
    const rowTop = y
    slopeScale(a.site_metrics.mean_slope_deg, limit, a.site_metrics.safe_area_fraction, M, W - 2 * M - gw - 12, rowTop)
    if (ground) {
      doc.setFont('helvetica', 'bold').setFontSize(8.5).setTextColor(...MUTED).text('THE GROUND AROUND IT', W - M - gw, rowTop + 3)
      doc.addImage(ground.data, 'JPEG', W - M - gw, rowTop + 6, gw, gw)
      doc.setFont('helvetica', 'normal').setFontSize(7.5).text(t.data_source === 'ldem_470m' ? 'NASA LOLA, about 470 m per pixel. Darker is lower.' : 'NASA LOLA, 40 m per pixel. Darker is lower.', W - M - gw, rowTop + gw + 10)
      y = rowTop + gw + 14
    } else {
      y = rowTop + 32
    }
    if (sky) {
      ensure(60)
      text('THE SKY FROM THE SITE', 8.5, MUTED, true)
      y += 1
      picture(sky, M, W - 2 * M)
      text('Terrain skyline with the hourly paths of the Sun (red) and Earth (green); faint dots are behind the terrain.', 8.5, MUTED)
      y += 4
    }
    if (lib && libText) {
      ensure(70)
      text("EARTH'S WOBBLE (LIBRATION)", 8.5, MUTED, true)
      y += 1
      picture(lib, M, 120)
      text('Earth height above the horizon over the month: solid = real, dashed = on a Moon that did not rock, grey = terrain in the way.', 8.5, MUTED)
      y += 1
      text(libText, 10.5, INK)
      y += 4
    }
    y += 4

    text('KEY NUMBERS', 8.5, MUTED, true)
    y += 1
    const m = a.site_metrics
    row('Sunlight', `${pct(q.window.sunlit_fraction)} of the month`)
    row('Earth in view', `${pct(q.window.dte_fraction)} of the month`)
    row('Longest night', span(q.window.longest_darkness.hours))
    row('Longest gap without Earth', span(q.window.longest_dte_outage.hours))
    row('Average slope', `${m.mean_slope_deg.toFixed(1)}°`)
    row('Landing area within slope limit', pct(m.safe_area_fraction))
    row('Terrain', `${t.class.replace(/_/g, ' ')}, ${Math.round(t.site_height_m)} m`)
    row('Terrain data', t.data_source === 'ldem_470m' ? 'NASA LOLA whole-Moon map, 470 m per pixel' : 'NASA LOLA polar map, 40 m per pixel')
    y += 4

    if (a.vetoes.length) {
      text('LIMITS THAT CAPPED THE SCORE', 8.5, MUTED, true)
      y += 1
      a.vetoes.forEach((v) => text(`${v.reason} (capped at ${Math.round(v.cap)})`, 10.5, INK))
      y += 4
    }
    if (q.findings.length) {
      text('NOTES FROM THE CHECKS', 8.5, MUTED, true)
      y += 1
      q.findings.slice(0, 8).forEach((f) => text(`${f.severity === 'info' ? '' : f.severity.toUpperCase() + ': '}${f.message}`, 10, INK))
      y += 4
    }
    const dq = a.data_quality
    text(`Vehicle data: ${dq.inputs_verified} inputs verified, ${dq.inputs_unverified} unverified, ${dq.inputs_unknown} not published.`, 9, MUTED)
  }

  doc.setFont('helvetica', 'normal').setFontSize(8).setTextColor(...MUTED)
  const pages = doc.getNumberOfPages()
  for (let i = 1; i <= pages; i++) {
    doc.setPage(i)
    doc.text(`Lunaris · page ${i} of ${pages}`, W / 2, 290, { align: 'center' })
  }
  doc.save(`lunaris-report-${items[0].scenario.date}.pdf`)
}
