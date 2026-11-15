/** Terrain patch helpers for the 360 view. The patch is n x n heights (m) above the observer's plane, north row first. */

export type Patch = { z: Float32Array; n: number; radiusM: number }

/** Bilinear height at offset (east, north) metres from the observer; the edge value outside the patch. */
export function heightAt(p: Patch, east: number, north: number): number {
  const step = (2 * p.radiusM) / (p.n - 1)
  const c = Math.min(p.n - 1.001, Math.max(0, (east + p.radiusM) / step))
  const r = Math.min(p.n - 1.001, Math.max(0, (p.radiusM - north) / step))
  const c0 = Math.floor(c), r0 = Math.floor(r)
  const fc = c - c0, fr = r - r0
  const g = (rr: number, cc: number) => p.z[rr * p.n + cc]
  return g(r0, c0) * (1 - fr) * (1 - fc) + g(r0, c0 + 1) * (1 - fr) * fc + g(r0 + 1, c0) * fr * (1 - fc) + g(r0 + 1, c0 + 1) * fr * fc
}

/** Elevation angle (deg) of the skyline in one direction, found by walking out along the heights from the eye. */
export function skylineElevation(p: Patch, azimuthDeg: number, eyeM = 2): number {
  const a = (azimuthDeg * Math.PI) / 180
  const se = Math.sin(a), sn = Math.cos(a)
  const step = (2 * p.radiusM) / (p.n - 1)
  let best = -90
  for (let d = step; d <= p.radiusM; d += step) {
    const el = Math.atan2(heightAt(p, d * se, d * sn) - eyeM, d)
    if (el > best) best = el
  }
  return (best * 180) / Math.PI
}

export type BodyView = 'above' | 'behind' | 'below'
/** Is a body at (azimuth, elevation) above the horizon, hidden behind terrain, or below the horizon? */
export function bodyStatus(p: Patch, az: number, el: number): { status: BodyView; skyline: number } {
  const skyline = skylineElevation(p, az)
  if (el < Math.min(0, skyline)) return { status: 'below', skyline }
  return { status: el > skyline ? 'above' : 'behind', skyline }
}
