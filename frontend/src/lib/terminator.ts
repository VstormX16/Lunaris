// The great circle 90 degrees from a "sub-point" (where the Sun, or Earth, stands overhead), on an
// equirectangular map drawn in a 360 x 180 box (x = lon + 180, y = 90 - lat). The same maths gives
// the day/night boundary for the Sun and the visible-from-Earth limb for Earth, on any world.
export type SubPoint = { lat: number; lon: number } // degrees, longitude east

const norm180 = (lon: number) => ((((lon + 180) % 360) + 360) % 360) - 180

function curvePoints(s: SubPoint): string[] {
  const phi = (Math.abs(s.lat) < 0.01 ? 0.01 : s.lat) * (Math.PI / 180)
  const pts: string[] = []
  for (let lon = -180; lon <= 180; lon += 2) {
    const d = ((lon - norm180(s.lon)) * Math.PI) / 180
    const lat = (Math.atan(-Math.cos(d) / Math.tan(phi)) * 180) / Math.PI // latitude of the boundary at this longitude
    pts.push(`${lon + 180},${90 - lat}`)
  }
  return pts
}

/** The boundary itself, as an open path. */
export function boundaryPath(s: SubPoint): string {
  return `M ${curvePoints(s).join(' L ')}`
}

/** The hemisphere on the far side of the boundary from the sub-point (night, or the Moon's far side). */
export function oppositeHemispherePath(s: SubPoint): string {
  const edge = s.lat >= 0 ? 180 : 0 // it lies south of the boundary when the sub-point is north, and vice versa
  return `${boundaryPath(s)} L 360,${edge} L 0,${edge} Z`
}

export const wrapLon = norm180
