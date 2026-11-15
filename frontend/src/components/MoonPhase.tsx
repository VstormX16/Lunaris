/**
 * The Moon as seen from Earth, drawn from the real geometry: `illumination` (0-1) and whether it is
 * waxing (lit side on the right, as seen from the northern hemisphere). The terminator is an ellipse
 * whose width follows the phase angle, so the shape is exact, not a stock icon.
 */
export default function MoonPhase({ illumination, waxing, size = 168 }: { illumination: number; waxing: boolean; size?: number }) {
  const R = 50
  const cosI = 2 * illumination - 1 // cos of the phase angle: +1 full, -1 new
  const rx = R * Math.abs(cosI)
  const gibbous = cosI > 0
  // right limb top -> bottom, then back along the terminator (bulging left when gibbous, right when crescent)
  const lit = `M 0 ${-R} A ${R} ${R} 0 0 1 0 ${R} A ${rx} ${R} 0 0 ${gibbous ? 1 : 0} 0 ${-R} Z`
  return (
    <svg viewBox="-54 -54 108 108" width={size} height={size} role="img" aria-label={`Moon, ${Math.round(illumination * 100)}% illuminated, ${waxing ? 'waxing' : 'waning'}`}>
      <circle r={R} fill="#34332f" />
      <g transform={waxing ? undefined : 'scale(-1 1)'}>
        <path d={lit} fill="#efece3" />
      </g>
      <circle r={R} fill="none" stroke="#24231f" strokeOpacity="0.35" strokeWidth="0.8" />
    </svg>
  )
}
