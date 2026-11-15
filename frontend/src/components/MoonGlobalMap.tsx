import { EARTH, SUN } from '../lib/palette'
import { boundaryPath, oppositeHemispherePath, wrapLon, type SubPoint } from '../lib/terminator'

// NASA LRO WAC global mosaic, served as tiles by NASA Moon Trek. Level 2 is an 8 x 4 grid (2048 x 1024 px),
// plate carree: longitude -180..180 left to right with the near side's centre (0 deg) in the middle.
const TILES = 'https://trek.nasa.gov/tiles/Moon/EQ/LRO_WAC_Mosaic_Global_303ppd_v02/1.0.0/default/default028mm/2'
const COLS = 8
const ROWS = 4

type Props = {
  subSolar: SubPoint
  subEarth: SubPoint
  /** Make the map clickable: a chosen place is marked and reported back (latitude, longitude east, degrees). */
  chosen?: { lat: number; lon: number }
  onPick?: (lat: number, lon: number) => void
}

const pct = (v: number, total: number) => `${(100 * v) / total}%`

/**
 * The whole Moon with the real day/night boundary (from the Sun's sub-solar point) and the limit of what
 * can be seen from Earth (the 90-degree circle around the sub-Earth point). Everything beyond that dashed
 * line is the far side; libration is why the line is not a straight meridian and moves from week to week.
 */
export default function MoonGlobalMap({ subSolar, subEarth, chosen, onPick }: Props) {
  const sun = { x: wrapLon(subSolar.lon) + 180, y: 90 - subSolar.lat }
  const earth = { x: wrapLon(subEarth.lon) + 180, y: 90 - subEarth.lat }
  return (
    <div>
      <div
        className={`relative aspect-[2/1] w-full select-none overflow-hidden rounded-[10px] border border-line bg-[#1c211f] ${onPick ? 'cursor-crosshair' : ''}`}
        role={onPick ? 'application' : 'img'}
        aria-label={onPick ? 'Map of the whole Moon. Click to choose a place.' : 'Map of the whole Moon with the day and night sides'}
        onClick={
          onPick
            ? (e) => {
                const r = e.currentTarget.getBoundingClientRect()
                const lon = ((e.clientX - r.left) / r.width) * 360 - 180
                const lat = 90 - ((e.clientY - r.top) / r.height) * 180
                onPick(Math.round(Math.max(-90, Math.min(90, lat)) * 100) / 100, Math.round(lon * 100) / 100)
              }
            : undefined
        }
      >
        <div className="absolute inset-0 grid" style={{ gridTemplateColumns: `repeat(${COLS}, 1fr)`, gridTemplateRows: `repeat(${ROWS}, 1fr)` }}>
          {Array.from({ length: ROWS * COLS }, (_, i) => (
            <img key={i} src={`${TILES}/${Math.floor(i / COLS)}/${i % COLS}.jpg`} alt="" draggable={false} className="block h-full w-full" />
          ))}
        </div>
        <svg viewBox="0 0 360 180" preserveAspectRatio="none" className="pointer-events-none absolute inset-0 h-full w-full" aria-hidden="true">
          <path data-region="far-side" d={oppositeHemispherePath(subEarth)} fill="#ffffff" fillOpacity="0.1" />
          <path data-region="night" d={oppositeHemispherePath(subSolar)} fill="#050912" fillOpacity="0.58" />
          <path d={boundaryPath(subEarth)} fill="none" stroke="#fff" strokeOpacity="0.75" strokeWidth="0.6" strokeDasharray="3 2" vectorEffect="non-scaling-stroke" />
          <line x1="0" x2="360" y1="175" y2="175" stroke="#fff" strokeOpacity="0.8" strokeWidth="1" strokeDasharray="4 3" vectorEffect="non-scaling-stroke" />
        </svg>
        <span className="pointer-events-none absolute -translate-x-1/2 -translate-y-1/2" style={{ left: pct(sun.x, 360), top: pct(sun.y, 180) }} title="Sun overhead">
          <span className="block h-4 w-4 rounded-full" style={{ background: SUN }} />
        </span>
        <span className="pointer-events-none absolute -translate-x-1/2 -translate-y-1/2" style={{ left: pct(earth.x, 360), top: pct(earth.y, 180) }} title="Earth overhead">
          <span className="block h-4 w-4 rounded-full" style={{ background: EARTH }} />
        </span>
        {chosen && (
          <span className="pointer-events-none absolute -translate-x-1/2 -translate-y-1/2" style={{ left: pct(wrapLon(chosen.lon) + 180, 360), top: pct(90 - chosen.lat, 180) }}>
            <span className="block h-4 w-4 rounded-full border-2 border-white shadow-[0_0_0_1.5px_#24231f]" />
          </span>
        )}
        <span className="pointer-events-none absolute bottom-[3.5%] left-2 text-[11px] text-white/90">85°S: the polar region with terrain data</span>
      </div>
      <div className="mt-2 flex flex-wrap items-center gap-x-6 gap-y-1 text-[13px] text-muted">
        <span className="flex items-center gap-2"><span className="inline-block h-3 w-3 rounded-full" style={{ background: SUN }} /> Sun overhead</span>
        <span className="flex items-center gap-2"><span className="inline-block h-3 w-3 rounded-full" style={{ background: EARTH }} /> Earth overhead</span>
        <span className="flex items-center gap-2"><span className="inline-block h-3 w-6 bg-[#050912]/60" /> night</span>
        <span className="flex items-center gap-2"><span className="inline-block h-0 w-6 border-t border-dashed border-ink" /> edge of what Earth can see</span>
        <span className="ml-auto text-[12px]">Image: NASA / GSFC / Arizona State University, LRO WAC mosaic, via Moon Trek</span>
      </div>
    </div>
  )
}
