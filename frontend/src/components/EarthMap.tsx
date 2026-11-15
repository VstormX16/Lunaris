import { oppositeHemispherePath } from '../lib/terminator'

type Props = { lat: number; lon: number; subsolar: { lat: number; lon: number } | null; onPick: (lat: number, lon: number) => void }

// NASA GIBS "Blue Marble" (shaded relief + bathymetry), whole globe in plate carree.
const BLUE_MARBLE =
  'https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi?SERVICE=WMS&REQUEST=GetMap&VERSION=1.3.0&LAYERS=BlueMarble_ShadedRelief_Bathymetry&STYLES=&FORMAT=image/jpeg&CRS=EPSG:4326&BBOX=-90,-180,90,180&WIDTH=1440&HEIGHT=720'

/** Blue Marble world map with the real day/night boundary. Click to choose a place. */
export default function EarthMap({ lat, lon, subsolar, onPick }: Props) {
  const norm = ((lon + 180) % 360 + 360) % 360 - 180 // the map runs -180..180
  return (
    <div>
      <div
        className="relative aspect-[2/1] w-full cursor-crosshair select-none overflow-hidden rounded-[10px] border border-line bg-[#1c211f]"
        onClick={(e) => {
          const r = e.currentTarget.getBoundingClientRect()
          const pickLon = ((e.clientX - r.left) / r.width) * 360 - 180
          const pickLat = 90 - ((e.clientY - r.top) / r.height) * 180
          onPick(Math.round(pickLat * 100) / 100, Math.round(pickLon * 100) / 100)
        }}
        role="application"
        aria-label="World map. Click to choose a place."
      >
        <img src={BLUE_MARBLE} alt="NASA Blue Marble world map" draggable={false} className="absolute inset-0 h-full w-full" />
        {subsolar && (
          <svg viewBox="0 0 360 180" preserveAspectRatio="none" className="pointer-events-none absolute inset-0 h-full w-full" aria-hidden="true">
            <path d={oppositeHemispherePath(subsolar)} fill="#0b1626" fillOpacity="0.48" />
          </svg>
        )}
        <span className="pointer-events-none absolute -translate-x-1/2 -translate-y-1/2" style={{ left: `${((norm + 180) / 360) * 100}%`, top: `${((90 - lat) / 180) * 100}%` }}>
          <span className="block h-4 w-4 rounded-full border-2 border-white shadow-[0_0_0_1.5px_#24231f]" />
        </span>
      </div>
      <div className="mt-2 text-[13px] tabular-nums text-muted">
        Chosen: <span className="font-semibold text-strong">{lat.toFixed(2)}°, {lon.toFixed(2)}°</span>
        <span className="ml-3">Shaded area is night</span>
      </div>
    </div>
  )
}
