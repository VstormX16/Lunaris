import L from 'leaflet'
import 'leaflet/dist/leaflet.css'
import { useEffect, useRef } from 'react'
import type { MoonSite, PointReport } from '../api'
import { compass } from '../lib/compass'
import { EARTH, SUN } from '../lib/palette'
import { wrapLon } from '../lib/terminator'
import type { Point } from './MoonSurfaceMap'

const TILES = 'https://trek.nasa.gov/tiles/Moon/EQ/LRO_WAC_Mosaic_Global_303ppd_v02/1.0.0/default/default028mm/{z}/{y}/{x}.jpg'
const INK = '#24231f'
const ZOOM = 6 // about 340 m per screen pixel at the equator; the picture itself goes down to about 170 m
const round = (v: number) => Math.round(v * 1000) / 1000

type Props = { center: Point; chosen: Point; sites: MoonSite[]; report: PointReport | null; onPick: (p: Point, s?: MoonSite) => void }

const SPAN_PX = 80 // length of a direction line on screen

/**
 * A close-up of the area around the chosen point, for placing it more exactly than the whole-Moon map allows.
 * Drag to look around, click to move the point. It re-centres only when the point is chosen on the big map.
 */
export default function PointLoupe({ center, chosen, sites, report, onPick }: Props) {
  const el = useRef<HTMLDivElement>(null)
  const map = useRef<L.Map | null>(null)
  const mark = useRef<L.CircleMarker | null>(null)
  const siteLayer = useRef<L.LayerGroup | null>(null)
  const rayLayer = useRef<L.LayerGroup | null>(null)
  const draw = useRef<() => void>(() => undefined)
  const handler = useRef(onPick)
  handler.current = onPick

  useEffect(() => {
    if (!el.current) return
    const m = L.map(el.current, { crs: L.CRS.EPSG4326, minZoom: 4, maxZoom: 7, zoomSnap: 1, attributionControl: false, scrollWheelZoom: false, doubleClickZoom: false })
    L.tileLayer(TILES, { noWrap: true, maxNativeZoom: 7, tileSize: 256 }).addTo(m)
    siteLayer.current = L.layerGroup().addTo(m)
    rayLayer.current = L.layerGroup().addTo(m)
    m.on('zoomend', () => draw.current())
    m.on('click', (e) => handler.current({ lat: round(Math.max(-90, Math.min(90, e.latlng.lat))), lon: round(wrapLon(e.latlng.lng)) }))
    map.current = m
    return () => {
      m.remove()
      map.current = null
    }
  }, [])

  useEffect(() => {
    map.current?.setView([center.lat, center.lon], ZOOM, { animate: false })
  }, [center])

  useEffect(() => {
    const g = siteLayer.current
    if (!g) return
    g.clearLayers()
    for (const s of sites) {
      const style: L.CircleMarkerOptions =
        s.kind === 'landing' ? { fillColor: '#fff', fillOpacity: 1, color: INK, weight: 2 }
        : s.kind === 'crash' ? { fillColor: INK, fillOpacity: 1, color: '#fff', weight: 1.5 }
        : { fillColor: '#3d7a68', fillOpacity: 1, color: '#fff', weight: 1.5 }
      const mk = L.circleMarker([s.lat, s.lon], { radius: 5.5, ...style }).bindTooltip(`<b>${s.name}</b>`, { direction: 'top', offset: [0, -6] })
      mk.on('click', (e) => {
        L.DomEvent.stopPropagation(e)
        handler.current({ lat: s.lat, lon: s.lon }, s)
      })
      mk.addTo(g)
    }
  }, [sites])

  useEffect(() => {
    const m = map.current
    if (!m) return
    mark.current?.remove()
    mark.current = L.circleMarker([chosen.lat, chosen.lon], { radius: 4, fillColor: INK, fillOpacity: 1, color: '#fff', weight: 2, interactive: false }).addTo(m)
  }, [chosen])

  // Sun and Earth directions, drawn once the analysis has an answer. On this map one pixel is the same number of
  // degrees both ways, but a degree of longitude is only cos(latitude) as long as a degree of latitude, so the true
  // compass direction (east, north) appears on screen as (east / cos(lat), north).
  useEffect(() => {
    draw.current = () => {
      const m = map.current
      const g = rayLayer.current
      if (!m || !g) return
      g.clearLayers()
      if (!report) return
      const stretch = 1 / Math.max(Math.cos((report.lat * Math.PI) / 180), 0.05)
      const origin = m.latLngToContainerPoint([report.lat, report.lon])
      const lines = [
        { color: SUN, az: report.sun.azimuth_deg, solid: report.sun.lit,
          text: report.sun.lit ? `Sun · ${Math.round(report.sun.azimuth_deg)}° ${compass(report.sun.azimuth_deg)} · ${report.sun.elevation_deg.toFixed(0)}° up`
            : report.sun.elevation_deg > 0 ? 'Sun · hidden by terrain' : 'Sun · below the horizon' },
        { color: EARTH, az: report.earth.azimuth_deg, solid: report.earth.visible,
          text: report.earth.visible ? `Earth · ${Math.round(report.earth.azimuth_deg)}° ${compass(report.earth.azimuth_deg)} · ${report.earth.elevation_deg.toFixed(0)}° up`
            : report.earth.blocked_by_terrain ? 'Earth · blocked by terrain' : 'Earth · below the horizon' },
      ]
      for (const ln of lines) {
        const a = (ln.az * Math.PI) / 180
        const vx = Math.sin(a) * stretch
        const vy = -Math.cos(a) // screen y points down
        const n = Math.hypot(vx, vy)
        const ux = vx / n
        const uy = vy / n
        const at = (d: number) => m.containerPointToLatLng(L.point(origin.x + ux * d, origin.y + uy * d))
        const opacity = ln.solid ? 1 : 0.8
        // a pale under-stroke keeps the line readable over both dark seas and bright highlands
        L.polyline([at(12), at(SPAN_PX)], { color: '#fff', weight: 5.5, opacity: 0.55, className: 'ray-line', interactive: false }).addTo(g)
        L.polyline([at(12), at(SPAN_PX)], { color: ln.color, weight: 3, opacity, dashArray: ln.solid ? undefined : '6 5', className: 'ray-line', interactive: false }).addTo(g)
        // the label sits beyond the arrow tip, never on top of the line
        const endX = origin.x + ux * SPAN_PX
        const width = m.getSize().x
        const horizontal = Math.abs(ux) > Math.abs(uy)
        const side: 'top' | 'bottom' | 'left' | 'right' = horizontal
          ? endX < 110 ? 'right' : endX > width - 110 ? 'left' : uy > 0 ? 'bottom' : 'top'
          : endX > width - 190 ? 'left' : 'right'
        const offset: L.PointTuple = side === 'top' ? [0, -8] : side === 'bottom' ? [0, 8] : side === 'left' ? [-8, 0] : [8, 0]
        const deg = (Math.atan2(ux, -uy) * 180) / Math.PI
        L.marker(at(SPAN_PX), {
          interactive: false,
          icon: L.divIcon({ className: 'ray-line', iconSize: [14, 14], iconAnchor: [7, 7],
            html: `<svg width="14" height="14" viewBox="0 0 14 14" style="transform:rotate(${deg}deg);opacity:${opacity}"><path d="M7 1 L12 12 L7 9.5 L2 12 Z" fill="${ln.color}"/></svg>` }),
        }).addTo(g).bindTooltip(ln.text, { permanent: true, direction: side, offset, className: 'ray-label' })
      }
    }
    draw.current()
  }, [report])

  return <div ref={el} style={{ background: '#1c211f' }} className="relative z-0 isolate h-[280px] w-full cursor-crosshair overflow-hidden rounded-[10px] border border-line" role="application" aria-label="Close-up of the chosen point. Click to move the point." />
}
