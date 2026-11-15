import L from 'leaflet'
import 'leaflet/dist/leaflet.css'
import { useEffect, useRef } from 'react'
import type { MoonSite } from '../api'
import { wrapLon } from '../lib/terminator'

// NASA LRO WAC global mosaic from Moon Trek. At zoom z the grid is 2^(z+1) x 2^z tiles of 256 px, which is exactly
// Leaflet's EPSG:4326 layout (longitude -180..180 left to right, latitude 90..-90 top to bottom).
const TILES = 'https://trek.nasa.gov/tiles/Moon/EQ/LRO_WAC_Mosaic_Global_303ppd_v02/1.0.0/default/default028mm/{z}/{y}/{x}.jpg'
const INK = '#24231f'
const BRAND = '#3d7a68'
const DATA_EDGE = -85 // LOLA elevation data covers the south pole from here down

export type Point = { lat: number; lon: number }
type Props = { sites: MoonSite[]; chosen: Point | null; onPick: (p: Point, site?: MoonSite) => void; onHover: (p: Point | null) => void }

const round = (v: number) => Math.round(v * 1000) / 1000
const dms = (v: number, pos: string, neg: string) => `${Math.abs(v).toFixed(3)}° ${v < 0 ? neg : pos}`

/** The whole Moon as a flat map, with the places we know marked on it. Click anywhere to choose a point. */
export default function MoonSurfaceMap({ sites, chosen, onPick, onHover }: Props) {
  const el = useRef<HTMLDivElement>(null)
  const map = useRef<L.Map | null>(null)
  const siteLayer = useRef<L.LayerGroup | null>(null)
  const mark = useRef<L.LayerGroup | null>(null)
  const handlers = useRef({ onPick, onHover })
  handlers.current = { onPick, onHover }

  useEffect(() => {
    if (!el.current) return
    const bounds: L.LatLngBoundsLiteral = [[-90, -180], [90, 180]]
    // a fixed picture of the whole Moon: no panning, no zooming, edge to edge
    const m = L.map(el.current, {
      crs: L.CRS.EPSG4326, zoomSnap: 0, attributionControl: false, zoomControl: false,
      dragging: false, scrollWheelZoom: false, doubleClickZoom: false, boxZoom: false, keyboard: false, touchZoom: false,
    })
    L.tileLayer(TILES, { noWrap: true, bounds, maxNativeZoom: 7, tileSize: 256 }).addTo(m)
    const fit = () => {
      m.invalidateSize()
      const z = m.getBoundsZoom(bounds, true) // largest zoom at which the whole Moon still fits, exactly
      m.setView([0, 0], z, { animate: false })
      m.setMinZoom(z)
      m.setMaxZoom(z)
    }
    fit()
    const watch = new ResizeObserver(fit)
    watch.observe(el.current)
    L.rectangle([[-90, -180], [DATA_EDGE, 180]], { color: '#fff', weight: 1, opacity: 0.8, dashArray: '5 5', fill: false, interactive: false }).addTo(m)
    siteLayer.current = L.layerGroup().addTo(m)
    m.on('click', (e) => handlers.current.onPick({ lat: round(Math.max(-90, Math.min(90, e.latlng.lat))), lon: round(wrapLon(e.latlng.lng)) }))
    m.on('mousemove', (e) => handlers.current.onHover({ lat: round(e.latlng.lat), lon: round(wrapLon(e.latlng.lng)) }))
    m.on('mouseout', () => handlers.current.onHover(null))
    map.current = m
    return () => {
      watch.disconnect()
      m.remove()
      map.current = null
    }
  }, [])

  // known places: dark dot = landing, ring = crash, green = south-pole candidate
  useEffect(() => {
    const g = siteLayer.current
    if (!g) return
    g.clearLayers()
    for (const s of sites) {
      const style: L.CircleMarkerOptions =
        s.kind === 'landing' ? { fillColor: '#fff', fillOpacity: 1, color: INK, weight: 2 }
        : s.kind === 'crash' ? { fillColor: INK, fillOpacity: 1, color: '#fff', weight: 1.5 }
        : { fillColor: BRAND, fillOpacity: 1, color: '#fff', weight: 1.5 }
      const mk = L.circleMarker([s.lat, s.lon], { radius: 5.5, ...style })
      const what = s.kind === 'landing' ? 'Landing site' : s.kind === 'crash' ? 'Crash site' : 'South-pole candidate'
      mk.bindTooltip(`<b>${s.name}</b><br>${what}<br>${dms(s.lat, 'N', 'S')}, ${dms(s.lon, 'E', 'W')}`, { direction: 'top', offset: [0, -6] })
      mk.on('click', (e) => {
        L.DomEvent.stopPropagation(e)
        handlers.current.onPick({ lat: s.lat, lon: s.lon }, s)
      })
      mk.addTo(g)
    }
  }, [sites])

  // the chosen point
  useEffect(() => {
    const m = map.current
    if (!m) return
    mark.current?.remove()
    mark.current = null
    if (chosen) {
      // a white ring with a dark edge: readable on dark seas and bright highlands alike
      const at: L.LatLngTuple = [chosen.lat, chosen.lon]
      mark.current = L.layerGroup([
        L.circleMarker(at, { radius: 8, fill: false, color: INK, weight: 1.5, interactive: false }),
        L.circleMarker(at, { radius: 6.5, fill: false, color: '#fff', weight: 3, interactive: false }),
      ]).addTo(m)
    }
  }, [chosen])

  return <div ref={el} style={{ background: '#1c211f' }} className="relative z-0 isolate aspect-[2/1] w-full cursor-crosshair overflow-hidden rounded-[10px] border border-line" role="application" aria-label="Map of the whole Moon. Click to choose a point." />
}
