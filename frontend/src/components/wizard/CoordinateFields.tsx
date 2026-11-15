import { useState } from 'react'
import { parseLat, parseLon } from '../../lib/input'
import { inputCls } from './ui'

type Props = { lat: string; lon: string; onLat: (v: string) => void; onLon: (v: string) => void; stacked?: boolean }

export default function CoordinateFields({ lat, lon, onLat, onLon, stacked = false }: Props) {
  const [touched, setTouched] = useState({ lat: false, lon: false })
  const latBad = touched.lat && parseLat(lat) === null
  const lonBad = touched.lon && parseLon(lon) === null

  const field = (label: string, value: string, set: (v: string) => void, key: 'lat' | 'lon', bad: boolean, hint: string, placeholder: string) => (
    <label className="block">
      <span className="text-[13px] uppercase tracking-wide text-muted">{label}</span>
      <input
        value={value}
        onChange={(e) => set(e.target.value)}
        onBlur={() => setTouched((t) => ({ ...t, [key]: true }))}
        inputMode="decimal"
        placeholder={placeholder}
        className={`${inputCls} mt-1.5 ${bad ? 'border-amber-600' : ''}`}
      />
      <span className={`mt-1 block h-5 text-[13px] ${bad ? 'text-amber-700' : 'text-muted'}`}>{hint}</span>
    </label>
  )

  return (
    <div className={`grid gap-3 ${stacked ? 'grid-cols-1' : 'grid-cols-2 gap-5'}`}>
      {field('Latitude', lat, onLat, 'lat', latBad, '−90 (south pole) to 90 (north pole)', '-85.987')}
      {field('Longitude', lon, onLon, 'lon', lonBad, '0 to 360, or −180 to 180 (east positive)', '2.08')}
    </div>
  )
}
