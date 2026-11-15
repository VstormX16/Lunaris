import { useEffect, useMemo, useState } from 'react'
import { getMoonSites, getSites, type MoonSite } from '../../api'
import { parseLat, parseLon } from '../../lib/input'
import Viewer from '../moon3d/Viewer'
import MoonSurfaceMap from '../MoonSurfaceMap'
import CoordinateFields from './CoordinateFields'
import { GlassButton, StepFooter } from './ui'

type Props = {
  lat: string
  lon: string
  onChange: (p: { lat: string; lon: string }) => void
  onBack: () => void
  onNext: () => void
  nextLabel?: string
}

type View = '3d' | '2d'
const VIEW_KEY = 'lunaris-location-view'
const readView = (): View => {
  try {
    return localStorage.getItem(VIEW_KEY) === '2d' ? '2d' : '3d'
  } catch {
    return '3d'
  }
}
const r3 = (v: number) => String(Math.round(v * 1000) / 1000)

export default function LocationStep({ lat, lon, onChange, onBack, onNext, nextLabel = 'Next' }: Props) {
  const [sites, setSites] = useState<{ name: string; lat: number; lon: number }[]>([])
  const [moonSites, setMoonSites] = useState<MoonSite[]>([])
  const [view, setViewState] = useState<View>(readView)
    const [fly, setFly] = useState<{ lat: number; lon: number; n: number } | undefined>()
  const [webglError, setWebglError] = useState('')
  useEffect(() => {
    getSites().then(setSites).catch(() => setSites([])) // shortcuts are optional
    getMoonSites().then(setMoonSites).catch(() => setMoonSites([]))
  }, [])

  const setView = (v: View) => {
    setViewState(v)
    try {
      localStorage.setItem(VIEW_KEY, v)
    } catch {
      /* not remembered */
    }
  }

  const la = parseLat(lat)
  const lo = parseLon(lon)
  const valid = la !== null && lo !== null
  const chosen = useMemo(() => (la !== null && lo !== null ? { lat: la, lon: lo > 180 ? lo - 360 : lo } : null), [la, lo])

  // any point on the Moon can be analysed
  const pick = (p: { lat: number; lon: number }) => {
    onChange({ lat: r3(Math.max(-90, Math.min(90, p.lat))), lon: r3(p.lon) })
  }
  const goSite = (s: { lat: number; lon: number }) => {
    onChange({ lat: String(s.lat), lon: String(s.lon) })
    setFly((f) => ({ lat: s.lat, lon: s.lon > 180 ? s.lon - 360 : s.lon, n: (f?.n ?? 0) + 1 }))
  }

  return (
    <>
      <div className="flex min-h-0 flex-1 flex-col gap-4 py-5">
        <div className="flex items-center justify-between">
          <div className="text-[13px] uppercase tracking-wide text-muted">Pick on the Moon</div>
          <div role="radiogroup" aria-label="Map view" className="grid grid-cols-2 rounded-[8px] border border-line p-0.5 text-[14px]">
            {(['3d', '2d'] as const).map((v) => (
              <button key={v} role="radio" aria-checked={view === v} onClick={() => setView(v)} className={`rounded-[6px] px-4 py-1 transition-colors ${view === v ? 'bg-active font-medium text-brand' : 'text-muted hover:text-ink'}`}>
                {v.toUpperCase()}
              </button>
            ))}
          </div>
        </div>

        {/* the map takes all the room there is; the 2D map keeps its 2:1 shape and is centred in it */}
        <div className="flex min-h-[340px] flex-1 items-center justify-center">
          {view === '3d' ? (
            <div className="h-full w-full overflow-hidden rounded-[12px] border border-line bg-[#05080c]">
              <Viewer initialLat={-55} initialLon={0} initialAlt={3000} mode="natural" sun={0} sunLat={-25} marker={chosen} focus={fly} showHint={false} onPick={(p) => pick(p)} onError={setWebglError} />
            </div>
          ) : (
            <div className="w-full max-h-full" style={{ maxWidth: 'min(100%, calc((100vh - 360px) * 2))' }}>
              <MoonSurfaceMap sites={moonSites} chosen={chosen} onPick={(p) => pick(p)} onHover={() => undefined} />
            </div>
          )}
        </div>

        <p className="text-[13px] leading-relaxed text-muted">
          {view === '3d' ? 'Drag to turn the Moon, scroll to come closer, click anywhere to choose the landing place.' : 'The whole Moon, flattened: click anywhere to choose the landing place.'}
          {webglError && <span className="ml-2 text-amber-700">{webglError}</span>}
        </p>

        <div className="grid grid-cols-[1fr_1fr_1.4fr] items-start gap-x-6 border-t border-line pt-4">
          <div className="col-span-2">
            <CoordinateFields lat={lat} lon={lon} onLat={(v) => onChange({ lat: v, lon })} onLon={(v) => onChange({ lat, lon: v })} />
          </div>
          <div>
            <div className="text-[13px] uppercase tracking-wide text-muted">Known sites</div>
            <div className="mt-2 flex flex-wrap gap-x-5 gap-y-1.5 text-[16px]">
              {sites.map((s) => (
                <button key={s.name} onClick={() => goSite(s)} className="capitalize text-ink hover:text-brand">{s.name.replace('_', ' ')}</button>
              ))}
            </div>
            {chosen && <p className="mt-3 text-[15px] tabular-nums text-ink">Chosen: {Math.abs(chosen.lat).toFixed(3)}°{chosen.lat < 0 ? 'S' : 'N'}, {Math.abs(chosen.lon).toFixed(3)}°{chosen.lon < 0 ? 'W' : 'E'}</p>}
          </div>
        </div>
      </div>
      <StepFooter onBack={onBack}>
        <GlassButton onClick={onNext} disabled={!valid}>{nextLabel}</GlassButton>
      </StepFooter>
    </>
  )
}
