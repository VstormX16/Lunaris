import type { PointReport } from '../api'
import { compass } from '../lib/compass'
import { EARTH, SUN } from '../lib/palette'

const deg = (v: number, digits = 1) => `${v.toFixed(digits)}°`

function Fact({ value, label }: { value: string; label: string }) {
  return (
    <div>
      <div className="text-[28px] font-bold leading-none tabular-nums text-strong">{value}</div>
      <div className="mt-1.5 text-[12px] uppercase tracking-wide text-muted">{label}</div>
    </div>
  )
}

function Section({ title, verdict, children }: { title: string; verdict?: { text: string; good: boolean }; children: React.ReactNode }) {
  return (
    <section className="border-t border-line pt-5">
      <div className="flex flex-wrap items-baseline gap-x-4 gap-y-1">
        <h3 className="text-[13px] uppercase tracking-wide text-muted">{title}</h3>
        {verdict && (
          <span className="flex items-center gap-2 text-[17px] font-bold text-strong">
            <span className={`h-2.5 w-2.5 rounded-full ${verdict.good ? 'bg-brand' : 'bg-[#ba1a1a]'}`} />
            {verdict.text}
          </span>
        )}
      </div>
      {children}
    </section>
  )
}

/** One row of 24 hourly cells for the chosen date. */
function Strip({ label, color, on, tips, now }: { label: string; color: string; on: boolean[]; tips: string[]; now: number }) {
  return (
    <div className="mt-3">
      <div className="mb-1 flex justify-between text-[13px] text-muted">
        <span>{label}</span>
        <span className="tabular-nums">{on.filter(Boolean).length} of 24 hours</span>
      </div>
      <div className="flex gap-[3px]">
        {on.map((v, h) => (
          <span
            key={h}
            title={`${String(h).padStart(2, '0')}:00 UTC · ${tips[h]}`}
            className="h-6 flex-1 rounded-[3px]"
            style={{ background: v ? color : 'rgba(85,83,78,0.14)', outline: h === now ? '2px solid #24231f' : undefined, outlineOffset: 1 }}
          />
        ))}
      </div>
    </div>
  )
}

export default function PointReportView({ r, hourUtc }: { r: PointReport; hourUtc: number }) {
  const { sun, earth, surface, day } = r
  const smooth = r.terrain === 'smooth'

  const sunVerdict = sun.lit ? 'Direct sunlight' : sun.elevation_deg > 0 ? 'Sun hidden behind terrain' : 'Night: the Sun is below the horizon'
  const earthVerdict = earth.visible ? 'Earth is in line of sight' : earth.blocked_by_terrain ? 'Blocked by terrain' : 'Earth is below the horizon'

  return (
    <div className="mt-8 space-y-7">
      <Section title="The Sun" verdict={{ text: sunVerdict, good: sun.lit }}>
        <div className="mt-4 grid grid-cols-2 gap-x-10 gap-y-5 sm:grid-cols-4">
          <Fact value={deg(sun.elevation_deg)} label="Height above the horizon" />
          <Fact value={`${deg(sun.azimuth_deg, 0)} ${compass(sun.azimuth_deg)}`} label="Direction" />
          <Fact value={deg(sun.incidence_flat_deg)} label="Angle of the rays on level ground" />
          <Fact value={sun.incidence_slope_deg === null ? 'Unknown' : deg(sun.incidence_slope_deg)} label="Angle of the rays on this slope" />
        </div>
        <p className="mt-4 max-w-[780px] text-[14px] leading-relaxed text-muted">
          The angle is measured from straight down: 0° is the Sun overhead, 90° is light skimming the ground.
          {sun.incidence_slope_deg !== null && sun.incidence_slope_deg > 90 && ' Above 90° the ground itself faces away from the Sun.'}
          {smooth
            ? ' There is no terrain data here, so the angle on the slope cannot be given.'
            : ` The skyline toward the Sun is ${deg(sun.skyline_deg)} high, so the Sun must climb above that to shine on this spot.`}
        </p>
      </Section>

      <Section title="Line of sight to Earth" verdict={{ text: earthVerdict, good: earth.visible }}>
        <div className="mt-4 grid grid-cols-2 gap-x-10 gap-y-5 sm:grid-cols-4">
          <Fact value={deg(earth.elevation_deg)} label="Earth above the horizon" />
          <Fact value={`${deg(earth.azimuth_deg, 0)} ${compass(earth.azimuth_deg)}`} label="Direction" />
          <Fact value={smooth ? 'Unknown' : deg(earth.skyline_deg)} label="Terrain skyline that way" />
          <Fact value={`${Math.round(earth.distance_km).toLocaleString('en')} km`} label="Distance" />
        </div>
        <p className="mt-4 max-w-[780px] text-[14px] leading-relaxed text-muted">
          {earth.visible && r.terrain === 'lola' && `Earth clears the skyline by ${deg(earth.margin_deg)}, so a direct-to-Earth radio link works at this moment.`}
          {earth.visible && r.terrain === 'global' && `Earth clears the coarse skyline by ${deg(earth.margin_deg)}. The grid is smooth at the scale of a few hundred metres, so a small ridge nearby could still hide it.`}
          {earth.visible && smooth && 'Earth is above the horizon. Without terrain data, hills or crater walls that could still hide it are not checked.'}
          {earth.blocked_by_terrain &&
            `Earth is ${deg(earth.elevation_deg)} up, but the terrain toward it rises to ${deg(earth.skyline_deg)}${earth.blocker_distance_m ? `, a ridge about ${(earth.blocker_distance_m / 1000).toFixed(1)} km away` : ''}. It would need a relay or a taller mast.`}
          {earth.below_horizon && 'Earth is below the horizon from here at this time. Only the near side ever faces Earth, so on the far side it never rises.'}
        </p>
      </Section>

      <Section title="The ground">
        {surface ? (
          <div className="mt-4 grid grid-cols-2 gap-x-10 gap-y-5 sm:grid-cols-4">
            <Fact value={deg(surface.slope_deg)} label="Slope" />
            <Fact value={`${deg(surface.downhill_azimuth_deg, 0)} ${compass(surface.downhill_azimuth_deg)}`} label="Downhill direction" />
            <Fact value={`${Math.round(surface.height_m).toLocaleString('en')} m`} label="Height vs mean Moon radius" />
          </div>
        ) : (
          <p className="mt-3 max-w-[780px] text-[15px] text-ink">
            Slope and height are unknown here: no elevation data is loaded for this place, and a value is never guessed.
          </p>
        )}
        {surface && (
          <p className="mt-4 max-w-[780px] text-[14px] leading-relaxed text-muted">
            {r.terrain === 'lola'
              ? `Slope is measured over ${surface.baseline_m} m on each side of the point, from NASA LOLA elevation data (40 m detail).`
              : `Slope is regional: measured over ${(surface.baseline_m / 1000).toFixed(0)} km on each side of the point, from the whole-Moon NASA LOLA grid (about 470 m per pixel). Steep spots smaller than that, such as crater walls or boulders, are smoothed away, so the real slope under a lander can be steeper.`}
          </p>
        )}
      </Section>

      <Section title="That day, hour by hour (UTC)">
        <Strip label="Sunlight" color={SUN} on={day.sun_lit} tips={day.sun_elevation_deg.map((e) => `Sun ${e.toFixed(1)}°`)} now={hourUtc} />
        <Strip label="Earth in line of sight" color={EARTH} on={day.earth_visible} tips={day.earth_elevation_deg.map((e) => `Earth ${e.toFixed(1)}°`)} now={hourUtc} />
        <div className="mt-1 flex justify-between text-[12px] tabular-nums text-muted">
          <span>00:00</span>
          <span>12:00</span>
          <span>23:00</span>
        </div>
        <p className="mt-3 text-[13px] text-muted">The outlined hour is the time you picked. Hover a cell for the angle.</p>
      </Section>
    </div>
  )
}
