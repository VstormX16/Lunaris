import type { Spacecraft } from '../data/spacecrafts'

export type FieldKey = 'battery_wh' | 'solar_w' | 'radioisotope_heat' | 'max_slope_deg' | 'survival_power_w' | 'payload_power_w' | 'comms'
export type Missing = { label: string; note: string }

const NOT_PUBLISHED = 'Yayımlanmamış'

/**
 * Why the landing analysis has no value for a field, in the garage's own terms. The garage often knows something about the vehicle
 * (battery chemistry, number of units, voltage) without knowing the one number the analysis needs; saying so is better than a bare "Unknown".
 */
export function missingNote(key: FieldKey, c: Spacecraft): Missing {
  switch (key) {
    case 'battery_wh': {
      const parts = [c.batteryType, c.batteryCount ? `${c.batteryCount} unit${c.batteryCount === '1' ? '' : 's'}` : '', c.voltage ? `${c.voltage} V` : ''].filter((p) => p && p !== 'Uygulanmaz')
      const known = parts.length ? `The garage lists ${parts.join(' · ')}, but ` : ''
      return !c.batteryCapacity || c.batteryCapacity === NOT_PUBLISHED
        ? { label: 'Not published', note: known ? `${known}no capacity, so the analysis cannot count the battery.` : 'The battery capacity is not published, so the analysis cannot count the battery.' }
        : { label: 'Not converted', note: `The garage lists "${c.batteryCapacity}", which could not be turned into watt-hours.` }
    }
    case 'radioisotope_heat':
      return c.rtgW === '0' || c.rtgW === ''
        ? { label: 'Not stated', note: 'The garage lists no radioisotope unit, but that does not prove there is no heater, so the analysis treats it as unknown rather than as "none".' }
        : { label: 'Unknown', note: '' }
    case 'max_slope_deg':
      return { label: 'Not published', note: 'No landing tilt limit is published for this vehicle; the analysis uses a generic 15° for the score and says so.' }
    case 'survival_power_w':
    case 'payload_power_w':
      return { label: 'Not published', note: 'The dataset has no figure for this.' }
    default:
      return { label: 'Unknown', note: '' }
  }
}
