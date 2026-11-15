import type { VehicleProfile } from '../api'
import type { Spacecraft } from '../data/spacecrafts'

const BLANK: Omit<Spacecraft, 'id' | 'no' | 'name' | 'year' | 'country' | 'type' | 'massKg'> = {
  outcome: '', massNote: '', height: '', width: '', legSpan: '', otherDims: '', powerSource: '', batteryType: '', batteryCount: '', batteryCapacity: '', solarW: '', rtgW: '',
  voltage: '', wheels: '', wheelDiameter: '', legs: '', landingMechanism: '', speed: '', computer: '', clockMhz: '', software: '', navSensors: '', cameras: '', instruments: '',
  results: '', propulsion: '', landerThrustN: '', ascentThrustN: '', serviceThrustN: '', rcs: '', otherPropulsion: '', antenna: '', band: '', dataRate: '', commDetails: '',
  sourceUrl: '', dimensions: '', survivalW: '', payloadW: '', slopeLimit: '', heatSource: '', datasetUrl: '', verification: '', checkLevel: 'unchecked', checkNote: '', corrections: [],
}

/** A garage-designed vehicle dressed as a spacecraft entry, so the operation wizard can list and show it like the others. */
export function customCraft(p: VehicleProfile): Spacecraft {
  return { ...BLANK, id: p.id, no: 0, name: p.name, year: new Date().getFullYear(), country: 'My vehicles', type: p.kind === 'rover' ? 'Rover (custom)' : 'Lander (custom)', massKg: p.mass_kg ? String(p.mass_kg) : '' }
}
