import raw from './spacecrafts.json'

// Source: "Uzay Araç Güncel Bilgileri.txt" (converted by data_engine/build_spacecrafts.py). Values keep the
// source's language and wording; "Uygulanmaz" (not applicable) and "Yayımlanmamış" (not published) are literal.
export type Spacecraft = {
  id: string
  no: number
  name: string
  year: number
  country: string
  type: string
  outcome: string
  massKg: string
  massNote: string
  height: string
  width: string
  legSpan: string
  otherDims: string
  powerSource: string
  batteryType: string
  batteryCount: string
  batteryCapacity: string
  solarW: string
  rtgW: string
  voltage: string
  wheels: string
  wheelDiameter: string
  legs: string
  landingMechanism: string
  speed: string
  computer: string
  clockMhz: string
  software: string
  navSensors: string
  cameras: string
  instruments: string
  results: string
  propulsion: string
  landerThrustN: string
  ascentThrustN: string
  serviceThrustN: string
  rcs: string
  otherPropulsion: string
  antenna: string
  band: string
  dataRate: string
  commDetails: string
  sourceUrl: string
  /** fields of the 2026-11-15 dataset */
  dimensions: string
  survivalW: string
  payloadW: string
  slopeLimit: string
  heatSource: string
  datasetUrl: string
  verification: string
  checkLevel: 'nasa' | 'wikipedia' | 'unchecked'
  checkNote: string
  corrections: { field: string; from: string; to: string; why: string }[]
}

export const spacecrafts = raw as Spacecraft[]
