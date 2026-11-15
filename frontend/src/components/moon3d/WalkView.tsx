import { useEffect, useRef } from 'react'
import * as T from 'three'

export type Direction = { az: number; el: number } // degrees, azimuth from north toward east
export type ViewState = { yaw: number; pitch: number; fov: number; aspect: number }

export type Grids = { far: Float32Array; nFar: number; farRadiusM: number; near: Float32Array; nNear: number; nearRadiusM: number }

type Props = {
  grids: Grids
  sun: Direction | null
  earth: Direction | null
  earthRot: number[] | null // Earth-fixed axes in east/north/up of the observer, row-major 3x3
  onView?: (v: ViewState) => void
  onError?: (m: string) => void
}

const EYE_M = 2
const SKY_R = 150_000
const SUN_HALF = (0.2665 * Math.PI) / 180 // angular radius of the Sun
const EARTH_HALF = (0.96 * Math.PI) / 180 // mean angular radius of Earth seen from the Moon
const HALO = 2.6 // the Sun's picture is this many discs wide; the extra is a very faint falloff, not a glow
/** Scene axes: x east, y up, -z north. */
const dir = (d: Direction) => {
  const a = (d.az * Math.PI) / 180, e = (d.el * Math.PI) / 180
  return new T.Vector3(Math.cos(e) * Math.sin(a), Math.sin(e), -Math.cos(e) * Math.cos(a))
}

/** The Sun's disc: white-hot centre, limb darkening toward a warmer rim, and a barely visible falloff beyond it. */
function sunTexture() {
  const size = 512, c = document.createElement('canvas')
  c.width = c.height = size
  const g = c.getContext('2d')!
  const img = g.createImageData(size, size)
  const mid = size / 2, disc = mid / HALO
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const r = Math.hypot(x + 0.5 - mid, y + 0.5 - mid) / disc // 1 at the limb
      let rgba: number[]
      if (r <= 1) {
        const mu = Math.sqrt(1 - r * r)
        const k = 1 - 0.45 * (1 - mu) // linear limb darkening
        const edge = Math.min(1, (1 - r) * disc * 0.9) // one pixel of anti-aliasing
        rgba = [255 * k, 250 * k * (0.9 + 0.1 * mu), 238 * k * (0.7 + 0.3 * mu), 255 * Math.max(0, edge) + 255 * 0]
      } else {
        const a = Math.max(0, 1 - (r - 1) / (HALO - 1))
        rgba = [255, 244, 225, 255 * 0.07 * a * a * a]
      }
      img.data.set(rgba.map((v) => Math.max(0, Math.min(255, v))), (y * size + x) * 4)
    }
  }
  g.putImageData(img, 0, 0)
  const t = new T.CanvasTexture(c)
  t.colorSpace = T.SRGBColorSpace
  return t
}

/** Fine grain for the ground, so it does not look like a smooth plastic sheet. It is a looks-only surface, not data. */
function grain() {
  const size = 256, c = document.createElement('canvas')
  c.width = c.height = size
  const g = c.getContext('2d')!
  const img = g.createImageData(size, size)
  let seed = 7
  const rand = () => { seed = (seed * 1664525 + 1013904223) >>> 0; return seed / 4294967296 }
  for (let i = 0; i < size * size; i++) {
    const v = 205 + (rand() - 0.5) * 70
    img.data.set([v, v * 0.985, v * 0.96, 255], i * 4)
  }
  g.putImageData(img, 0, 0)
  const t = new T.CanvasTexture(c)
  t.wrapS = t.wrapT = T.RepeatWrapping
  t.colorSpace = T.SRGBColorSpace
  t.anisotropy = 8
  return t
}

/** First-person view from the surface: real terrain around one point, the Sun and Earth where they truly are, real shadows. */
export default function WalkView({ grids, sun, earth, earthRot, onView, onError }: Props) {
  const host = useRef<HTMLDivElement>(null)
  const latest = useRef({ sun, earth, earthRot, onView, onError })
  latest.current = { sun, earth, earthRot, onView, onError }

  useEffect(() => {
    const el = host.current!
    let renderer: T.WebGLRenderer
    try {
      renderer = new T.WebGLRenderer({ antialias: true, logarithmicDepthBuffer: true })
    } catch {
      latest.current.onError?.('WebGL could not start. Turn on hardware acceleration in the browser.')
      return
    }
    renderer.setPixelRatio(Math.min(devicePixelRatio, 1.75))
    renderer.setClearColor(0x000000)
    renderer.shadowMap.enabled = true
    renderer.shadowMap.type = T.PCFShadowMap
    renderer.autoClear = false
    el.appendChild(renderer.domElement)

    // two passes sharing one camera: the wide area (with its own wide shadow map), then the nearest square kilometres on top
    // (with a sharp shadow map). The wide terrain also casts into the near pass, as an invisible caster, so a far ridge still
    // throws its shadow across the ground at your feet.
    const scene = new T.Scene()
    const nearScene = new T.Scene()
    const camera = new T.PerspectiveCamera(60, 1, 0.3, 400_000)
    camera.rotation.order = 'YXZ'
    camera.position.set(0, EYE_M, 0)

    const { far, nFar, farRadiusM, near, nNear, nearRadiusM } = grids
    const texture = grain()
    const material = new T.MeshStandardMaterial({ map: texture, color: 0xa9a59d, roughness: 1, metalness: 0 })

    // x east, y height, z -north; row 0 is the north edge
    const build = (z: Float32Array, n: number, radius: number, hole: number) => {
      const step = (2 * radius) / (n - 1)
      const pos = new Float32Array(n * n * 3), uv = new Float32Array(n * n * 2)
      for (let r = 0; r < n; r++) {
        for (let c = 0; c < n; c++) {
          const i = r * n + c, e = -radius + c * step, no = radius - r * step
          pos.set([e, z[i], -no], i * 3)
          uv.set([e / 14, no / 14], i * 2)
        }
      }
      const idx: number[] = []
      for (let r = 0; r < n - 1; r++) {
        for (let c = 0; c < n - 1; c++) {
          // the wide mesh leaves out cells fully inside the near square
          const e0 = -radius + c * step, e1 = e0 + step, n0 = radius - (r + 1) * step, n1 = n0 + step
          if (hole > 0 && e0 >= -hole && e1 <= hole && n0 >= -hole && n1 <= hole) continue
          const a = r * n + c, b2 = a + 1, d = a + n, f = d + 1
          idx.push(a, d, b2, b2, d, f)
        }
      }
      const g = new T.BufferGeometry()
      g.setAttribute('position', new T.BufferAttribute(pos, 3))
      g.setAttribute('uv', new T.BufferAttribute(uv, 2))
      g.setIndex(new T.BufferAttribute(new Uint32Array(idx), 1))
      g.computeVertexNormals()
      return g
    }
    const farGeo = build(far, nFar, farRadiusM, nearRadiusM)
    const nearGeo = build(near, nNear, nearRadiusM, 0)
    const farMesh = new T.Mesh(farGeo, material)
    farMesh.position.y = -0.6 // where the two meshes overlap by a cell, the near one wins
    farMesh.castShadow = farMesh.receiveShadow = true
    scene.add(farMesh)
    const nearMesh = new T.Mesh(nearGeo, material)
    nearMesh.castShadow = nearMesh.receiveShadow = true
    nearScene.add(nearMesh)
    const casterMat = new T.MeshBasicMaterial({ colorWrite: false, depthWrite: false })
    const farCaster = new T.Mesh(farGeo, casterMat)
    farCaster.castShadow = true
    nearScene.add(farCaster)
    const radiusM = farRadiusM

    // light: the Sun is the only real light; a faint fill stands for light scattered off the ground and Earthshine; lifted a little so shadowed terrain stays readable
    const makeLight = (half: number, depth: number, res: number, normalBias: number) => {
      const l = new T.DirectionalLight(0xfff6e8, 3.2)
      l.castShadow = true
      l.shadow.mapSize.set(res, res)
      const c = l.shadow.camera
      c.left = c.bottom = -half
      c.right = c.top = half
      c.near = 1
      c.far = depth
      l.shadow.bias = -0.0004
      l.shadow.normalBias = normalBias
      return l
    }
    const light = makeLight(radiusM * 1.05, radiusM * 4, 4096, 1.5)
    const nearLight = makeLight(nearRadiusM * 1.4, radiusM * 4, 4096, 0.25)
    scene.add(light, light.target, new T.AmbientLight(0xdfe6f0, 0.22))
    nearScene.add(nearLight, nearLight.target, new T.AmbientLight(0xdfe6f0, 0.22))

    // the sky has its own scene, drawn first: stars, the Sun and Earth. Terrain drawn after it hides whatever is behind a ridge.
    const skyScene = new T.Scene()
    const skyLight = new T.DirectionalLight(0xffffff, 3.2) // lights Earth from the real Sun direction, whatever the local horizon
    skyScene.add(skyLight, skyLight.target, new T.AmbientLight(0xffffff, 0.015))

    // the Sun: a white-hot disc, a little darker and redder toward the rim (limb darkening); it is bright, not coloured
    const sunTex = sunTexture()
    const sunR = SKY_R * Math.tan(SUN_HALF)
    const sunDisk = new T.Mesh(new T.PlaneGeometry(sunR * 2 * HALO, sunR * 2 * HALO), new T.MeshBasicMaterial({ map: sunTex, transparent: true, toneMapped: false, depthWrite: false }))
    skyScene.add(sunDisk)

    // Earth: NASA's map of the surface and clouds, turned the way Earth really is turned at this moment and lit by the real Sun
    const earthTex = new T.TextureLoader().load('/data/earth-color.jpg')
    earthTex.colorSpace = T.SRGBColorSpace
    earthTex.anisotropy = 8
    const earthR = SKY_R * Math.sin(EARTH_HALF)
    const earthGeo = new T.SphereGeometry(earthR, 128, 64)
    const earth = new T.Mesh(earthGeo, new T.MeshStandardMaterial({ map: earthTex, roughness: 0.95, metalness: 0 }))
    earth.matrixAutoUpdate = false
    skyScene.add(earth)
    const orient = (d: Direction | null, rot: number[] | null) => {
      earth.visible = !!d
      if (!d) return
      // Earth-fixed axes in east/north/up (row-major M); sphere axes -> Earth-fixed -> east/north/up -> scene (x east, y up, -z north)
      const m = rot && rot.length === 9 ? rot : [1, 0, 0, 0, 1, 0, 0, 0, 1]
      const q = [[1, 0, 0], [0, 0, -1], [0, 1, 0]]
      const mq = [0, 1, 2].map((r) => [0, 1, 2].map((c) => m[r * 3] * q[0][c] + m[r * 3 + 1] * q[1][c] + m[r * 3 + 2] * q[2][c]))
      const p = dir(d).multiplyScalar(SKY_R)
      earth.matrix.set(mq[0][0], mq[0][1], mq[0][2], p.x, mq[2][0], mq[2][1], mq[2][2], p.y, -mq[1][0], -mq[1][1], -mq[1][2], p.z, 0, 0, 0, 1)
      earth.matrixWorldNeedsUpdate = true
    }
    const placeSun = (d: Direction | null) => {
      sunDisk.visible = !!d
      if (!d) return
      sunDisk.position.copy(dir(d).multiplyScalar(SKY_R))
      sunDisk.lookAt(0, 0, 0)
    }

    // background stars: a looks-only backdrop, not a star catalogue
    const stars: number[] = []
    let seed = 11
    const rand = () => { seed = (seed * 1664525 + 1013904223) >>> 0; return seed / 4294967296 }
    for (let i = 0; i < 1600; i++) {
      const u = rand() * 2 - 1, a = rand() * Math.PI * 2, s = Math.sqrt(1 - u * u)
      stars.push(300_000 * s * Math.cos(a), 300_000 * u, 300_000 * s * Math.sin(a))
    }
    const starGeo = new T.BufferGeometry()
    starGeo.setAttribute('position', new T.Float32BufferAttribute(stars, 3))
    const starMat = new T.PointsMaterial({ color: 0xb8c0c8, size: 1.6, sizeAttenuation: false })
    skyScene.add(new T.Points(starGeo, starMat))

    // looking around
    let yaw = 0, pitch = 3, fov = 60
    const start = latest.current.sun
    if (start && start.el > -5) yaw = start.az
    let down: { x: number; y: number } | null = null
    const clampPitch = (p: number) => Math.max(-35, Math.min(80, p))
    const pd = (e: PointerEvent) => { down = { x: e.clientX, y: e.clientY }; el.setPointerCapture(e.pointerId) }
    const pm = (e: PointerEvent) => {
      if (!down) return
      const k2 = fov / el.clientHeight // degrees per pixel
      yaw = (((yaw - (e.clientX - down.x) * k2) % 360) + 360) % 360
      pitch = clampPitch(pitch + (e.clientY - down.y) * k2)
      down = { x: e.clientX, y: e.clientY }
    }
    const pu = () => { down = null }
    const wheel = (e: WheelEvent) => { e.preventDefault(); fov = Math.max(3, Math.min(90, fov * Math.exp(e.deltaY * 0.0012))) }
    const key = (e: KeyboardEvent) => {
      if (e.key === 'ArrowLeft') yaw = (yaw + 358) % 360
      if (e.key === 'ArrowRight') yaw = (yaw + 2) % 360
      if (e.key === 'ArrowUp') pitch = clampPitch(pitch + 2)
      if (e.key === 'ArrowDown') pitch = clampPitch(pitch - 2)
    }
    el.addEventListener('pointerdown', pd)
    el.addEventListener('pointermove', pm)
    el.addEventListener('pointerup', pu)
    el.addEventListener('pointercancel', pu)
    el.addEventListener('wheel', wheel, { passive: false })
    el.addEventListener('keydown', key)

    const resize = () => {
      renderer.setSize(el.clientWidth, el.clientHeight)
      camera.aspect = el.clientWidth / el.clientHeight
      camera.updateProjectionMatrix()
    }
    const observer = new ResizeObserver(resize)
    observer.observe(el)
    resize()

    let raf = 0, lastView = ''
    const animate = () => {
      raf = requestAnimationFrame(animate)
      const { sun: s, earth: ea, onView: report } = latest.current
      placeSun(s)
      orient(ea, latest.current.earthRot)
      if (s) {
        skyLight.position.copy(dir(s).multiplyScalar(SKY_R))
        skyLight.target.position.set(0, 0, 0)
        skyLight.target.updateMatrixWorld()
        skyLight.visible = true
      } else skyLight.visible = false
      for (const l of [light, nearLight]) {
        if (s) {
          l.position.copy(dir(s).multiplyScalar(radiusM * 2))
          l.target.position.set(0, 0, 0)
          l.target.updateMatrixWorld()
          l.visible = s.el > -3 // a Sun well below the horizon lights nothing nearby
        } else l.visible = false
      }
      camera.fov = fov
      camera.updateProjectionMatrix()
      camera.rotation.set((pitch * Math.PI) / 180, (-yaw * Math.PI) / 180, 0)
      const v = `${yaw.toFixed(1)}|${pitch.toFixed(1)}|${fov.toFixed(1)}|${camera.aspect.toFixed(2)}`
      if (v !== lastView) {
        lastView = v
        report?.({ yaw, pitch, fov, aspect: camera.aspect })
      }
      renderer.clear()
      renderer.render(skyScene, camera)
      renderer.render(scene, camera)
      renderer.render(nearScene, camera)
    }
    animate()

    return () => {
      cancelAnimationFrame(raf)
      observer.disconnect()
      el.removeEventListener('pointerdown', pd)
      el.removeEventListener('pointermove', pm)
      el.removeEventListener('pointerup', pu)
      el.removeEventListener('pointercancel', pu)
      el.removeEventListener('wheel', wheel)
      el.removeEventListener('keydown', key)
      farGeo.dispose()
      nearGeo.dispose()
      casterMat.dispose()
      texture.dispose()
      material.dispose()
      starGeo.dispose()
      starMat.dispose()
      sunDisk.geometry.dispose()
      sunTex.dispose()
      earthGeo.dispose()
      earthTex.dispose()
      light.shadow.map?.dispose()
      nearLight.shadow.map?.dispose()
      renderer.dispose()
      el.removeChild(renderer.domElement)
    }
  }, [grids])

  return <div ref={host} tabIndex={0} aria-label="360 degree view from the Moon's surface" className="h-full w-full cursor-grab outline-none active:cursor-grabbing" />
}
