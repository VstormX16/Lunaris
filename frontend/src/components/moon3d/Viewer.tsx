import { useEffect, useRef } from 'react';
import * as T from 'three';
import {Terrain} from './terrain';
import {R,xyz,latlon,wrapLon,type Point} from './geo';
import {API_BASE} from '../../api';

export type ViewerProps = {
 initialLat?: number;
 initialLon?: number;
 initialAlt?: number;
 mode?: 'natural' | 'relief' | 'elevation';
 wire?: boolean;
 grid?: boolean;
 sun?: number;
 sunLat?: number;
 marker?: { lat: number; lon: number; height?: number } | null;
 focus?: { lat: number; lon: number; n: number }; // look at this point whenever n changes
 showHint?: boolean;
 onPick?: (p: Point) => void;
 onError?: (err: string) => void;
};

export default function Viewer(props: ViewerProps = {}) {
 const host = useRef<HTMLDivElement>(null);
 const latest = useRef(props);
 latest.current = props;

 useEffect(() => {
  const el = host.current!;
  let renderer: T.WebGLRenderer;
  try {
   renderer = new T.WebGLRenderer({ antialias: true, logarithmicDepthBuffer: true, powerPreference: 'high-performance' });
  } catch {
   latest.current.onError?.('WebGL could not start. Turn on hardware acceleration in the browser.');
   return;
  }

  renderer.setPixelRatio(Math.min(devicePixelRatio, 1.75));
  renderer.setClearColor(0x05080c);
  el.appendChild(renderer.domElement);

  const scene = new T.Scene();
  const camera = new T.PerspectiveCamera(42, 1, 0.02, 60000);
  const terrain = new Terrain();
  scene.add(terrain.group);

  // Lunar vacuum environment: High contrast directional sun & subtle cosmic ambient
  const ambient = new T.AmbientLight(0xd5e0f0, 0.10);
  const sun = new T.DirectionalLight(0xfff7e6, 2.8);
  scene.add(ambient, sun);

  // Background starry sky
  const starsGeo = new T.BufferGeometry(), stars: number[] = [];
  let seed = 42;
  const rand = () => { seed = (seed * 1664525 + 1013904223) >>> 0; return seed / 4294967296; };
  for (let i = 0; i < 1400; i++) {
   const z = rand() * 2 - 1, a = rand() * Math.PI * 2, r = 24000;
   stars.push(r * Math.sqrt(1 - z * z) * Math.cos(a), r * z, r * Math.sqrt(1 - z * z) * Math.sin(a));
  }
  starsGeo.setAttribute('position', new T.Float32BufferAttribute(stars, 3));
  const starsMat = new T.PointsMaterial({ color: 0x8e9aa8, size: 8, sizeAttenuation: true });
  scene.add(new T.Points(starsGeo, starsMat));

  // the chosen point: a small flat dot with a white rim, kept the same size on screen
  const markerGroup = new T.Group();
  const rim = new T.Mesh(new T.SphereGeometry(1, 20, 14), new T.MeshBasicMaterial({ color: 0xffffff, depthTest: false }));
  const dot = new T.Mesh(new T.SphereGeometry(0.62, 20, 14), new T.MeshBasicMaterial({ color: 0x4d756a, depthTest: false }));
  rim.renderOrder = 10; dot.renderOrder = 11;
  markerGroup.add(rim, dot);
  markerGroup.visible = false;
  scene.add(markerGroup);

  // Optional coordinate grid
  const gridGroup = new T.Group();
  scene.add(gridGroup);
  const gridMat = new T.LineBasicMaterial({ color: 0x516879, transparent: true, opacity: 0.26, depthTest: true });
  for (let lat = -60; lat <= 60; lat += 30) {
   const ps = [];
   for (let lon = -180; lon <= 180; lon++) ps.push(new T.Vector3(...xyz(lat, lon, R + 12)));
   gridGroup.add(new T.Line(new T.BufferGeometry().setFromPoints(ps), gridMat));
  }
  for (let lon = -180; lon < 180; lon += 30) {
   const ps = [];
   for (let lat = -90; lat <= 90; lat++) ps.push(new T.Vector3(...xyz(lat, lon, R + 12)));
   gridGroup.add(new T.Line(new T.BufferGeometry().setFromPoints(ps), gridMat));
  }

  let lat = props.initialLat ?? 10;
  let lon = props.initialLon ?? -15;
  let alt = props.initialAlt ?? 6400;
  let targetLat = lat, targetLon = lon, targetAlt = alt;
  let tilt = 0;
  let height = 0, raf = 0, frame = 0;
  let down: { x: number; y: number; lx: number; ly: number; button: number } | null = null;

  const resize = () => {
   renderer.setSize(el.clientWidth, el.clientHeight);
   camera.aspect = el.clientWidth / el.clientHeight;
   camera.updateProjectionMatrix();
  };
  const observer = new ResizeObserver(resize);
  observer.observe(el);
  resize();

  const ray = new T.Raycaster(), pointer = new T.Vector2();
  const pick = (e: PointerEvent) => {
   const rect = el.getBoundingClientRect();
   pointer.set(((e.clientX - rect.left) / rect.width) * 2 - 1, -((e.clientY - rect.top) / rect.height) * 2 + 1);
   ray.setFromCamera(pointer, camera);
   const meshes = terrain.group.children.filter(m => m.visible);
   const hits = ray.intersectObjects(meshes, false);
   if (hits.length) {
    const p = latlon(hits[0].point.x, hits[0].point.y, hits[0].point.z);
    latest.current.onPick?.(p);
   }
  };

  const wheel = (e: WheelEvent) => {
   e.preventDefault();
   targetAlt = T.MathUtils.clamp(targetAlt * Math.exp(e.deltaY * 0.0015), 1.5, 16000);
  };

  const pd = (e: PointerEvent) => {
   down = { x: e.clientX, y: e.clientY, lx: e.clientX, ly: e.clientY, button: e.button };
   el.setPointerCapture(e.pointerId);
  };

  const pm = (e: PointerEvent) => {
   if (!down) return;
   const dx = e.clientX - down.lx, dy = e.clientY - down.ly;
   down.lx = e.clientX; down.ly = e.clientY;
   if (down.button === 2) {
    tilt = T.MathUtils.clamp(tilt + dy * 0.3, 0, 78);
   } else {
    const scale = Math.max(0.0003, (alt / R) * 0.055);
    const cosLat = Math.cos(lat * (Math.PI / 180));
    targetLon = wrapLon(targetLon - (dx * scale) / Math.max(0.08, cosLat));
    targetLat = T.MathUtils.clamp(targetLat + dy * scale, -89.999, 89.999);
   }
  };

  const pu = (e: PointerEvent) => {
   if (down && Math.hypot(e.clientX - down.x, e.clientY - down.y) < 5 && down.button === 0) {
    pick(e);
   }
   down = null;
  };
  const context = (e: Event) => e.preventDefault();

  el.addEventListener('wheel', wheel, { passive: false });
  el.addEventListener('pointerdown', pd);
  el.addEventListener('pointermove', pm);
  el.addEventListener('pointerup', pu);
  el.addEventListener('pointercancel', () => { down = null; });
  el.addEventListener('contextmenu', context);

  let lastHeightRequest = 0;
  let lastStyle = '';
  let lastFocus = -1;

  const animate = () => {
   raf = requestAnimationFrame(animate);
   frame++;
   const p = latest.current;

   if (p.focus && p.focus.n !== lastFocus) {
    lastFocus = p.focus.n;
    targetLat = p.focus.lat;
    targetLon = wrapLon(p.focus.lon);
   }
   lat += (targetLat - lat) * 0.12;
   lon = wrapLon(lon + wrapLon(targetLon - lon) * 0.12);
   alt += (targetAlt - alt) * 0.12;

   const phi = lat * (Math.PI / 180), lam = lon * (Math.PI / 180);
   const cosP = Math.cos(phi), sinP = Math.sin(phi);
   const cosL = Math.cos(lam), sinL = Math.sin(lam);
   const n = new T.Vector3(cosP * cosL, sinP, -cosP * sinL);

   // Zero gimbal-lock continuous tangent north across lunar poles
   const north = new T.Vector3();
   if (Math.abs(sinP) < 0.999) {
    north.set(-sinP * cosL, cosP, sinP * sinL).normalize();
   } else {
    const sign = sinP >= 0 ? 1 : -1;
    const east = new T.Vector3(-sinL, 0, -cosL);
    if (east.lengthSq() < 1e-5) east.set(0, 0, -1);
    else east.normalize();
    north.crossVectors(east, n).multiplyScalar(sign).normalize();
   }

   const focus = n.clone().multiplyScalar(R + height / 1000);
   const angle = tilt * (Math.PI / 180);
   camera.position.copy(focus).addScaledVector(n, alt * Math.cos(angle)).addScaledVector(north, -alt * Math.sin(angle));
   camera.up.copy(north);
   camera.lookAt(focus);
   camera.updateMatrixWorld();

   const curMode = p.mode ?? 'natural';
   const curWire = p.wire ?? false;
   const styleKey = `${curMode}/${curWire}`;
   if (lastStyle !== styleKey) {
    terrain.setStyle(curMode, curWire);
    lastStyle = styleKey;
   }

   sun.position.set(...xyz(p.sunLat ?? 12, p.sun ?? -35, 10000));
   gridGroup.visible = p.grid ?? false;
   if (p.marker) {
    markerGroup.visible = true;
    markerGroup.position.set(...xyz(p.marker.lat, p.marker.lon, R + (p.marker.height ?? 0) / 1000));
    markerGroup.scale.setScalar(Math.max(0.01, alt * 0.012));
   } else markerGroup.visible = false;

   if (frame % 6 === 0) {
    terrain.update(camera, focus, alt);
    if (terrain.failed > 3) {
     p.onError?.('Terrain data could not be loaded. Check that the analysis server is running.');
    }
   }

   if (frame - lastHeightRequest > 120) {
    lastHeightRequest = frame;
    fetch(`${API_BASE}/moon-height?lat=${lat}&lon=${lon}`)
     .then(r => r.json())
     .then(d => { if (Number.isFinite(d.height)) height = d.height; })
     .catch(() => {});
   }

   renderer.render(scene, camera);
  };
  animate();

  return () => {
   rim.geometry.dispose(); dot.geometry.dispose();
   cancelAnimationFrame(raf);
   observer.disconnect();
   el.removeEventListener('wheel', wheel);
   el.removeEventListener('pointerdown', pd);
   el.removeEventListener('pointermove', pm);
   el.removeEventListener('pointerup', pu);
   el.removeEventListener('contextmenu', context);
   terrain.dispose();
   starsGeo.dispose();
   starsMat.dispose();
   gridGroup.children.forEach(m => (m as T.Line).geometry.dispose());
   gridMat.dispose();
   renderer.dispose();
   el.removeChild(renderer.domElement);
  };
 }, []);

 return (
  <div className="viewer-wrapper" style={{ width: '100%', height: '100%', position: 'relative' }}>
   <div ref={host} className="viewer" style={{ width: '100%', height: '100%' }} aria-label="Interactive 3D Moon surface" />
   {props.showHint !== false && (
    <div className="interaction-hint">
     Sol sürükle: Döndür <span>·</span> Kaydır: Yakınlaş <span>·</span> Sağ sürükle: Eğim
    </div>
   )}
  </div>
 );
}
