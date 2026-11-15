export const R = 1737.4;
export const RAD = Math.PI / 180;
export type Point = {lat:number; lon:number; height?:number; name?:string};
export const wrapLon = (lon:number) => ((lon+180)%360+360)%360-180;
export function xyz(lat:number,lon:number,r=R):[number,number,number]{const a=lat*RAD,b=lon*RAD;return [r*Math.cos(a)*Math.cos(b),r*Math.sin(a),-r*Math.cos(a)*Math.sin(b)];}
export function latlon(x:number,y:number,z:number):Point{const h=Math.hypot(x,z);return {lat:Math.atan2(y,h)/RAD,lon:h<1e-7?0:wrapLon(Math.atan2(-z,x)/RAD),height:(Math.hypot(x,y,z)-R)*1000};}
export function distance(a:Point,b:Point){const p=(b.lat-a.lat)*RAD,q=(b.lon-a.lon)*RAD;return 2*R*Math.asin(Math.min(1,Math.sqrt(Math.sin(p/2)**2+Math.cos(a.lat*RAD)*Math.cos(b.lat*RAD)*Math.sin(q/2)**2)));}
export function interpolate(a:Point,b:Point,t:number):Point {const av=xyz(a.lat,a.lon,1),bv=xyz(b.lat,b.lon,1);const omega=Math.acos(Math.max(-1,Math.min(1,av.reduce((s,v,i)=>s+v*bv[i],0))));if(omega<1e-9)return {...a}; if(Math.PI-omega<1e-6)throw new Error('Karşıt noktalar arasında tek bir en kısa rota yok.');const s=Math.sin(omega),u=Math.sin((1-t)*omega)/s,v=Math.sin(t*omega)/s;return latlon(av[0]*u+bv[0]*v,av[1]*u+bv[1]*v,av[2]*u+bv[2]*v);}
export const places:Point[]=[{name:'Tycho',lat:-43.31,lon:-11.36},{name:'Copernicus',lat:9.62,lon:-20.08},{name:'Shackleton (Güney Kutbu)',lat:-89.9,lon:0},{name:'Schrödinger',lat:-75,lon:132.4},{name:'Apollo 11',lat:.67408,lon:23.47297},{name:'Tsiolkovskiy',lat:-20.4,lon:129.1},{name:'Mare Imbrium',lat:32.8,lon:-15.6},{name:'Kuzey Kutbu (Hermite kenarı)',lat:89.9,lon:0}].slice(0,8);
