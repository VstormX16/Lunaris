import * as T from 'three';
import {R,xyz} from './geo';
import {API_BASE} from '../../api';
const N=90;
const RAD=Math.PI/180;
const TWO_DY=947.6047008; // 2 * (1737400 * pi / 180 * 1/64)
type Tile={key:string;z:number;x:number;y:number;mesh?:T.Mesh;pending?:boolean;used:number;center:T.Vector3;size:number};

export class Terrain {
 group=new T.Group(); tiles=new Map<string,Tile>(); active=0; failed=0; mode='natural';wire=false;frame=0;
 material:T.MeshStandardMaterial; texture:T.Texture;disposed=false; controllers=new Set<AbortController>();
 constructor(){this.texture=new T.TextureLoader().load('/data/moon-color.jpg');this.texture.colorSpace=T.SRGBColorSpace;this.texture.wrapS=T.RepeatWrapping;this.material=new T.MeshStandardMaterial({map:this.texture,roughness:0.96,metalness:0.02});}
 tile(z:number,x:number,y:number){const key=`${z}/${x}/${y}`;let tile=this.tiles.get(key);if(!tile){const size=180/2**z;tile={key,z,x,y,size,used:0,center:new T.Vector3(...xyz(90-(y+.5)*size,-180+(x+.5)*size))};this.tiles.set(key,tile);}return tile;}
 request(tile:Tile){if(tile.pending||tile.mesh||this.active>=8||this.disposed)return;tile.pending=true;this.active++;const abort=new AbortController();this.controllers.add(abort);
 fetch(`${API_BASE}/moon-tile/${tile.key}`,{signal:abort.signal}).then(r=>{if(!r.ok)throw Error('Terrain service not responding');return r.arrayBuffer();}).then(b=>{if(this.disposed)return;const h=new Uint16Array(b);if(h.length!==5*(N+1)**2)throw Error('Invalid terrain tile');const geo=this.geometry(tile,h);const mesh=new T.Mesh(geo,this.material);mesh.userData={tile:true,heights:h};mesh.visible=false;tile.mesh=mesh;this.group.add(mesh);}).catch(e=>{if(e.name!=='AbortError')this.failed++;}).finally(()=>{tile.pending=false;this.active--;this.controllers.delete(abort);});}
 geometry(tile:Tile,h:Uint16Array){
  const pos:number[]=[],uv:number[]=[],col:number[]=[],ind:number[]=[];
  const color=new T.Color();
  const vertex=(i:number,j:number,skirt=false)=>{
   const k=j*(N+1)+i,lat=90-(tile.y+j/N)*tile.size,lon=-180+(tile.x+i/N)*tile.size,height=h[k]*.5-10000;
   const r=R+height/1000-(skirt?Math.max(.04,tile.size*.04):0);
   if(lat>=89.999){
    pos.push(0,r,0);
   }else if(lat<=-89.999){
    pos.push(0,-r,0);
   }else{
    pos.push(...xyz(lat,lon,r));
   }
   uv.push((lon+180)/360,(lat+90)/180);
   color.setHSL(.68-Math.max(0,Math.min(1,(height+9000)/19000))*.68,.48,.42);
   col.push(color.r,color.g,color.b);
   return pos.length/3-1;
  };
  for(let j=0;j<=N;j++)for(let i=0;i<=N;i++)vertex(i,j);
  for(let j=0;j<N;j++)for(let i=0;i<N;i++){const a=j*(N+1)+i,b=a+1,c=a+N+1,d=c+1;ind.push(a,c,b,b,c,d);}
  const edges=[Array.from({length:N+1},(_,i)=>[i,0]),Array.from({length:N+1},(_,j)=>[N,j]),Array.from({length:N+1},(_,i)=>[N-i,N]),Array.from({length:N+1},(_,j)=>[0,N-j])];
  for(const edge of edges){let last=-1;for(let k=0;k<edge.length;k++){const [i,j]=edge[k],v=vertex(i,j,true),a=j*(N+1)+i;if(k){const [pi,pj]=edge[k-1],p=pj*(N+1)+pi;ind.push(p,last,a,a,last,v);}last=v;}}
  const g=new T.BufferGeometry();
  g.setAttribute('position',new T.Float32BufferAttribute(pos,3));
  g.setAttribute('uv',new T.Float32BufferAttribute(uv,2));
  g.setAttribute('color',new T.Float32BufferAttribute(col,3));
  g.setIndex(ind);

  // Exact differential geometry normals from orthodromic tangent samples
  const normals=new Float32Array(pos.length);
  const count=(N+1)**2;
  for(let j=0;j<=N;j++)for(let i=0;i<=N;i++){
   const k=j*(N+1)+i,lat=90-(tile.y+j/N)*tile.size,lon=-180+(tile.x+i/N)*tile.size;
   const hn=h[count+k]*.5-10000,hs=h[2*count+k]*.5-10000;
   const he=h[3*count+k]*.5-10000,hw=h[4*count+k]*.5-10000;
   const sy=(hn-hs)/TWO_DY,sx=(he-hw)/TWO_DY;

   const phi=lat*RAD,lam=lon*RAD;
   const cosP=Math.cos(phi),sinP=Math.sin(phi);
   const cosL=Math.cos(lam),sinL=Math.sin(lam);

   // Radial surface unit vector (+Y up, -Z east)
   const rx=cosP*cosL,ry=sinP,rz=-cosP*sinL;

   let ex=-sinL,ey=0,ez=-cosL;
   let nx=-sinP*cosL,ny=cosP,nz=sinP*sinL;

   if(Math.abs(ry)>=0.9995){
    const sign=ry>=0?1:-1;
    ex=0;ey=0;ez=-1;
    nx=-sign;ny=0;nz=0;
   }

   // True surface normal = r - sx*east - sy*north
   let mx=rx-sx*ex-sy*nx;
   let my=ry-sx*ey-sy*ny;
   let mz=rz-sx*ez-sy*nz;
   const len=Math.hypot(mx,my,mz)||1;
   normals[k*3]=mx/len;normals[k*3+1]=my/len;normals[k*3+2]=mz/len;
  }
  // Skirt vertices copy base normal
  for(let idx=(N+1)**2;idx<pos.length/3;idx++){
   normals[idx*3]=0;normals[idx*3+1]=1;normals[idx*3+2]=0;
  }
  g.setAttribute('normal',new T.Float32BufferAttribute(normals,3));
  g.computeBoundingSphere();
  return g;
 }
 update(_camera:T.PerspectiveCamera,focus:T.Vector3,alt:number){
  this.frame++;for(const t of this.tiles.values())if(t.mesh)t.mesh.visible=false;
  const visit=(z:number,x:number,y:number):boolean=>{
   const t=this.tile(z,x,y);t.used=this.frame;this.request(t);
   const distance=t.center.distanceTo(focus);
   const width=t.size*Math.PI/180*R;
   const refine=z<2||(z<7&&distance<width*1.5&&width>Math.max(alt*.8,8));
   let childReady=false;
   if(refine){
    const children=[this.tile(z+1,x*2,y*2),this.tile(z+1,x*2+1,y*2),this.tile(z+1,x*2,y*2+1),this.tile(z+1,x*2+1,y*2+1)];
    children.forEach(c=>this.request(c));
    childReady=children.every(c=>c.mesh);
    if(childReady)children.forEach(c=>visit(c.z,c.x,c.y));
   }
   if(!childReady&&t.mesh)t.mesh.visible=true;
   return !!t.mesh;
  };
  visit(0,0,0);visit(0,1,0);
  if(this.tiles.size>360){
   for(const [k,t] of this.tiles){
    if(t.z>1&&!t.pending&&this.frame-t.used>50){
     if(t.mesh){this.group.remove(t.mesh);t.mesh.geometry.dispose();}
     this.tiles.delete(k);
     if(this.tiles.size<=300)break;
    }
   }
  }
 }
 setStyle(mode:string,wire:boolean){
  this.mode=mode;this.wire=wire;
  this.material.map=mode==='natural'?this.texture:null;
  this.material.vertexColors=mode==='elevation';
  this.material.color.set(mode==='relief'?0xb4b8bf:0xffffff);
  this.material.wireframe=wire;
  this.material.needsUpdate=true;
 }
 dispose(){this.disposed=true;this.controllers.forEach(c=>c.abort());this.tiles.forEach(t=>t.mesh?.geometry.dispose());this.material.dispose();this.texture.dispose();}
}
