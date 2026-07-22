import * as ort from "onnxruntime-web/webgpu";
export type Manifest={kind:"sd-turbo"|"longlive"|"memflow";width:number;height:number;latentScale:number;models:Record<string,string>;io?:Record<string,string>;scheduler?:{trainSteps:number};memory?:{capacity:number;topK:number;keySize:number}};
// Structured progress reporter passed through load(): status() for free text,
// file() for per-asset byte progress so the UI can draw a bar and flag cache hits.
export type LoadProgress={status(message:string):void;file(name:string,loaded:number,total:number,cached:boolean):void};
export const MODEL_CACHE="bvl-models-v1";
export async function manifest(url:string){const r=await fetch(url);if(!r.ok)throw new Error(`Manifest ${r.status}: ${url}`);const m=await r.json() as Manifest;const base=new URL(url,location.href);for(const k of Object.keys(m.models))m.models[k]=new URL(m.models[k],base).href;return m}
// Fetch a model file, serving from (and populating) Cache Storage, while streaming
// byte progress to the reporter. Returns the raw bytes for ort to compile.
export async function fetchModel(url:string,name:string,p:LoadProgress){
 const cache=typeof caches!=='undefined'?await caches.open(MODEL_CACHE):null;
 let res=await cache?.match(url)??undefined,cached=!!res;
 if(!res){res=await fetch(url);if(!res.ok)throw new Error(`${name} ${res.status}: ${url}`);if(cache)await cache.put(url,res.clone());}
 const total=+(res.headers.get('content-length')||0);
 const reader=res.body?.getReader();
 if(!reader){const buf=new Uint8Array(await res.arrayBuffer());p.file(name,buf.byteLength,buf.byteLength,cached);return buf;}
 const chunks:Uint8Array[]=[];let loaded=0;p.file(name,0,total,cached);
 for(;;){const {done,value}=await reader.read();if(done)break;chunks.push(value);loaded+=value.byteLength;p.file(name,loaded,total||loaded,cached);}
 const out=new Uint8Array(loaded);let off=0;for(const c of chunks){out.set(c,off);off+=c.byteLength;}
 p.file(name,loaded,loaded,cached);return out;
}
// Adapt transformers.js download events onto our reporter so tokenizer files
// (which are fetched and cached internally by transformers.js) show up in the bar too.
export function tokenizerProgress(p:LoadProgress){return (e:any)=>{if(!e||!e.file)return;if(e.status==='progress'||e.status==='download')p.file(`tokenizer/${e.file}`,e.loaded??0,e.total??0,false);else if(e.status==='done')p.file(`tokenizer/${e.file}`,e.total??1,e.total??1,e.cache_hit??false);}}
export async function session(url:string,name:string,p:LoadProgress){const bytes=await fetchModel(url,name,p);return ort.InferenceSession.create(bytes,{executionProviders:["webgpu"],graphOptimizationLevel:"all",enableMemPattern:false})}
export function normal(size:number,seed:number){let s=seed>>>0;const out=new Float32Array(size);for(let i=0;i<size;i+=2){s=(1664525*s+1013904223)>>>0;const u=(s+1)/4294967297;s=(1664525*s+1013904223)>>>0;const v=(s+1)/4294967297;const r=Math.sqrt(-2*Math.log(u));out[i]=r*Math.cos(2*Math.PI*v);if(i+1<size)out[i+1]=r*Math.sin(2*Math.PI*v)}return out}
export function rgbaFromNchw(data:Float32Array,w:number,h:number){const o=new Uint8ClampedArray(w*h*4),plane=w*h;for(let i=0;i<plane;i++){o[i*4]=255*Math.max(0,Math.min(1,data[i]/2+.5));o[i*4+1]=255*Math.max(0,Math.min(1,data[plane+i]/2+.5));o[i*4+2]=255*Math.max(0,Math.min(1,data[2*plane+i]/2+.5));o[i*4+3]=255}return o}
export {ort};
