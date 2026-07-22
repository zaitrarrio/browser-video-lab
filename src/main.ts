import "./style.css";
type Mode="sd"|"longlive"|"memflow";
let mode:Mode="sd";
const MODEL_CACHE="bvl-models-v1";
const root=document.querySelector<HTMLDivElement>("#app")!;
const fmtBytes=(n:number)=>n>=1e9?`${(n/1e9).toFixed(1)} GB`:n>=1e6?`${(n/1e6).toFixed(1)} MB`:n>=1e3?`${(n/1e3).toFixed(0)} KB`:`${n} B`;

function render(){root.innerHTML=`<main><h1>Browser Video Lab</h1><div class="muted">WebGPU-first SD-Turbo, LongLive, and adaptive-memory MemFlow inference</div><nav><button data-mode="sd" class="${mode==='sd'?'active':''}">SD-Turbo</button><button data-mode="longlive" class="${mode==='longlive'?'active':''}">LongLive</button><button data-mode="memflow" class="${mode==='memflow'?'active':''}">MemFlow</button></nav><section class="card"><label>Model manifest URL<input id="manifest" style="width:100%" value="${import.meta.env.BASE_URL}models/${mode}/manifest.json"></label><label><div style="margin-top:14px">Prompt</div><textarea id="prompt">A cinematic tracking shot of a silver robot walking through Austin at sunset</textarea></label><div class="grid"><label>Seed<input id="seed" type="number" value="42" style="width:100%"></label><label>${mode==='sd'?'Steps<input id="steps" type="number" value="4" min="1" max="12" style="width:100%">':'Chunks<input id="steps" type="number" value="4" min="1" max="32" style="width:100%">'}</label></div><div class="actions"><button id="load">Load model</button><button id="run">Generate</button><button id="stop">Stop</button><button id="clear" class="ghost">Clear cache</button></div><div class="progress" id="progress" hidden><div class="bar"><div class="fill" id="bar"></div></div><div class="progress-meta"><span id="pct">0%</span><span id="detail" class="muted"></span></div><ul class="files" id="files"></ul></div><div class="status" id="status">WebGPU: ${'gpu' in navigator?'available':'unavailable'}</div><canvas id="output" width="512" height="512"></canvas></section></main>`;
 document.querySelectorAll<HTMLButtonElement>("[data-mode]").forEach(b=>b.onclick=()=>{mode=b.dataset.mode as Mode;render()});
 const worker=new Worker(new URL("./worker.ts",import.meta.url),{type:"module"});
 const status=document.querySelector<HTMLDivElement>("#status")!, canvas=document.querySelector<HTMLCanvasElement>("#output")!;
 const progress=document.querySelector<HTMLDivElement>("#progress")!, bar=document.querySelector<HTMLDivElement>("#bar")!, pct=document.querySelector<HTMLSpanElement>("#pct")!, detail=document.querySelector<HTMLSpanElement>("#detail")!, filesEl=document.querySelector<HTMLUListElement>("#files")!;
 worker.onmessage=(e)=>{const m=e.data;
  if(m.type==='status'){status.textContent=m.message;status.classList.toggle('error',!!m.error);}
  if(m.type==='progress'){progress.hidden=false;bar.style.width=`${m.percent}%`;bar.classList.toggle('done',!!m.done);pct.textContent=`${m.percent}%`;
   detail.textContent=m.done?(m.cached?'Loaded from cache ⚡':'Model ready'):m.cached?`Restoring from cache · ${fmtBytes(m.loaded)}`:`Downloading · ${fmtBytes(m.loaded)}${m.total?` / ${fmtBytes(m.total)}`:''}`;
   if(m.message)status.textContent=m.message;
   filesEl.innerHTML=(m.files||[]).map((f:any)=>{const p=f.total?Math.min(100,Math.round(f.loaded/f.total*100)):(f.loaded?100:0);return `<li><span class="fname">${f.name}${f.cached?' <em>cached</em>':''}</span><span class="fpct">${p}%</span></li>`}).join('');}
  if(m.type==='frame'){canvas.width=m.width;canvas.height=m.height;canvas.getContext('2d')!.putImageData(new ImageData(new Uint8ClampedArray(m.rgba),m.width,m.height),0,0)}};
 const payload=()=>({mode,manifestUrl:(document.querySelector<HTMLInputElement>('#manifest')!).value,prompt:(document.querySelector<HTMLTextAreaElement>('#prompt')!).value,seed:+(document.querySelector<HTMLInputElement>('#seed')!).value,steps:+(document.querySelector<HTMLInputElement>('#steps')!).value});
 document.querySelector<HTMLButtonElement>('#load')!.onclick=()=>{progress.hidden=false;bar.style.width='0%';bar.classList.remove('done');pct.textContent='0%';detail.textContent='';filesEl.innerHTML='';worker.postMessage({type:'load',...payload()});};
 document.querySelector<HTMLButtonElement>('#run')!.onclick=()=>worker.postMessage({type:'run',...payload()});
 document.querySelector<HTMLButtonElement>('#stop')!.onclick=()=>worker.postMessage({type:'stop'});
 document.querySelector<HTMLButtonElement>('#clear')!.onclick=async()=>{if('caches' in window)await caches.delete(MODEL_CACHE);status.textContent='Model cache cleared — next load will re-download.';};
}
render();
