from __future__ import annotations


PLAYABLE_PROBE_HTML = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>CausalRAG · Playable Probe</title>
<style>
:root{color-scheme:dark;--bg:#090b10;--panel:#11151d;--line:#273041;--text:#edf2f7;--muted:#8f9bad;--accent:#7dd3fc;--warn:#fbbf24;--bad:#fb7185;--good:#86efac}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px/1.45 ui-monospace,SFMono-Regular,Menlo,monospace}
header{display:flex;align-items:center;justify-content:space-between;padding:16px 22px;border-bottom:1px solid var(--line);position:sticky;top:0;background:rgba(9,11,16,.94);backdrop-filter:blur(10px);z-index:2}
h1{font-size:16px;margin:0}.tag{color:var(--accent);font-size:12px}.layout{display:grid;grid-template-columns:330px 1fr;min-height:calc(100vh - 58px)}
aside{border-right:1px solid var(--line);padding:18px;position:sticky;top:58px;height:calc(100vh - 58px);overflow:auto}.main{padding:18px;min-width:0}
label{display:block;color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.08em;margin:14px 0 6px}textarea,input,select,button{width:100%;border:1px solid var(--line);background:#0c1017;color:var(--text);padding:9px;border-radius:7px;font:inherit}textarea{min-height:92px;resize:vertical}button{margin-top:14px;background:#d8f3ff;color:#071018;font-weight:700;cursor:pointer;border:0}button:disabled{opacity:.45;cursor:wait}
.row{display:grid;grid-template-columns:1fr 1fr;gap:8px}.checks{display:grid;grid-template-columns:1fr 1fr;gap:7px;margin-top:6px}.check{display:flex;align-items:center;gap:6px;color:var(--muted);font-size:12px}.check input{width:auto}
.summary{display:grid;grid-template-columns:repeat(5,minmax(120px,1fr));gap:10px;margin-bottom:14px}.card,.panel{border:1px solid var(--line);background:var(--panel);border-radius:9px}.card{padding:12px}.card b{display:block;font-size:18px;margin-top:3px}.small{font-size:11px;color:var(--muted)}
.grid{display:grid;grid-template-columns:1.15fr .85fr;gap:12px}.panel{padding:13px;min-width:0}.panel h2{font-size:12px;margin:0 0 10px;color:var(--muted);text-transform:uppercase;letter-spacing:.08em}
.timeline{display:flex;flex-direction:column;gap:6px;max-height:55vh;overflow:auto}.event{border-left:2px solid var(--line);padding:7px 9px;background:#0d1118;cursor:pointer}.event:hover,.event.active{border-left-color:var(--accent);background:#101924}.event .name{font-weight:700}.event .meta{color:var(--muted);font-size:11px}.event.warn{border-left-color:var(--warn)}.event.bad{border-left-color:var(--bad)}.event.good{border-left-color:var(--good)}
.hyp{display:grid;grid-template-columns:1fr auto;gap:6px;padding:7px 0;border-bottom:1px dashed var(--line)}.bar{height:5px;background:#263143;border-radius:4px;overflow:hidden;grid-column:1/3}.bar i{display:block;height:100%;background:var(--accent)}
pre{white-space:pre-wrap;word-break:break-word;background:#090d13;border:1px solid var(--line);padding:10px;border-radius:7px;max-height:38vh;overflow:auto;color:#c8d2df}.empty{color:var(--muted);padding:18px;text-align:center}.status{margin-top:10px;color:var(--muted);min-height:20px}.danger{color:var(--bad)}
@media(max-width:900px){.layout{grid-template-columns:1fr}aside{position:static;height:auto;border-right:0;border-bottom:1px solid var(--line)}.summary{grid-template-columns:repeat(2,1fr)}.grid{grid-template-columns:1fr}}
</style>
</head>
<body>
<header><h1>CausalRAG · Playable Probe</h1><div class="tag">experience → belief → decision → action → outcome</div></header>
<div class="layout">
<aside>
<label>Goal</label><textarea id="goal">Diagnose the situation, gather discriminating evidence, and act only when justified.</textarea>
<label>Model lane</label><select id="lane"><option value="frontier">Frontier model</option><option value="small">Small / local model</option></select>
<div class="row"><div><label>Provider</label><select id="provider"><option>openai</option><option>local</option><option>anthropic</option></select></div><div><label>Max steps</label><input id="steps" type="number" value="8" min="1" max="64" /></div></div>
<label>Model</label><input id="model" value="gpt-5.6-terra" />
<label>Runtime ablation record</label>
<div class="checks">
<label class="check"><input id="eig" type="checkbox" checked /> EIG</label>
<label class="check"><input id="evsi" type="checkbox" checked /> EVSI</label>
<label class="check"><input id="temporal" type="checkbox" checked /> temporal</label>
<label class="check"><input id="openworld" type="checkbox" checked /> open-world</label>
<label class="check"><input id="retrieval" type="checkbox" /> retrieval</label>
<label class="check"><input id="causal" type="checkbox" checked /> causal runtime</label>
</div>
<button id="run">Run probe</button>
<div id="status" class="status">Ready.</div>
</aside>
<main class="main">
<div class="summary">
<div class="card"><span class="small">run</span><b id="runid">—</b></div>
<div class="card"><span class="small">decisions</span><b id="decisions">0</b></div>
<div class="card"><span class="small">observations</span><b id="observations">0</b></div>
<div class="card"><span class="small">mismatches</span><b id="mismatches">0</b></div>
<div class="card"><span class="small">stop</span><b id="stop">—</b></div>
</div>
<div class="grid">
<section class="panel"><h2>Decision / causal timeline</h2><div id="timeline" class="timeline"><div class="empty">Run a probe to inspect the causal loop.</div></div></section>
<section>
<div class="panel"><h2>Hypothesis state at selected event</h2><div id="hypotheses" class="empty">No hypothesis snapshot.</div></div>
<div class="panel" style="margin-top:12px"><h2>Selected event payload</h2><pre id="payload">{}</pre></div>
</section>
</div>
<div class="panel" style="margin-top:12px"><h2>Final answer / outcome</h2><pre id="answer">—</pre></div>
</main>
</div>
<script>
const $=id=>document.getElementById(id);let current=[];
$('lane').onchange=()=>{if($('lane').value==='small'){ $('provider').value='local'; $('model').value='qwen3:8b'; }else{ $('provider').value='openai'; $('model').value='gpt-5.6-terra'; }};
function cls(name){if(name==='model_mismatch'||name==='tool.failed')return ' bad';if(name==='hypothesis.discovered')return ' warn';if(name==='posterior.updated'||name==='run.completed')return ' good';return ''}
function hypothesesFrom(ev){const p=ev?.payload||{};if(Array.isArray(p.hypotheses))return p.hypotheses;if(p.world_model?.hypotheses)return p.world_model.hypotheses;return null}
function showEvent(i){document.querySelectorAll('.event').forEach(x=>x.classList.remove('active'));const el=document.querySelector(`[data-i="${i}"]`);if(el)el.classList.add('active');const ev=current[i];$('payload').textContent=JSON.stringify(ev?.payload||{},null,2);const hs=hypothesesFrom(ev);if(!hs){$('hypotheses').innerHTML='<div class="empty">No hypothesis snapshot on this event.</div>';return}$('hypotheses').innerHTML=hs.map(h=>`<div class="hyp"><div><b>${h.id||h.hypothesis_id}</b> <span class="small">${h.status||''}${h.origin?' · '+h.origin:''}</span><div class="small">${h.statement||''}</div></div><div>${Number(h.probability||0).toFixed(3)}</div><div class="bar"><i style="width:${Math.max(0,Math.min(100,Number(h.probability||0)*100))}%"></i></div></div>`).join('')}
function render(data){current=data.events||[];$('runid').textContent=(data.run_id||'—').slice(0,10);$('decisions').textContent=data.result?.steps??0;$('observations').textContent=data.result?.observations?.length??0;$('mismatches').textContent=current.filter(e=>e.name==='model_mismatch').length;$('stop').textContent=data.result?.stop_reason||'—';$('answer').textContent=data.result?.answer||'—';$('timeline').innerHTML=current.map((e,i)=>`<div class="event${cls(e.name)}" data-i="${i}"><div><span class="name">${e.name}</span> <span class="meta">#${e.sequence}${e.step!==null&&e.step!==undefined?' · step '+e.step:''}</span></div><div class="meta">${e.payload?.action_name||e.payload?.experiment_id||e.payload?.stop_reason||''}</div></div>`).join('')||'<div class="empty">No events.</div>';document.querySelectorAll('.event').forEach(el=>el.onclick=()=>showEvent(Number(el.dataset.i)));if(current.length)showEvent(current.length-1)}
$('run').onclick=async()=>{const btn=$('run');btn.disabled=true;$('status').className='status';$('status').textContent='Running…';const body={goal:$('goal').value,max_steps:Number($('steps').value),lane:$('lane').value,provider:$('provider').value,model:$('model').value,features:{causal_runtime:$('causal').checked,eig:$('eig').checked,evsi:$('evsi').checked,temporal_attribution:$('temporal').checked,open_world_discovery:$('openworld').checked,retrieval:$('retrieval').checked}};try{const res=await fetch('/probe/run',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});const data=await res.json();if(!res.ok)throw new Error(data.detail||JSON.stringify(data));render(data);$('status').textContent='Completed.'}catch(err){$('status').className='status danger';$('status').textContent=String(err)}finally{btn.disabled=false}};
</script>
</body></html>'''
