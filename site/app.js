(function(){
"use strict";

var $ = function(id){ return document.getElementById(id); };
var esc = function(v){ return String(v == null ? "" : v).replace(/[&<>"]/g,function(m){return {"&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;"}[m];}); };
var clamp = function(v,min,max){ return Math.max(min,Math.min(max,v)); };
var uid = function(){ return "S-" + Math.random().toString(36).slice(2,8).toUpperCase(); };

var preset = "hvac";
var state = null;

var presets = {
  hvac: {
    title:"CO₂ / ventilation",
    goal:"卧室 CO₂ 一直升高，但窗户已经开了。判断真正原因，并决定下一步应该验证什么。",
    hypotheses:[
      {id:"H1",name:"CO₂ 传感器漂移",p:.28},
      {id:"H2",name:"开窗但没有形成有效穿堂气流",p:.44},
      {id:"H3",name:"排风/回风路径受阻",p:.28}
    ]
  },
  temporal: {
    title:"Delayed effect",
    goal:"阀门已经打开，但流量读数没有变化。判断是执行器失效、系统真实延迟，还是传感器滞后，并避免把过早读取当成证据。",
    hypotheses:[
      {id:"H1",name:"执行器正常，但物理效果有延迟",p:.50},
      {id:"H2",name:"阀门/执行器实际卡住",p:.30},
      {id:"H3",name:"流量传感器自身存在滞后",p:.20}
    ]
  },
  retrieval: {
    title:"RAG contradiction",
    goal:"一次模型与语料刷新后，检索质量下降。判断是语料版本漂移、embedding 分布变化，还是 reranker 行为变化，并找出最值得先验证的原因。",
    hypotheses:[
      {id:"H1",name:"语料版本/切分发生漂移",p:.38},
      {id:"H2",name:"embedding 分布整体迁移",p:.37},
      {id:"H3",name:"reranker 排序逻辑变化",p:.25}
    ]
  },
  custom: {
    title:"Custom task",
    goal:"",
    hypotheses:[
      {id:"H1",name:"观测本身存在偏差或测量问题",p:.30},
      {id:"H2",name:"最直接的主因机制",p:.45},
      {id:"H3",name:"被忽略的约束或隐藏变量",p:.25}
    ]
  }
};

function scenarioForCurrentGoal(){
  var base = presets[preset] || presets.custom;
  var hs = base.hypotheses.map(function(h){ return {id:h.id,name:h.name,p:h.p}; });
  if(preset === "custom"){
    var g = $("goal").value.trim();
    if(/检索|rag|embedding|召回|排序/i.test(g)){
      hs = presets.retrieval.hypotheses.map(function(h){return {id:h.id,name:h.name,p:h.p};});
    }else if(/延迟|阀门|等待|时序|sensor lag|stale/i.test(g)){
      hs = presets.temporal.hypotheses.map(function(h){return {id:h.id,name:h.name,p:h.p};});
    }else if(/co2|通风|窗|风量|空气/i.test(g)){
      hs = presets.hvac.hypotheses.map(function(h){return {id:h.id,name:h.name,p:h.p};});
    }
  }
  return hs;
}

function makeCandidate(id,kind,name,reason,s){
  s = s || {};
  var score = {
    model:s.model == null ? .60 : s.model,
    eig:s.eig == null ? .35 : s.eig,
    evsi:s.evsi == null ? .30 : s.evsi,
    utility:s.utility == null ? .25 : s.utility,
    cost:s.cost == null ? .08 : s.cost,
    risk:s.risk == null ? .05 : s.risk,
    temporal:s.temporal == null ? 1 : s.temporal
  };
  score.total = clamp(
    score.eig*.30 + score.evsi*.28 + score.utility*.23 + score.model*.12 + score.temporal*.12 - score.cost*.13 - score.risk*.18,
    -1,1
  );
  return {
    id:id,kind:kind,name:name,reason:reason,score:score,
    valid:s.valid !== false,
    invalidReason:s.invalidReason || "",
    observation:s.observation || "",
    posterior:s.posterior || null,
    cost:s.cost == null ? .08 : s.cost,
    cf:s.cf || ""
  };
}

function candidatesHVAC(){
  if(state.step === 0){
    return [
      makeCandidate("reference_sensor","observe","用独立参考仪复测 CO₂","直接检验“读数是不是假的”，信息量中等但成本低。",{model:.76,eig:.58,evsi:.46,utility:.22,cost:.05,risk:.01,observation:"参考仪读数与主传感器接近：主传感器 1260 ppm，参考仪 1215 ppm。",posterior:{H1:.08,H2:.57,H3:.35},cf:"很快排除传感器漂移，但对 H2/H3 区分有限。"}),
      makeCandidate("airflow_test","observe","测窗边与门缝压差/风速","如果窗开着却没有跨房间压差，这个观测能直接区分 H2 与 H3。",{model:.71,eig:.84,evsi:.72,utility:.24,cost:.09,risk:.02,observation:"窗边局部有风，但门缝压差接近 0；房间没有形成稳定穿堂气流。",posterior:{H1:.08,H2:.78,H3:.14},cf:"最快把后验推向‘没有形成有效气流路径’。"}),
      makeCandidate("open_door","intervene","把门完全打开 5 分钟","直接改变流路。如果 CO₂ 下降，就能验证有效通风路径的重要性。",{model:.88,eig:.41,evsi:.55,utility:.66,cost:.10,risk:.10,observation:"门打开后 5 分钟，CO₂ 从 1260 ppm 降到 1015 ppm。",posterior:{H1:.07,H2:.84,H3:.09},cf:"高概率直接改善结果，但会牺牲部分诊断纯度。"})
    ];
  }
  if(state.step === 1){
    return [
      makeCandidate("door_crack","observe","比较门关 / 门开两种状态下的 CO₂ 斜率","做最小干预，比较同一房间在两种连通条件下的变化率。",{model:.69,eig:.76,evsi:.72,utility:.31,cost:.07,risk:.01,observation:"门关时 CO₂ +18 ppm/min；门开时 CO₂ -31 ppm/min。",posterior:{H1:.04,H2:.90,H3:.06},cf:"给 H2 很强的因果支持。"}),
      makeCandidate("exhaust_check","observe","检查排风/回风口实际流量","排除 H3：如果回风正常，就不该继续花预算修排风。",{model:.62,eig:.51,evsi:.44,utility:.20,cost:.06,risk:.01,observation:"回风口流量在正常范围，未发现明显堵塞。",posterior:{H1:.07,H2:.82,H3:.11},cf:"排除回风堵塞，但信息价值低于门状态 A/B。"}),
      makeCandidate("keep_door_open","intervene","保持门开启并联动窗户","根据当前最高后验，直接建立进风+出风路径。",{model:.78,eig:.10,evsi:.27,utility:.90,cost:.04,risk:.08,observation:"10 分钟后 CO₂ 降至 820 ppm，下降持续且稳定。",posterior:{H1:.03,H2:.94,H3:.03},cf:"最快结束任务，效果可观察且可逆。"})
    ];
  }
  return [
    makeCandidate("keep_door_open","intervene","采用门窗联动形成稳定通风路径","当前证据已经足够，继续诊断的 EVSI 低于直接干预收益。",{model:.82,eig:.06,evsi:.12,utility:.94,cost:.03,risk:.06,observation:"通风联动持续后 CO₂ 稳定在 780–850 ppm。",posterior:{H1:.02,H2:.96,H3:.02},cf:"完成目标。"}),
    makeCandidate("more_probe","observe","再做一次低价值确认测量","可减少少量不确定性，但不太可能改变决策。",{model:.58,eig:.18,evsi:.05,utility:.08,cost:.08,risk:.01,observation:"新增观测与当前 H2 一致，但没有改变最佳动作。",posterior:{H1:.03,H2:.93,H3:.04},cf:"增加成本，几乎不改变决策。"})
  ];
}

function candidatesTemporal(){
  if(state.step === 0){
    return [
      makeCandidate("read_now","observe","立刻读取流量","模型倾向马上看结果，但当前 observation 还没有进入有效 causal window。",{model:.91,eig:.78,evsi:.49,utility:.10,cost:.03,risk:.05,temporal:.05,valid:false,invalidReason:"Temporal guard: intervention effect window has not opened yet.",observation:"当前读数仍接近基线，但这个结果被标记为 stale / premature。",posterior:null,cf:"如果把这条 stale observation 当证据，会错误提升 H2。"}),
      makeCandidate("wait90","wait","等待 90 秒进入有效观测窗口","等待本身不提供信息，但让下一条 observation 具有因果归因资格。",{model:.42,eig:.05,evsi:.71,utility:.52,cost:.01,risk:.00,temporal:1,observation:"已等待 90 秒。系统进入预设 observation window。",posterior:{H1:.50,H2:.30,H3:.20},cf:"不改变 belief，但保护后续证据质量。"}),
      makeCandidate("inspect_actuator","observe","读取执行器位置反馈","可以提前检验机械执行是否到位，但不能替代等待物理量稳定。",{model:.67,eig:.55,evsi:.42,utility:.20,cost:.05,risk:.01,temporal:.72,observation:"执行器位置反馈显示阀门已到 96% 开度。",posterior:{H1:.63,H2:.12,H3:.25},cf:"缩小 H2，但仍需等到物理效果窗口。"})
    ];
  }
  if(state.step === 1){
    return [
      makeCandidate("read_after","observe","在有效窗口内读取流量","现在 observation 可以合法归因到前一步 intervention。",{model:.82,eig:.86,evsi:.80,utility:.28,cost:.03,risk:.01,temporal:1,observation:"流量从 0.42 升到 0.81 m³/s，变化发生在有效窗口内。",posterior:{H1:.86,H2:.05,H3:.09},cf:"明确支持真实延迟，而非阀门卡死。"}),
      makeCandidate("sensor_crosscheck","observe","用第二测点交叉验证流量","检验 H3，但成本略高。",{model:.60,eig:.58,evsi:.39,utility:.18,cost:.08,risk:.01,temporal:1,observation:"第二测点也记录到相同方向的流量上升。",posterior:{H1:.78,H2:.07,H3:.15},cf:"提高 H1，但不如窗口内主测点直接。"}),
      makeCandidate("replace_valve","intervene","直接更换阀门执行器","高成本且当前没有足够证据支持 H2。",{model:.49,eig:.03,evsi:.05,utility:.25,cost:.42,risk:.32,temporal:1,observation:"更换后系统恢复，但无法证明原执行器就是主因。",posterior:{H1:.55,H2:.25,H3:.20},cf:"代价高，并污染诊断。"})
    ];
  }
  return [
    makeCandidate("accept_delay","intervene","把延迟窗口写入控制策略","将 90 秒 settling window 作为下一次判断前置条件，避免 premature read。",{model:.84,eig:.04,evsi:.16,utility:.93,cost:.04,risk:.03,temporal:1,observation:"策略更新后，后续动作不再在窗口开启前读取并归因。",posterior:{H1:.91,H2:.03,H3:.06},cf:"完成任务并修复控制逻辑。"})
  ];
}

function candidatesRetrieval(){
  if(state.step === 0){
    return [
      makeCandidate("centroid_shift","observe","比较刷新前后的 embedding centroid / NN turnover","直接检验表征空间是否整体迁移。",{model:.75,eig:.82,evsi:.70,utility:.25,cost:.08,risk:.01,observation:"embedding centroid 明显移动，nearest-neighbor turnover 为 18%。",posterior:{H1:.18,H2:.70,H3:.12},cf:"快速锁定 embedding drift。"}),
      makeCandidate("corpus_diff","observe","对比语料版本、chunk 数量与来源分布","检验是不是语料本身换了，而不是模型空间变了。",{model:.78,eig:.66,evsi:.59,utility:.21,cost:.06,risk:.01,observation:"语料总量与来源分布变化很小，chunk 数仅 +1.8%。",posterior:{H1:.12,H2:.61,H3:.27},cf:"排除大规模 corpus drift。"}),
      makeCandidate("reranker_off","intervene","临时关闭 reranker 做 A/B","快速判断 reranker 是否是质量下降主因。",{model:.63,eig:.55,evsi:.62,utility:.58,cost:.07,risk:.06,observation:"关闭 reranker 后 NDCG@10 仅恢复 0.7%，主要下降仍存在。",posterior:{H1:.17,H2:.72,H3:.11},cf:"对 H3 形成反证。"})
    ];
  }
  if(state.step === 1){
    return [
      makeCandidate("old_embed_ab","observe","旧 embedding vs 新 embedding 同语料 A/B","冻结 corpus 和 query，只替换 embedding 版本，隔离变量。",{model:.81,eig:.89,evsi:.86,utility:.29,cost:.10,risk:.01,observation:"旧 embedding 恢复 NDCG@10 +6.4%，新 embedding 仍下降。",posterior:{H1:.06,H2:.90,H3:.04},cf:"强因果证据指向 embedding 版本。"}),
      makeCandidate("reranker_trace","observe","检查 reranker score distribution","进一步排查 H3，但改变最终决策的概率不高。",{model:.58,eig:.33,evsi:.17,utility:.12,cost:.05,risk:.01,observation:"reranker score 分布与旧版本近似，未见结构性漂移。",posterior:{H1:.10,H2:.83,H3:.07},cf:"确认 H3 较弱。"}),
      makeCandidate("rollback_embed","intervene","回滚 embedding 模型并重建索引","基于当前后验执行可逆回滚。",{model:.83,eig:.04,evsi:.23,utility:.88,cost:.18,risk:.08,observation:"回滚并重建后 NDCG@10 恢复 6.6%，Recall@50 恢复 5.1%。",posterior:{H1:.04,H2:.93,H3:.03},cf:"快速恢复系统质量。"})
    ];
  }
  return [
    makeCandidate("rollback_embed","intervene","固定旧 embedding 并安排兼容性迁移","当前 posterior 已足以支持可逆 rollback；继续采样的 EVSI 很低。",{model:.86,eig:.03,evsi:.10,utility:.94,cost:.15,risk:.07,observation:"线上回滚完成，核心检索指标恢复到刷新前区间。",posterior:{H1:.03,H2:.95,H3:.02},cf:"结束事故并保留后续迁移窗口。"})
  ];
}

function candidatesCustom(){
  var top = topHypothesis();
  if(state.step === 0){
    return [
      makeCandidate("discriminating_probe","observe","收集最能区分竞争假设的一条证据","优先购买会改变 posterior 的 observation，而不是重复确认当前直觉。",{model:.74,eig:.82,evsi:.68,utility:.24,cost:.07,risk:.01,observation:"获得一条高区分度观测：它更支持“"+top.name+"”，同时削弱另两个解释。",posterior:{H1:.16,H2:.68,H3:.16},cf:"最快减少不确定性。"}),
      makeCandidate("cheap_check","observe","先做一个低成本 sanity check","便宜，但对假设区分有限。",{model:.67,eig:.32,evsi:.19,utility:.16,cost:.02,risk:.01,observation:"sanity check 没发现明显异常，但没有真正区分主因。",posterior:{H1:.26,H2:.49,H3:.25},cf:"成本低，但可能浪费一步。"}),
      makeCandidate("direct_action","intervene","直接对当前最可能主因采取可逆动作","若时间价值更高，可以牺牲部分诊断纯度换取快速反馈。",{model:.76,eig:.18,evsi:.41,utility:.70,cost:.09,risk:.12,observation:"可逆干预后目标指标明显改善，说明当前 leading hypothesis 具有实际解释力。",posterior:{H1:.10,H2:.80,H3:.10},cf:"更快得到现实反馈，但解释性稍弱。"})
    ];
  }
  return [
    makeCandidate("confirm_probe","observe","做一次反事实式确认","专门寻找会推翻当前 leading hypothesis 的证据。",{model:.68,eig:.65,evsi:.56,utility:.20,cost:.06,risk:.01,observation:"反证探针没有推翻当前 leading hypothesis，后验进一步集中。",posterior:{H1:.09,H2:.82,H3:.09},cf:"提高置信度。"}),
    makeCandidate("direct_action","intervene","执行可逆干预并观察真实结果","当前信息已足以行动，把剩余不确定性交给现实反馈。",{model:.80,eig:.10,evsi:.24,utility:.87,cost:.08,risk:.10,observation:"干预后目标指标朝预期方向变化，且没有触发安全约束。",posterior:{H1:.06,H2:.88,H3:.06},cf:"结束任务。"})
  ];
}

function generateCandidates(){
  var arr;
  if(preset === "hvac") arr = candidatesHVAC();
  else if(preset === "temporal") arr = candidatesTemporal();
  else if(preset === "retrieval") arr = candidatesRetrieval();
  else arr = candidatesCustom();

  if(state.operatorMessage){
    var msg = state.operatorMessage;
    arr.unshift(makeCandidate(
      "operator_probe_"+state.step,"observe",
      "Operator-directed check",
      "Human operator asked: “"+msg+"”. Runtime converts it into a safe information-gathering step rather than treating it as true evidence.",
      {model:.72,eig:.70,evsi:.66,utility:.22,cost:.05,risk:.01,
       observation:"已按 operator message 执行检查。结果作为 observation 写入，而不是把 operator message 本身当作证据。",
       posterior:null,cf:"尊重人的方向，但仍保持证据边界。"}
    ));
  }

  arr.forEach(function(c,i){ c.index=i; });
  var valid = arr.filter(function(c){return c.valid;});
  var recommended = valid.slice().sort(function(a,b){return b.score.total-a.score.total;})[0];
  state.candidates = arr;
  state.recommendedId = recommended ? recommended.id : null;
  state.selectedId = state.recommendedId;
}

function start(){
  var goal = $("goal").value.trim();
  if(!goal){ $("goal").focus(); return; }
  state = {
    id:uid(),goal:goal,step:0,phase:"gate",budget:3,cost:0,
    hypotheses:scenarioForCurrentGoal(),candidates:[],selectedId:null,recommendedId:null,
    history:[],trace:[],operatorMessage:"",startedAt:new Date().toISOString(),pendingObservation:null
  };
  state.trace.push({type:"goal",label:"Goal accepted",detail:goal});
  state.trace.push({type:"world_model",label:"World model initialized",detail:"Competing hypotheses are explicit and defeasible."});
  generateCandidates();
  $("workbench").classList.remove("hidden");
  $("finalCard").classList.add("hidden");
  $("observationCard").classList.add("hidden");
  $("humanGate").classList.remove("hidden");
  renderAll();
  setTimeout(function(){ $("workbench").scrollIntoView({behavior:"smooth",block:"start"}); },50);
}

function topHypothesis(){
  if(!state || !state.hypotheses.length) return {id:"H?",name:"unknown",p:0};
  return state.hypotheses.slice().sort(function(a,b){return b.p-a.p;})[0];
}

function normalizeBeliefs(){
  var sum = state.hypotheses.reduce(function(a,h){return a+h.p;},0) || 1;
  state.hypotheses.forEach(function(h){ h.p = h.p/sum; });
}

function applyPosterior(posterior){
  if(!posterior) return;
  state.hypotheses.forEach(function(h){
    if(posterior[h.id] != null) h.p = posterior[h.id];
  });
  normalizeBeliefs();
}

function scoreRows(c){
  if(!c) return "";
  var s = c.score;
  var rows = [
    ["model proposal",s.model],
    ["Bayesian EIG",s.eig],
    ["EVSI",s.evsi],
    ["utility",s.utility],
    ["temporal validity",s.temporal],
    ["cost penalty",-s.cost],
    ["risk penalty",-s.risk]
  ];
  return rows.map(function(r){
    var val = r[1];
    var width = Math.round(Math.min(1,Math.abs(val))*100);
    return '<div class="score-row"><span>'+esc(r[0])+'</span><div class="score-bar"><i style="width:'+width+'%"></i></div><strong>'+Number(val).toFixed(2)+'</strong></div>';
  }).join("") + '<div class="score-row"><span>TOTAL</span><div class="score-bar"><i style="width:'+Math.round(Math.max(0,c.score.total)*100)+'%"></i></div><strong>'+c.score.total.toFixed(3)+'</strong></div>';
}

function renderBeliefs(){
  $("beliefs").innerHTML = state.hypotheses.map(function(h){
    return '<div class="belief-card"><div class="belief-top"><b>'+esc(h.id)+'</b><span>'+Math.round(h.p*100)+'%</span></div><p>'+esc(h.name)+'</p><div class="belief-bar"><i style="width:'+Math.round(h.p*100)+'%"></i></div></div>';
  }).join("");
}

function renderCandidates(){
  var rec = state.recommendedId;
  $("candidates").innerHTML = state.candidates.map(function(c){
    var cls = "candidate";
    if(c.id === state.selectedId) cls += " selected";
    if(c.id === rec) cls += " recommended";
    if(!c.valid) cls += " invalid";
    return '<button class="'+cls+'" data-id="'+esc(c.id)+'" type="button"'+(!c.valid?' disabled':'')+'>'+
      '<div class="candidate-head"><span class="kind">'+esc(c.kind.toUpperCase())+'</span><h3>'+esc(c.name)+'</h3></div>'+
      '<p>'+esc(c.reason)+(c.valid?'':' · '+esc(c.invalidReason))+'</p>'+
      '<div class="score-strip"><span>total '+c.score.total.toFixed(3)+'</span><span>EIG '+c.score.eig.toFixed(2)+'</span><span>EVSI '+c.score.evsi.toFixed(2)+'</span><span>risk '+c.score.risk.toFixed(2)+'</span><span>cost '+c.score.cost.toFixed(2)+'</span></div>'+
      '</button>';
  }).join("");
  Array.prototype.forEach.call(document.querySelectorAll(".candidate"),function(btn){
    btn.addEventListener("click",function(){
      state.selectedId = btn.getAttribute("data-id");
      renderCandidates();
      renderInspector();
    });
  });
}

function selectedCandidate(){
  if(!state) return null;
  return state.candidates.find(function(c){return c.id===state.selectedId;}) || null;
}

function renderInspector(){
  var c = selectedCandidate();
  $("ctxSelected").textContent = c ? c.kind+" · "+c.name : "—";
  $("ctxWhy").textContent = c ? c.reason : "选择一个 candidate 后，这里会显示完整 runtime score provenance。";
  $("scoreTable").innerHTML = scoreRows(c);
  if(c){
    $("mathBody").textContent =
      "score = 0.30·EIG + 0.28·EVSI + 0.23·utility + 0.12·model + 0.12·temporal\n"+
      "        - 0.13·cost - 0.18·risk\n\n"+
      "EIG      = "+c.score.eig.toFixed(3)+"\n"+
      "EVSI     = "+c.score.evsi.toFixed(3)+"\n"+
      "utility  = "+c.score.utility.toFixed(3)+"\n"+
      "temporal = "+c.score.temporal.toFixed(3)+"\n"+
      "TOTAL    = "+c.score.total.toFixed(3)+"\n\n"+
      (c.valid ? "runtime validation: PASS" : "runtime validation: REJECTED — "+c.invalidReason);
  }else{
    $("mathBody").textContent = "No candidate selected yet.";
  }
}

function renderTrace(){
  $("traceList").innerHTML = state.trace.map(function(t,i){
    return '<div class="trace-item '+(i===state.trace.length-1?'active':'')+'"><span>#'+i+' · '+esc(t.type.toUpperCase())+'</span><b>'+esc(t.label)+'</b><span>'+esc(t.detail||"")+'</span></div>';
  }).join("");
}

function renderForks(){
  if(!state.history.length){
    $("forkControls").innerHTML = '<p class="muted">还没有 frozen step。先执行至少一步。</p>';
    $("forkResult").innerHTML = "";
    return;
  }
  $("forkControls").innerHTML = state.history.map(function(h,idx){
    var alt = h.candidates.filter(function(c){return c.id!==h.chosen.id && c.valid;}).sort(function(a,b){return b.score.total-a.score.total;})[0];
    return '<div class="fork-step"><strong>Step '+(idx+1)+' · actual: '+esc(h.chosen.name)+'</strong>'+
      '<div class="muted">'+esc(h.observation)+'</div>'+
      (alt?'<button class="tiny forkBtn" data-step="'+idx+'" data-alt="'+esc(alt.id)+'" type="button">Fork with: '+esc(alt.name)+'</button>':'')+
      '</div>';
  }).join("");
  Array.prototype.forEach.call(document.querySelectorAll(".forkBtn"),function(btn){
    btn.addEventListener("click",function(){
      showFork(Number(btn.getAttribute("data-step")),btn.getAttribute("data-alt"));
    });
  });
}

function showFork(stepIndex,altId){
  var h = state.history[stepIndex];
  if(!h) return;
  var alt = h.candidates.find(function(c){return c.id===altId;});
  if(!alt) return;
  var actualTop = h.posteriorAfter ? Object.keys(h.posteriorAfter).sort(function(a,b){return h.posteriorAfter[b]-h.posteriorAfter[a];})[0] : h.topBefore;
  var altOutcome = alt.cf || alt.observation || "Alternative branch changes the evidence path.";
  $("forkResult").innerHTML =
    '<div class="fork-compare">'+
      '<div class="fork-branch"><strong>ACTUAL FUTURE</strong><p>'+esc(h.chosen.name)+'</p><p>'+esc(h.observation)+'</p><p>leading posterior: '+esc(actualTop||"—")+'</p></div>'+
      '<div class="fork-branch"><strong>COUNTERFACTUAL FUTURE</strong><p>'+esc(alt.name)+'</p><p>'+esc(altOutcome)+'</p><p>estimated branch score: '+alt.score.total.toFixed(3)+'</p></div>'+
    '</div>';
}

function renderLifecycle(){
  var labels = ["Goal","Hypotheses","Propose","Runtime score","Human gate","Execute","Observe","Update belief"];
  var active = 4;
  if(state.phase==="observation") active = 6;
  if(state.phase==="final") active = 7;
  var lis = $("lifecycle").querySelectorAll("li");
  Array.prototype.forEach.call(lis,function(li,i){
    li.classList.remove("active","done");
    if(i < active) li.classList.add("done");
    if(i === active) li.classList.add("active");
  });
}

function renderAll(){
  if(!state) return;
  $("sessionTitle").textContent = state.id+" · "+(presets[preset]?presets[preset].title:"Custom");
  $("sessionMeta").textContent = "step "+state.step+" · "+state.phase+" · "+state.history.length+" frozen decisions";
  $("budget").textContent = state.budget+" / 3";
  $("cost").textContent = state.cost.toFixed(2);
  $("stageKicker").textContent = "STEP "+(state.step+1)+" · "+(state.phase==="gate"?"RUNTIME ARBITRATION":state.phase.toUpperCase());
  $("stageTitle").textContent = state.phase==="gate" ? "Runtime 已评分，等待你的决定" : state.phase==="observation" ? "世界已经返回 observation" : "Trajectory complete";
  renderBeliefs();
  renderCandidates();
  renderInspector();
  renderTrace();
  renderForks();
  renderLifecycle();
  $("rawJson").textContent = JSON.stringify(state,null,2);
}

function execute(candidate){
  if(!candidate || !candidate.valid) return;
  var before = {};
  state.hypotheses.forEach(function(h){before[h.id]=h.p;});
  var frozenCandidates = state.candidates.map(function(c){return JSON.parse(JSON.stringify(c));});
  state.cost += candidate.cost;
  if(candidate.kind==="observe" || candidate.kind==="retrieve") state.budget = Math.max(0,state.budget-1);

  var observation = candidate.observation || "World returned an observation.";
  state.pendingObservation = {
    candidate:candidate,
    text:observation,
    posterior:candidate.posterior,
    before:before,
    candidates:frozenCandidates
  };
  state.phase = "observation";
  state.trace.push({type:"execute",label:candidate.kind+" · "+candidate.name,detail:"Tool/capability executed after Human Gate."});
  state.trace.push({type:"observe",label:"World observation",detail:observation});
  $("humanGate").classList.add("hidden");
  $("observationCard").classList.remove("hidden");
  $("observationTitle").textContent = candidate.name;
  $("observationBody").textContent = observation;
  renderAll();
}

function continueAfterObservation(){
  if(!state.pendingObservation) return;
  var p = state.pendingObservation;
  applyPosterior(p.posterior);
  var after = {};
  state.hypotheses.forEach(function(h){after[h.id]=h.p;});
  var top = topHypothesis();

  state.trace.push({type:"belief_update",label:"Posterior updated",detail:top.id+" "+Math.round(top.p*100)+"% · "+top.name});
  state.history.push({
    step:state.step,
    topBefore:Object.keys(p.before).sort(function(a,b){return p.before[b]-p.before[a];})[0],
    beliefsBefore:p.before,
    candidates:p.candidates,
    chosen:JSON.parse(JSON.stringify(p.candidate)),
    observation:p.text,
    posteriorAfter:after
  });
  var wasIntervention = p.candidate.kind==="intervene";
  state.pendingObservation = null;
  state.operatorMessage = "";
  state.step += 1;

  if(wasIntervention || state.step>=4 || (state.budget===0 && top.p>=.70)){
    finish();
    return;
  }

  state.phase = "gate";
  generateCandidates();
  $("observationCard").classList.add("hidden");
  $("humanGate").classList.remove("hidden");
  renderAll();
}

function finish(){
  state.phase = "final";
  var top = topHypothesis();
  state.trace.push({type:"final",label:"Trajectory complete",detail:"Leading hypothesis: "+top.id+" "+Math.round(top.p*100)+"%"});
  $("observationCard").classList.add("hidden");
  $("humanGate").classList.add("hidden");
  $("finalCard").classList.remove("hidden");
  $("finalTitle").textContent = top.id+" · "+top.name;
  $("finalText").textContent = "最终 posterior "+Math.round(top.p*100)+"%。系统不是因为“模型说它对”而结束，而是因为当前证据、行动价值与剩余预算已经使继续采样的价值下降。";
  $("finalMetrics").innerHTML =
    '<span>steps '+state.history.length+'</span>'+
    '<span>cost '+state.cost.toFixed(2)+'</span>'+
    '<span>remaining probes '+state.budget+'</span>'+
    '<span>posterior '+Math.round(top.p*100)+'%</span>';
  renderAll();
}

function replan(){
  if(!state || state.phase!=="gate") return;
  var msg = $("operatorMessage").value.trim();
  if(!msg) return;
  state.operatorMessage = msg;
  state.trace.push({type:"operator",label:"Operator requested replan",detail:msg});
  generateCandidates();
  $("operatorMessage").value = "";
  renderAll();
}

function addHypothesis(){
  if(!state) return;
  var statement = window.prompt("新增一个 provisional hypothesis：");
  if(!statement || !statement.trim()) return;
  var id = "H"+(state.hypotheses.length+1);
  state.hypotheses.forEach(function(h){h.p *= .86;});
  state.hypotheses.push({id:id,name:statement.trim(),p:.14});
  normalizeBeliefs();
  state.trace.push({type:"human_hypothesis",label:id+" added by human",detail:statement.trim()+" · provisional, not evidence"});
  renderAll();
}

function exportSession(){
  if(!state) return;
  var blob = new Blob([JSON.stringify(state,null,2)],{type:"application/json"});
  var a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "causalrag-"+state.id+".json";
  document.body.appendChild(a);
  a.click();
  setTimeout(function(){URL.revokeObjectURL(a.href);a.remove();},0);
}

function parseFirstJson(text){
  var start = text.indexOf("{");
  if(start<0) return null;
  var depth=0,quoted=false,escaped=false;
  for(var i=start;i<text.length;i++){
    var c=text[i];
    if(quoted){
      if(escaped) escaped=false;
      else if(c==="\\") escaped=true;
      else if(c==="\"") quoted=false;
      continue;
    }
    if(c==="\"") quoted=true;
    else if(c==="{") depth++;
    else if(c==="}" && --depth===0){
      try{return JSON.parse(text.slice(start,i+1));}catch(e){return null;}
    }
  }
  return null;
}

function loadEvidence(){
  fetch("data/runtime.json").then(function(r){
    if(!r.ok) throw new Error(String(r.status));
    return r.json();
  }).then(function(data){
    var suite = parseFirstJson(data.hidden_world_suite || "");
    $("build").textContent = "VERIFIED · "+String(data.commit||"current").slice(0,8)+" · PYTHON "+(data.python||"");
    if(suite){
      $("accuracy").textContent = Math.round(suite.identification_accuracy*100)+"%";
      $("posterior").textContent = Number(suite.mean_true_hypothesis_posterior).toFixed(3);
      $("regret").textContent = Number(suite.mean_causal_regret).toFixed(3);
      $("probes").textContent = Number(suite.mean_probes).toFixed(1);
    }
    $("verifiedRaw").textContent = JSON.stringify({
      commit:data.commit,
      playable_ab:data.playable_ab,
      hidden_world_suite:suite,
      bayesian_experiment:data.bayesian_experiment
    },null,2);
  }).catch(function(err){
    $("build").textContent = "WORKBENCH READY · VERIFIED EVIDENCE UNAVAILABLE";
    $("verifiedRaw").textContent = "Could not load CI-generated runtime evidence: "+err;
  });
}

Array.prototype.forEach.call(document.querySelectorAll(".preset"),function(btn){
  btn.addEventListener("click",function(){
    preset = btn.getAttribute("data-preset");
    Array.prototype.forEach.call(document.querySelectorAll(".preset"),function(b){b.classList.toggle("active",b===btn);});
    if(preset!=="custom") $("goal").value = presets[preset].goal;
    else {
      $("goal").focus();
      if(!$("goal").value.trim()) $("goal").placeholder = "输入任意现实任务…";
    }
  });
});

Array.prototype.forEach.call(document.querySelectorAll(".tab"),function(btn){
  btn.addEventListener("click",function(){
    var tab=btn.getAttribute("data-tab");
    Array.prototype.forEach.call(document.querySelectorAll(".tab"),function(b){b.classList.toggle("active",b===btn);});
    ["context","trace","counterfactual"].forEach(function(name){
      $("tab-"+name).classList.toggle("hidden",name!==tab);
    });
    if(state){renderTrace();renderForks();}
  });
});

$("startSession").addEventListener("click",start);
$("resetAll").addEventListener("click",function(){
  state=null;
  $("workbench").classList.add("hidden");
  $("goal").value=presets[preset].goal || "";
});
$("approve").addEventListener("click",function(){
  if(!state) return;
  var c=state.candidates.find(function(x){return x.id===state.recommendedId;});
  state.selectedId=state.recommendedId;
  execute(c);
});
$("executeSelected").addEventListener("click",function(){ execute(selectedCandidate()); });
$("replan").addEventListener("click",replan);
$("operatorMessage").addEventListener("keydown",function(e){ if(e.key==="Enter"){e.preventDefault();replan();} });
$("continueBtn").addEventListener("click",continueAfterObservation);
$("addHypothesis").addEventListener("click",addHypothesis);
$("exportSession").addEventListener("click",exportSession);
$("toggleRaw").addEventListener("click",function(){
  if(!state) return;
  $("rawState").classList.toggle("hidden");
  $("rawJson").textContent=JSON.stringify(state,null,2);
});
$("restartSame").addEventListener("click",start);

loadEvidence();
})();