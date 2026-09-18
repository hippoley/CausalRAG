const $ = id => document.getElementById(id);
let finished = false;

function setStages(active) {
  [...document.querySelectorAll('#stages span')].forEach((el, i) => el.classList.toggle('active', i <= active));
}

function reset() {
  finished = false;
  $('h1v').textContent = $('h2v').textContent = '50%';
  $('h1bar').style.width = $('h2bar').style.width = '50%';
  $('beliefNote').textContent = 'Equal priors. The explainer has not observed a discriminating outcome.';
  $('trace').innerHTML = '<li class="current">World model initialized</li><li>Waiting for an observation</li><li>Posterior not updated</li>';
  document.querySelectorAll('.action').forEach(b => b.disabled = false);
  setStages(0);
}

function choose(kind) {
  if (finished) return;
  setStages(1);
  document.querySelectorAll('.action').forEach(b => b.disabled = true);
  if (kind === 'temperature') {
    $('trace').innerHTML = '<li class="done">World model initialized</li><li class="done">Temperature observed: 25°C</li><li class="current">Equal likelihood under H1 and H2</li>';
    $('beliefNote').textContent = 'No update: the observation does not distinguish filter restriction from fan weakness.';
    setStages(2);
    setTimeout(() => { document.querySelectorAll('.action').forEach(b => b.disabled = false); }, 700);
    return;
  }
  finished = true;
  setTimeout(() => {
    $('h1v').textContent = '10%'; $('h2v').textContent = '90%';
    $('h1bar').style.width = '10%'; $('h2bar').style.width = '90%';
    $('beliefNote').textContent = 'Observed normal pressure. Bayes update: P(H2 | normal) = 0.90.';
    $('trace').innerHTML = '<li class="done">World model initialized</li><li class="done">Pressure test selected by runtime EIG</li><li class="done">Observed: normal</li><li class="current">Weak fan becomes the leading mechanism</li>';
    setStages(3);
  }, 350);
}

function parseFirstJson(text) {
  const start = text.indexOf('{');
  if (start < 0) return null;
  let depth = 0, quoted = false, escaped = false;
  for (let i = start; i < text.length; i++) {
    const c = text[i];
    if (quoted) { if (escaped) escaped = false; else if (c === '\\') escaped = true; else if (c === '"') quoted = false; continue; }
    if (c === '"') quoted = true;
    else if (c === '{') depth++;
    else if (c === '}' && --depth === 0) return JSON.parse(text.slice(start, i + 1));
  }
  return null;
}

function selectedActions(episode) {
  return (episode.decisions || []).map((decision, index) => {
    const selected = decision.selected || {};
    return {index, kind: selected.kind || '—', name: selected.name || '—', rationale: selected.rationale || decision.rationale || ''};
  });
}

function armMetrics(episode) {
  const m = episode.metrics || {};
  return [
    ['success', m.success ? 'YES' : 'NO', m.success ? 'good' : 'bad'],
    ['regret', Number(m.causal_regret || 0).toFixed(3), ''],
    ['cost', Number(m.total_cost || 0).toFixed(3), ''],
    ['true posterior', Number(m.true_hypothesis_posterior || 0).toFixed(3), ''],
  ].map(([label, value, cls]) => `<div class="mini-metric"><span>${label}</span><strong class="${cls}">${value}</strong></div>`).join('');
}

function armActions(episode) {
  return selectedActions(episode).map(row => `<div class="action-row"><span>#${row.index}</span><b>${row.kind} · ${row.name}</b><small>${row.rationale}</small></div>`).join('');
}

function renderAB(report) {
  const d = report.first_divergence;
  if (d) {
    const v = d.vanilla ? `${d.vanilla.kind} · ${d.vanilla.name}` : 'ended';
    const c = d.causal ? `${d.causal.kind} · ${d.causal.name}` : 'ended';
    $('divergence').innerHTML = `<b>First divergence: step ${d.step}</b><span>vanilla → ${v}</span><span>causal → ${c}</span>`;
  } else {
    $('divergence').innerHTML = '<b>No action-sequence divergence in this replay.</b>';
  }
  $('vanillaMetrics').innerHTML = armMetrics(report.vanilla);
  $('causalMetrics').innerHTML = armMetrics(report.causal);
  $('vanillaActions').innerHTML = armActions(report.vanilla);
  $('causalActions').innerHTML = armActions(report.causal);
  const delta = report.metric_deltas_causal_minus_vanilla || {};
  $('deltas').innerHTML = [
    ['Δ success', report.success_delta ?? 0],
    ['Δ regret', Number(delta.causal_regret || 0).toFixed(3)],
    ['Δ cost', Number(delta.total_cost || 0).toFixed(3)],
    ['Δ probes', Number(delta.probes || 0).toFixed(1)],
  ].map(([k,v]) => `<span><b>${k}</b> ${v}</span>`).join('');
}

async function loadEvidence() {
  try {
    const data = await fetch('data/runtime.json').then(r => { if (!r.ok) throw new Error(r.status); return r.json(); });
    const suite = parseFirstJson(data.hidden_world_suite);
    $('build').textContent = `VERIFIED · ${data.commit} · PYTHON ${data.python}`;
    $('accuracy').textContent = `${Math.round(suite.identification_accuracy * 100)}%`;
    $('posterior').textContent = suite.mean_true_hypothesis_posterior.toFixed(3);
    $('regret').textContent = suite.mean_causal_regret.toFixed(3);
    $('probes').textContent = suite.mean_probes.toFixed(1);
    renderAB(data.playable_ab);
    $('raw').textContent = JSON.stringify({
      playable_ab: data.playable_ab,
      bayesian_experiment: data.bayesian_experiment,
      hidden_world_suite: suite,
    }, null, 2);
  } catch (error) {
    $('build').textContent = 'BUILD EVIDENCE UNAVAILABLE';
    $('raw').textContent = `Could not load generated runtime evidence: ${error}`;
    $('divergence').textContent = 'A/B replay evidence unavailable.';
  }
}

document.querySelectorAll('.action').forEach(b => b.addEventListener('click', () => choose(b.dataset.action)));
$('reset').addEventListener('click', reset);
loadEvidence();
