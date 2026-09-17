const $ = id => document.getElementById(id);
let finished = false;

function setStages(active) {
  [...document.querySelectorAll('#stages span')].forEach((el, i) => el.classList.toggle('active', i <= active));
}

function reset() {
  finished = false;
  $('h1v').textContent = $('h2v').textContent = '50%';
  $('h1bar').style.width = $('h2bar').style.width = '50%';
  $('beliefNote').textContent = 'Equal priors. The runtime has not observed a discriminating outcome.';
  $('trace').innerHTML = '<li class="current">World model initialized</li><li>Waiting for an observation</li><li>Posterior not updated</li>';
  document.querySelectorAll('.action').forEach(b => b.disabled = false);
  setStages(0);
}

function choose(kind) {
  if (finished) return;
  setStages(1);
  document.querySelectorAll('.action').forEach(b => b.disabled = true);
  if (kind === 'temperature') {
    $('trace').innerHTML = '<li class="done">World model initialized</li><li class="done">Temperature observed: 25°C</li><li class="current">Outcome has equal likelihood under H1 and H2</li>';
    $('beliefNote').textContent = 'No update: room temperature does not distinguish filter restriction from fan weakness.';
    setStages(2);
    setTimeout(() => { document.querySelectorAll('.action').forEach(b => b.disabled = false); }, 700);
    return;
  }
  finished = true;
  setTimeout(() => {
    $('h1v').textContent = '10%'; $('h2v').textContent = '90%';
    $('h1bar').style.width = '10%'; $('h2bar').style.width = '90%';
    $('beliefNote').textContent = 'Observed normal pressure. Bayes update: P(H2 | normal) = 0.90.';
    $('trace').innerHTML = '<li class="done">World model initialized</li><li class="done">Pressure test selected by runtime EIG</li><li class="done">Observed: normal · 12 Pa</li><li class="current">Stop: weak fan is the leading mechanism</li>';
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

async function loadEvidence() {
  try {
    const data = await fetch('data/runtime.json').then(r => { if (!r.ok) throw new Error(r.status); return r.json(); });
    const suite = parseFirstJson(data.hidden_world_suite);
    $('build').textContent = `VERIFIED · ${data.commit} · PYTHON ${data.python}`;
    $('accuracy').textContent = `${Math.round(suite.identification_accuracy * 100)}%`;
    $('posterior').textContent = suite.mean_true_hypothesis_posterior.toFixed(3);
    $('regret').textContent = suite.mean_causal_regret.toFixed(3);
    $('probes').textContent = suite.mean_probes.toFixed(1);
    $('raw').textContent = data.bayesian_experiment + '\n\n' + data.hidden_world_suite;
  } catch (error) {
    $('build').textContent = 'BUILD EVIDENCE UNAVAILABLE';
    $('raw').textContent = `Could not load generated runtime evidence: ${error}`;
  }
}

document.querySelectorAll('.action').forEach(b => b.addEventListener('click', () => choose(b.dataset.action)));
$('reset').addEventListener('click', reset);
$('start').addEventListener('click', () => { document.querySelector('.lab').scrollIntoView({behavior:'smooth'}); setTimeout(() => choose('pressure'), 650); });
loadEvidence();
