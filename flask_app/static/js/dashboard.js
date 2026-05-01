/*dashboard.js*/
/* MedSpec Classifier v6 */

// Theme
const html = document.documentElement;
const themeBtn = document.getElementById('themeToggle');
const applyTheme = (t) => {
  html.setAttribute('data-theme', t);
  localStorage.setItem('msai-theme', t);
  const sun = themeBtn?.querySelector('.icon-sun');
  const moon = themeBtn?.querySelector('.icon-moon');
  if (sun && moon) { sun.style.display=t==='dark'?'block':'none'; moon.style.display=t==='light'?'block':'none'; }
};
applyTheme(localStorage.getItem('msai-theme') || 'dark');
themeBtn?.addEventListener('click', () => applyTheme(html.getAttribute('data-theme')==='dark'?'light':'dark'));

// Tabs
const switchTab = (name) => {
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.toggle('active', b.dataset.tab===name));
  document.querySelectorAll('.tab-panel').forEach(p => p.classList.toggle('active', p.id===`tab-${name}`));
  window.scrollTo({top:0,behavior:'smooth'});
  if (name==='results' && !window._chartInit) { initChart(); window._chartInit=true; }
};
document.querySelectorAll('.tab-btn').forEach(b => b.addEventListener('click', () => switchTab(b.dataset.tab)));
window.switchTab = switchTab;

// Counters
const animateNum = (el) => {
  const target = parseFloat(el.dataset.target);
  const small = target < 10;
  const dur = 1200, t0 = performance.now();
  const tick = (now) => {
    const p = Math.min((now-t0)/dur, 1), e = 1-Math.pow(1-p, 3);
    el.textContent = small ? (target*e).toFixed(4) : target%1!==0 ? (target*e).toFixed(2) : Math.round(target*e);
    if (p < 1) requestAnimationFrame(tick);
    else el.textContent = small ? target.toFixed(4) : target%1!==0 ? target.toFixed(2) : target;
  };
  requestAnimationFrame(tick);
};
new IntersectionObserver((entries, obs) => {
  if (entries[0].isIntersecting) { document.querySelectorAll('.stat-val').forEach(animateNum); obs.disconnect(); }
}, {threshold:0.5}).observe(document.querySelector('.stat-strip') || document.body);

// Vertical bar chart
const initChart = () => {
  const ctx = document.getElementById('expChart');
  if (!ctx) return;
  const cs = getComputedStyle(document.documentElement);
  const dim    = cs.getPropertyValue('--dim').trim();
  const blue   = cs.getPropertyValue('--blue').trim();
  const accent = cs.getPropertyValue('--accent').trim();
  const muted  = cs.getPropertyValue('--muted').trim();
  const text   = cs.getPropertyValue('--text').trim();
  const card   = cs.getPropertyValue('--bg-card').trim();
  const border = cs.getPropertyValue('--border').trim();

  new Chart(ctx, {
    type: 'bar',
    data: {
      labels: ['Exp 1\nRaw·TTS', 'Exp 2\nRaw·GSS', 'Exp 3\nAug20·TTS', 'Exp 4\nAug20·GSS', 'Exp 5\nAug15·TTS', 'Exp 6\nAug15·GSS ★'],
      datasets: [{
        label: 'Accuracy (%)',
        data: [39.4, 44.1, 58.3, 61.8, 79.4, 83.04],
        backgroundColor: ['#64748b','#64748b', blue, blue, accent, accent],
        borderRadius: 8,
        borderSkipped: false,
        borderWidth: 0,
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: card,
          borderColor: border,
          borderWidth: 1,
          titleColor: text,
          bodyColor: muted,
          titleFont: { family: 'Syne', weight: 'bold' },
          callbacks: {
            title: (items) => {
              const labels = ['Raw · TTS', 'Raw · GSS', 'Aug20 · TTS', 'Aug20 · GSS', 'Aug15 · TTS', 'Aug15 · GSS'];
              return `Experiment ${items[0].dataIndex + 1} — ${labels[items[0].dataIndex]}`;
            },
            label: (item) => `  Accuracy: ${item.parsed.y}%`,
            afterLabel: (item) => {
              const f1s = ['0.427','0.475','0.572','0.624','0.801','0.8306'];
              return `  Macro F1: ${f1s[item.dataIndex]}`;
            }
          }
        }
      },
      scales: {
        x: {
          grid: { display: false },
          ticks: { color: muted, font: { family: 'Syne', size: 11 } }
        },
        y: {
          min: 0, max: 100,
          grid: { color: border + '60' },
          ticks: { color: muted, font: { family: 'JetBrains Mono', size: 11 }, callback: v => v + '%' }
        }
      },
      animation: { duration: 800, easing: 'easeOutQuart', delay: (c) => c.dataIndex * 80 }
    }
  });
};

// Demo
const MODEL_LABELS = { tfidf_lr:'TF-IDF + LR', tfidf_svm:'TF-IDF + SVM', cnn_lstm:'CNN-LSTM', bert:'BioM-BERT-Large' };
const $ = id => document.getElementById(id);

const setLoading = (on) => {
  const btn = $('sampleBtn'); if (!btn) return;
  btn.disabled = on;
  btn.querySelector('.btn-icon').style.display = on ? 'none' : 'flex';
  btn.querySelector('.btn-spin').style.display  = on ? 'flex'  : 'none';
  btn.querySelector('.btn-text').textContent     = on ? 'Running inference...' : 'Try Random Sample';
};

let fullExp = false;
$('toggleFull')?.addEventListener('click', () => {
  fullExp = !fullExp;
  $('transcriptPreview').style.display = fullExp ? 'none'  : 'block';
  $('transcriptFull').style.display    = fullExp ? 'block' : 'none';
  $('toggleFull').textContent = fullExp ? 'Show less ▲' : 'Show full ▼';
});

$('sampleBtn')?.addEventListener('click', async () => {
  setLoading(true); fullExp = false;
  try {
    const data = await fetch('/api/random-sample').then(r => r.json());
    if (!data.success) throw new Error(data.error || 'Inference failed');
    $('resultTruth').textContent       = data.ground_truth;
    $('wordCount').textContent         = `${data.word_count} words`;
    $('transcriptPreview').textContent = data.transcription_preview;
    $('transcriptFull').textContent    = data.transcription_full;
    $('transcriptPreview').style.display = 'block';
    $('transcriptFull').style.display    = 'none';
    if ($('toggleFull')) $('toggleFull').textContent = 'Show full ▼';

    const cont = $('predCards'); cont.innerHTML = '';
    let correct = 0;
    const preds = data.predictions, total = Object.keys(preds).length;
    Object.entries(preds).forEach(([key, val]) => {
      if (val.correct) correct++;
      const c = document.createElement('div');
      c.className = `pred-card ${val.correct ? 'correct' : 'wrong'}`;
      c.innerHTML = `<div class="pred-icon">${val.correct ? '✓' : '✗'}</div><div class="pred-info"><div class="pred-model">${MODEL_LABELS[key]||key}</div><div class="pred-value">${val.pred}</div></div>`;
      cont.appendChild(c);
    });

    const cls = correct===total ? 'score-all' : correct>0 ? 'score-some' : 'score-none';
    const lbl = correct===total ? 'All correct' : correct>0 ? `${correct}/${total} correct` : 'All incorrect';
    $('scoreBar').innerHTML = `<span class="score-num ${cls}">${correct}/${total}</span><span>${lbl}</span>`;
    $('demoEmpty').style.display  = 'none';
    $('demoResult').style.display = 'block';
  } catch(err) { alert(`Error: ${err.message}`); console.error(err); }
  finally { setLoading(false); }
});