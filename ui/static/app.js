/**
 * ARGUS Shield - Desktop Instrument Controller
 */

const STATE_COLORS = {
  TRUSTED: '#5fa878',
  FLAGGED_WITH_WARNING: '#d99a3d',
  ABSTAIN: '#c4554a'
};

let currentImageData = null;
let currentResult = null;
let isHeatmapVisible = false;

document.addEventListener('DOMContentLoaded', () => {
  initTabs();
  initDropZone();
  initSampleButtons();
  initToolbar();
  initSettings();
  loadSystemStatus();
  loadEvalSummary();
});

// --- Tab Switching ---
function initTabs() {
  const tabs = document.querySelectorAll('.tab-btn');
  tabs.forEach(tab => {
    tab.addEventListener('click', () => {
      tabs.forEach(t => t.classList.remove('active'));
      document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
      
      tab.classList.add('active');
      const target = document.getElementById(tab.dataset.tab);
      if (target) target.classList.add('active');

      if (tab.dataset.tab === 'evalTab') {
        loadEvalSummary();
      }
    });
  });
}

// --- Status & Config Loading ---
async function loadSystemStatus() {
  try {
    const res = await fetch('/shield/health');
    if (!res.ok) throw new Error(`Health HTTP ${res.status}`);
    const data = await res.json();
    
    const isMock = data.adapter_type && data.adapter_type.toLowerCase().includes('mock');
    const badge = document.getElementById('adapterBadge');
    const cfgDeployStatus = document.getElementById('cfgDeployStatus');
    const cfgAdapterType = document.getElementById('cfgAdapterType');

    if (isMock) {
      badge.className = 'badge badge-mock';
      badge.textContent = 'MOCK MODE: FASTER-RCNN';
      if (cfgDeployStatus) {
        cfgDeployStatus.className = 'kv-val badge badge-mock';
        cfgDeployStatus.textContent = 'MOCK MODE ACTIVE';
      }
    } else {
      badge.className = 'badge badge-live';
      badge.textContent = 'LIVE MODE: ' + (data.adapter_type || 'HTTP');
      if (cfgDeployStatus) {
        cfgDeployStatus.className = 'kv-val badge badge-live';
        cfgDeployStatus.textContent = 'LIVE ENDPOINT ACTIVE';
      }
    }

    if (cfgAdapterType && data.adapter_type) {
      cfgAdapterType.textContent = data.adapter_type;
    }

    // Populate thresholds in settings
    if (data.thresholds) {
      renderThresholds(data.thresholds);
    }
  } catch (err) {
    console.warn('Unable to load /shield/health:', err);
    document.getElementById('systemStatus').textContent = 'OFFLINE / LOCAL';
  }
}

function renderThresholds(thresholds) {
  const container = document.getElementById('thresholdsList');
  if (!container) return;
  container.innerHTML = '';
  for (const [k, v] of Object.entries(thresholds)) {
    const item = document.createElement('div');
    item.className = 'key-value-item';
    item.innerHTML = `<span class="kv-key">${escapeHtml(k)}</span><span class="kv-val">${escapeHtml(String(v))}</span>`;
    container.appendChild(item);
  }
}

// --- Drag & Drop / Upload ---
function initDropZone() {
  const dropZone = document.getElementById('dropZone');
  const fileInput = document.getElementById('fileInput');

  ['dragenter', 'dragover'].forEach(name => {
    dropZone.addEventListener(name, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropZone.classList.add('dragover');
    });
  });

  ['dragleave', 'drop'].forEach(name => {
    dropZone.addEventListener(name, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropZone.classList.remove('dragover');
    });
  });

  dropZone.addEventListener('drop', (e) => {
    const files = e.dataTransfer.files;
    if (files && files.length > 0) {
      processSelectedFile(files[0]);
    }
  });

  fileInput.addEventListener('change', (e) => {
    if (e.target.files && e.target.files.length > 0) {
      processSelectedFile(e.target.files[0]);
    }
  });
}

function initSampleButtons() {
  const sampleMap = {
    btnSampleClean: '/static/samples/car_sample.png',
    btnSamplePerson: '/static/samples/person_sample.png',
    btnSampleStop: '/static/samples/stop_sample.png'
  };

  for (const [btnId, path] of Object.entries(sampleMap)) {
    const btn = document.getElementById(btnId);
    if (!btn) continue;
    btn.addEventListener('click', async () => {
      try {
        setStatus('FETCHING SAMPLE...');
        const res = await fetch(path);
        if (!res.ok) throw new Error('Sample not found');
        const blob = await res.blob();
        processSelectedFile(blob, path.split('/').pop());
      } catch (e) {
        console.error('Error fetching sample:', e);
        setStatus('SAMPLE FETCH FAILED');
      }
    });
  }
}

function processSelectedFile(file, filename = 'image.png') {
  const reader = new FileReader();
  reader.onload = (e) => {
    currentImageData = e.target.result;
    submitToShield(file, filename);
  };
  reader.readAsDataURL(file);
}

// --- API Execution ---
async function submitToShield(file, filename) {
  setStatus('ANALYZING DEFENSE...');
  
  const apiKey = getApiKey();
  const formData = new FormData();
  formData.append('image', file, filename);

  try {
    const started = performance.now();
    const res = await fetch('/shield/detect', {
      method: 'POST',
      headers: {
        'X-Shield-Key': apiKey
      },
      body: formData
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || `Server responded with ${res.status}`);
    }

    const data = await res.json();
    currentResult = data;
    renderResult(currentImageData, currentResult);
    setStatus('READY');
  } catch (err) {
    console.error('Shield detect error:', err);
    alert('Shield detection error: ' + err.message);
    setStatus('ERROR');
  }
}

// --- Rendering Result View ---
function renderResult(imgDataUrl, result) {
  document.getElementById('dropZoneContainer').classList.add('hidden');
  document.getElementById('resultViewer').classList.remove('hidden');

  // Load image into main canvas
  const img = new Image();
  img.onload = () => {
    const mainCanvas = document.getElementById('mainCanvas');
    const heatmapCanvas = document.getElementById('heatmapCanvas');
    
    mainCanvas.width = img.naturalWidth;
    mainCanvas.height = img.naturalHeight;
    heatmapCanvas.width = img.naturalWidth;
    heatmapCanvas.height = img.naturalHeight;

    document.getElementById('imageDimensions').textContent = `${img.naturalWidth} × ${img.naturalHeight} PX`;

    const ctx = mainCanvas.getContext('2d');
    ctx.drawImage(img, 0, 0);

    // Render Bounding Boxes
    drawDetections(ctx, result.detections || []);

    // Render Heatmap on secondary canvas
    if (result.anomaly_heatmap_base64) {
      const hmImg = new Image();
      hmImg.onload = () => {
        const hmCtx = heatmapCanvas.getContext('2d');
        hmCtx.clearRect(0, 0, heatmapCanvas.width, heatmapCanvas.height);
        hmCtx.drawImage(hmImg, 0, 0, heatmapCanvas.width, heatmapCanvas.height);
      };
      hmImg.src = result.anomaly_heatmap_base64.startsWith('data:') 
        ? result.anomaly_heatmap_base64 
        : 'data:image/png;base64,' + result.anomaly_heatmap_base64;
    }

    updateTelemetry(result);
  };
  img.src = imgDataUrl;
}

function drawDetections(ctx, detections) {
  ctx.save();
  
  detections.forEach(det => {
    const [x1, y1, x2, y2] = det.box;
    const w = x2 - x1;
    const h = y2 - y1;
    const decision = det.decision || 'TRUSTED';
    const color = STATE_COLORS[decision] || STATE_COLORS.TRUSTED;

    // Hairline box
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.strokeRect(x1, y1, w, h);

    // Header label tag
    const labelText = `${det.label.toUpperCase()} ${(det.score * 100).toFixed(0)}% [${decision}]`;
    ctx.font = '500 11px -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, monospace';
    const textMetrics = ctx.measureText(labelText);
    const tagHeight = 18;
    const tagWidth = textMetrics.width + 10;

    ctx.fillStyle = 'rgba(13, 13, 15, 0.85)';
    ctx.fillRect(x1, Math.max(0, y1 - tagHeight), tagWidth, tagHeight);

    ctx.fillStyle = color;
    ctx.fillText(labelText, x1 + 5, Math.max(12, y1 - 4));
  });

  ctx.restore();
}

function updateTelemetry(result) {
  // Determine dominant decision
  const overallEl = document.getElementById('overallDecision');
  const decisions = (result.detections || []).map(d => d.decision);
  let overall = 'TRUSTED';
  if (decisions.includes('ABSTAIN')) {
    overall = 'ABSTAIN';
  } else if (decisions.includes('FLAGGED_WITH_WARNING')) {
    overall = 'FLAGGED_WITH_WARNING';
  } else if (decisions.length === 0) {
    overall = 'NO_DETECTIONS';
  }

  overallEl.textContent = overall;
  overallEl.className = 'decision-value ' + (
    overall === 'TRUSTED' ? 'color-trusted' :
    overall === 'ABSTAIN' ? 'color-abstain' : 'color-flagged'
  );

  document.getElementById('valAnomalyScore').textContent = Number(result.anomaly_score || 0).toFixed(3);
  document.getElementById('valAgreementScore').textContent = Number(result.agreement_score || 0).toFixed(3);
  
  const latencyVal = Number(result.total_latency_ms || 0);
  const latencyEl = document.getElementById('valLatency');
  if (latencyVal > 250) {
    latencyEl.textContent = `${latencyVal.toFixed(1)} ms`;
    latencyEl.title = `Latency: ${latencyVal.toFixed(1)} ms (Contract target budget: 250 ms)`;
    latencyEl.style.color = '#d99a3d';
  } else {
    latencyEl.textContent = `${latencyVal.toFixed(1)} ms`;
    latencyEl.title = 'Within contract latency budget (<= 250 ms)';
    latencyEl.style.color = '#5fa878';
  }
  document.getElementById('valDetectionsCount').textContent = (result.detections || []).length;

  // Detections side list
  const list = document.getElementById('detectionsList');
  list.innerHTML = '';

  if (!result.detections || result.detections.length === 0) {
    list.innerHTML = '<div style="color: var(--text-dim); font-size: 11px; padding: 12px 0;">No detections found in image.</div>';
    return;
  }

  result.detections.forEach((det, idx) => {
    const decision = det.decision || 'TRUSTED';
    const stateClass = decision === 'TRUSTED' ? 'state-trusted' :
      decision === 'ABSTAIN' ? 'state-abstain' : 'state-flagged';
    
    const row = document.createElement('div');
    row.className = `detection-row ${stateClass}`;
    row.innerHTML = `
      <div class="detection-top">
        <span class="detection-name">#${idx + 1} ${escapeHtml(det.label.toUpperCase())}</span>
        <span class="badge ${decision === 'TRUSTED' ? 'badge-trusted' : decision === 'ABSTAIN' ? 'badge-abstain' : 'badge-flagged'}">
          ${escapeHtml(decision)}
        </span>
      </div>
      <div class="detection-meta">
        <span>Conf: ${(det.score * 100).toFixed(1)}%</span>
        <span>Agreement: ${Number(det.agreement_score || result.agreement_score || 0).toFixed(2)}</span>
      </div>
      <div class="detection-meta" style="color: var(--text-dim); font-size: 10px;">
        Box: [${det.box.map(n => Math.round(n)).join(', ')}]
      </div>
    `;
    list.appendChild(row);
  });
}

function initToolbar() {
  const btnReset = document.getElementById('btnResetImage');
  const btnToggle = document.getElementById('btnToggleHeatmap');
  const heatmapCanvas = document.getElementById('heatmapCanvas');

  btnReset.addEventListener('click', () => {
    document.getElementById('resultViewer').classList.add('hidden');
    document.getElementById('dropZoneContainer').classList.remove('hidden');
    currentImageData = null;
    currentResult = null;
    isHeatmapVisible = false;
    heatmapCanvas.style.display = 'none';
    btnToggle.textContent = 'SHOW ANOMALY HEATMAP';
    document.getElementById('fileInput').value = '';
  });

  btnToggle.addEventListener('click', () => {
    isHeatmapVisible = !isHeatmapVisible;
    heatmapCanvas.style.display = isHeatmapVisible ? 'block' : 'none';
    btnToggle.textContent = isHeatmapVisible ? 'HIDE ANOMALY HEATMAP' : 'SHOW ANOMALY HEATMAP';
  });
}

// --- Batch / Eval Summary ---
async function loadEvalSummary() {
  const tbody = document.getElementById('evalTableBody');
  try {
    const res = await fetch('/shield/eval-summary');
    if (!res.ok) throw new Error('No evaluation report available yet');
    const data = await res.json();
    renderEvalTable(data);
    if (data.latency_series) {
      renderLatencyChart(data.latency_series);
    }
  } catch (err) {
    tbody.innerHTML = `
      <tr>
        <td colspan="4" style="text-align: center; color: var(--text-dim); padding: 24px;">
          No evaluation runs found. Click "RUN HARNESS ON MANIFEST" to execute full raw-vs-shield validation.
        </td>
      </tr>
    `;
  }
}

function renderEvalTable(data) {
  const tbody = document.getElementById('evalTableBody');
  tbody.innerHTML = '';
  const rows = data.rows || [];
  if (rows.length === 0) {
    tbody.innerHTML = '<tr><td colspan="4" style="text-align:center; padding: 20px;">No records</td></tr>';
    return;
  }

  rows.forEach(r => {
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td style="font-weight: 500;">${escapeHtml(r.family_or_metric)}</td>
      <td>${escapeHtml(r.raw_val)}</td>
      <td style="color: var(--state-trusted); font-weight: 600;">${escapeHtml(r.shield_val)}</td>
      <td style="color: var(--text-secondary);">${escapeHtml(r.context || '-')}</td>
    `;
    tbody.appendChild(tr);
  });
}

function renderLatencyChart(series) {
  const canvas = document.getElementById('latencyChart');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  
  canvas.width = rect.width * dpr;
  canvas.height = rect.height * dpr;
  ctx.scale(dpr, dpr);

  const w = rect.width;
  const h = rect.height;
  const padding = 20;

  ctx.clearRect(0, 0, w, h);

  if (!series || series.length === 0) return;

  const maxVal = Math.max(...series, 250);
  const stepX = (w - padding * 2) / Math.max(1, series.length - 1);

  // Budget threshold line (250ms)
  const budgetY = h - padding - (250 / maxVal) * (h - padding * 2);
  ctx.strokeStyle = 'rgba(196, 85, 74, 0.4)';
  ctx.setLineDash([4, 4]);
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(padding, budgetY);
  ctx.lineTo(w - padding, budgetY);
  ctx.stroke();
  ctx.setLineDash([]);

  ctx.fillStyle = 'rgba(196, 85, 74, 0.8)';
  ctx.font = '10px monospace';
  ctx.fillText('BUDGET 250 MS', w - 100, budgetY - 4);

  // Plot line
  ctx.strokeStyle = '#5fa878';
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  series.forEach((val, i) => {
    const x = padding + i * stepX;
    const y = h - padding - (val / maxVal) * (h - padding * 2);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();

  // Points
  ctx.fillStyle = '#5fa878';
  series.forEach((val, i) => {
    const x = padding + i * stepX;
    const y = h - padding - (val / maxVal) * (h - padding * 2);
    ctx.fillRect(x - 2, y - 2, 4, 4);
  });
}

document.getElementById('btnRunEval')?.addEventListener('click', async () => {
  const btn = document.getElementById('btnRunEval');
  btn.disabled = true;
  btn.textContent = 'RUNNING HARNESS...';
  setStatus('RUNNING EVAL HARNESS...');

  try {
    const res = await fetch('/shield/run-eval', {
      method: 'POST',
      headers: { 'X-Shield-Key': getApiKey() }
    });
    if (!res.ok) throw new Error('Evaluation failed to run');
    await loadEvalSummary();
    setStatus('EVAL COMPLETE');
  } catch (err) {
    alert('Eval execution: ' + err.message);
    setStatus('READY');
  } finally {
    btn.disabled = false;
    btn.textContent = 'RUN HARNESS ON MANIFEST';
  }
});

// --- Settings & Auth ---
function initSettings() {
  const input = document.getElementById('inputApiKey');
  const stored = localStorage.getItem('argus_shield_api_key');
  if (stored && input) {
    input.value = stored;
  }
  input?.addEventListener('change', () => {
    localStorage.setItem('argus_shield_api_key', input.value.trim());
  });
}

function getApiKey() {
  const input = document.getElementById('inputApiKey');
  return (input ? input.value.trim() : null) || localStorage.getItem('argus_shield_api_key') || 'argus-shield-local';
}

function setStatus(msg) {
  const el = document.getElementById('systemStatus');
  if (el) el.textContent = msg;
}

function escapeHtml(str) {
  if (str == null) return '';
  return String(str).replace(/[&<>"']/g, m => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
  })[m]);
}
