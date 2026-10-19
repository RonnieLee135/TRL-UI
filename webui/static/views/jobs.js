// View: 任务 — a list on the left, and on the right the exact command, the generated
// artifacts, a live log stream (SSE) and the metric curves parsed out of it.
//
// The detail pane is built once per selection and then *updated in place*: the terminal
// must not be rebuilt while a job is streaming.

import { $, h, render, toast, copyText, fmtTime, fmtBytes, STATUS_TEXT } from '../dom.js';
import { state, api, refreshJobs } from '../app.js';

const SERIES_COLORS = ['#1a73e8', '#16a34a', '#d97706', '#dc2626', '#7c3aed', '#0891b2', '#4b5563'];
const SERIES_ORDER = ['loss', 'reward', 'rewards/margins', 'accuracy', 'entropy', 'kl'];
const MAX_LOG_LINES = 4000;

let refs = {};
let stream = null;
let autoscroll = true;
let modalText = '';
let lastStatus = null;
let resizeBound = false;

/* --------------------------------------------------------------- shell */
export function renderJobs() {
  return h(
    'div',
    { class: 'jobs-layout' },
    h(
      'div',
      { class: 'job-list-wrap' },
      h(
        'div',
        { class: 'panel-head' },
        h('h2', { class: 'section-title', text: '任务' }),
        h('button', { class: 'btn sm', id: 'refresh-jobs', type: 'button', text: '刷新' }),
      ),
      h('div', { class: 'job-list', id: 'job-list', 'aria-live': 'polite' }),
    ),
    h('div', { class: 'job-detail', id: 'job-detail' }),
  );
}

/* ---------------------------------------------------------------- list */
function renderJobList() {
  const box = $('#job-list');
  if (!box) return;
  if (!state.jobs.length) {
    render(
      box,
      h(
        'div',
        { class: 'empty-state' },
        h('p', { class: 'eyebrow', text: '暂无任务' }),
        h('h2', { text: '还没有训练任务' }),
        h('p', { class: 'muted', text: '去「新建训练」挑一个算法，填几步就能生成命令并启动。' }),
        h('button', { class: 'btn primary sm', type: 'button', onclick: () => { location.hash = '#/new'; }, text: '新建训练' }),
      ),
    );
    return;
  }
  render(box, state.jobs.map(jobItem));
}

function jobItem(job) {
  const algo = state.catalog?.algorithms.find((a) => a.id === job.algorithm);
  return h(
    'button',
    {
      class: 'job-item',
      type: 'button',
      'aria-current': String(state.selectedJob === job.id),
      onclick: () => void openJob(job.id),
    },
    h('h3', { text: job.name }),
    h(
      'div',
      { class: 'meta' },
      h('span', { class: `status ${job.status}`, text: STATUS_TEXT[job.status] || job.status }),
      h('span', { text: algo?.label || job.algorithm }),
      h('span', { text: job.mode === 'simulate' ? '模拟' : '真实' }),
      h('span', { text: fmtTime(job.created_at) }),
    ),
  );
}

/* -------------------------------------------------------------- detail */
export async function openJob(jobId) {
  state.selectedJob = jobId;
  state.events = [];
  state.lastEventId = 0;
  state.metrics = {};
  state.hiddenSeries = new Set();
  lastStatus = null;
  closeStream();
  renderJobList();
  renderDetailSkeleton();

  let detail;
  try {
    detail = await api(`jobs/${jobId}`);
  } catch (err) {
    const box = $('#job-detail');
    if (box) render(box, h('p', { class: 'item error', text: `加载任务失败：${err.message}` }));
    return;
  }
  if (state.selectedJob !== jobId) return;
  state.jobDetail = detail;

  const [metrics, logs] = await Promise.all([
    api(`jobs/${jobId}/metrics`).catch(() => ({ metrics: {} })),
    api(`jobs/${jobId}/logs?since=0`).catch(() => ({ events: [], last_id: 0 })),
  ]);
  if (state.selectedJob !== jobId) return;
  state.metrics = metrics.metrics || {};
  state.events = logs.events || [];
  state.lastEventId = logs.last_id || 0;

  renderJobDetail();
  if (detail.running || detail.job.status === 'running') connectStream(jobId);
}

function renderDetailSkeleton() {
  const box = $('#job-detail');
  if (!box) return;
  render(
    box,
    h(
      'div',
      { class: 'panel' },
      h('div', { class: 'skeleton', style: { height: '24px', width: '38%' } }),
      h('div', { class: 'skeleton', style: { marginTop: '14px' } }),
      h('div', { class: 'skeleton', style: { marginTop: '8px', width: '72%' } }),
    ),
  );
}

function renderEmptyDetail() {
  const box = $('#job-detail');
  if (!box) return;
  render(
    box,
    h(
      'div',
      { class: 'empty-state' },
      h('p', { class: 'eyebrow', text: '任务' }),
      h('h2', { text: '还没有选中任务' }),
      h('p', { class: 'muted', text: '左边选一个任务查看命令、日志与曲线；或去「新建训练」创建一个。' }),
    ),
  );
}

function renderJobDetail() {
  const box = $('#job-detail');
  const detail = state.jobDetail;
  if (!box || !detail) return;
  const job = detail.job;
  const algo = detail.algorithm || state.catalog?.algorithms.find((a) => a.id === job.algorithm);

  refs = {};
  refs.status = h('span', { class: `status ${job.status}`, text: STATUS_TEXT[job.status] || job.status });
  refs.mode = h('span', { class: 'chip', text: job.mode === 'simulate' ? '模拟模式' : '真实训练' });
  refs.duration = h('span', { class: 'mono small', text: durationText(job) });
  refs.message = h('div', { class: 'note', text: job.message || '' });
  refs.terminal = h('div', {
    class: 'terminal',
    id: 'log-terminal',
    tabindex: '0',
    role: 'log',
    'aria-label': '训练日志',
  });
  refs.legend = h('div', { class: 'chart-legend', id: 'chart-legend' });
  refs.artifacts = h('div', { id: 'artifact-list' });

  const canvas = h('canvas', { id: 'chart', role: 'img', 'aria-label': '训练指标曲线' });
  const canvasWrap = h('div', { class: 'chart-canvas-wrap' }, canvas, h('div', { class: 'chart-empty', text: '' }));
  refs.chart = canvas;
  refs.chartEmpty = canvasWrap.querySelector('.chart-empty');

  refs.startBtn = h('button', { class: 'btn primary sm', type: 'button', onclick: () => void startJob(job.id, refs.startBtn), text: '启动' });
  refs.stopBtn = h('button', { class: 'btn sm', type: 'button', onclick: () => void stopJob(job.id), text: '停止' });
  refs.deleteBtn = h('button', { class: 'btn sm danger', type: 'button', onclick: () => void deleteJob(job.id, job.name), text: '删除' });

  render(
    box,
    h(
      'div',
      { class: 'detail-head' },
      h(
        'div',
        {},
        h('p', { class: 'eyebrow', text: algo ? `${algo.label} · ${algo.status === 'stable' ? '稳定' : '实验'}` : job.algorithm }),
        h('h2', { class: 'detail-title', text: job.name }),
        h('p', { class: 'small muted mono', text: job.id }),
      ),
      h('div', { class: 'card-actions' }, refs.status, refs.mode, refs.startBtn, refs.stopBtn, refs.deleteBtn),
    ),
    h(
      'div',
      { class: 'detail-meta' },
      metaCell('算法', algo ? `${algo.label} · ${algo.config_class}` : job.algorithm),
      metaCell('训练器', algo?.trainer_class || '—'),
      metaCell('运行时长', refs.duration, true),
      metaCell('创建时间', fmtTime(job.created_at)),
      metaCell('开始 / 结束', `${fmtTime(job.started_at)} → ${fmtTime(job.finished_at)}`),
      metaCell('退出码', job.exit_code === null || job.exit_code === undefined ? '—' : String(job.exit_code)),
      metaCell('输出目录', job.output_dir || '—'),
      metaCell('启动命令', job.launch === 'cli' || algo?.launch?.kind === 'cli' ? 'trl CLI' : 'Python 脚本'),
    ),
    refs.message,
    sectionHead('命令', h('button', { class: 'btn sm', type: 'button', onclick: () => copyText(detail.command, '命令已复制'), text: '复制' })),
    h('pre', { class: 'cmd', text: detail.command }),
    sectionHead(
      `产物（${detail.files.length}）`,
      h('p', { class: 'small muted', text: detail.files.length ? '「查看」在弹窗里预览，或直接下载。' : '还没有产物文件。' }),
    ),
    refs.artifacts,
    sectionHead(
      '指标',
      h('p', { class: 'small muted', text: '点图例显示/隐藏曲线；多条曲线时各自按自身范围缩放（图例给出实际范围）。' }),
    ),
    h('div', { class: 'chart-wrap' }, refs.legend, canvasWrap),
    sectionHead('日志', logTools()),
    refs.terminal,
    h('p', {
      class: 'small muted',
      text: '日志与曲线来自任务进程的真实 stdout；模拟模式的行由内置模拟器产生，不代表任何训练结果。',
    }),
  );

  renderArtifacts();
  for (const event of state.events) appendLog(event);
  renderLegend();
  drawChart();
  updateButtons();
}

function sectionHead(title, tools) {
  return h(
    'div',
    { class: 'panel-head' },
    h('h3', { class: 'section-title', style: { fontSize: '15px' }, text: title }),
    tools,
  );
}

function metaCell(label, value, isNode = false) {
  return h(
    'div',
    {},
    h('span', { text: label }),
    isNode ? h('b', {}, value) : h('b', { text: String(value) }),
  );
}

function renderArtifacts() {
  if (!refs.artifacts || !state.jobDetail) return;
  const files = state.jobDetail.files || [];
  if (!files.length) {
    render(refs.artifacts, h('p', { class: 'small muted', text: '任务创建后会在这里列出 command.sh、params.json、README.md 等文件。' }));
    return;
  }
  render(
    refs.artifacts,
    files.map((file) =>
      h(
        'div',
        { class: 'artifact' },
        h('div', { class: 'artifact-name' }, h('code', { text: file.name }), h('span', { class: 'artifact-size', text: fmtBytes(file.size) })),
        h(
          'div',
          { class: 'card-actions' },
          h('button', { class: 'btn sm', type: 'button', onclick: () => void previewFile(file.name), text: '查看' }),
          h('a', { class: 'btn sm', href: `/api/jobs/${state.selectedJob}/files/${encodeURIComponent(file.name)}`, download: file.name, text: '下载' }),
        ),
      ),
    ),
  );
}

/* ------------------------------------------------------------- actions */
async function startJob(jobId, button) {
  const original = button?.textContent || '启动';
  if (button) { button.disabled = true; button.textContent = '启动中…'; }
  try {
    const result = await api(`jobs/${jobId}/start`, { method: 'POST' });
    if (!result.ok) toast(result.message || '启动失败', 'error');
    else toast('已开始运行');
  } catch (err) {
    toast(`启动失败：${err.message}`, 'error');
  } finally {
    if (button) button.textContent = original;
  }
  if (state.selectedJob === jobId) await openJob(jobId);
  void refreshJobs({ silent: true });
}

async function stopJob(jobId) {
  try {
    const result = await api(`jobs/${jobId}/stop`, { method: 'POST' });
    toast(result.message || (result.ok ? '已停止' : '任务没有在运行'), result.ok ? 'ok' : 'error');
  } catch (err) {
    toast(`停止失败：${err.message}`, 'error');
  }
  if (state.selectedJob === jobId) await openJob(jobId);
  void refreshJobs({ silent: true });
}

async function deleteJob(jobId, name) {
  if (!window.confirm(`删除任务记录「${name}」？\n只会删除控制台里的这条记录，runs/${jobId}/ 里的产物文件保持不动。`)) return;
  try {
    const result = await api(`jobs/${jobId}/delete`, { method: 'POST' });
    if (!result.ok) {
      toast(result.message || '删除失败', 'error');
      return;
    }
    toast('已删除任务记录');
  } catch (err) {
    toast(`删除失败：${err.message}`, 'error');
    return;
  }
  closeStream();
  state.selectedJob = null;
  state.jobDetail = null;
  renderEmptyDetail();
  void refreshJobs({ silent: true });
}

/* --------------------------------------------------------------- modal */
async function previewFile(name) {
  try {
    const response = await fetch(`/api/jobs/${state.selectedJob}/files/${encodeURIComponent(name)}`);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const text = await response.text();
    modalText = text;
    $('#modal-title').textContent = name;
    $('#modal-body').textContent = text;
    $('#modal-backdrop').hidden = false;
    $('#modal-close').focus();
  } catch (err) {
    toast(`读取文件失败：${err.message}`, 'error');
  }
}

/* ----------------------------------------------------------------- log */
function logTools() {
  return h(
    'div',
    { class: 'log-tools' },
    h(
      'label',
      { class: 'switch sm' },
      h('input', { type: 'checkbox', checked: autoscroll, onchange: (event) => { autoscroll = event.target.checked; } }),
      h('span', { class: 'small muted', text: '自动滚动' }),
    ),
    h('button', { class: 'btn sm', type: 'button', onclick: () => { if (refs.terminal) refs.terminal.textContent = ''; }, text: '清屏' }),
    h('button', { class: 'btn sm', type: 'button', onclick: () => { if (refs.terminal) refs.terminal.scrollTop = 0; }, text: '回到顶部' }),
  );
}

function streamClass(stream) {
  if (stream === 'system') return 'sys';
  if (stream === 'stderr') return 'err';
  return '';
}

function appendLog(event) {
  const node = refs.terminal;
  if (!node) return;
  node.append(h('div', { class: streamClass(event.stream), text: event.line }));
  while (node.childElementCount > MAX_LOG_LINES) node.firstElementChild.remove();
  if (autoscroll) node.scrollTop = node.scrollHeight;
}

function connectStream(jobId) {
  closeStream();
  const source = new EventSource(`/api/jobs/${jobId}/events?since=${state.lastEventId}`);
  stream = source;
  source.addEventListener('log', (event) => {
    let data;
    try { data = JSON.parse(event.data); } catch { return; }
    if (state.selectedJob !== jobId) return;
    state.events.push(data);
    state.lastEventId = Math.max(state.lastEventId, data.id || 0);
    appendLog(data);
  });
  source.addEventListener('status', (event) => {
    let data;
    try { data = JSON.parse(event.data); } catch { return; }
    if (state.selectedJob !== jobId) return;
    applyStatus(data);
  });
  source.addEventListener('done', () => {
    closeStream();
    void refreshJobs({ silent: true });
  });
  source.addEventListener('gone', closeStream);
}

export function closeJobStream() {
  closeStream();
}

function closeStream() {
  if (stream) {
    stream.close();
    stream = null;
  }
}

/* -------------------------------------------------------------- status */
function applyStatus(data) {
  const detail = state.jobDetail;
  if (!detail) return;
  detail.running = Boolean(data.running);
  detail.job.status = data.status;
  detail.job.message = data.message;
  detail.job.exit_code = data.exit_code;
  detail.job.finished_at = data.finished_at;
  state.metrics = data.metrics || {};
  updateButtons();
  renderLegend();
  drawChart();
  if (lastStatus !== data.status) {
    lastStatus = data.status;
    const item = state.jobs.find((job) => job.id === detail.job.id);
    if (item) item.status = data.status;
    renderJobList();
  }
}

function updateButtons() {
  const detail = state.jobDetail;
  if (!detail || !refs.startBtn) return;
  const job = detail.job;
  const running = Boolean(detail.running) || job.status === 'running';
  refs.startBtn.disabled = running;
  refs.stopBtn.disabled = !running;
  refs.deleteBtn.disabled = running || !['created', 'finished', 'failed', 'stopped'].includes(job.status);
  refs.status.className = `status ${job.status}`;
  refs.status.textContent = STATUS_TEXT[job.status] || job.status;
  refs.duration.textContent = durationText(job);
  refs.message.hidden = !job.message;
  refs.message.textContent = job.message || '';
}

function durationText(job) {
  if (!job.started_at) return '—';
  const seconds = Math.max(0, (job.finished_at || Date.now() / 1000) - job.started_at);
  const minutes = Math.floor(seconds / 60);
  const body = minutes ? `${minutes} 分 ${String(Math.round(seconds % 60)).padStart(2, '0')} 秒` : `${seconds.toFixed(1)} 秒`;
  return job.finished_at ? body : `运行中 · ${body}`;
}

/* --------------------------------------------------------------- chart */
const seriesNames = () =>
  Object.keys(state.metrics || {})
    .filter((name) => (state.metrics[name] || []).length)
    .sort((a, b) => {
      const ia = SERIES_ORDER.indexOf(a);
      const ib = SERIES_ORDER.indexOf(b);
      return (ia === -1 ? 99 : ia) - (ib === -1 ? 99 : ib) || a.localeCompare(b);
    });

const colorFor = (name) => SERIES_COLORS[seriesNames().indexOf(name) % SERIES_COLORS.length] || SERIES_COLORS[0];

function fmtNum(value) {
  if (!Number.isFinite(value)) return '—';
  const abs = Math.abs(value);
  if (abs === 0) return '0';
  if (abs >= 1000) return value.toFixed(0);
  if (abs >= 10) return value.toFixed(2);
  if (abs >= 1) return value.toFixed(3);
  return value.toPrecision(3);
}

function renderLegend() {
  if (!refs.legend) return;
  const names = seriesNames();
  if (!names.length) {
    render(refs.legend, h('span', { class: 'small muted', text: '还没有指标数据。' }));
    return;
  }
  render(
    refs.legend,
    names.map((name) => {
      const points = state.metrics[name] || [];
      const values = points.map((p) => p.value);
      const hidden = state.hiddenSeries.has(name);
      return h(
        'button',
        {
          class: 'legend-item',
          type: 'button',
          'aria-pressed': String(!hidden),
          title: `${name}：${fmtNum(Math.min(...values))} → ${fmtNum(Math.max(...values))}（${points.length} 个点）`,
          onclick: () => {
            if (hidden) state.hiddenSeries.delete(name);
            else state.hiddenSeries.add(name);
            renderLegend();
            drawChart();
          },
        },
        h('span', { class: 'dot', style: { background: colorFor(name) } }),
        h('span', { text: name }),
        h('span', { class: 'range', text: `${fmtNum(Math.min(...values))}→${fmtNum(Math.max(...values))}` }),
      );
    }),
  );
}

function drawChart() {
  const canvas = refs.chart;
  if (!canvas || !document.body.contains(canvas)) return;
  const names = seriesNames().filter((name) => !state.hiddenSeries.has(name));
  const width = Math.max(280, canvas.parentElement?.clientWidth || 280);
  const height = 220;
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.round(width * dpr);
  canvas.height = Math.round(height * dpr);
  canvas.style.height = `${height}px`;
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, width, height);

  const styles = getComputedStyle(document.documentElement);
  const grid = styles.getPropertyValue('--border').trim() || '#e5e7eb';
  const faint = styles.getPropertyValue('--fg-faint').trim() || '#6b7280';
  const padL = 52;
  const padR = 12;
  const padT = 12;
  const padB = 26;
  const plotW = width - padL - padR;
  const plotH = height - padT - padB;

  ctx.font = '11px ui-monospace, SFMono-Regular, Menlo, monospace';
  ctx.lineWidth = 1;

  if (!names.length) {
    if (refs.chartEmpty) refs.chartEmpty.textContent = seriesNames().length ? '全部曲线已隐藏，点图例重新显示。' : '还没有指标数据。';
    return;
  }
  if (refs.chartEmpty) refs.chartEmpty.textContent = '';

  const ranges = new Map();
  for (const name of names) {
    const values = (state.metrics[name] || []).map((p) => p.value);
    let lo = Math.min(...values);
    let hi = Math.max(...values);
    if (!Number.isFinite(lo) || !Number.isFinite(hi)) { lo = 0; hi = 1; }
    if (hi === lo) { hi = lo + (Math.abs(lo) || 1) * 0.5; lo -= (Math.abs(lo) || 1) * 0.5; }
    ranges.set(name, [lo, hi]);
  }
  const gridlines = 4;
  const yFor = (name, value) => {
    const [lo, hi] = ranges.get(name);
    return padT + (1 - (value - lo) / (hi - lo)) * plotH;
  };

  // grid + y labels
  ctx.strokeStyle = grid;
  ctx.fillStyle = faint;
  for (let i = 0; i <= gridlines; i += 1) {
    const y = padT + (plotH * i) / gridlines;
    ctx.beginPath();
    ctx.moveTo(padL, y + 0.5);
    ctx.lineTo(padL + plotW, y + 0.5);
    ctx.stroke();
    const label = names.length === 1
      ? fmtNum(ranges.get(names[0])[1] - ((ranges.get(names[0])[1] - ranges.get(names[0])[0]) * i) / gridlines)
      : `${100 - (100 / gridlines) * i}%`;
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    ctx.fillText(label, padL - 8, y);
  }

  // x axis labels
  const first = state.metrics[names[0]] || [];
  const stepMin = first[0]?.step ?? 0;
  const stepMax = first[first.length - 1]?.step ?? stepMin;
  const xFor = (step) => padL + ((step - stepMin) / Math.max(1, stepMax - stepMin)) * plotW;
  ctx.textBaseline = 'top';
  ctx.textAlign = 'left';
  ctx.fillText(String(stepMin), padL, padT + plotH + 6);
  ctx.textAlign = 'right';
  ctx.fillText(String(stepMax), padL + plotW, padT + plotH + 6);

  for (const name of names) {
    const points = state.metrics[name] || [];
    if (!points.length) continue;
    ctx.strokeStyle = colorFor(name);
    ctx.lineWidth = 1.6;
    ctx.setLineDash([]);
    ctx.beginPath();
    points.forEach((point, index) => {
      const x = xFor(point.step);
      const y = yFor(name, point.value);
      if (index === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
  }
}

/* -------------------------------------------------------------- refresh */
/** Called by the app shell after the job list is refetched (state.jobs is fresh). */
export function syncJobsView() {
  renderJobList();
  const exists = state.selectedJob && state.jobs.some((job) => job.id === state.selectedJob);
  if (exists) {
    void refreshDetailHead();
    return;
  }
  if (!state.selectedJob && state.jobs.length) {
    void openJob(state.jobs[0].id);
    return;
  }
  if (state.selectedJob) {
    closeStream();
    state.selectedJob = null;
    state.jobDetail = null;
  }
  renderEmptyDetail();
}

async function refreshDetailHead() {
  const jobId = state.selectedJob;
  if (!jobId) return;
  try {
    const detail = await api(`jobs/${jobId}`);
    if (state.selectedJob !== jobId) return;
    state.jobDetail = detail;
    updateButtons();
    renderArtifacts();
    if ((detail.running || detail.job.status === 'running') && !stream) connectStream(jobId);
  } catch {
    /* the list is still usable even if one detail refresh fails */
  }
}

export function initJobs() {
  const refreshBtn = $('#refresh-jobs');
  if (refreshBtn) {
    refreshBtn.onclick = async () => {
      refreshBtn.disabled = true;
      await refreshJobs({ silent: false });
      refreshBtn.disabled = false;
      toast('已刷新');
    };
  }
  const closeBtn = $('#modal-close');
  if (closeBtn) closeBtn.onclick = () => { $('#modal-backdrop').hidden = true; };
  const copyBtn = $('#modal-copy');
  if (copyBtn) copyBtn.onclick = () => copyText(modalText, '文件内容已复制');
  const backdrop = $('#modal-backdrop');
  if (backdrop) backdrop.onclick = (event) => { if (event.target === backdrop) backdrop.hidden = true; };

  if (!resizeBound) {
    resizeBound = true;
    window.addEventListener('resize', () => { if (state.view === 'jobs') drawChart(); });
  }

  renderJobList();
  if (state.selectedJob && state.jobs.some((job) => job.id === state.selectedJob)) void openJob(state.selectedJob);
  else if (state.jobs.length) void openJob(state.jobs[0].id);
  else renderEmptyDetail();
}
