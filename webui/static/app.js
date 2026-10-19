// App shell: state, API client, router, theme, nav. Views live in ./views/*.

import { $, $$, h, render, toast, applyHelpPreference } from './dom.js';
import { renderCatalog, initCatalog } from './views/catalog.js';
import { renderWizard, initWizard } from './views/wizard.js';
import { renderJobs, initJobs, syncJobsView, closeJobStream } from './views/jobs.js';
import { renderEnv, initEnv } from './views/env.js';

export const state = {
  catalog: null,
  doctor: null,
  envVars: null,
  envVarsError: null,
  view: 'catalog',
  family: 'all',
  search: '',
  selectedAlgorithm: null,
  algorithms: new Map(), // id -> full schema (lazily fetched)
  jobs: [],
  selectedJob: null,
  jobDetail: null,
  events: [],
  lastEventId: 0,
  metrics: {},
  hiddenSeries: new Set(),
  stream: null,
  wizard: {
    step: 1,
    algorithm: null,
    values: {},
    mode: 'simulate',
    name: '',
    accelerate: null,
    numProcesses: 1,
    validation: null,
    showAdvanced: false,
    presetApplied: false,
  },
};

export async function api(path, options = {}) {
  const response = await fetch(`/api/${path.replace(/^\//, '')}`, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  });
  const text = await response.text();
  let payload = null;
  try {
    payload = text ? JSON.parse(text) : null;
  } catch {
    throw new Error(`服务返回了非 JSON 内容：${text.slice(0, 120)}`);
  }
  if (!response.ok) {
    // `errors` is a list of {field, message} entries; anything else (a per-field map, say) is
    // not ours to render here, so fall back to the plain `error` message.
    const message =
      payload?.error ||
      (Array.isArray(payload?.errors) ? payload.errors.map((e) => e.message).join('；') : null) ||
      `HTTP ${response.status}`;
    const error = new Error(message);
    error.payload = payload;
    throw error;
  }
  return payload;
}

/* ------------------------------------------------------------------ routing */
const VIEWS = {
  catalog: { render: renderCatalog, init: initCatalog, el: '#view-catalog' },
  new: { render: renderWizard, init: initWizard, el: '#view-new' },
  jobs: { render: renderJobs, init: initJobs, el: '#view-jobs', dispose: closeJobStream },
  env: { render: renderEnv, init: initEnv, el: '#view-env' },
};

export function navigate(view) {
  if (!VIEWS[view]) view = 'catalog';
  VIEWS[state.view]?.dispose?.();
  state.view = view;
  for (const [name, def] of Object.entries(VIEWS)) {
    $(def.el).hidden = name !== view;
  }
  $$('.tabs a').forEach((tab) => tab.classList.toggle('active', tab.dataset.tab === view));
  $('.tabs')?.classList.remove('open');
  $('#burger')?.setAttribute('aria-expanded', 'false');
  if (location.hash !== `#/${view}`) history.replaceState(null, '', `#/${view}`);
  const def = VIEWS[view];
  render($(def.el), def.render());
  def.init();
  window.scrollTo({ top: 0, behavior: 'instant' in window ? 'instant' : 'auto' });
}

/* -------------------------------------------------------------------- theme */
function initTheme() {
  const stored = localStorage.getItem('trl-studio-theme');
  const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
  setTheme(stored || (prefersDark ? 'dark' : 'light'));
  $('#theme-toggle').addEventListener('click', () => {
    setTheme(document.documentElement.classList.contains('dark') ? 'light' : 'dark');
  });
}

function setTheme(theme) {
  document.documentElement.classList.toggle('dark', theme === 'dark');
  localStorage.setItem('trl-studio-theme', theme);
  const toggle = $('#theme-toggle');
  toggle?.setAttribute('aria-pressed', String(theme === 'dark'));
  toggle?.setAttribute('aria-label', theme === 'dark' ? '切换到浅色模式' : '切换到深色模式');
  const sun = document.querySelector('.icon-sun');
  const moon = document.querySelector('.icon-moon');
  if (sun && moon) {
    sun.style.display = theme === 'dark' ? 'none' : '';
    moon.style.display = theme === 'dark' ? '' : 'none';
  }
}

/* --------------------------------------------------------------------- boot */
function updateModeChip() {
  const chip = $('#mode-chip');
  const doctor = state.doctor;
  if (!doctor) {
    chip.textContent = '检测中…';
    return;
  }
  const mode = doctor.recommended_mode;
  chip.textContent = mode === 'real' ? `可实训 · ${doctor.devices.length} 设备` : '演示模式 · 仅模拟';
  chip.className = `mode-chip ${mode}`;
  chip.title =
    mode === 'real'
      ? `后端 ${doctor.backend}，${doctor.devices.length} 个设备`
      : '本机缺少 torch/加速器，任务默认以模拟模式提交（命令与产物照常生成）';
}

function updateJobCount() {
  const chip = $('#job-count');
  const active = state.jobs.filter((job) => job.status === 'running').length;
  const total = state.jobs.length;
  if (!total) {
    chip.hidden = true;
    return;
  }
  chip.hidden = false;
  chip.textContent = active ? `${active} 运行中` : String(total);
}

export async function refreshJobs({ silent = true } = {}) {
  try {
    const data = await api('jobs');
    state.jobs = data.jobs;
    updateJobCount();
    if (state.view === 'jobs') syncJobsView();
    return data.jobs;
  } catch (err) {
    if (!silent) toast(`任务列表加载失败：${err.message}`, 'error');
    return state.jobs;
  }
}

export async function loadAlgorithm(id) {
  if (state.algorithms.has(id)) return state.algorithms.get(id);
  const algo = await api(`algorithms/${id}`);
  state.algorithms.set(id, algo);
  return algo;
}

async function boot() {
  initTheme();
  applyHelpPreference(); // 参数说明默认收起，除非用户上次打开过
  $('#burger')?.addEventListener('click', (event) => {
    const tabs = $('.tabs');
    const open = tabs.classList.toggle('open');
    event.currentTarget.setAttribute('aria-expanded', String(open));
  });
  window.addEventListener('hashchange', () => navigate((location.hash || '#/catalog').slice(2)));

  try {
    const [catalog, doctor] = await Promise.all([api('catalog'), api('doctor')]);
    state.catalog = catalog;
    state.doctor = doctor;
    state.wizard.mode = doctor.recommended_mode;
    updateModeChip();
  } catch (err) {
    render(
      $('#main'),
      h('div', { class: 'panel' }, h('div', { class: 'validation' }, h('p', { class: 'item error', text: `初始化失败：${err.message}` }))),
    );
    return;
  }

  navigate((location.hash || '#/catalog').slice(2));
  await refreshJobs();
  setInterval(() => {
    if (state.view !== 'jobs') refreshJobs();
  }, 20000);
}

boot();
