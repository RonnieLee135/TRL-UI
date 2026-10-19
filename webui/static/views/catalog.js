// View: algorithm catalogue — families, the 8-stage pipeline, per-algorithm cards and detail drawer.

import { $, $$, h, render, clear, toast, copyText } from '../dom.js';
import { state, navigate, loadAlgorithm } from '../app.js';

const STATUS_TEXT = { stable: '稳定', experimental: '实验' };

export function renderCatalog() {
  const catalog = state.catalog;
  const families = catalog.families;
  const stable = catalog.algorithms.filter((a) => a.status === 'stable').length;
  const fields = catalog.algorithms.reduce((sum, a) => sum + (a.counts?.fields || 0), 0);

  return [
    renderHero(catalog, stable, fields),
    h(
      'section',
      { class: 'panel' },
      h('h2', { class: 'section-title', text: '训练流程抽象' }),
      h('p', {
        class: 'section-sub',
        text: '所有 trainer 共用同一条八阶段管线；换算法只替换其中几段，模型与运行平面原样复用。',
      }),
      h('div', { class: 'pipeline' }, catalog.stages.map(stageCard)),
    ),
    h(
      'section',
      { class: 'panel' },
      h(
        'div',
        { class: 'panel-head' },
        h('div', {}, h('h2', { class: 'section-title', text: '算法' }), h('p', { class: 'section-sub', id: 'grid-sub' })),
        h(
          'div',
          { class: 'filter-row', role: 'group', 'aria-label': '按家族筛选' },
          h(
            'div',
            { class: 'pills', id: 'family-filter' },
            [{ id: 'all', label: '全部' }, ...families].map((fam) =>
              h('button', {
                class: 'pill',
                type: 'button',
                'aria-pressed': String(state.family === fam.id),
                dataset: { family: fam.id },
                text: fam.id === 'all' ? `全部 ${catalog.algorithms.length}` : `${fam.label} ${catalog.algorithms.filter((a) => a.family === fam.id).length}`,
              }),
            ),
          ),
          h(
            'label',
            { class: 'search' },
            h('span', { class: 'sr-only', text: '搜索算法' }),
            h('input', {
              type: 'search',
              id: 'algo-search',
              placeholder: '搜索算法、类名、参数…',
              autocomplete: 'off',
              value: state.search,
            }),
          ),
        ),
      ),
      h('div', { class: 'card-grid', id: 'algo-grid', 'aria-live': 'polite' }),
    ),
  ];
}

function renderHero(catalog, stable, fields) {
  const meta = catalog.meta;
  const doctor = state.doctor || {};
  const stats = [
    { value: catalog.algorithms.length, label: '算法（trainer）' },
    { value: `${stable} / ${catalog.algorithms.length - stable}`, label: '稳定 / 实验' },
    { value: fields, label: '参数条目（含四平面）' },
    { value: doctor.can_train ? `${doctor.devices.length} × ${String(doctor.backend).toUpperCase()}` : '模拟', label: doctor.can_train ? '可用设备' : '本机运行模式' },
  ];
  return h(
    'div',
    { class: 'hero' },
    h('p', { class: 'eyebrow', text: `TRL ${meta.trl_version} · ${meta.trl_commit.commit}` }),
    h('h1', { text: '强化学习算法与入参，一处看清' }),
    h('p', {
      class: 'lede',
      text:
        '参数、类型、默认值与分组全部来自 TRL 源码本身：dataclass 声明 + docstring 段落。挑一个算法，填表、校验、生成命令，' +
        '然后在这里启动——真机训练或模拟演示。',
    }),
    h('div', { class: 'stats' }, stats.map((s) => h('div', { class: 'stat' }, h('b', { text: String(s.value) }), h('span', { text: s.label })))),
  );
}

function stageCard(stage) {
  return h(
    'div',
    { class: 'stage' },
    h('div', { class: 'stage-top' }, h('span', { class: 'stage-id', text: stage.id.toUpperCase() }), h('span', { class: 'stage-name', text: stage.label })),
    h('p', { text: stage.summary }),
  );
}

function matches(algo, query) {
  if (!query) return true;
  const haystack = [
    algo.label,
    algo.full_name,
    algo.id,
    algo.summary,
    algo.config_class,
    algo.trainer_class,
    ...(algo.key_params || []),
  ]
    .join(' ')
    .toLowerCase();
  return haystack.includes(query.toLowerCase());
}

function renderGrid() {
  const grid = $('#algo-grid');
  if (!grid) return;
  const list = state.catalog.algorithms.filter(
    (algo) => (state.family === 'all' || algo.family === state.family) && matches(algo, state.search),
  );
  const sub = $('#grid-sub');
  if (sub) {
    sub.textContent = list.length
      ? `${list.length} 个算法 · 点击卡片看参数与流程映射`
      : '没有匹配的算法';
  }
  if (!list.length) {
    render(
      grid,
      h(
        'div',
        { class: 'empty-state' },
        h('p', { class: 'eyebrow', text: '空结果' }),
        h('h2', { text: '没有匹配的算法' }),
        h('p', { class: 'muted', text: '换个关键词，或点「全部」重新筛选。' }),
      ),
    );
    return;
  }
  render(grid, list.map(algoCard));
}

function algoCard(algo) {
  const family = state.catalog.families.find((f) => f.id === algo.family);
  const card = h(
    'article',
    { class: 'card' },
    h(
      'div',
      { class: 'card-top' },
      h('div', {}, h('h3', { text: algo.label }), h('div', { class: 'sub', text: algo.full_name })),
      h('span', { class: `chip ${algo.status}`, text: STATUS_TEXT[algo.status] }),
    ),
    h('div', { class: 'card-meta' }, h('span', { class: 'chip brand', text: family?.label || algo.family }), h('span', { class: 'chip mono', text: algo.config_class })),
    h('p', { class: 'summary', text: algo.summary }),
    h(
      'div',
      { class: 'key-params' },
      (algo.key_params || []).slice(0, 6).map((name) => h('span', { class: 'chip mono', text: name })),
      (algo.key_params || []).length > 6 ? h('span', { class: 'chip', text: `+${algo.key_params.length - 6}` }) : null,
    ),
    h(
      'div',
      { class: 'card-actions' },
      h('button', { class: 'btn sm', type: 'button', onclick: () => openDrawer(algo.id), text: '查看详情' }),
      h('button', {
        class: 'btn sm primary',
        type: 'button',
        onclick: () => startWizard(algo.id),
        text: '用这个算法训练',
      }),
    ),
  );
  card.addEventListener('click', (event) => {
    if (event.target.closest('button')) return;
    openDrawer(algo.id);
  });
  return card;
}

export function startWizard(algorithmId) {
  state.wizard.algorithm = algorithmId;
  state.wizard.step = 1;
  navigate('new');
}

/* --------------------------------------------------------------- drawer */
async function openDrawer(algorithmId) {
  const drawer = $('#drawer');
  const backdrop = $('#drawer-backdrop');
  drawer.hidden = false;
  backdrop.hidden = false;
  render($('#drawer-body'), loadingBlock());
  let algo;
  try {
    algo = await loadAlgorithm(algorithmId);
  } catch (err) {
    render($('#drawer-body'), h('p', { class: 'item error', text: `加载失败：${err.message}` }));
    return;
  }
  render($('#drawer-body'), drawerBody(algo));
  $('#drawer-close').focus();
}

function loadingBlock() {
  return h(
    'div',
    { class: 'drawer-loading' },
    h('div', { class: 'skeleton', style: { width: '40%', height: '24px' } }),
    h('div', { class: 'skeleton', style: { marginTop: '14px' } }),
    h('div', { class: 'skeleton', style: { marginTop: '8px', width: '80%' } }),
    h('div', { class: 'skeleton', style: { marginTop: '8px', width: '60%' } }),
  );
}

function closeDrawer() {
  $('#drawer').hidden = true;
  $('#drawer-backdrop').hidden = true;
}

function drawerBody(algo) {
  const family = state.catalog.families.find((f) => f.id === algo.family);
  const requires = algo.requires || {};
  const requiredChips = Object.entries(requires)
    .filter(([, value]) => value === true || value === 'required')
    .map(([key]) => key);
  const optionalChips = Object.entries(requires)
    .filter(([, value]) => value === 'optional' || value === 'self' || value === 'remote')
    .map(([key, value]) => `${key}:${value}`);

  const fields = algo.fields || [];
  const keyFields = fields.filter((f) => f.tier === 'key');
  const groups = [];
  for (const field of fields) {
    const last = groups[groups.length - 1];
    const label = `${stageLabel(field.stage)} · ${field.group}`;
    if (last && last.label === label) last.fields.push(field);
    else groups.push({ label, fields: [field] });
  }

  return h(
    'div',
    {},
    h('p', { class: 'eyebrow', text: `${family?.label || algo.family} · ${STATUS_TEXT[algo.status]}` }),
    h('h2', { text: algo.label }),
    h('p', { class: 'drawer-sub', text: algo.full_name }),
    h('p', { class: 'muted', style: { marginTop: '10px' }, text: algo.summary }),
    h(
      'dl',
      { class: 'kv' },
      h('dt', { text: '配置类' }),
      h('dd', { text: `${algo.config_class}（${(algo.config_chain || []).join(' → ')}）` }),
      h('dt', { text: '训练器' }),
      h('dd', { text: algo.trainer_class }),
      h('dt', { text: '模块' }),
      h('dd', { text: algo.trainer_module }),
      h('dt', { text: '启动' }),
      h('dd', { text: algo.launch?.kind === 'cli' ? `trl ${algo.launch.cli}` : 'Python API（实验算法，界面会生成脚本）' }),
      h('dt', { text: '数据契约' }),
      h('dd', { text: (algo.data?.roles || []).join(' + ') }),
      h('dt', { text: '文档' }),
      h('dd', {}, h('span', { class: 'mono small', text: algo.doc || '—' })),
      algo.paper ? h('dt', { text: '论文' }) : null,
      algo.paper
        ? h('dd', {}, h('a', { href: `https://huggingface.co/papers/${algo.paper}`, target: '_blank', rel: 'noreferrer', text: `huggingface.co/papers/${algo.paper}` }))
        : null,
    ),
    h(
      'div',
      { class: 'card-meta', style: { marginTop: '14px' } },
      requiredChips.map((name) => h('span', { class: 'chip brand', text: `需要 ${name}` })),
      optionalChips.map((name) => h('span', { class: 'chip', text: name })),
    ),
    h('h3', { text: '训练流程映射' }),
    h(
      'div',
      {},
      state.catalog.stages
        .filter((stage) => algo.stages?.[stage.id])
        .map((stage) =>
          h('div', { class: 'stage-row' }, h('b', { text: stage.label }), h('span', { text: algo.stages[stage.id] })),
        ),
    ),
    h('h3', { text: `关键参数（${keyFields.length}）` }),
    h(
      'div',
      { style: { overflowX: 'auto' } },
      h(
        'table',
        { class: 'param-table' },
        h('thead', {}, h('tr', {}, ['参数', '类型', '默认', '可选值', '说明'].map((t) => h('th', { text: t })))),
        h(
          'tbody',
          {},
          keyFields.map((field) =>
            h(
              'tr',
              {},
              h('td', {}, h('code', { text: field.name })),
              h('td', { text: field.type }),
              h('td', {}, h('code', { text: field.default === null ? '—' : String(field.default) })),
              h('td', { class: 'choices', text: (field.choices || []).join(' | ') || '—' }),
              h('td', { text: clamp(field.help, 160) }),
            ),
          ),
        ),
      ),
    ),
    h('h3', { text: `全部参数（${fields.length}）` }),
    h('p', {
      class: 'small muted',
      text: '分组标题直接取自 TRL docstring 里的 “> Parameters that control …”，阶段映射见上方。',
    }),
    h(
      'div',
      {},
      groups.map((group) =>
        h(
          'details',
          { style: { marginTop: '10px' } },
          h('summary', { class: 'small', style: { cursor: 'pointer' }, text: `${group.label}（${group.fields.length}）` }),
          h(
            'table',
            { class: 'param-table' },
            h('tbody', {}, group.fields.map((field) => paramRow(field))),
          ),
        ),
      ),
    ),
    algo.pros?.length ? h('h3', { text: '优势' }) : null,
    algo.pros?.length ? h('ul', { class: 'list-tight' }, algo.pros.map((t) => h('li', { text: t }))) : null,
    algo.cons?.length ? h('h3', { text: '代价' }) : null,
    algo.cons?.length ? h('ul', { class: 'list-tight' }, algo.cons.map((t) => h('li', { text: t }))) : null,
    algo.pitfalls?.length ? h('h3', { text: '容易踩的坑' }) : null,
    algo.pitfalls?.length ? h('ul', { class: 'list-tight' }, algo.pitfalls.map((t) => h('li', { text: t }))) : null,
    h('div', { class: 'note', text: '这里的每一项都来自本机 TRL 检出的源码；docs/02-algorithm-params.md 是同一份数据的可读版本。' }),
    h(
      'div',
      { style: { marginTop: '18px', display: 'flex', gap: '8px', flexWrap: 'wrap' } },
      h('button', { class: 'btn primary', type: 'button', onclick: () => { closeDrawer(); startWizard(algo.id); }, text: '用这个算法新建训练' }),
      h('button', {
        class: 'btn',
        type: 'button',
        onclick: () => copyText(`trl ${algo.launch?.cli || algo.id} --help`, '命令已复制'),
        text: '复制命令',
      }),
    ),
  );
}

function paramRow(field) {
  return h(
    'tr',
    {},
    h('td', {}, h('code', { text: field.name }), field.tier === 'key' ? h('span', { class: 'field-flag', text: ' ★' }) : null),
    h('td', { text: field.type }),
    h('td', {}, h('code', { text: field.default === null ? '—' : String(field.default) })),
    h('td', { class: 'choices', text: (field.choices || []).join(' | ') }),
    h('td', { text: clamp(field.help, 220) }),
  );
}

const clamp = (text, limit) => (text && text.length > limit ? `${text.slice(0, limit - 1)}…` : text || '—');
const stageLabel = (id) => state.catalog.stages.find((s) => s.id === id)?.label || id;

/* ------------------------------------------------------------------- init */
export function initCatalog() {
  renderGrid();
  $$('#family-filter .pill').forEach((pill) => {
    pill.addEventListener('click', () => {
      state.family = pill.dataset.family;
      $$('#family-filter .pill').forEach((other) => other.setAttribute('aria-pressed', String(other === pill)));
      renderGrid();
    });
  });
  let timer = null;
  $('#algo-search')?.addEventListener('input', (event) => {
    clearTimeout(timer);
    const value = event.target.value;
    timer = setTimeout(() => {
      state.search = value;
      renderGrid();
    }, 120);
  });
  $('#drawer-close')?.addEventListener('click', closeDrawer);
  $('#drawer-backdrop')?.addEventListener('click', closeDrawer);
  document.addEventListener('keydown', (event) => {
    if (event.key !== 'Escape') return;
    if (!$('#drawer').hidden) closeDrawer();
    if (!$('#modal-backdrop').hidden) {
      $('#modal-backdrop').hidden = true;
    }
  });
}
