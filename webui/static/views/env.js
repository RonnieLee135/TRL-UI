// View: 运行环境 — what this machine can actually run: reachability checks, the packages
// TRL needs, the detected accelerators and the paths the console points at.
//
// The 环境变量 section is the one part of this page the user writes to: HF_HOME / HF_ENDPOINT /
// HF_TOKEN / CUDA_VISIBLE_DEVICES are stored server-side (`/api/env-vars`), injected into every
// real training job, and handed to the probe — so editing them can change the device list and
// the package picture right below.

import { $, h, render, toast, helpSwitch } from '../dom.js';
import { state, api } from '../app.js';

const CHECK_CLASS = { ok: 'ok', warn: 'warn', fail: 'error' };
const inputId = (name) => `envv-${name}`;
const errorId = (name) => `env-err-${name}`;

// A secret is never sent to the browser, so an empty token box must mean "keep the stored one".
// This flag is the only thing that turns it into "delete it".
let tokenCleared = false;

export function renderEnv() {
  const doctor = state.doctor;
  // The form is always rebuilt from the server's state, so a pending "delete the token" mark
  // must not survive a re-render (it would delete a token the user never asked to remove).
  tokenCleared = false;
  return h(
    'div',
    { class: 'panel' },
    h(
      'div',
      { class: 'panel-head' },
      h(
        'div',
        {},
        h('h2', { class: 'section-title', text: '运行环境' }),
        h('p', { class: 'section-sub', text: '直接探测本机，决定任务默认以真实训练还是模拟模式提交。' }),
      ),
      h('button', { class: 'btn sm', id: 'refresh-env', type: 'button', text: '重新探测' }),
    ),
    h('div', { id: 'env-body' }, envBody(doctor)),
  );
}

function envBody(doctor) {
  if (!doctor) {
    return h(
      'div',
      {},
      h('div', { class: 'skeleton', style: { height: '18px', width: '45%' } }),
      h('div', { class: 'skeleton', style: { marginTop: '12px' } }),
      h('div', { class: 'skeleton', style: { marginTop: '8px', width: '70%' } }),
    );
  }

  const devices = doctor.devices || [];
  const mode = doctor.recommended_mode;
  const stats = [
    { value: `Python ${doctor.python}`, label: doctor.platform },
    { value: doctor.backend === 'unknown' ? '未知' : String(doctor.backend).toUpperCase(), label: devices.length ? `${devices.length} 个加速器` : '加速器' },
    { value: mode === 'real' ? '可真实训练' : '仅模拟', label: '推荐模式' },
    { value: `${(doctor.packages || []).filter((p) => p.installed).length} / ${(doctor.packages || []).length}`, label: '已安装依赖' },
  ];

  return h(
    'div',
    {},
    h('div', { class: 'stats' }, stats.map((item) => h('div', { class: 'stat' }, h('b', { text: item.value }), h('span', { text: item.label })))),
    mode === 'simulate'
      ? h('div', {
          class: 'note',
          text: '这台机器跑不了真实训练：任务默认以模拟模式提交——命令、产物、日志解析与曲线全部照常，但日志由内置模拟器产生。把任务模式改成 real 可以看到真实的报错。',
        })
      : h('div', { class: 'note', text: '检测到可用的加速器，任务默认以真实训练提交；失败时原始报错会出现在任务日志里。' }),

    envVarsSection(),

    h('h3', { class: 'section-title', style: { fontSize: '15px', marginTop: '28px' }, text: '检查项' }),
    h('div', { class: 'validation check-list' }, (doctor.checks || []).map(checkItem)),

    h('h3', { class: 'section-title', style: { fontSize: '15px', marginTop: '28px' }, text: '依赖' }),
    h(
      'div',
      { style: { overflowX: 'auto' } },
      h(
        'table',
        { class: 'param-table pkg-table' },
        h('thead', {}, h('tr', {}, ['包', '版本', '用途', '类别'].map((title) => h('th', { text: title })))),
        h(
          'tbody',
          {},
          (doctor.packages || []).map((pkg) =>
            h(
              'tr',
              {},
              h('td', {}, h('code', { text: pkg.name })),
              h('td', { class: pkg.installed ? '' : pkg.essential ? 'pkg-missing' : 'pkg-optional', text: pkg.installed ? pkg.version : '未安装' }),
              h('td', { text: pkg.purpose }),
              h('td', { class: 'choices', text: pkg.essential ? '必需' : '可选' }),
            ),
          ),
        ),
      ),
    ),

    h('h3', { class: 'section-title', style: { fontSize: '15px', marginTop: '28px' }, text: '设备' }),
    devices.length
      ? h(
          'div',
          { class: 'card-grid' },
          devices.map((device) =>
            h(
              'div',
              { class: 'stat' },
              h('b', { text: `#${device.index} ${device.name}` }),
              h('span', {
                text: [
                  device.memory_gb ? `${device.memory_gb} GB 显存` : null,
                  device.capability ? `compute ${device.capability}` : null,
                  (doctor.dtype_hints || []).length ? `建议 ${doctor.dtype_hints.join('/')}` : null,
                ]
                  .filter(Boolean)
                  .join(' · ') || '信息不可用',
              }),
            ),
          ),
        )
      : h('p', { class: 'muted', text: '没有探测到 CUDA / NPU 设备，torch 只能使用 CPU。' }),

    h('h3', { class: 'section-title', style: { fontSize: '15px', marginTop: '28px' }, text: '路径' }),
    h(
      'dl',
      { class: 'kv' },
      h('dt', { text: 'TRL 源码' }),
      h('dd', { text: doctor.trl?.path || '—' }),
      h('dt', { text: 'trl 命令' }),
      h('dd', { text: doctor.trl?.cli || 'PATH 中没有，退化为 python -m trl.cli.main' }),
      h('dt', { text: '产物目录' }),
      h('dd', { text: doctor.runs_dir || '—' }),
      h('dt', { text: '解释器' }),
      h('dd', { text: doctor.executable || '—' }),
    ),
  );
}

/* --------------------------------------------------------- 环境变量 (editable) */

function envVarsSection() {
  const payload = state.envVars;
  return h(
    'div',
    {},
    h('h3', { class: 'section-title', style: { fontSize: '15px', marginTop: '28px' }, text: '环境变量' }),
    h('p', {
      class: 'section-sub',
      text: '这里设置的变量会注入每个真实训练任务的进程，也决定本页探测看到哪些卡、模型从哪里下载。留空表示不设置。',
    }),
    payload
      ? h(
          'div',
          { id: 'env-vars-form' },
          h('div', { class: 'field-grid wide' }, (payload.fields || []).map(envField)),
          h(
            'div',
            { class: 'form-actions' },
            h('button', { class: 'btn primary sm', id: 'save-env', type: 'button', text: '保存' }),
            h('span', { class: 'hint', text: '保存后立即生效：探测会重新跑一次，之后新建的任务都带着这些变量。' }),
            helpSwitch('env-help'),
          ),
        )
      : state.envVarsError
        ? h('div', {
            class: 'note',
            text: `环境变量接口当前不可用（${state.envVarsError}）。如果服务是在这版界面之前启动的，重启 webui/server.py 后本页就能读写这四个变量。`,
          })
        : h('div', { id: 'env-vars-form' }, h('div', { class: 'skeleton', style: { height: '38px' } }), h('div', { class: 'skeleton', style: { height: '38px', marginTop: '10px' } })),
  );
}

function envField(field) {
  const placeholder = field.secret && field.set ? `${field.preview}（留空保持不变）` : field.placeholder;
  return h(
    'div',
    { class: 'field', id: `envf-${field.name}` },
    h(
      'span',
      { class: 'field-label' },
      h('code', { text: field.name }),
      h('span', { class: 'type', text: field.label }),
      field.secret ? h('span', { class: 'field-flag', text: '敏感' }) : null,
    ),
    h('input', {
      id: inputId(field.name),
      type: field.secret ? 'password' : 'text',
      value: field.value || '',
      placeholder,
      dataset: { placeholder },
      autocomplete: 'off',
      spellcheck: 'false',
      oninput: () => {
        if (field.secret && tokenCleared) resetTokenClear();
      },
    }),
    h('span', { class: 'help', text: field.purpose }),
    h('span', { class: 'help', text: sourceHint(field) }),
    field.secret && field.set
      ? h('button', { class: 'btn sm', id: 'clear-HF_TOKEN', type: 'button', text: '清除已保存的令牌' })
      : null,
    h('span', { class: 'error-text', id: errorId(field.name), hidden: true }),
  );
}

function sourceHint(field) {
  if (field.source === 'user') return field.secret ? `已保存在本机（${field.preview}），只有保存时才会传给训练进程` : '已保存，本机重启后依然有效';
  if (field.source === 'process') return `当前未设置；启动本服务的进程里有 ${field.process_value || '（空）'}`;
  return '当前未设置';
}

function resetTokenClear() {
  tokenCleared = false;
  const button = $('#clear-HF_TOKEN');
  const input = $(`#${inputId('HF_TOKEN')}`);
  if (button) button.textContent = '清除已保存的令牌';
  if (input) input.placeholder = input.dataset.placeholder || '';
}

function toggleTokenClear() {
  const button = $('#clear-HF_TOKEN');
  const input = $(`#${inputId('HF_TOKEN')}`);
  if (!button || !input) return;
  tokenCleared = !tokenCleared;
  if (tokenCleared) {
    input.value = '';
    input.placeholder = '保存后删除';
    button.textContent = '取消清除';
  } else {
    resetTokenClear();
  }
}

function setFieldError(name, message) {
  const box = $(`#${errorId(name)}`);
  if (box) {
    box.textContent = message;
    box.hidden = !message;
  }
  $(`#envf-${name}`)?.classList.toggle('invalid', Boolean(message));
}

async function saveEnvVars(button) {
  const fields = state.envVars?.fields || [];
  const values = {};
  for (const field of fields) {
    setFieldError(field.name, '');
    const input = $(`#${inputId(field.name)}`);
    const raw = (input?.value || '').trim();
    if (field.secret) {
      // no value in the browser and no explicit delete: leave the stored token alone
      if (tokenCleared) values[field.name] = '';
      else if (raw) values[field.name] = raw;
    } else {
      values[field.name] = raw; // an empty box is "unset this variable"
    }
  }

  const original = button.textContent;
  button.disabled = true;
  button.textContent = '保存中…';
  try {
    const data = await api('env-vars', { method: 'POST', body: JSON.stringify({ values }) });
    state.envVars = { fields: data.fields };
    state.envVarsError = null;
    tokenCleared = false;
    toast('环境变量已保存');
    await detect(); // the variables change what the probe sees
  } catch (err) {
    const fieldErrors = err.payload?.field_errors;
    if (fieldErrors && typeof fieldErrors === 'object') {
      Object.entries(fieldErrors).forEach(([name, message]) => setFieldError(name, message));
      toast('有变量没通过校验，请看输入框下方的提示', 'error');
    } else {
      toast(`保存失败：${err.message}`, 'error');
    }
    button.disabled = false;
    button.textContent = original;
  }
}

/* ------------------------------------------------------------------ probe */

function checkItem(check) {
  return h(
    'div',
    { class: `item ${CHECK_CLASS[check.level] || ''}` },
    h('b', { text: check.title }),
    h('span', { text: check.detail }),
    check.hint ? h('span', { class: 'small', text: `建议：${check.hint}` }) : null,
  );
}

function rerenderEnv() {
  const view = $('#view-env');
  if (!view) return;
  render(view, renderEnv());
  bindEnv();
}

async function detect(button) {
  const original = button?.textContent || '重新探测';
  const probe = button || $('#refresh-env');
  if (probe) {
    probe.disabled = true;
    probe.textContent = '探测中…';
  }
  try {
    state.doctor = await api('doctor?refresh=1');
    if (button) toast('环境已重新探测');
  } catch (err) {
    toast(`探测失败：${err.message}`, 'error');
  } finally {
    if (probe) {
      probe.disabled = false;
      probe.textContent = original;
    }
    rerenderEnv();
  }
}

async function loadEnvVars() {
  try {
    state.envVars = await api('env-vars');
    state.envVarsError = null;
  } catch (err) {
    state.envVarsError = err.message;
  }
  rerenderEnv();
}

async function refreshEnv(button) {
  await detect(button);
  // the endpoint can come back after a server restart — retry the load with the same click
  if (!state.envVars) await loadEnvVars();
}

function bindEnv() {
  const button = $('#refresh-env');
  if (button) button.onclick = () => void refreshEnv(button);
  const save = $('#save-env');
  if (save) save.onclick = () => void saveEnvVars(save);
  const clear = $('#clear-HF_TOKEN');
  if (clear) clear.onclick = () => toggleTokenClear();
}

export function initEnv() {
  bindEnv();
  if (!state.doctor) void detect();
  if (!state.envVars) void loadEnvVars();
}
