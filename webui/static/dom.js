// Tiny DOM helpers — no framework, no build step.

export function h(tag, props = {}, ...children) {
  const el = document.createElement(tag);
  for (const [key, value] of Object.entries(props || {})) {
    if (value === null || value === undefined || value === false) continue;
    if (key === 'class') el.className = value;
    else if (key === 'dataset') Object.assign(el.dataset, value);
    else if (key === 'style') Object.assign(el.style, value);
    else if (key.startsWith('on') && typeof value === 'function') el.addEventListener(key.slice(2), value);
    else if (key === 'html') el.innerHTML = value;
    else if (key === 'text') el.textContent = value;
    else if (key === 'value') el.value = value;
    else if (value === true) el.setAttribute(key, '');
    else el.setAttribute(key, value);
  }
  for (const child of children.flat(4)) {
    if (child === null || child === undefined || child === false) continue;
    el.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return el;
}

export const $ = (sel, root = document) => root.querySelector(sel);
export const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

export function clear(node) {
  while (node.firstChild) node.removeChild(node.firstChild);
  return node;
}

export function render(node, ...children) {
  clear(node);
  for (const child of children.flat(4)) {
    if (child === null || child === undefined || child === false) continue;
    node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return node;
}

export function toast(message, kind = 'ok', ms = 3600) {
  const box = $('#toasts');
  const el = h('div', { class: `toast ${kind === 'error' ? 'error' : ''}`, text: message });
  box.append(el);
  setTimeout(() => el.remove(), ms);
}

/* ------------------------------------------------- 参数说明 (field help text) */
// Every parameter input can carry a line of explanation. It is hidden by default so a form
// stays scannable; this switch, offered next to each form's own actions, brings all of them
// back and remembers the choice.
const HELP_KEY = 'trl-studio-show-help';

export const helpEnabled = () => localStorage.getItem(HELP_KEY) === '1';

export function applyHelpPreference() {
  document.body.classList.toggle('show-help', helpEnabled());
}

export function setHelpEnabled(on) {
  localStorage.setItem(HELP_KEY, on ? '1' : '0');
  applyHelpPreference();
}

export function helpSwitch(id) {
  return h(
    'label',
    { class: 'switch sm help-switch', title: '在每个参数输入框下面显示说明文字' },
    h('input', {
      type: 'checkbox',
      id,
      checked: helpEnabled(),
      onchange: (event) => setHelpEnabled(event.target.checked),
    }),
    h('span', { class: 'small muted', text: '显示参数说明' }),
  );
}

export async function copyText(text, label = '已复制') {
  try {
    if (navigator.clipboard && window.isSecureContext) await navigator.clipboard.writeText(text);
    else {
      const area = document.createElement('textarea');
      area.value = text;
      area.style.position = 'fixed';
      area.style.opacity = '0';
      document.body.append(area);
      area.select();
      document.execCommand('copy');
      area.remove();
    }
    toast(label);
  } catch (err) {
    toast(`复制失败：${err.message}`, 'error');
  }
}

export const fmtTime = (ts) => (ts ? new Date(ts * 1000).toLocaleString('zh-CN', { hour12: false }) : '—');
export const fmtBytes = (n) => (n < 1024 ? `${n} B` : n < 1024 * 1024 ? `${(n / 1024).toFixed(1)} KB` : `${(n / 1024 / 1024).toFixed(1)} MB`);
export const STATUS_TEXT = {
  created: '待启动',
  running: '运行中',
  finished: '已完成',
  failed: '失败',
  stopped: '已停止',
};
