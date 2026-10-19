// View: 新建训练 — a four-step wizard driven entirely by the catalogue:
//   1 算法  2 数据与信号（script 平面）  3 模型（model 平面）  4 训练与资源（algorithm + runtime 平面）
// The right-hand rail always shows the validation result and the exact command that will run.

import { $, $$, h, render, clear, toast, copyText, helpSwitch } from '../dom.js';
import { state, api, navigate, loadAlgorithm, refreshJobs } from '../app.js';

const STEP_TITLES = ['选择算法', '数据与信号', '模型', '训练与资源'];
const REQUIRED_ALWAYS = new Set(['model_name_or_path', 'dataset_name']);

export function renderWizard() {
  const wizard = state.wizard;
  const algo = wizard.algorithm ? state.algorithms.get(wizard.algorithm) : null;
  return h(
    'div',
    { class: 'wizard' },
    h(
      'div',
      { class: 'wizard-main' },
      h('ol', { class: 'steps', id: 'steps' }, stepsList()),
      h('div', { class: 'step-body', id: 'step-body', 'aria-live': 'polite' }, stepBody(algo)),
      h(
        'div',
        { class: 'step-nav' },
        h('button', { class: 'btn', id: 'step-prev', type: 'button', disabled: wizard.step === 1, text: '上一步' }),
        h(
          'div',
          { class: 'step-nav-right' },
          h('button', { class: 'btn', id: 'step-next', type: 'button', disabled: wizard.step === 4, text: '下一步' }),
          // Step 1 has no parameter fields, so a 「显示参数说明」 switch there would do nothing.
          wizard.step === 1 ? null : helpSwitch('wizard-help'),
        ),
      ),
    ),
    sideRail(algo),
  );
}

/** Re-render the whole wizard in place and rebind the controls that live on the shell. */
function refreshWizard() {
  const view = $('#view-new');
  if (!view) return;
  render(view, renderWizard());
  bindControls();
}

function sideRail(algo) {
  const wizard = state.wizard;
  return h(
    'aside',
    { class: 'wizard-side', 'aria-label': '命令预览与校验' },
    h(
      'div',
      { class: 'side-block' },
      h(
        'label',
        { class: 'field' },
        h('span', { class: 'field-label', text: '任务名' }),
        h('input', {
          type: 'text',
          id: 'job-name',
          placeholder: '例如 grpo-qwen3-4b-code',
          autocomplete: 'off',
          value: wizard.name || '',
        }),
      ),
      h(
        'fieldset',
        { class: 'mode-pick' },
        h('legend', { class: 'field-label', text: '执行模式' }),
        ['simulate', 'real'].map((mode) =>
          h(
            'label',
            {},
            h('input', { type: 'radio', name: 'mode', value: mode, checked: wizard.mode === mode }),
            mode === 'simulate' ? ' 模拟（不真训，演示用）' : ' 真实训练',
          ),
        ),
        h('span', {
          class: 'small muted',
          text:
            wizard.mode === 'real'
              ? '真跑需要本机装好 trl + torch + 加速器；失败会在日志里显示原始报错。'
              : '模拟模式只演示界面与产物，不会产生任何训练结果。',
        }),
      ),
    ),
    h(
      'div',
      { class: 'side-block' },
      h(
        'div',
        { class: 'side-head' },
        h('h3', { text: '校验' }),
        h('button', { class: 'link-btn', id: 'validate-btn', type: 'button', text: '重新校验' }),
      ),
      h('div', { class: 'validation', id: 'validation', 'aria-live': 'polite' }),
    ),
    h(
      'div',
      { class: 'side-block' },
      h(
        'div',
        { class: 'side-head' },
        h('h3', { text: '命令预览' }),
        h('button', { class: 'link-btn', id: 'copy-cmd', type: 'button', text: '复制' }),
      ),
      h('pre', { class: 'cmd', id: 'cmd-preview', text: wizard.validation?.command || '—' }),
      h('p', { class: 'small muted', text: algo ? `产物目录：runs/<任务 id>/（生成后可见 command.sh、params.json、README.md${algo.launch?.kind === 'cli' ? '' : ' 和训练脚本'}）` : '' }),
    ),
    h(
      'div',
      { class: 'side-actions' },
      h('button', {
        class: 'btn primary lg',
        id: 'submit-job',
        type: 'button',
        disabled: !algo || !wizard.validation?.ok,
        text: '创建任务',
      }),
      h('button', {
        class: 'btn lg',
        id: 'submit-run',
        type: 'button',
        disabled: !algo || !wizard.validation?.ok,
        text: '创建并启动',
      }),
      h('button', {
        class: 'btn lg',
        id: 'use-preset',
        type: 'button',
        disabled: !algo || !state.catalog.presets?.[algo?.id],
        text: '套用起步预设',
      }),
    ),
  );
}

function stepsList() {
  const current = state.wizard.step;
  return ['1', '2', '3', '4'].map((step, index) => {
    const number = index + 1;
    const stateName = number === current ? 'current' : number < current ? 'done' : 'todo';
    return h(
      'li',
      { dataset: { step: String(number), state: stateName } },
      h('span', { class: 'step-num', text: String(number) }),
      h('span', { text: STEP_TITLES[index] }),
    );
  });
}

function stepBody(algo) {
  const step = state.wizard.step;
  if (step === 1) return algorithmPicker();
  if (!algo) {
    return h(
      'div',
      { class: 'empty-state' },
      h('p', { class: 'eyebrow', text: '第一步' }),
      h('h2', { text: '先选一个算法' }),
      h('p', { class: 'muted', text: '回到上一步选择算法后，这里会自动带出它需要的参数。' }),
      h('button', { class: 'btn', type: 'button', onclick: () => goStep(1), text: '去选算法' }),
    );
  }
  if (step === 2) return dataStep(algo);
  if (step === 3) return modelStep(algo);
  return trainingStep(algo);
}

/* -------------------------------------------------------------- step 1 */
// One dropdown instead of a card per algorithm: the descriptions live on the 算法总览 page,
// where they can be compared, and this step is only about picking one.
function algorithmPicker() {
  const { families, algorithms } = state.catalog;
  const selected = state.wizard.algorithm;
  return h(
    'div',
    {},
    h('h2', { class: 'section-title', text: '选择算法' }),
    h(
      'p',
      { class: 'section-sub' },
      '按家族分组，选中后会套用起步预设并进入下一步。想看每个算法在做什么，去 ',
      h('a', { href: '#/catalog', text: '算法总览' }),
      '。',
    ),
    h(
      'div',
      { class: 'field', style: { maxWidth: '32rem' } },
      h('span', { class: 'field-label', text: '算法' }),
      h(
        'select',
        {
          id: 'algorithm-select',
          'aria-label': '选择算法',
          onchange: (event) => {
            if (event.target.value) void pickAlgorithm(event.target.value);
          },
        },
        h('option', { value: '', text: '选择一个算法…', selected: !selected }),
        ...families
          .map((family) => [family, algorithms.filter((algo) => algo.family === family.id)])
          .filter(([, items]) => items.length)
          .map(([family, items]) =>
            h(
              'optgroup',
              { label: family.label },
              items.map((algo) =>
                h('option', {
                  value: algo.id,
                  selected: selected === algo.id,
                  text: `${algo.label}（${algo.status === 'stable' ? '稳定' : '实验'}）`,
                }),
              ),
            ),
          ),
      ),
    ),
  );
}

async function pickAlgorithm(id) {
  state.wizard.algorithm = id;
  try {
    await loadAlgorithm(id);
  } catch (err) {
    toast(`加载 ${id} 的参数失败：${err.message}`, 'error');
    return;
  }
  applyPreset({ silent: true, render: false });
  state.wizard.step = 2;
  refreshWizard();
  void schedulePreview();
}

/* -------------------------------------------------------------- step 2 */
function dataStep(algo) {
  const fields = algo.schema_fields.filter((f) => f.stage === 'data' || f.stage === 'signal');
  const examples = state.catalog.dataset_examples?.[algo.id] || [];
  const roles = algo.data?.roles || [];
  return h(
    'div',
    {},
    h('h2', { class: 'section-title', text: '数据与信号' }),
    h('p', { class: 'section-sub', text: `${algo.label} 的数据契约：${roles.join(' + ') || '—'}。${algo.data?.format || ''}` }),
    examples.length
      ? h(
          'div',
          { class: 'card-meta', style: { marginBottom: '16px' } },
          h('span', { class: 'small muted', text: '文档里的例子：' }),
          examples.map((name) =>
            h('button', {
              class: 'chip mono',
              type: 'button',
              style: { cursor: 'pointer' },
              onclick: () => {
                state.wizard.values.dataset_name = name;
                refreshWizard();
                void schedulePreview();
              },
              text: name,
            }),
          ),
        )
      : null,
    fields.length ? groupsFor(fields, algo) : h('p', { class: 'muted', text: '该算法在这一步没有额外参数。' }),
    algo.requires?.reward_funcs
      ? h('div', { class: 'note', text: '奖励函数在 rollout 时由数据集的其它列（如 answer、test_cases）作为 kwargs 调用；写错不会让训练报错，只会把模型带偏，建议先用小样本抽查。' })
      : null,
    algo.requires?.teacher
      ? h('div', { class: 'note', text: `该算法需要教师模型（${algo.requires.teacher}）。教师与学生的 tokenizer 不一致时考虑 GOLD，教师太大放不下可以用 Server Distillation。` })
      : null,
  );
}

/* -------------------------------------------------------------- step 3 */
function modelStep(algo) {
  const fields = algo.schema_fields.filter((f) => f.stage === 'model');
  return h(
    'div',
    {},
    h('h2', { class: 'section-title', text: '模型' }),
    h('p', { class: 'section-sub', text: '模型平面（ModelConfig）：换算法时这一层原样复用。' }),
    groupsFor(fields, algo),
    h('div', {
      class: 'note',
      text: '显存不够时的顺序：先降 max_completion_length → 开 gradient_checkpointing（默认已开）→ LoRA（use_peft）→ 4bit 量化 → DeepSpeed/FSDP 分片。',
    }),
  );
}

/* -------------------------------------------------------------- step 4 */
function trainingStep(algo) {
  const algorithmStages = ['env', 'rollout', 'estimate', 'update', 'output', 'algorithm'];
  const algoFields = algo.schema_fields.filter((f) => algorithmStages.includes(f.stage));
  const runtimeFields = algo.schema_fields.filter((f) => f.plane === 'runtime');
  const wizard = state.wizard;
  const showAdvanced = wizard.showAdvanced;
  const visible = (list) => (showAdvanced ? list : list.filter((f) => f.tier === 'key'));
  const algorithmKeyCount = algoFields.filter((f) => f.tier === 'key').length;
  const hiddenCount = algoFields.length - algorithmKeyCount;

  return h(
    'div',
    {},
    h('h2', { class: 'section-title', text: '训练与资源' }),
    h('p', { class: 'section-sub', text: '算法平面与运行平面：采样、优势、损失，以及并行、精度与日志。' }),

    h(
      'fieldset',
      { class: 'field-group' },
      h('h3', {}, '运行方式', h('span', { class: 'chip mono', text: 'accelerate' })),
      h(
        'div',
        { class: 'field-grid' },
        selectField({
          name: '__accelerate_config',
          label: '并行配置（accelerate_config）',
          help:
            'TRL 自带 trl/accelerate_configs/：single_gpu / multi_gpu / zero1-3 / fsdp1-2；' +
            'none 表示不套 accelerate，直接用 python 跑（仅适用于生成的脚本）。CLI 算法经由 `trl <algo>` 启动，' +
            '该值会被转成 `--accelerate_config`。',
          options: ['single_gpu', 'multi_gpu', 'zero1', 'zero2', 'zero3', 'fsdp1', 'fsdp2', 'none'],
          value: wizard.accelerate || defaultAccelerate(),
          onChange: (value) => {
            wizard.accelerate = value;
            void schedulePreview();
          },
        }),
        numberField({
          name: '__num_processes',
          label: '进程数（num_processes）',
          help: 'accelerate launch 的进程数，通常等于 GPU 数；GRPO 系要求 有效 batch 能被 num_generations 整除。',
          value: wizard.numProcesses,
          min: 1,
          onChange: (value) => {
            wizard.numProcesses = Math.max(1, Number(value) || 1);
            void schedulePreview();
          },
        }),
      ),
    ),

    groupsFor(
      visible(algoFields),
      algo,
      !showAdvanced && hiddenCount > 0
        ? h(
            'p',
            { class: 'small muted' },
            `已隐藏 ${hiddenCount} 个高级参数（保留 ${algorithmKeyCount} 个关键项），用下方按钮展开全部 ${algoFields.length} 个。`,
          )
        : null,
    ),
    h(
      'div',
      { style: { margin: '4px 0 22px' } },
      h('button', {
        class: 'advanced-toggle',
        type: 'button',
        onclick: () => {
          wizard.showAdvanced = !wizard.showAdvanced;
          refreshWizard();
        },
        text: showAdvanced ? '收起高级参数（只看关键项）' : `显示全部参数（${algoFields.length} 个算法参数 + ${runtimeFields.length} 个运行参数）`,
      }),
    ),
    groupsFor(visible(runtimeFields), algo),
    h('div', {
      class: 'note',
      text: '留空的参数不会出现在命令里——这样 TRL 升级后的新默认值会自动生效；要显式固定某个值就直接填上它。',
    }),
  );
}

/* ----------------------------------------------------------- field render */
function groupsFor(fields, algo, extra) {
  if (!fields.length) return h('p', { class: 'muted', text: '这一层没有可填参数。' });
  const groups = [];
  for (const field of fields) {
    const label = `${stageLabel(field.stage)} · ${field.group}`;
    const sub = planeLabel(field.plane);
    const last = groups[groups.length - 1];
    if (last && last.label === label && last.sub === sub) last.fields.push(field);
    else groups.push({ label, sub, fields: [field] });
  }
  return h(
    'div',
    {},
    ...groups.map((group) =>
      h(
        'fieldset',
        { class: 'field-group' },
        h('h3', {}, group.label, h('span', { class: 'chip', text: group.sub }), h('span', { class: 'small muted', text: `${group.fields.length} 项` })),
        h('div', { class: 'field-grid' }, group.fields.map((field) => fieldControl(field, algo))),
      ),
    ),
    extra || null,
  );
}

function fieldControl(field) {
  const wizard = state.wizard;
  const value = wizard.values[field.name];
  const required = REQUIRED_ALWAYS.has(field.name) || (field.name === 'reward_funcs' && field.tier === 'key' && isRewardRequired());
  const onChange = (next) => {
    if (next === '' || next === null || (Array.isArray(next) && !next.length)) delete wizard.values[field.name];
    else wizard.values[field.name] = next;
    void schedulePreview();
  };
  // The container is a div, not a label: these fields hold either one control or a
  // switch, and a label may not wrap more than one form control. The control is bound
  // to the field name through aria-labelledby / aria-describedby instead.
  const labelId = `fl-${field.name.replace(/[^A-Za-z0-9_-]/g, '_')}`;
  const helpId = `${labelId}-help`;
  const helpText = field.help || '';  const aria = { 'aria-labelledby': labelId, 'aria-describedby': helpText ? helpId : null };
  const label = h(
    'span',
    { class: 'field-label', id: labelId },
    h('code', { text: field.name }),
    h('span', { class: 'type', text: field.type }),
    field.tier === 'key' ? h('span', { class: 'field-flag', text: '★ 关键' }) : null,
    required ? h('span', { class: 'required-mark', text: '必填' }) : null,
  );
  const help = helpText ? h('span', { class: 'help', id: helpId, text: helpText }) : null;
  const control =
    field.ui === 'toggle'
      ? toggleControl(field, value, onChange, aria)
      : field.ui === 'select'
        ? selectControl(field, value, onChange, aria)
        : field.ui === 'json'
          ? jsonControl(field, value, onChange, aria)
          : numberOrText(field, value, onChange, aria);

  return h(
    'div',
    { class: 'field', dataset: { field: field.name } },
    label,
    control,
    // One explanation line per input, collapsed unless the 「显示参数说明」 switch is on.
    help,
  );
}

function toggleControl(field, value, onChange, aria) {
  const input = h('input', {
    type: 'checkbox',
    checked: value === true,
    ...aria,
    onchange: (event) => onChange(event.target.checked),
  });
  return h(
    'label',
    { class: 'switch' },
    input,
    h('span', { class: 'small muted', text: value === true ? '已开启' : field.default === null ? '未设置（框架默认）' : `默认 ${field.default}` }),
  );
}

function selectControl(field, value, onChange, aria) {
  return h(
    'select',
    { ...aria, onchange: (event) => onChange(event.target.value) },
    h('option', { value: '', text: field.default === null ? '不传（框架默认）' : `不传（TRL 默认 ${field.default}）` }),
    (field.choices || []).map((choice) =>
      h('option', { value: choice, selected: String(value) === String(choice), text: choice }),
    ),
  );
}

function jsonControl(field, value, onChange, aria) {
  return h('textarea', {
    ...aria,
    placeholder: field.default === null ? 'JSON，例如 {"key": "value"}' : `默认 ${field.default}`,
    value: value === undefined ? '' : typeof value === 'string' ? value : JSON.stringify(value),
    onchange: (event) => onChange(event.target.value.trim()),
  });
}

function numberOrText(field, value, onChange, aria) {
  if (field.ui === 'int' || field.ui === 'number') {
    return h('input', {
      ...aria,
      type: 'number',
      step: field.ui === 'int' ? '1' : 'any',
      placeholder: field.default === null ? '未设置' : `默认 ${field.default}`,
      value: value === undefined ? '' : String(value),
      oninput: (event) => onChange(event.target.value),
    });
  }
  if (field.ui === 'list') {
    const asText = Array.isArray(value) ? value.join(' ') : value || '';
    return h('input', {
      ...aria,
      type: 'text',
      placeholder: '多个值用空格或逗号分隔',
      value: asText,
      oninput: (event) => onChange(splitList(event.target.value)),
    });
  }
  return h('input', {
    ...aria,
    type: 'text',
    placeholder: field.default === null ? '未设置' : `默认 ${field.default}`,
    value: value ?? '',
    oninput: (event) => onChange(event.target.value),
  });
}

function selectField({ name, label, help, options, value, onChange }) {
  return h(
    'label',
    { class: 'field' },
    h('span', { class: 'field-label' }, h('code', { text: name.replace(/^__/, '') }), h('span', { class: 'type', text: 'str' })),
    h('select', { onchange: (event) => onChange(event.target.value) }, options.map((option) => h('option', { value: option, selected: option === value, text: option }))),
    h('span', { class: 'help', text: help }),
  );
}

function numberField({ name, label, help, value, min, onChange }) {
  return h(
    'label',
    { class: 'field' },
    h('span', { class: 'field-label' }, h('code', { text: name.replace(/^__/, '') }), h('span', { class: 'type', text: 'int' })),
    h('input', { type: 'number', min: String(min), value: String(value), oninput: (event) => onChange(event.target.value) }),
    h('span', { class: 'help', text: help }),
  );
}

const splitList = (text) => text.split(/[,\s]+/).map((v) => v.trim()).filter(Boolean);
const stageLabel = (id) => state.catalog.stages.find((s) => s.id === id)?.label || id;
const planeLabel = (id) => state.catalog.planes?.find((p) => p.id === id)?.label || id;
const isRewardRequired = () => {
  const algo = state.algorithms.get(state.wizard.algorithm);
  return Boolean(algo?.requires?.reward_funcs);
};
const defaultAccelerate = () => {
  const doctor = state.doctor || {};
  return (doctor.devices?.length || 0) > 1 ? 'multi_gpu' : 'single_gpu';
};

/* --------------------------------------------------- preset / preview / submit */
export function applyPreset({ silent = false, render: shouldRender = true } = {}) {
  const wizard = state.wizard;
  const preset = state.catalog.presets?.[wizard.algorithm];
  if (!preset) {
    if (!silent) toast('这个算法没有起步预设，按源码默认值填写即可', 'error');
    return;
  }
  Object.assign(wizard.values, preset.params);
  wizard.presetApplied = true;
  if (!silent) toast('已套用起步预设');
  if (shouldRender) refreshWizard();
  void schedulePreview();
}

let previewTimer = null;
let previewToken = 0;

export async function schedulePreview(delay = 350) {
  clearTimeout(previewTimer);
  previewTimer = setTimeout(runPreview, delay);
}

async function runPreview() {
  const wizard = state.wizard;
  if (!wizard.algorithm) return;
  const token = ++previewToken;
  try {
    const result = await api('preview', {
      method: 'POST',
      body: JSON.stringify({
        job_id: wizard.candidateId,
        algorithm: wizard.algorithm,
        params: wizard.values,
        resource: { accelerate_config: wizard.accelerate || defaultAccelerate(), num_processes: wizard.numProcesses },
      }),
    });
    if (token !== previewToken) return;
    wizard.candidateId = result.job_id;
    wizard.validation = result;
    $('#cmd-preview').textContent = result.command || '—';
    $('#submit-job').disabled = !result.ok;
    $('#submit-run').disabled = !result.ok;
    renderValidation();
  } catch (err) {
    if (token !== previewToken) return;
    wizard.validation = { ok: false, errors: [{ level: 'error', field: '', message: err.message }], warnings: [] };
    renderValidation();
  }
}

function renderValidation() {
  const box = $('#validation');
  const result = state.wizard.validation;
  if (!result) {
    render(box, h('p', { class: 'muted', text: '选择算法后自动校验。' }));
    return;
  }
  const items = [
    ...(result.errors || []).map((item) => ({ ...item, level: 'error' })),
    ...(result.warnings || []).map((item) => ({ ...item, level: 'warn' })),
  ];
  if (!items.length && result.ok) {
    render(box, h('p', { class: 'item ok', text: '参数检查通过，可以提交。' }));
    return;
  }
  if (!items.length) {
    render(box, h('p', { class: 'muted', text: '参数检查中…' }));
    return;
  }
  render(
    box,
    items.map((item) => h('p', { class: `item ${item.level}` }, item.field ? h('code', { text: `${item.field} ` }) : null, item.message)),
    result.ok ? h('p', { class: 'item ok', text: '除上述提醒外没有阻塞问题。' }) : null,
  );
}

async function submit({ start }) {
  const wizard = state.wizard;
  if (!wizard.algorithm) return;
  const button = $(start ? '#submit-run' : '#submit-job');
  const original = button.textContent;
  button.disabled = true;
  button.textContent = '提交中…';
  try {
    const created = await api('jobs', {
      method: 'POST',
      body: JSON.stringify({
        job_id: wizard.candidateId,
        name: $('#job-name').value.trim() || undefined,
        algorithm: wizard.algorithm,
        mode: wizard.mode,
        params: wizard.values,
        resource: { accelerate_config: wizard.accelerate || defaultAccelerate(), num_processes: wizard.numProcesses },
      }),
    });
    toast(created.warnings?.length ? `任务已创建（${created.warnings.length} 条提醒）` : '任务已创建');
    if (start) {
      await api(`jobs/${created.job.id}/start`, { method: 'POST' });
      toast('已开始运行');
    }
    wizard.candidateId = null;
    state.selectedJob = created.job.id;
    await refreshJobs();
    navigate('jobs');
  } catch (err) {
    toast(`提交失败：${err.message}`, 'error');
    if (err.payload?.errors) {
      wizard.validation = { ok: false, errors: err.payload.errors, warnings: err.payload.warnings || [] };
      renderValidation();
    }
  } finally {
    button.disabled = false;
    button.textContent = original;
  }
}

function goStep(step) {
  state.wizard.step = Math.min(4, Math.max(1, step));
  state.wizard.showAdvanced = false;
  refreshWizard();
}

/* ------------------------------------------------------------------- init */
/** Bind the wizard shell controls; called again after every in-place re-render. */
function bindControls() {
  const wizard = state.wizard;
  const prev = $('#step-prev');
  const next = $('#step-next');
  if (prev) prev.onclick = () => goStep(wizard.step - 1);
  if (next) next.onclick = () => goStep(wizard.step + 1);
  const preset = $('#use-preset');
  if (preset) preset.onclick = () => applyPreset();
  const create = $('#submit-job');
  if (create) create.onclick = () => submit({ start: false });
  const run = $('#submit-run');
  if (run) run.onclick = () => submit({ start: true });
  const copy = $('#copy-cmd');
  if (copy) copy.onclick = () => copyText($('#cmd-preview').textContent, '命令已复制');
  const validate = $('#validate-btn');
  if (validate) {
    validate.onclick = () => {
      void runPreview();
      toast('已重新校验');
    };
  }
  const name = $('#job-name');
  if (name) name.oninput = (event) => { wizard.name = event.target.value; };
  $$('input[name="mode"]').forEach((input) => {
    input.onchange = (event) => {
      wizard.mode = event.target.value;
      if (wizard.mode === 'simulate') toast('模拟模式：命令与产物照常生成，日志与曲线由内置模拟器产生');
    };
  });
  $$('#steps li').forEach((item) => {
    item.style.cursor = 'pointer';
    item.onclick = () => goStep(Number(item.dataset.step));
  });
}

export function initWizard() {
  bindControls();
  const id = state.wizard.algorithm;
  if (!id) return;
  if (!state.algorithms.has(id)) {
    void loadAlgorithm(id)
      .then(() => { if (state.wizard.algorithm === id) refreshWizard(); })
      .catch((err) => toast(`加载 ${id} 的参数失败：${err.message}`, 'error'));
    return;
  }
  void schedulePreview(0);
}
