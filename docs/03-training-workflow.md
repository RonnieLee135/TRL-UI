# 强化学习训练流程的抽象

> 目的：把 TRL 里 21 个 trainer 的差异收敛成**一条可复用的流水线 + 一组能力开关**，让「换算法」不再等于「换一套代码」，也让 WebUI 能用同一套表单渲染所有算法。
> 抽象不是凭空的：每一层都能在 TRL 源码里找到对应物（下文都给了文件与类名）。参数清单见 [02-algorithm-params.md](02-algorithm-params.md)。

## 1. 抽象的依据：TRL 自己就是三层结构

打开 `trl/scripts/grpo.py`（以及 dpo/rloo/kto/sft/reward/distillation 六个脚本），每个脚本的 `make_parser()` 完全同形：

```python
dataclass_types = (GRPOScriptArguments, GRPOConfig, ModelConfig, DatasetMixtureConfig)
```

这四件事就是 TRL 的四个正交平面——它们互相不知道对方的存在，只是被拼在同一个命令行里：

| 平面 | 类 | 回答的问题 | 换算法时是否变化 |
| --- | --- | --- | --- |
| **任务平面** | `ScriptArguments` / `<Algo>ScriptArguments` | 数据从哪来、切哪一列、奖励函数叫什么 | 部分（奖励函数只有在线族有） |
| **算法平面** | `<Algo>Config`（继承 `_BaseConfig` ← `TrainingArguments`） | 采样多少、怎么算优势、用什么损失 | ✅ 整体替换 |
| **模型平面** | `ModelConfig` | 多大的模型、什么精度、要不要 LoRA/量化 | ❌ 不变 |
| **运行平面** | `TrainingArguments` + `accelerate_configs/*.yaml` | 几张卡、怎么并行、显存怎么省 | ❌ 不变 |

**这就是最有用的一条抽象**：换算法只动「算法平面 + 任务平面的一句话」，模型与运行平面原样复用。WebUI 的表单结构直接照抄这四层（见第 5 节）。

## 2. 抽象的依据：所有 trainer 共用一条管线

把 `GRPOTrainer`、`RLOOTrainer`、`DPOTrainer`、`SFTTrainer`、`DistillationTrainer` 的方法名排在一起看，骨架是同一个（`trl/trainer/base_trainer.py` 的 `_BaseTrainer` ← `transformers.Trainer`）：

- 构造：`__init__`
- 数据：`_prepare_dataset`（旧 API）/ `get_train_dataloader`、`_set_signature_columns_if_needed`
- 采样与打分（在线族/蒸馏族才有）：`_generate_and_score_completions`、`_generate`、`_calculate_rewards`
- 算损失：`compute_loss` → `_compute_loss`
- 记录：`log`、`prediction_step`
- 产物：`_save_checkpoint`、`save_model`

差别只在**同一位置的实现**：GRPO 的 `_generate_and_score_completions` 做组内标准化，RLOO 做 leave-one-out，DPO 干脆没有这一步而用 `_precompute_ref_log_probs` 预计算参考对数概率。于是可以抽出一条固定的八阶段流水线：

## 3. 八阶段流水线

| # | 阶段 | 输入 → 输出 | 不变的部分 | 随算法变化的部分 |
| --- | --- | --- | --- | --- |
| S0 | **环境与资源** `env` | 配置 → 可运行的进程组 | accelerate 配置、混合精度、梯度检查点、LoRA/量化、vLLM/DeepSpeed/FSDP 选择 | 是否需要 vLLM server、是否需要额外的教师 server |
| S1 | **模型与参考/教师** `model` | 模型 id → 内存中的模型组 | `ModelConfig`：加载、精度、LoRA | 是否加载参考模型（`beta>0`）、教师模型（蒸馏）、奖励模型 |
| S2 | **数据与提示** `data` | 数据集 → 规范化样本 | chat template、截断、padding/packing | 需要的列不同（prompt / 对 / label / messages / privileged_context） |
| S3 | **监督信号** `signal` | 样本 → 标量或分布 | 无（这是算法差异的源头） | 奖励函数 / 奖励模型 / 偏好标签 / 教师分布 |
| S4 | **采样** `rollout` | prompt → N 条补全 | 生成参数（temperature/top_p/top_k/min_p）、vLLM 加速 | 每条 prompt 采样几条、是否多轮工具、是否异步 |
| S5 | **优势/偏好估计** `estimate` | 补全+信号 → 学习信号 | 无 | 组内标准化 / leave-one-out / 隐式奖励差 / 教师-学生分布差 / 离线价值 |
| S6 | **损失与更新** `update` | 学习信号 → 梯度 | 优化器、调度器、裁剪 | 损失函数族（`loss_type`）、KL 项（`beta`）、熵奖励 |
| S7 | **记录与产物** `output` | 指标 → 可交付的模型 | 日志、checkpoint、save_model、push_to_hub | 额外记录的指标（clip_ratio、rewards/margins、蒸馏损失等） |

这条流水线有三条「捷径」需要注意，它们正是三族算法的分界：

- **偏好对齐族跳过 S4/S5**：没有采样，S2 出来的成对样本直接进 S6（Online DPO、TPO 例外，它们补上了 S4）。
- **蒸馏族的 S3 不是标量奖励**，而是教师的分布，S5 变成「对齐两份分布」。
- **S1 是可选的**：`beta=0.0` 时 GRPO 不加载参考模型；ORPO/CPO 也完全不需要参考模型。少一个模型往往就是显存能不能放下的区别。

## 4. 能力开关（Capability flags）

抽象的另一半是给每个算法一张「需要什么」的清单，WebUI 用它决定显示哪些面板、校验哪些必填。声明在 `catalog/algorithms.json` 的 `requires` 字段里，取值 `true` / `false` / `"optional"` / `"required"`：

| 开关 | 含义 | 影响的阶段 |
| --- | --- | --- |
| `reward_funcs` | 需要奖励函数 | S3 |
| `reward_model` | 可选地用一个奖励模型（同时也能当奖励函数挂上） | S3 |
| `ref_model` | 需要参考模型（KL 锚点/隐式奖励） | S1、S6 |
| `teacher` | 需要教师模型（`true` / `"self"` 自我教师 / `"remote"` 远程） | S1、S3 |
| `preference_pairs` | 数据必须是成对的（chosen/rejected） | S2 |
| `generation` | 训练时要生成（有 S4） | S4 |
| `vllm` | 是否/必须用 vLLM（`"optional"` / `"required"`） | S0 |
| `tools` | 支持工具调用/环境交互 | S4 |

## 5. 抽象落到 WebUI

WebUI（`webui/`）不对 21 个算法写 21 套表单，而是把上面的结构直接映射成组件：

| 抽象 | WebUI 里的样子 |
| --- | --- |
| 四个平面 | 新建任务向导的四步：**数据 → 算法 → 模型 → 资源**（顺序按依赖排：算法决定数据契约，模型与资源最后） |
| 八阶段 | 算法步里的参数分组（`stage` 字段），每组标题就是 TRL docstring 里的分组名 |
| `requires` 开关 | 决定「奖励函数」输入框是否出现/必填、是否出现教师模型、是否出现 vLLM 面板 |
| 数据契约 | 数据步按算法给的 `roles` 显示可填的列名，缺失时前端立刻标红 |
| `key_params` / `advanced` | 每个分组默认只显示关键参数，展开能看到全部字段（888 个字段全部来自源码，一个不少） |
| 参数元数据 | 表单控件类型、默认值、可选值、说明文字全部来自 `catalog/catalog.json`，与 `docs/02` 同源 |
| 命令生成 | 提交时按「平面 → 命令行参数」拼出等价于 `trl <algo> ...` 的命令，并写入 `runs/<job>/command.sh` |

参数到命令行的映射规则（与 TRL 的 `HfArgumentParser` 一致）：

1. 参数名即 flag：`--num_generations 8`；
2. `bool` 只在其为 `True` 时传 `--flag`（`False` 一律省略，让 TRL 用自己的默认值）；
3. `list` 用空格分隔（`--reward_funcs accuracy_reward code_reward`），进程启动时用 `shlex.quote` 逐个转义；
4. 未填写的字段**不传**，与「显式传默认值」区分开——这样升级 TRL 后行为跟着新默认走；
5. 只有用户显式写在「额外参数」里的未知 flag 才会被原样透传。

## 6. 每个算法的阶段映射

下面是流水线的紧凑视图（`●` = 该算法在此阶段有实质动作，`○` = 复用/继承，`—` = 不适用）。完整的逐算法文字说明见 [02-algorithm-params.md](02-algorithm-params.md) 的「训练流程映射」小节。

| 算法 | S0 环境 | S1 模型 | S2 数据 | S3 信号 | S4 采样 | S5 估计 | S6 更新 | S7 产物 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GRPO | ○ | ○ | ● | ● 奖励函数 | ● 组采样 | ● 组内标准化 | ● 裁剪+KL | ○ |
| RLOO | ○ | ○ | ○ | ● 奖励函数 | ● k 条 | ● leave-one-out | ● 序列级裁剪 | ○ |
| GMPO | ○ | ○ | ○ | ○ 继承 | ○ | ○ | ● 几何平均 | ○ |
| SDPO | ○ | ● 自我教师 | ○ | ● 奖励函数 | ● 可重提示 | ● 优势+蒸馏权重 | ● 裁剪+蒸馏 | ● 诊断指标 |
| A\*-PO | ○ | ● 参考策略 | ● 全量采样 | ● 奖励函数 | ○ 每题 1 条 | ● 离线 V\* | ● 平方误差回归 | ○ |
| Async GRPO | ● vLLM server | ● 权重同步 | ● 流式投喂 | ● 奖励函数 | ● 后台 worker | ● 组内标准化 | ● 裁剪 | ● 吞吐指标 |
| DPO | ○ | ● 参考模型 | ● 成对数据 | ● 偏好标签 | — | ● 隐式奖励差 | ● 15 种 loss | ● rewards/* |
| KTO | ○ | ● 参考模型 | ● label 列 | ● 二元标签 | — | ● 前景效用+KL | ● kto/apo | ● rewards/kl |
| ORPO | ○ | ● 无参考模型 | ● 成对数据 | ● 偏好标签 | — | ● odds 比 | ● SFT+odds | ○ |
| CPO | ○ | ● 无参考模型 | ● 成对数据 | ● 偏好标签 | — | ● 对比项 | ● 对比+SFT 项 | ○ |
| TPO | ○ | ● 无参考模型 | ● 成对数据 | ● 偏好+自生成 | ● 生成弱样本 | ● 三路对比 | ● TPO 损失 | ○ |
| Online DPO | ○ | ● 奖励模型 | ● prompt | ● 在线 AI 反馈 | ● 2 条 | ● 在线偏好 | ● DPO 损失 | ● 偏好准确率 |
| Distillation | ○ | ● 教师 | ○ | ● 教师分布 | ● 学生生成 | ● 分布差 | ● 广义 JSD | ○ |
| GKD | ○ | ● 教师 | ● 可带补全 | ● 教师分布+补全 | ● 按 lmbda | ● 分布差 | ● JSD/序列 KD | ○ |
| GOLD | ○ | ● 教师+教师 tokenizer | ○ | ● 跨词表 logits | ○ | ● 文本片段对齐 | ● ULD 混合损失 | ○ |
| SDFT | ○ | ● 自我教师 | ● privileged_context | ● 教师分布 | ● 可教师采样 | ● 分布差 | ● 蒸馏损失 | ○ |
| IW-OPD | ○ | ● 教师（可远程） | ○ | ● top-k logprobs | ○ | ● 前缀漂移加权 | ● 蒸馏/策略梯度 | ○ |
| Async / Server Distillation | ● vLLM server | ● 远程教师 | ○ | ● 教师 logprobs | ● 后台 worker | ● 稀疏散度 | ● 同 Distillation | ● 吞吐指标 |
| Reward | ○ | ● 序列分类头 | ● 成对数据 | ● 偏好标签 | — | ● 分数差 | ● pairwise 排序损失 | ● 可复用奖励模型 |
| SFT | ○ | ○ | ● 指令数据 | ● 标准答案 | — | — | ● 交叉熵 | ○ |

## 7. 这套抽象带来的三个直接好处

1. **一张表单跑 21 个算法**：新增算法只需要在 `catalog/algorithms.json` 里加一条 + 重新构建目录，前端不用改。
2. **可校验**：`requires` 决定必填项，数据契约决定列名，`choices` 决定下拉选项——提交前就能拦住「DPO 任务没填 chosen/rejected」「GRPO 没挂奖励函数」这类错误。
3. **可解释**：每个参数旁都带着它在哪个阶段起作用、默认值是多少、TRL 原文怎么描述，训练崩了能立刻定位是哪个阶段的问题（例如 loss 曲线震荡 → S6 的 `epsilon`/`loss_type`；奖励全对全错 → S4 的 `num_generations`/`temperature`；OOM → S0/S1 的资源与参考模型）。
