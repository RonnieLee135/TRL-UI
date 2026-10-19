# TRL 的强化学习算法全景

> 事实来源：本机 `../trl`（`TRL-UI` 的兄弟目录）检出的 **TRL 1.16.0.dev0**，commit `8f39f8fc`（2026-10-14）。
> 所有结论都来自该检出目录里的源码与 `docs/source/*.md`，参数清单见 [02-algorithm-params.md](02-algorithm-params.md)（自动生成）。

## 1. TRL 把「强化学习」分成三族 + 两个辅助任务

TRL 自己的分类标准不是论文名称，而是**训练时要不要采样、要不要奖励、要不要参考模型**。按这个标准看源码目录，21 个 trainer 清楚地分成五类：

| 家族 | 训练时是否采样 | 监督信号从哪来 | 是否需要参考模型 | 成员 |
| --- | --- | --- | --- | --- |
| **在线强化学习**（on-policy RL） | ✅ 当前策略生成 | 奖励函数 / 奖励模型 | 可选（KL 正则） | GRPO、RLOO、GMPO、SDPO、A*-PO、Async GRPO |
| **偏好对齐**（preference optimization） | ❌（Online DPO、TPO 除外） | 成对/带标签的偏好数据 | 多数需要，ORPO/CPO 不需要 | DPO、KTO、ORPO、CPO、TPO、Online DPO |
| **蒸馏**（distillation） | ✅（学生自己生成） | 教师的下一 token 分布 | 不需要参考模型，需要教师 | Distillation、GKD、GOLD、SDFT、IW-OPD、Async/Server Distillation |
| **奖励模型**（辅助） | ❌ | 偏好标签 → 标量打分器 | 否 | Reward（ORM） |
| **监督微调**（辅助） | ❌ | 标准答案 | 否 | SFT |

关键判断：**只有第一族是教科书意义上的 RL**（策略 → 采样 → 奖励 → 优势 → 策略梯度）。第二族是把 RLHF 的目标闭式改写成分类损失，第三族是把奖励换成分布监督。做 RL 训练选型时，先问自己「我有没有一个可执行的判题器或奖励模型」——有就走第一族，只有偏好数据就走第二族，只有一个更强的模型就走第三族。

## 2. 在线强化学习族

### 2.1 共同骨架

在线族的四个 trainer（`GRPOTrainer` / `RLOOTrainer` / `SDPOTrainer` / `AsyncGRPOTrainer`）共享同一套方法名，这也正是后面「训练流程抽象」里各阶段的落点：

| 阶段 | 方法 | 作用 |
| --- | --- | --- |
| 构造 | `__init__` | 建模型/参考模型、奖励函数、优化器；决定 vLLM 走 colocate 还是 server |
| 数据 | `get_train_dataloader`、`_get_train_sampler`、`_tokenize_prompts` | 按 `generation_batch_size × steps_per_generation` 切批次并 tokenize |
| 采样 | `_generate`、`_generate_single_turn`、`_tool_call_loop` | 每条 prompt 采 `num_generations` 条补全，可选工具调用 |
| 打分 | `_calculate_rewards` | 调奖励函数/奖励模型，得到 `(B×G, num_reward_funcs)` 的奖励矩阵 |
| 估计 | `_generate_and_score_completions` 内 | 组内标准化（GRPO）或 leave-one-out 基线（RLOO）算出优势 |
| 更新 | `compute_loss` → `_compute_loss` | 重要性比 + 裁剪 + KL/熵，反向传播 |
| 记录 | `log`、`_log_metric`、`_log_completion_extra` | 指标与补全样例 |
| 产物 | `_save_checkpoint`、`save_model` | checkpoint 与最终模型 |

### 2.2 GRPO：事实上的默认选择

对同一 prompt 采一组（默认 `num_generations=8`）补全，用**组内奖励的相对大小**构造优势，因此不需要 PPO 的价值网络。TRL 里 GRPO 一个 trainer 覆盖了一个「算法族」——通过三个参数切换：

| 变体 | 激活方式 | 动机 |
| --- | --- | --- |
| GRPO（原始） | `loss_type="grpo"` | 按序列长度归一化；有长度偏置，不推荐 |
| **DAPO** | `loss_type="dapo"`（默认） | token 级归一化，消除长度偏置 |
| Dr. GRPO | `loss_type="dr_grpo"` + `scale_rewards="none"` | 除以常数，彻底去长度偏置；同时不按标准差缩放奖励以减少难度偏置 |
| BNPO | `loss_type="bnpo"` | 只在局部 batch 内按有效 token 数归一化 |
| CISPO | `loss_type="cispo"` | 裁剪重要性权重而不是「优势×权重」 |
| SAPO | `loss_type="sapo"` | 用平滑软门控替代硬裁剪 |
| VESPO | `loss_type="vespo"` | 序列级重要性比 + 非对称 Gamma 加权 |
| LUSPO | `loss_type="luspo"` | GSPO 的改造，序列损失按长度缩放（要求 `importance_sampling_level="sequence"`） |
| **GSPO** | `importance_sampling_level="sequence"` | 重要性比在序列级计算 |
| **GDPO** | `multi_objective_aggregation="normalize_then_sum"` | 多奖励先各自归一化再求和，避免奖励间尺度干扰 |
| GMPO | 换 trainer（`GMPOTrainer`） | 几何平均替代算术平均，裁剪区间可以放宽 |

要点：`beta=0.0` 是默认值，即**默认不加 KL 惩罚**（省一份参考模型显存），这也是当前社区（Open-Reasoner-Zero、DAPO 等）的主流做法；要约束漂移就把 `beta` 设成 0.05~0.1。

### 2.3 RLOO：更便宜的 REINFORCE

`num_generations` 就是论文里的 k，优势 = 本样本奖励 − 其余 k−1 条的平均奖励，再乘 PPO 式的序列级裁剪比。默认 `k=2`，生成成本约为 GRPO(8) 的四分之一；代价是没有工具调用能力、基线方差略大。注意 TRL 默认 `beta=0.05>0`（会加载参考模型），复现论文需显式设 `beta=0.0`。

### 2.4 其余在线成员

| 算法 | 一句话 | 什么时候用 |
| --- | --- | --- |
| SDPO | 把自己的成功轨迹当「自我教师」，把带反馈条件下的下一 token 预测蒸馏回来，稀疏奖励变密集信号 | 长 CoT、奖励很稀疏、需要更快收敛 |
| A\*-PO | 先用参考策略离线估计最优价值 `V*`，训练时每题只生成 1 条补全做回归 | 生成是瓶颈、能接受一次离线采样开销 |
| Async GRPO | 后台 rollout worker 持续从 vLLM server 生成，训练循环只消费 | 多卡以上、要压满 GPU 吞吐；接受一定 off-policy |

## 3. 偏好对齐族

| 算法 | 监督信号 | 参考模型 | 生成 | 损失形式 |
| --- | --- | --- | --- | --- |
| DPO | chosen/rejected 成对 | 需要 | 否 | 15 种 `loss_type`（sigmoid/hinge/ipo/…/discopop） |
| KTO | 单样本 desirable/undesirable 二元标签 | 需要（KL 项） | 否 | `kto` / `apo_zero_unpaired` |
| ORPO | 成对 | **不需要** | 否 | SFT 交叉熵 + odds 比惩罚，单步完成 |
| CPO | 成对 | 可不需要 | 否 | 对比损失 + `cpo_alpha` 的 SFT 项；可选 SimPO 式长度归一化 |
| TPO | 成对 + 自生成弱样本 | 不需要 | ✅ | 三路对比（`tpo_alpha`/`tpo_l_gamma`） |
| Online DPO | 在线 AI 反馈（奖励函数/奖励模型） | 可选 | ✅ | DPO 损失，偏好标签在训练中产生 |

DPO 的 `loss_type` 是这一族里最值得展开的参数，它同时吸收了 IPO/CPO/BCo/SimPO 等论文的目标：

```
sigmoid（默认） hinge  ipo  exo_pair  nca_pair  robust  bco_pair
sppo_hard  aot  aot_unpaired  apo_zero  apo_down  discopop  sft  sigmoid_norm
```

配合 `f_divergence_type`（`reverse_kl`/`forward_kl`/`js_divergence`/`alpha_divergence`）还能给策略加 f-散度正则。多损失可以加权组合：`loss_type=["sigmoid","bco_pair"]` + `loss_weights=[0.8,0.2]`。

**KTO 的一个硬约束**：损失里的 KL 项是用同 batch 内的邻居样本配对估计的，所以 `train_sampling_strategy` 必须是 `sequential`（除非用 `apo_zero_unpaired`）。

## 4. 蒸馏族

| 算法 | 教师 | 关键参数 | 特点 |
| --- | --- | --- | --- |
| Distillation（稳定） | 本地模型 | `beta`（0=前向 KL，1=反向 KL，中间插值 JSD）、`distillation_objective`（`jsd`/`iw_opd`） | 学生 on-policy 生成，学教师的下一 token 分布 |
| GKD | 本地模型 | `lmbda`（on/off-policy 比例）、`beta`、`seq_kd` | 经典 GKD，继承 SFTConfig |
| GOLD | 本地模型（**词表可不同**） | `use_uld_loss`、`teacher_tokenizer_name_or_path`、`uld_*` | 跨 tokenizer 的 logit 蒸馏 |
| SDFT | 自己的「特权上下文」版本 | `teacher_model_kind`（base/live/ema）、`distillation_mode` | 持续学习，抗遗忘 |
| IW-OPD | 本地或远程 | `iw_opd_gamma`、`reverse_kl_top_1_mode` | 按前缀漂移加权，缓解位置偏置 |
| Async / Server Distillation | vLLM server | `teacher_server_urls`、`loss_top_k` | 本地不放教师，显存最省 / 吞吐最高 |

一句话选型：教师与学生**同 tokenizer** → `Distillation`；**不同 tokenizer** → `GOLD`；教师太大放不下 → `Server Distillation`；要极限吞吐 → `Async Distillation`。

## 5. 两个辅助任务 + 一个技巧

- **Reward（ORM）**：`trl reward` 训一个打分器，产物就是在线 RL 的 `reward_model_name_or_path`。它的输出是排序分数、不是概率，下游 RL 里要注意 `center_rewards_coefficient` 之类的尺度处理。
- **SFT**：RL 之前把格式拉正的冷启动。TRL 的 SFTConfig 支持 `packing`/`padding_free`/`assistant_only_loss`，长上下文 RL 前的 SFT 基本都用 packing 省显存。
- **BEMA（`BEMACallback`）**：不是独立算法，而是给 DPO/GRPO 的**参考模型**做指数移动平均的技巧（BEMA 论文），参考模型跟着策略慢慢走，缓解 KL 锚点过旧或过新。

## 6. 选型决策树

```
有可执行的判题器 / 规则奖励，或已训好奖励模型？
├── 是 ──► 在线强化学习
│          ├── 显存和吞吐优先、能少采样        ──► RLOO（k=2）
│          ├── 需要多轮工具/环境交互           ──► GRPO（tools/environment_factory）
│          ├── 奖励极稀疏、长 CoT              ──► SDPO
│          ├── 生成是瓶颈                      ──► A*-PO / Async GRPO
│          └── 常规首选                        ──► GRPO（loss_type=dapo，vLLM 加速）
└── 否 ──►
     ├── 只有成对偏好数据 ──► DPO（+ 参考模型）／ORPO（无参考模型、显存最省）
     ├── 只有单样本好坏标签 ──► KTO
     ├── 偏好数据 + 能生成 ──► Online DPO / TPO
     └── 只有一个更强的模型 ──► 蒸馏族（同 tokenizer：Distillation；不同：GOLD；远程：Server Distillation）
```

## 7. 数据契约

不同 trainer 吃的列不一样，换算法时**数据必须跟着换**（TRL 的 `_prepare_dataset` 会直接报缺列）：

| 列 | 用在哪 | 说明 |
| --- | --- | --- |
| `prompt` | GRPO、RLOO、SDPO、A\*-PO、Online DPO、蒸馏族 | 对话列表（推荐）或纯文本 |
| `chosen` / `rejected` | DPO、ORPO、CPO、TPO、Reward | 成对偏好；TRL 会做数据完整性检查与自动 unpair |
| `label`（bool） | KTO | `True` = desirable |
| `messages` | SFT、GOLD | ChatML 消息列表 |
| `privileged_context` | SDFT | 只有教师能看到的额外信息（标准答案/文档） |
| 任意额外列 | GRPO/RLOO 的奖励函数 | 奖励函数通过 `**kwargs` 拿到数据集的所有其他列（如 `answer`、`test_cases`）——`remove_unused_columns=False`（默认）是前提 |

## 8. 奖励函数契约（在线族）

这是把「训练」和「业务」接起来的地方，签名在 `GRPOTrainer._calculate_rewards` 里固定：

```python
def my_reward(prompts, completions, completion_ids, **kwargs) -> list[float | None]:
    # completions[i] 是一条补全（对话格式时是 [{"role":"assistant","content":...}]）
    # kwargs 里是数据集的其他列 + trainer_state + log_extra + log_metric
    # 返回：每个 (prompt, completion) 一个分数；返回 None 表示「这条不可打分」，会被排除出基线
    ...
```

- 支持 **async** 奖励函数（`asyncio` 并发执行），适合调用外部判题服务；
- 一个任务可以挂多个奖励函数，用 `reward_weights` 加权、用 `multi_objective_aggregation` 决定聚合顺序；
- 内置奖励：`accuracy_reward`、`reasoning_accuracy_reward`、`get_cosine_scaled_reward`、`think_format_reward`、`get_repetition_penalty_reward`、`get_soft_overlong_punishment`；
- 脚本里 `--reward_funcs` 既可以写内置名字，也可以写 `my_lib.rewards.code_reward` 这样的点分导入路径（`trl/scripts/grpo.py` 会把当前工作目录加进 `sys.path`）；
- 奖励函数在 **rollout worker 里执行**，写坏了不会让训练报错，只会安静地把模型带偏——这是 RL 训练最常见的坑。

## 9. 从 TRL v0 迁移过来会踩的点

依据本机检出里的 `MIGRATION.md`（v0 → v1）与源码核对：

- **PPO 已经不在 TRL v1 里**：整个仓库搜不到 `PPOConfig`/`PPOTrainer`，在线 RL 的入口只剩 GRPO/RLOO 及其变体。需要经典 PPO 得换框架。
- **默认值变了**：`GRPOConfig.vllm_mode` 与 `RLOOConfig.vllm_mode` 的默认从 `"server"` 改成 `"colocate"`——`use_vllm=True` 且不写明模式时，vLLM 现在跑在同一进程内，不再是连一个独立 server。
- **选项改名**：`packing="bfd-requeue"` → `"bfd_split"`。
- **实验性 trainer 不在 CLI 里**：CLI 只有 `distillation / dpo / grpo / kto / reward / rloo / sft` 七个子命令；`GKDConfig`、`GOLDTrainer`、`SDPOConfig`、`ORPOTrainer`、`CPOTrainer`、`TPOTrainer`、`OnlineDPOTrainer`、`Async*` 等都在 `trl.experimental.*` 下，只能用 Python API 启动。
- **不再自动清理数据集里的 `None`**：SFT/DPO/Reward 不再对样本做 `remove_none_values`；老数据集（`datasets<4.7.0` 或没用 Json 特征保存过）需要自己
  `dataset.with_transform(remove_none_values)`。
