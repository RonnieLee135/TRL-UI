# TRL 算法入参总表（自动生成）

> 本文件由 `tools/gen_params_doc.py` 从 `catalog/catalog.json` 生成，后者又由 `tools/extract_source_catalog.py` 直接解析 TRL 源码得到，**请勿手工编辑**。

- TRL 版本：`1.16.0.dev0`（commit `8f39f8fc`，Wed Oct 14 17:18:02 2026 -0400）
- 生成时间：2026-10-17T12:17:35Z
- 算法数：21，参数条目：888，通用训练参数：36


每个参数的取值来源：类型/默认值来自 `field(...)` 声明，说明来自 `metadata['help']` 或类 docstring 的 `Parameters:` 段落，分组标题直接采用 docstring 里的 `> Parameters that control ...`。`stage` 列是这些分组映射到[训练流程抽象](03-training-workflow.md)的阶段。


## 算法总览

| 算法 | 全称 | 家族 | 状态 | 配置类 | 参数数（关键/弃用） | 启动方式 |
| --- | --- | --- | --- | --- | --- | --- |
| [GRPO](#grpo) | Group Relative Policy Optimization | 在线强化学习 | 稳定 | `GRPOConfig` | 85（19/1） | `trl grpo` |
| [RLOO](#rloo) | REINFORCE Leave-One-Out | 在线强化学习 | 稳定 | `RLOOConfig` | 59（10/1） | `trl rloo` |
| [GMPO](#gmpo) | Geometric-Mean Policy Optimization | 在线强化学习 | 实验 | `GMPOConfig` | 85（5/1） | Python API（实验） |
| [SDPO](#sdpo) | Self-Distillation Policy Optimization | 在线强化学习 | 实验 | `SDPOConfig` | 69（13/0） | Python API（实验） |
| [A*-PO](#a2po) | Optimal Advantage Regression Policy Optimization | 在线强化学习 | 实验 | `A2POConfig` | 19（6/0） | Python API（实验） |
| [Async GRPO](#async_grpo) | Asynchronous GRPO | 在线强化学习 | 实验 | `AsyncGRPOConfig` | 37（8/0） | Python API（实验） |
| [DPO](#dpo) | Direct Preference Optimization | 偏好对齐 | 稳定 | `DPOConfig` | 32（13/0） | `trl dpo` |
| [KTO](#kto) | Kahneman-Tversky Optimization | 偏好对齐 | 稳定 | `KTOConfig` | 25（6/0） | `trl kto` |
| [ORPO](#orpo) | Odds Ratio Preference Optimization | 偏好对齐 | 实验 | `ORPOConfig` | 17（4/0） | Python API（实验） |
| [CPO](#cpo) | Contrastive Preference Optimization | 偏好对齐 | 实验 | `CPOConfig` | 21（8/0） | Python API（实验） |
| [TPO](#tpo) | Triple Preference Optimization | 偏好对齐 | 实验 | `TPOConfig` | 19（6/0） | Python API（实验） |
| [Online DPO](#online_dpo) | Online Direct Preference Optimization | 偏好对齐 | 实验 | `OnlineDPOConfig` | 38（6/0） | Python API（实验） |
| [Distillation](#distillation) | On-policy Knowledge Distillation | 蒸馏 | 稳定 | `DistillationConfig` | 44（6/0） | `trl distillation` |
| [GKD](#gkd) | Generalized Knowledge Distillation | 蒸馏 | 实验 | `GKDConfig` | 36（6/0） | Python API（实验） |
| [GOLD](#gold) | General Online Logit Distillation | 蒸馏 | 实验 | `GOLDConfig` | 73（11/0） | Python API（实验） |
| [SDFT](#sdft) | Self-Distilled Fine-Tuning | 蒸馏 | 实验 | `SDFTConfig` | 51（9/0） | Python API（实验） |
| [IW-OPD](#iw_opd) | Importance-Weighted On-Policy Distillation | 蒸馏 | 实验 | `IWOPDConfig` | 48（8/0） | Python API（实验） |
| [Async Distillation](#async_distillation) | Asynchronous On-policy Distillation | 蒸馏 | 实验 | `AsyncDistillationConfig` | 36（7/0） | Python API（实验） |
| [Server Distillation](#server_distillation) | Teacher-server On-policy Distillation | 蒸馏 | 实验 | `ServerDistillationConfig` | 48（6/0） | Python API（实验） |
| [Reward](#reward) | Outcome Reward Model (ORM) | 奖励模型 | 稳定 | `RewardConfig` | 18（4/0） | `trl reward` |
| [SFT](#sft) | Supervised Fine-Tuning | 监督微调 | 稳定 | `SFTConfig` | 28（8/0） | `trl sft` |

## 通用训练参数

这些是 `transformers.TrainingArguments` 的字段，TRL 自己不重新声明，但对任何算法都生效；WebUI 把它们归到「训练资源」面板。留空即不传给训练脚本，使用框架默认值。

| 参数 | 类型 | 阶段 | 默认 | 证据 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `output_dir` | str | 记录与产物 | — | `docs/source/grpo_trainer.md` | 模型与 checkpoint 的输出目录；RL 训练必填，通常与任务 id 对齐。 |
| `run_name` | str | None | 记录与产物 | — | `docs/source/rloo_trainer.md` | 实验名，用于日志平台（trackio/wandb/tensorboard）上的运行标识。 |
| `report_to` | str | list[str] | 记录与产物 | — | `trl/experimental/async_distillation/async_distillation_trainer.py` | 日志后端，如 trackio、wandb、tensorboard、none。 |
| `seed` | int | 环境与资源 | — | `trl/experimental/async_distillation/async_distillation_trainer.py` | 全局随机种子。 |
| `num_train_epochs` | float | 环境与资源 | — | `docs/source/distillation_trainer.md` | 训练轮数；RL 训练通常 1 轮左右，用 max_steps 更直观。 |
| `max_steps` | int | 环境与资源 | — | `trl/experimental/async_distillation/async_distillation_trainer.py` | 总优化步数；设了它 num_train_epochs 就被忽略。 |
| `per_device_train_batch_size` | int | 环境与资源 | — | `trl/experimental/a2po/a2po_trainer.py` | 单卡训练 batch；GRPO 系要满足 有效 batch = per_device × 进程数 × 梯度累积 能被 num_generations 整除。 |
| `per_device_eval_batch_size` | int | 环境与资源 | — | `trl/experimental/sdft/sdft_config.py` | 单卡评估 batch。 |
| `gradient_accumulation_steps` | int | 环境与资源 | — | `trl/experimental/async_distillation/async_distillation_trainer.py` | 梯度累积步数，用来把小 batch 拼成有效的组大小。 |
| `lr_scheduler_type` | str | 损失与更新 | — | `docs/source/lora_without_regret.md` | 学习率调度器，如 linear、cosine、constant。 |
| `warmup_steps` | float | 损失与更新 | — | `docs/source/lora_without_regret.md` | 预热步数（可为比例）。 |
| `warmup_ratio` | float | 损失与更新 | — | `docs/source/paper_index.md` | 预热占总步数的比例。 |
| `weight_decay` | float | 损失与更新 | — | `examples/grpo_sudoku/grpo_sudoku.py` | 权重衰减。 |
| `max_grad_norm` | float | 损失与更新 | — | `docs/source/lora_without_regret.md` | 梯度裁剪范数，RL 训练里常用 1.0。 |
| `optim` | str | 损失与更新 | — | `trl/experimental/a2po/a2po_trainer.py` | 优化器类型，如 adamw_torch、adamw_bnb_8bit；长补全显存紧张时用 8bit。 |
| `fp16` | bool | 环境与资源 | `False` | `trl/trainer/base_config.py` | 半精度训练；与 bf16 互斥（bf16 未显式设置时由 fp16 推导）。 |
| `label_smoothing_factor` | float | 损失与更新 | — | `trl/trainer/sft_trainer.py` | 标签平滑系数。 |
| `gradient_checkpointing_kwargs` | dict | str | 环境与资源 | — | `trl/experimental/cpo/cpo_trainer.py` | 梯度检查点参数，如 {"use_reentrant": false}。 |
| `dataloader_num_workers` | int | 环境与资源 | — | `trl/experimental/gold/gold_trainer.py` | 数据加载进程数。 |
| `dataloader_pin_memory` | bool | 环境与资源 | — | `trl/experimental/gold/gold_trainer.py` | 锁页内存加速 H2D 拷贝。 |
| `dataloader_drop_last` | bool | 环境与资源 | — | `trl/experimental/gold/gold_trainer.py` | 丢弃最后一个不完整 batch。 |
| `average_tokens_across_devices` | bool | 环境与资源 | — | `trl/trainer/sft_trainer.py` | 跨设备平均 token 数，多卡下让损失尺度一致。 |
| `accelerator_config` | dict | str | 环境与资源 | — | `trl/experimental/async_distillation/async_distillation_config.py` | accelerate 配置（如 gradient_accumulation_plugin）。 |
| `deepspeed` | str | 环境与资源 | — | `docs/source/clis.md` | DeepSpeed 配置文件路径；TRL 自带 trl/accelerate_configs/zero{1,2,3}.yaml 可配合 accelerate 使用。 |
| `fsdp` | str | list | 环境与资源 | — | `docs/source/clis.md` | FSDP 分片策略；TRL 提供 fsdp1/fsdp2 accelerate 配置。 |
| `ddp_find_unused_parameters` | bool | 环境与资源 | — | `trl/trainer/distillation_trainer.py` | DDP 下查找未使用参数；有多个可选分支（如只训部分模块）时设为 False/True 视报错而定。 |
| `save_strategy` | str | 记录与产物 | — | `trl/experimental/async_distillation/async_distillation_trainer.py` | 保存策略：no / steps / epoch。 |
| `save_steps` | int | 记录与产物 | — | `docs/source/online_dpo_trainer.md` | 每多少步保存一次 checkpoint。 |
| `save_total_limit` | int | 记录与产物 | — | `docs/source/jobs_training.md` | 最多保留几个 checkpoint，防止磁盘被吃满。 |
| `eval_strategy` | str | 记录与产物 | — | `trl/experimental/sdft/sdft_config.py` | 评估策略：no / steps / epoch。 |
| `eval_steps` | int | 记录与产物 | — | `docs/source/sdpo_trainer.md` | 每多少步评估一次。 |
| `logging_strategy` | str | 记录与产物 | — | `trl/trainer/base_config.py` | 日志策略：steps / epoch。 |
| `push_to_hub` | bool | 记录与产物 | `False` | `trl/experimental/merge_model_callback.py` | 训练结束后推送模型到 Hub（需要 token）。 |
| `hub_model_id` | str | 记录与产物 | — | `trl/experimental/cpo/cpo_trainer.py` | 推送目标仓库 id。 |
| `hub_private_repo` | bool | 记录与产物 | — | `trl/trainer/grpo_trainer.py` | 推送的仓库设为私有。 |
| `resume_from_checkpoint` | str | bool | 记录与产物 | — | `trl/scripts/distillation.py` | 从 checkpoint 续训。 |

## <a id="grpo"></a>GRPO（`grpo`）

- **全称**：Group Relative Policy Optimization（在线强化学习，稳定）
- **配置类**：`GRPOConfig` @ `trl/trainer/grpo_config.py`（继承链：`GRPOConfig` → `_BaseConfig`）
- **训练器**：`GRPOTrainer` @ `trl/trainer/grpo_trainer.py`
- **文档**：`docs/source/grpo_trainer.md` · 论文：<https://huggingface.co/papers/2402.03300>
- **启动**：`trl grpo`
- **脚本参数**：`reward_funcs`, `reward_model_name_or_path`
- **数据契约**：prompt；prompt（对话列表或纯文本，列名 prompt 可由 remove_unused_columns=False 时的其他列补充）

> 对每个 prompt 采样一组（num_generations 条）补全，用组内奖励的相对大小构造优势，去掉 PPO 的价值网络（critic），显存与调参成本大幅下降，是当前开源 RLHF/RLVR 的默认选择。


### 训练流程映射

| 阶段 | 该算法在此阶段做什么 |
| --- | --- |
| 环境与资源 | TrainingArguments + accelerate 配置（单卡/多卡/DeepSpeed ZeRO-3）；可选 vLLM colocate 或 server 模式加速生成。 |
| 模型与参考/教师 | 策略模型；beta=0.0（默认）时不加载参考模型，显存更省；disable_dropout 保证 logp 可复现。 |
| 数据与提示 | prompt 数据集按 generation_batch_size × steps_per_generation 切分，可 shuffle_dataset。 |
| 监督信号 | reward_funcs：内置（accuracy_reward / reasoning_accuracy_reward / think_format_reward / get_soft_overlong_punishment）、任意 dotted path 的 Python 函数，或一个奖励模型路径。 |
| 采样（rollout） | 每组采样 num_generations 条，max_completion_length 截断，temperature/top_p/top_k/min_p 控制多样性；支持工具调用与 rollout_func 自定义。 |
| 优势/偏好估计 | 组内标准化优势：scale_rewards=group 时用组内标准差，读取多奖励时按 reward_weights 聚合（multi_objective_aggregation）。 |
| 损失与更新 | 按 loss_type 选择聚合方式（grpo/dapo/dr_grpo/bnpo/cispo/sapo/luspo/vespo），重要性比裁剪 epsilon/epsilon_high，beta>0 时加 KL 项，可选熵奖励。 |
| 记录与产物 | completion 日志、clip_ratio/entropy/reward 指标、checkpoint 与 save_model/push_to_hub。 |

### 参数


**algorithm · 算法其他参数**

| 参数 | 类型 | 默认 | 关键 | 可选值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `logging_steps` | float | `10` |  | — | Log every X updates steps. Should be an integer or a float in range `[0,1)`. If smaller than 1, will be interpreted as ratio of total training steps. |
| `gradient_checkpointing` | bool | `True` |  | — | Enable gradient checkpointing to trade compute for memory. Reduces memory at the cost of ~20%% slower training. |
| `bf16` | bool | None | `None` |  | — | Whether to use bf16 (mixed) precision instead of 32-bit. Requires Ampere or higher NVIDIA architecture or Intel XPU or using CPU (use_cpu) or Ascend NPU. If not set, it defaults to `True` if `fp16` is not set. |
| `lr_scheduler_kwargs` | dict | str | None | `None` |  | — | Additional parameters for the lr_scheduler, such as {'num_cycles': 1} for cosine with hard restarts. See the documentation of each scheduler for possible values. |
| `use_liger_kernel` | bool | `False` |  | — | Enable Liger Kernel optimizations: transformers patches the model's layers with Liger's Triton kernels. Deprecated in `SFTTrainer`, `DPOTrainer`, `KTOTrainer`, `GRPOTrainer` and `RLOOTrainer`, and will be removed in v2.0.0: use `model_init… |
| `torch_empty_cache_steps` | int | None | `None` |  | — | Number of steps to wait before calling `torch.<device>.empty_cache()`. Helps avoid CUDA OOM at a cost of ~10%% slower performance. If None, cache will not be emptied. |

**损失与更新 · 训练超参**

| 参数 | 类型 | 默认 | 关键 | 可选值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `learning_rate` | float | `1e-06` |  | — | The initial learning rate for AdamW. |

**模型与参考/教师 · Parameters that control the model and reference model**

| 参数 | 类型 | 默认 | 关键 | 可选值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `model_init_kwargs` | dict[str, Any] | str | None | `None` |  | — | Keyword arguments for `transformers.AutoModelForCausalLM.from_pretrained`, used when the `model` argument of the `GRPOTrainer` is provided as a string. The `revision` value is also used when loading processing classes. |
| `trust_remote_code` | bool | `False` |  | — | Whether to allow loading models and tokenizers that ship custom Python code from the Hub. Forwarded to `AutoModelForCausalLM.from_pretrained` and `AutoProcessor.from_pretrained`. Also applied to reward-model and reward-tokenizer loads. |
| `router_aux_loss_coef` | float | None | `None` |  | — | Coefficient of the load-balancing auxiliary loss for Mixture-of-Experts (MoE) models, added to the training loss with this weight. When not set, the value declared by the model config is used. Set to `0.0` to disable it. Fails when used wi… |
| `disable_dropout` | bool | `False` |  | — | Whether to disable dropout in the model. This is useful for training with a reference model, as it prevents the model from generating different logprobs for the same input. |
| `cast_lm_head_to_fp32` | bool | `False` |  | — | Whether to cast the language modeling head of the policy and reference models to float32. As recommended by the [ScaleRL](https://huggingface.co/papers/2510.13786) recipe. This flag is only supported when the model has untied word embeddin… |

**数据与提示 · Parameters that control the data preprocessing**

| 参数 | 类型 | 默认 | 关键 | 可选值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `remove_unused_columns` | bool | None | `False` |  | — | Whether to only keep the column 'prompt' in the dataset. If you use a custom reward function that requires any column other than 'prompts' and 'completions', you should keep this to `False`. |
| `num_generations` | int | None | `8` | ★ | — | Number of generations to sample. The effective batch size (num_processes * per_device_batch_size * gradient_accumulation_steps) must be evenly divisible by this value. |
| `num_generations_eval` | int | None | `None` |  | — | Number of generations to sample during evaluation. This allows using fewer generations during evaluation to save computation. If `None`, uses the value of `num_generations`. |
| `max_completion_length` | int | None | `512` | ★ | — | Maximum length of the generated completion. |
| `ds3_gather_for_generation` | bool | `True` |  | — | This setting applies to DeepSpeed ZeRO-3. If enabled, the policy model weights are gathered for generation, improving generation speed. However, disabling this option allows training models that exceed the VRAM capacity of a single GPU, al… |
| `shuffle_dataset` | bool | None | `True` |  | — | Whether to shuffle the training dataset. |
| `pad_to_multiple_of` | int | str | None | `auto` |  | — | If set, the prompts ids and completions ids will be padded to a multiple of this value. `'auto'` pads to a multiple of 128 on Neuron (AWS Trainium), which compiles once per input shape, and does not pad elsewhere. |

**采样 · Parameters that control generation**

| 参数 | 类型 | 默认 | 关键 | 可选值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `generation_batch_size` | int | None | `None` |  | — | Batch size to use for generation. If `None`, it defaults to the effective training batch size: `per_device_train_batch_size * num_processes * steps_per_generation`. |
| `steps_per_generation` | int | None | `None` | ★ | — | Number of steps per generation. If `None`, it defaults to `gradient_accumulation_steps`. |
| `temperature` | float | `1.0` | ★ | — | Temperature for sampling. The higher the temperature, the more random the completions. |
| `top_p` | float | `1.0` | ★ | — | Float that controls the cumulative probability of the top tokens to consider. Must be in (0, 1]. Set to 1.0 to consider all tokens. |
| `top_k` | int | `0` |  | — | Number of highest probability vocabulary tokens to keep for top-k-filtering. If `0`, top-k-filtering is disabled and all tokens are considered. |
| `min_p` | float | None | `None` |  | — | Minimum token probability, which will be scaled by the probability of the most likely token. It must be a value between 0.0 and 1.0. Typical values are in the 0.01-0.2 range. |
| `generation_kwargs` | dict | str | None | `None` |  | — | Additional keyword arguments to pass to `GenerationConfig` (if using transformers) or `SamplingParams` (if using vLLM) when sampling completions. This can be used to further customize the generation behavior, such as setting `suppress_toke… |
| `chat_template_kwargs` | dict | str | None | `None` |  | — | Additional keyword arguments to pass to the `apply_chat_template` function when generating completions. |
| `repetition_penalty` | float | `1.0` |  | — | Float that penalizes new tokens based on whether they appear in the prompt and the generated text so far. Values > 1.0 encourage the model to use new tokens, while values < 1.0 encourage the model to repeat tokens. |
| `cache_implementation` | str | None | `None` |  | — | Implementation of the cache method for faster generation when use_vllm is set to False. |

**环境与资源 · Parameters that control generation acceleration powered by vLLM**

| 参数 | 类型 | 默认 | 关键 | 可选值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `use_vllm` | bool | `False` | ★ | — | Whether to use vLLM for generating completions. If set to `True`, the trainer will use vLLM for generation instead of the default model.generate(). Requires `vllm` to be installed. |
| `vllm_mode` | str | `colocate` | ★ | `colocate`, `server` | Mode to use for vLLM integration when `use_vllm` is set to `True`. Must be one of `'server'` or `'colocate'`. `'server'`: The trainer will send generation requests to a separate vLLM server. Make sure a vLLM server is running (start with `… |
| `vllm_model_impl` | str | `vllm` |  | `vllm`, `transformers` | Model implementation to use for vLLM. Must be one of `transformers` or `vllm`. `transformers`: Use the `transformers` backend for model implementation. `vllm`: Use the `vllm` library for model implementation. |
| `vllm_enable_sleep_mode` | bool | `False` |  | — | Enable vLLM sleep mode to offload weights/cache during the optimizer step. Keeps GPU memory usage low, but waking the engine adds host–device transfer latency. |
| `vllm_structured_outputs_regex` | str | None | `None` |  | — | Regex for vLLM structured outputs. If `None` (default), structured outputs is disabled. |

**环境与资源 · Parameters that control the vLLM server (only used when `vllm_mode` is `"server"`)**

| 参数 | 类型 | 默认 | 关键 | 可选值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `vllm_server_base_url` | str | None | `None` |  | — | Base URL for the vLLM server (e.g., 'http://localhost:8000'). If provided, `vllm_server_host` and `vllm_server_port` are ignored. |
| `vllm_server_host` | str | `0.0.0.0` |  | — | Host of the vLLM server to connect to. Ignored if vllm_server_base_url is provided. |
| `vllm_server_port` | int | `8000` |  | — | Port of the vLLM server to connect to. Ignored if vllm_server_base_url is provided. |
| `vllm_server_timeout` | float | `240.0` |  | — | Total timeout duration in seconds to wait for the vLLM server to be up. If the server is not up after the timeout, a `ConnectionError` is raised. |
| `vllm_group_port` | int | `51216` |  | — | Port number for the weight update group. This is used to communicate with the vLLM server. Unless the port is occupied, there is no need to change it. |

**环境与资源 · Parameters that control colocated vLLM execution (only used when `vllm_mode` is `"colocate"`)**

| 参数 | 类型 | 默认 | 关键 | 可选值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `vllm_gpu_memory_utilization` | float | `0.3` |  | — | Control the GPU memory utilization for vLLM. This setting only applies when `vllm_mode` is set to `'colocate'`. If you are using `vllm_mode='server'`, this parameter must be passed separately when launching the vLLM server via the `--vllm_… |
| `vllm_max_model_length` | int | None | `None` |  | — | Context window for vLLM. Set it to at least the maximum prompt length in the dataset plus `max_completion_length`; if omitted, it is inferred from the model config. |
| `vllm_tensor_parallel_size` | int | `1` |  | — | Control the tensor parallel size for vLLM. This setting only applies when `vllm_mode` is set to `'colocate'`. If you are using `vllm_mode='server'`, this parameter must be passed separately when launching the vLLM server via the `--vllm_te… |

**损失与更新 · Parameters that control the training**

| 参数 | 类型 | 默认 | 关键 | 可选值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `beta` | float | `0.0` | ★ | — | KL coefficient. If `0.0` (default), the reference model is not loaded, reducing memory usage and improving training speed. [DeepSeek-R1 incentivizes reasoning in LLMs through reinforcement learning](https://huggingface.co/papers/2501.12948… |
| `num_iterations` | int | `1` | ★ | — | Number of iterations per batch (denoted as μ in the algorithm). |
| `epsilon` | float | `0.2` | ★ | — | Epsilon value for clipping. |
| `delta` | float | None | `None` |  | — | Enables the upper clipping bound in two-sided GRPO loss when set to a float. If `None` (default), standard GRPO clipping is used. Recommended to be greater than `1 + ε` when enabled. This method is introduced in the [INTELLECT-2 tech repor… |
| `epsilon_high` | float | None | `None` | ★ | — | Upper-bound epsilon value for clipping. If not specified, it defaults to the same value as the lower-bound specified in argument `epsilon`. Paper DAPO recommends `0.28`. When used with `loss_type='cispo'`, this corresponds to the ε_max par… |
| `sapo_temperature_neg` | float | `1.05` |  | — | Temperature for tokens with non-positive advantage scores used in the `sapo` loss function. This parameter is introduced in the [Soft Adaptive Policy Optimization paper](https://huggingface.co/papers/2511.20347). |
| `sapo_temperature_pos` | float | `1.0` |  | — | Temperature for tokens with positive advantage scores used in the `sapo` loss function. This parameter is introduced in the [Soft Adaptive Policy Optimization paper](https://huggingface.co/papers/2511.20347). |
| `vespo_k_pos` | float | `2.0` |  | — | k parameter for positive advantages, it is the power exponent in the VESPO loss. Controls how aggressively we down-weight samples with low importance weights (when the importance sampling ratio < 1). |
| `vespo_lambda_pos` | float | `3.0` |  | — | lambda parameter for positive advantages, it is the decay factor in the VESPO loss. Controls how aggressively we down-weight samples with high importance weights (when the importance sampling ratio > 1). |
| `vespo_k_neg` | float | `3.0` |  | — | k parameter for negative advantages, it is the power exponent in the VESPO loss. Controls how aggressively we down-weight samples with low importance weights (when the importance sampling ratio < 1). |
| `vespo_lambda_neg` | float | `2.0` |  | — | lambda parameter for negative advantages, it is the exponential decay factor in the VESPO loss. Controls how aggressively we down-weight samples with high importance weights (when the importance sampling ratio > 1). |
| `importance_sampling_level` | str | `token` | ★ | `token`, `sequence` | Controls whether importance sampling ratios are computed at the `'token'` or `'sequence'` level. `'token'` keeps the raw per-token log-probability ratios (one weight per token). `'sequence'` averages the log-probability ratios across valid… |
| `reward_weights` | list[float] | None | `None` | ★ | — | Weights for each reward function. Must match the number of reward functions. If `None`, all rewards are weighted equally with weight `1.0`. |
| `multi_objective_aggregation` | str | `sum_then_normalize` | ★ | `sum_then_normalize`, `normalize_then_sum` | Method to aggregate multiple reward functions. Supported values are: `'sum_then_normalize'` (default): First sums the weighted rewards from each reward function, then applies reward scaling/normalization as specified by `scale_rewards` (se… |
| `scale_rewards` | str | `group` | ★ | `group`, `batch`, `none` | Specifies the scaling strategy for rewards. Supported values are: `True` or `group'` (default): rewards are scaled by the standard deviation within each group, ensuring unit variance within a group. `'batch'`: rewards are scaled by the sta… |
| `loss_type` | str | `dapo` | ★ | `grpo`, `dr_grpo`, `dapo`, `bnpo`, `cispo`, `sapo`, `luspo`, `vespo` | Specifies the loss formulation to use. Supported values are 'grpo', 'dr_grpo', 'dapo', 'bnpo', 'cispo', 'sapo', 'luspo', and 'vespo'. 'grpo': Aggregates token-level losses by normalizing over sequence length. Not recommended due to length… |
| `mask_truncated_completions` | bool | `False` | ★ | — | When enabled, truncated completions are excluded from the loss calculation, preventing them from being incorrectly penalized and introducing noise during training. According to the DAPO paper, this is a good practice for training stability. |
| `sync_ref_model` | bool | `False` |  | — | Whether to synchronize the reference model with the active model every `ref_model_sync_steps` steps, using the `ref_model_mixup_alpha` parameter. |
| `ref_model_mixup_alpha` | float | `0.6` |  | — | α parameter from the TR-DPO paper, which controls the mix between the current policy and the previous reference policy during updates. The reference policy is updated according to the equation: `π_ref = α * π_θ + (1 - α) * π_ref_prev`. To… |
| `ref_model_sync_steps` | int | `512` |  | — | τ parameter from the TR-DPO paper, which determines how frequently the current policy is synchronized with the reference policy. To use this parameter, you must set `sync_ref_model=True`. |
| `top_entropy_quantile` | float | `1.0` |  | — | ρ parameter from Beyond the 80/20 Rule. Keeps in the policy loss term only the top-ρ quantile of tokens by entropy of the probability distribution at each sequence position, improving results. Range: [0.0-1.0]. A value of `0.0` masks all b… |
| `entropy_coef` | float | `0.0` | ★ | — | Coefficient of the entropy regularization term in the loss. A positive value adds an entropy bonus that encourages exploration. The bonus is always the mean per-token entropy regardless of `loss_type` (not rescaled to a loss type's policy… |
| `use_adaptive_entropy` | bool | `False` |  | — | Whether to use adaptive entropy control, introduced in Skywork-OR1 (https://huggingface.co/papers/2505.22312). When enabled, `entropy_coef` is incremented by `entropy_coef_delta` when entropy is below `entropy_target`, and decremented othe… |
| `entropy_coef_min` | float | `0.0` |  | — | Lower bound for the entropy coefficient when using adaptive entropy control. |
| `entropy_coef_max` | float | `1.0` |  | — | Upper bound for the entropy coefficient when using adaptive entropy control. |
| `entropy_coef_delta` | float | `0.005` |  | — | Step size for adjusting the entropy coefficient during adaptive entropy control. |
| `entropy_target` | float | `0.2` |  | — | Target mean per-token entropy (nats) for adaptive entropy control. The coefficient is only applied when current entropy is at or below this value. Measured over the same token set as the policy loss (all completion tokens, or the high-entr… |
| `max_tool_calling_iterations` | int | None | `None` |  | — | Maximum number of tool-calling turns when training an agent. If `None`, there is no limit and generation stops when the model generates a response turn with no tool calls or when the total response length reaches `max_model_length`. |
| `vllm_importance_sampling_correction` | bool | `True` |  | — | Whether to apply Importance Sampling (IS) to correct for the mismatch between vLLM completion logprobs and recomputed training logprobs. If set to `False`, no IS is applied regardless of `vllm_importance_sampling_mode`. When `True`, the se… |
| `vllm_importance_sampling_mode` | str | `sequence_mask` |  | — | Specifies how Importance Sampling (IS) is performed when vllm_importance_sampling_correction=True. Modes are defined along two orthogonal dimensions: (1) constraint, which determines how to handle ratios outside the bounds [C_min, C_max]—e… |
| `vllm_importance_sampling_clip_max` | float | None | `3.0` |  | — | Importance sampling upper bound C_max used by `vllm_importance_sampling_mode`. For '*_truncate' modes, ratios are clipped from above at C_max. For '*_mask' modes, ratios larger than C_max are set to zero. |
| `vllm_importance_sampling_clip_min` | float | None | `None` |  | — | Importance sampling lower bound C_min used by `vllm_importance_sampling_mode`. For `*_truncate` modes, ratios are clipped from below at C_min. For `*_mask` modes, ratios below C_min are set to zero. To strictly mask ratios below C_min with… |
| `off_policy_mask_threshold` | float | None | `None` |  | — | Threshold for off-policy sequence masking. If `None`, off-policy sequence masking is disabled. When set, sequences with negative advantages and high KL divergence are masked out to stabilize training. This parameter corresponds to the `del… |
| `use_bias_correction_kl` | bool | `True` |  | — | Whether to multiply the KL term by the importance sampling ratio, so that the KL gradient becomes the unbiased reverse-KL gradient, as described in the [DeepSeek-V3.2 paper](https://huggingface.co/papers/2512.02556). This changes the KL gr… |

**记录与产物 · Parameters that control the logging**

| 参数 | 类型 | 默认 | 关键 | 可选值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `log_completions` | bool | `False` | ★ | — | Whether to log a sample of (prompt, completion) pairs every `logging_steps` steps. If `rich` is installed, it prints the sample. If `wandb` logging is enabled, it logs it to `wandb`. |
| `log_multimodal` | bool | `True` |  | — | Whether to log multimodal content (images, videos, etc.) together with completions. Disable this to reduce log size when using high-resolution multimodal data. |
| `num_completions_to_print` | int | None | `None` |  | — | Number of completions to print with `rich`. If `None`, all completions are logged. |
| `log_unique_prompts` | bool | `False` |  | — | Whether to log unique prompts. If `True`, only unique prompts are logged. If `False`, all prompts are logged. |
| `log_completions_hub_repo` | str | None | `None` |  | — | Hugging Face Hub repository to save the completions. Should be a complete repository name like `'username/reponame'` or `'orgname/reponame'`, or just `'reponame'` in which case the repository will be created in the currently-logged-in Hugg… |

**环境与资源 · Parameters that control generation acceleration powered by transformers continuous batching**

| 参数 | 类型 | 默认 | 关键 | 可选值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `use_transformers_continuous_batching` | bool | `False` |  | — | Whether to use transformers' continuous batching engine for generating completions. Requires transformers>=5.8.0. |
| `transformers_continuous_batching_config` | dict | str | None | `None` |  | — | Keyword arguments for `transformers.generation.ContinuousBatchingConfig`. |

**已弃用 · 已弃用**

| 参数 | 类型 | 默认 | 关键 | 可选值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `use_transformers_paged` | bool | `False` | 弃用 | — | Deprecated. Use `use_transformers_continuous_batching` instead. |

**损失与更新 · vLLM 重要性采样**

| 参数 | 类型 | 默认 | 关键 | 可选值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `vllm_importance_sampling_cap` | float | None | `None` |  | — | Deprecated, use `vllm_importance_sampling_clip_max` instead. Importance sampling cap C used by `vllm_importance_sampling_mode`. For '*_truncate' modes, ratios are clipped from above at C. For '*_mask' modes, ratios larger than C are set to… |

### 同一配置内的算法变体

- `loss_type`：
  - grpo (长度归一化，偏短回答)
  - dapo (默认，token 级归一化，去长度偏置)
  - dr_grpo (除以常数，Dr. GRPO 去偏)
  - bnpo (局部 batch 同 token 数归一化)
  - cispo (裁剪重要性权重而非优势)
  - sapo (软门控替代硬裁剪)
  - luspo (序列级 + 长度缩放，需 importance_sampling_level=sequence)
  - vespo (Gamma 加权序列级重要性比)
- `importance_sampling_level`：
  - token (默认)
  - sequence → 即 GSPO
- `scale_rewards`：
  - group (默认)
  - batch (PPO Lite)
  - none / False (Dr. GRPO 推荐)
- `multi_objective_aggregation`：
  - sum_then_normalize (默认)
  - normalize_then_sum → 即 GDPO

### 取舍与坑

- ✅ 无需 critic/价值网络，省显存、少一套超参
- ✅ 奖励可以是任意可执行代码（判题、格式检查、规则），RLVR 场景最实用
- ✅ 同一 trainer 通过 loss_type/importance_sampling_level 覆盖 DAPO/GSPO/Dr.GRPO/CISPO/SAPO 等变体
- ⚠️ 依赖组内奖励方差：一组全对或全错时优势为 0，没有梯度信号
- ⚠️ 生成成本高（每题 num_generations 条补全），长 CoT 场景尤其吃显存
- ⚠️ 奖励函数写错会安静地把模型带偏，必须自己做抽查
- 🔻 有效 batch（per_device_train_batch_size × num_processes × gradient_accumulation_steps）必须能被 num_generations 整除
- 🔻 beta 默认 0.0：不加 KL 就是纯奖励最大化，模型可能跑偏；需要约束就设 beta>0，代价是要多留一份参考模型显存
- 🔻 num_generations=8 且长补全时，显存需求最高的是生成阶段，先调 max_completion_length 再调 batch
- 🔻 用 vLLM 时 vllm_gpu_memory_utilization 与训练进程共享显存，设太高会 OOM

## <a id="rloo"></a>RLOO（`rloo`）

- **全称**：REINFORCE Leave-One-Out（在线强化学习，稳定）
- **配置类**：`RLOOConfig` @ `trl/trainer/rloo_config.py`（继承链：`RLOOConfig` → `_BaseConfig`）
- **训练器**：`RLOOTrainer` @ `trl/trainer/rloo_trainer.py`
- **文档**：`docs/source/rloo_trainer.md` · 论文：<https://huggingface.co/papers/2402.14740>
- **启动**：`trl rloo`
- **脚本参数**：`reward_funcs`, `reward_model_name_or_path`
- **数据契约**：prompt；prompt

> REINFORCE 式在线 RL：每个 prompt 采样 k 条补全（num_generations），用“其余 k-1 条的平均奖励”作为 leave-one-out 基线，比组内标准化更温和，并保留 PPO 式序列级概率比裁剪。


### 训练流程映射

| 阶段 | 该算法在此阶段做什么 |
| --- | --- |
| 环境与资源 | 同 GRPO，支持 vLLM colocate/server。 |
| 模型与参考/教师 | 策略模型；beta=0.05 默认>0，因此默认会加载参考模型（要省显存就设 beta=0.0）。 |
| 数据与提示 | prompt 数据集。 |
| 监督信号 | reward_funcs（内置/自定义/奖励模型），可用 reward_weights 与 reward_clip_range。 |
| 采样（rollout） | 每个 prompt 采样 num_generations 条补全。 |
| 优势/偏好估计 | leave-one-out 基线：优势 = 本样本奖励 − 其余样本平均奖励；不可打分的样本标记为 NaN 并从基线里剔除。可选 normalize_advantages 做批归一化。 |
| 损失与更新 | 序列级重要性比裁剪（epsilon/epsilon_high）后的 REINFORCE 损失，加上 beta 控制的 KL 项。 |
| 记录与产物 | clip_ratio/*、entropy、reward 指标与 checkpoint。 |

### 参数


**algorithm · 算法其他参数**

| 参数 | 类型 | 默认 | 关键 | 可选值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `logging_steps` | float | `10` |  | — | Log every X updates steps. Should be an integer or a float in range `[0,1)`. If smaller than 1, will be interpreted as ratio of total training steps. |
| `gradient_checkpointing` | bool | `True` |  | — | Enable gradient checkpointing to trade compute for memory. Reduces memory at the cost of ~20%% slower training. |
| `bf16` | bool | None | `None` |  | — | Whether to use bf16 (mixed) precision instead of 32-bit. Requires Ampere or higher NVIDIA architecture or Intel XPU or using CPU (use_cpu) or Ascend NPU. If not set, it defaults to `True` if `fp16` is not set. |
| `lr_scheduler_kwargs` | dict | str | None | `None` |  | — | Additional parameters for the lr_scheduler, such as {'num_cycles': 1} for cosine with hard restarts. See the documentation of each scheduler for possible values. |
| `use_liger_kernel` | bool | `False` |  | — | Enable Liger Kernel optimizations: transformers patches the model's layers with Liger's Triton kernels. Deprecated in `SFTTrainer`, `DPOTrainer`, `KTOTrainer`, `GRPOTrainer` and `RLOOTrainer`, and will be removed in v2.0.0: use `model_init… |
| `torch_empty_cache_steps` | int | None | `None` |  | — | Number of steps to wait before calling `torch.<device>.empty_cache()`. Helps avoid CUDA OOM at a cost of ~10%% slower performance. If None, cache will not be emptied. |

**损失与更新 · 训练超参**

| 参数 | 类型 | 默认 | 关键 | 可选值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `learning_rate` | float | `1e-06` |  | — | The initial learning rate for AdamW. |

**模型与参考/教师 · Parameters that control the model and reference model**

| 参数 | 类型 | 默认 | 关键 | 可选值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `model_init_kwargs` | dict[str, Any] | str | None | `None` |  | — | Keyword arguments for `transformers.AutoModelForCausalLM.from_pretrained`, used when the `model` argument of the `RLOOTrainer` is provided as a string. The `revision` value is also used when loading processing classes. |
| `trust_remote_code` | bool | `False` |  | — | Whether to allow loading models and tokenizers that ship custom Python code from the Hub. Forwarded to `AutoModelForCausalLM.from_pretrained` and `AutoProcessor.from_pretrained`. Also applied to reward-model and reward-tokenizer loads. |
| `router_aux_loss_coef` | float | None | `None` |  | — | Coefficient of the load-balancing auxiliary loss for Mixture-of-Experts (MoE) models, added to the training loss with this weight. When not set, the value declared by the model config is used. Set to `0.0` to disable it. Fails when used wi… |
| `disable_dropout` | bool | `False` |  | — | Whether to disable dropout in the model. This is useful for training with a reference model, as it prevents the model from generating different logprobs for the same input. |

**数据与提示 · Parameters that control the data preprocessing**

| 参数 | 类型 | 默认 | 关键 | 可选值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `remove_unused_columns` | bool | None | `False` |  | — | Whether to only keep the column 'prompt' in the dataset. If you use a custom reward function that requires any column other than 'prompts' and 'completions', you should keep this to `False`. |
| `num_generations` | int | None | `2` | ★ | — | Number of generations to sample. The effective batch size (num_processes * per_device_batch_size * gradient_accumulation_steps) must be evenly divisible by this value. |
| `num_generations_eval` | int | None | `None` |  | — | Number of generations to sample during evaluation. This allows using fewer generations during evaluation to save computation. If `None`, uses the value of `num_generations`. |
| `max_completion_length` | int | None | `512` | ★ | — | Maximum length of the generated completion. |
| `ds3_gather_for_generation` | bool | `True` |  | — | This setting applies to DeepSpeed ZeRO-3. If enabled, the policy model weights are gathered for generation, improving generation speed. However, disabling this option allows training models that exceed the VRAM capacity of a single GPU, al… |
| `shuffle_dataset` | bool | None | `True` |  | — | Whether to shuffle the training dataset. |
| `pad_to_multiple_of` | int | str | None | `auto` |  | — | If set, the prompts ids and completions ids will be padded to a multiple of this value. `'auto'` pads to a multiple of 128 on Neuron (AWS Trainium), which compiles once per input shape, and does not pad elsewhere. |

**采样 · Parameters that control generation**

| 参数 | 类型 | 默认 | 关键 | 可选值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `generation_batch_size` | int | None | `None` |  | — | Batch size to use for generation. If `None`, it defaults to the effective training batch size: `per_device_train_batch_size * num_processes * steps_per_generation`. |
| `steps_per_generation` | int | None | `None` |  | — | Number of steps per generation. If `None`, it defaults to `gradient_accumulation_steps`. |
| `temperature` | float | `1.0` |  | — | Temperature for sampling. The higher the temperature, the more random the completions. |
| `top_p` | float | `1.0` |  | — | Float that controls the cumulative probability of the top tokens to consider. Must be in (0, 1]. Set to 1.0 to consider all tokens. |
| `top_k` | int | `0` |  | — | Number of highest probability vocabulary tokens to keep for top-k-filtering. If `0`, top-k-filtering is disabled and all tokens are considered. |
| `min_p` | float | None | `None` |  | — | Minimum token probability, which will be scaled by the probability of the most likely token. It must be a value between 0.0 and 1.0. Typical values are in the 0.01-0.2 range. |
| `generation_kwargs` | dict | str | None | `None` |  | — | Additional keyword arguments to pass to `GenerationConfig` (if using transformers) or `SamplingParams` (if using vLLM) when sampling completions. This can be used to further customize the generation behavior, such as setting `suppress_toke… |
| `chat_template_kwargs` | dict | str | None | `None` |  | — | Additional keyword arguments to pass to the `apply_chat_template` function when generating completions. |
| `repetition_penalty` | float | `1.0` |  | — | Float that penalizes new tokens based on whether they appear in the prompt and the generated text so far. Values > 1.0 encourage the model to use new tokens, while values < 1.0 encourage the model to repeat tokens. |
| `cache_implementation` | str | None | `None` |  | — | Implementation of the cache method for faster generation when use_vllm is set to False. |

**环境与资源 · Parameters that control generation acceleration powered by vLLM**

| 参数 | 类型 | 默认 | 关键 | 可选值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `use_vllm` | bool | `False` | ★ | — | Whether to use vLLM for generating completions. If set to `True`, the trainer will use vLLM for generation instead of the default model.generate(). Requires `vllm` to be installed. |
| `vllm_mode` | str | `colocate` |  | `colocate`, `server` | Mode to use for vLLM integration when `use_vllm` is set to `True`. Must be one of `'server'` or `'colocate'`. `'server'`: The trainer will send generation requests to a separate vLLM server. Make sure a vLLM server is running (start with `… |
| `vllm_model_impl` | str | `vllm` |  | `vllm`, `transformers` | Model implementation to use for vLLM. Must be one of `transformers` or `vllm`. `transformers`: Use the `transformers` backend for model implementation. `vllm`: Use the `vllm` library for model implementation. |
| `vllm_enable_sleep_mode` | bool | `False` |  | — | Enable vLLM sleep mode to offload weights/cache during the optimizer step. Keeps GPU memory usage low, but waking the engine adds host–device transfer latency. |
| `vllm_structured_outputs_regex` | str | None | `None` |  | — | Regex for vLLM structured outputs. If `None` (default), structured outputs is disabled. |

**环境与资源 · Parameters that control the vLLM server (only used when `vllm_mode` is `"server"`)**

| 参数 | 类型 | 默认 | 关键 | 可选值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `vllm_server_base_url` | str | None | `None` |  | — | Base URL for the vLLM server (e.g., 'http://localhost:8000'). If provided, `vllm_server_host` and `vllm_server_port` are ignored. |
| `vllm_server_host` | str | `0.0.0.0` |  | — | Host of the vLLM server to connect to. Ignored if vllm_server_base_url is provided. |
| `vllm_server_port` | int | `8000` |  | — | Port of the vLLM server to connect to. Ignored if vllm_server_base_url is provided. |
| `vllm_server_timeout` | float | `240.0` |  | — | Total timeout duration in seconds to wait for the vLLM server to be up. If the server is not up after the timeout, a `ConnectionError` is raised. |
| `vllm_group_port` | int | `51216` |  | — | Port number for the weight update group. This is used to communicate with the vLLM server. Unless the port is occupied, there is no need to change it. |

**环境与资源 · Parameters that control colocated vLLM execution (only used when `vllm_mode` is `"colocate"`)**

| 参数 | 类型 | 默认 | 关键 | 可选值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `vllm_gpu_memory_utilization` | float | `0.3` |  | — | Control the GPU memory utilization for vLLM. This setting only applies when `vllm_mode` is set to `'colocate'`. If you are using `vllm_mode='server'`, this parameter must be passed separately when launching the vLLM server via the `--vllm_… |
| `vllm_max_model_length` | int | None | `None` |  | — | Context window for vLLM. Set it to at least the maximum prompt length in the dataset plus `max_completion_length`; if omitted, it is inferred from the model config. |
| `vllm_tensor_parallel_size` | int | `1` |  | — | Control the tensor parallel size for vLLM. This setting only applies when `vllm_mode` is set to `'colocate'`. If you are using `vllm_mode='server'`, this parameter must be passed separately when launching the vLLM server via the `--vllm_te… |

**损失与更新 · Parameters that control the training**

| 参数 | 类型 | 默认 | 关键 | 可选值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `beta` | float | `0.05` | ★ | — | KL coefficient. If `0.0`, the reference model is not loaded, reducing memory usage and improving training speed. |
| `num_iterations` | int | `1` | ★ | — | Number of iterations per batch (denoted as μ in the algorithm). |
| `epsilon` | float | `0.2` | ★ | — | Epsilon value for clipping. |
| `epsilon_high` | float | None | `None` | ★ | — | Upper-bound epsilon value for clipping. If not specified, it defaults to the same value as the lower-bound specified in argument `epsilon`. Paper DAPO recommends `0.28`. |
| `reward_weights` | list[float] | None | `None` | ★ | — | Weights for each reward function. Must match the number of reward functions. If `None`, all rewards are weighted equally with weight `1.0`. |
| `normalize_advantages` | bool | `False` | ★ | — | Whether to normalize advantages. Normalization is done per generation batch to have mean `0.0` and standard deviation of `1.0`. |
| `reward_clip_range` | tuple[float, float] | None | `None` | ★ | — | Clip range for rewards as (min, max). If None, no clipping is applied. |
| `mask_truncated_completions` | bool | `False` |  | — | When enabled, truncated completions are excluded from the loss calculation, preventing them from being incorrectly penalized and introducing noise during training. According to the DAPO paper, this is a good practice for training stability. |
| `sync_ref_model` | bool | `False` |  | — | Whether to synchronize the reference model with the active model every `ref_model_sync_steps` steps, using the `ref_model_mixup_alpha` parameter. |
| `ref_model_mixup_alpha` | float | `0.6` |  | — | α parameter from the TR-DPO paper, which controls the mix between the current policy and the previous reference policy during updates. The reference policy is updated according to the equation: `π_ref = α * π_θ + (1 - α) * π_ref_prev`. To… |
| `ref_model_sync_steps` | int | `512` |  | — | τ parameter from the TR-DPO paper, which determines how frequently the current policy is synchronized with the reference policy. To use this parameter, you must set `sync_ref_model=True`. |

**记录与产物 · Parameters that control the logging**

| 参数 | 类型 | 默认 | 关键 | 可选值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `log_completions` | bool | `False` |  | — | Whether to log a sample of (prompt, completion) pairs every `logging_steps` steps. If `rich` is installed, it prints the sample. If `wandb` logging is enabled, it logs it to `wandb`. |
| `log_multimodal` | bool | `True` |  | — | Whether to log multimodal content (images, videos, etc.) together with completions. Disable this to reduce log size when using high-resolution multimodal data. |
| `num_completions_to_print` | int | None | `None` |  | — | Number of completions to print with `rich`. If `None`, all completions are logged. |
| `log_unique_prompts` | bool | `False` |  | — | Whether to log unique prompts. If `True`, only unique prompts are logged. If `False`, all prompts are logged. |

**环境与资源 · Parameters that control generation acceleration powered by transformers continuous batching**

| 参数 | 类型 | 默认 | 关键 | 可选值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `use_transformers_continuous_batching` | bool | `False` |  | — | Whether to use transformers' continuous batching engine for generating completions. Requires transformers>=5.8.0. |
| `transformers_continuous_batching_config` | dict | str | None | `None` |  | — | Keyword arguments for `transformers.generation.ContinuousBatchingConfig`. |

**已弃用 · 已弃用**

| 参数 | 类型 | 默认 | 关键 | 可选值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `use_transformers_paged` | bool | `False` | 弃用 | — | Deprecated. Use `use_transformers_continuous_batching` instead. |

### 取舍与坑

- ✅ 比 GRPO 更简单：没有组内标准差、不需要多奖励聚合逻辑
- ✅ k=2 就能训，生成成本约为 GRPO(num_generations=8) 的 1/4
- ✅ 保留了序列级裁剪，更新幅度可控
- ⚠️ 基线方差比组标准化 GRPO 稍大，奖励噪声大时更抖
- ⚠️ 不支持工具调用/多轮交互（GRPOTrainer 才支持）
- ⚠️ 同样受“全对/全错组无梯度”的影响
- 🔻 论文里的 RLOO 不含 KL，但 TRL 默认 beta=0.05；要复现论文设 beta=0.0
- 🔻 num_generations 就是论文中的 k，调大能降方差但线性增加生成成本

## <a id="gmpo"></a>GMPO（`gmpo`）

- **全称**：Geometric-Mean Policy Optimization（在线强化学习，实验）
- **配置类**：`GMPOConfig` @ `trl/experimental/gmpo/gmpo_config.py`（继承链：`GMPOConfig` → `GRPOConfig` → `_BaseConfig`）
- **训练器**：`GMPOTrainer` @ `trl/experimental/gmpo/gmpo_trainer.py`
- **文档**：`docs/source/gmpo.md` · 论文：<https://huggingface.co/papers/2507.20673>
- **启动**：`examples 中的写法：python 脚本内构造 GMPOTrainer（无 CLI 子命令）`
- **数据契约**：prompt；与 GRPO 相同

> GRPO 的变体：把 token 级重要性比的算术平均换成几何平均，几何平均对离群比值远不敏感，因此可以用比 GRPO 大得多的裁剪区间而保持稳定。


### 训练流程映射

| 阶段 | 该算法在此阶段做什么 |
| --- | --- |
| 模型与参考/教师 | 同 GRPO（继承全部字段）。 |
| 数据与提示 | 同 GRPO。 |
| 监督信号 | 同 GRPO 的 reward_funcs。 |
| 采样（rollout） | 同 GRPO。 |
| 优势/偏好估计 | 同 GRPO 的组内相对优势。 |
| 损失与更新 | 唯一区别在 _compute_loss：几何平均的重要性比，裁剪区间可放宽（默认 epsilon 更大）。 |
| 记录与产物 | 同 GRPO。 |

### 参数


**algorithm · 算法其他参数**

| 参数 | 类型 | 默认 | 关键 | 可选值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `logging_steps` | float | `10` |  | — | Log every X updates steps. Should be an integer or a float in range `[0,1)`. If smaller than 1, will be interpreted as ratio of total training steps. |
| `gradient_checkpointing` | bool | `True` |  | — | Enable gradient checkpointing to trade compute for memory. Reduces memory at the cost of ~20%% slower training. |
| `bf16` | bool | None | `None` |  | — | Whether to use bf16 (mixed) precision instead of 32-bit. Requires Ampere or higher NVIDIA architecture or Intel XPU or using CPU (use_cpu) or Ascend NPU. If not set, it defaults to `True` if `fp16` is not set. |
| `lr_scheduler_kwargs` | dict | str | None | `None` |  | — | Additional parameters for the lr_scheduler, such as {'num_cycles': 1} for cosine with hard restarts. See the documentation of each scheduler for possible values. |
| `use_liger_kernel` | bool | `False` |  | — | Enable Liger Kernel optimizations: transformers patches the model's layers with Liger's Triton kernels. Deprecated in `SFTTrainer`, `DPOTrainer`, `KTOTrainer`, `GRPOTrainer` and `RLOOTrainer`, and will be removed in v2.0.0: use `model_init… |
| `torch_empty_cache_steps` | int | None | `None` |  | — | Number of steps to wait before calling `torch.<device>.empty_cache()`. Helps avoid CUDA OOM at a cost of ~10%% slower performance. If None, cache will not be emptied. |

**损失与更新 · 训练超参**

| 参数 | 类型 | 默认 | 关键 | 可选值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `learning_rate` | float | `1e-06` |  | — | The initial learning rate for AdamW. |

**模型与参考/教师 · Parameters that control the model and reference model**

| 参数 | 类型 | 默认 | 关键 | 可选值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `model_init_kwargs` | dict[str, Any] | str | None | `None` |  | — | Keyword arguments for `transformers.AutoModelForCausalLM.from_pretrained`, used when the `model` argument of the `GRPOTrainer` is provided as a string. The `revision` value is also used when loading processing classes. |
| `trust_remote_code` | bool | `False` |  | — | Whether to allow loading models and tokenizers that ship custom Python code from the Hub. Forwarded to `AutoModelForCausalLM.from_pretrained` and `AutoProcessor.from_pretrained`. Also applied to reward-model and reward-tokenizer loads. |
| `router_aux_loss_coef` | float | None | `None` |  | — | Coefficient of the load-balancing auxiliary loss for Mixture-of-Experts (MoE) models, added to the training loss with this weight. When not set, the value declared by the model config is used. Set to `0.0` to disable it. Fails when used wi… |
| `disable_dropout` | bool | `False` |  | — | Whether to disable dropout in the model. This is useful for training with a reference model, as it prevents the model from generating different logprobs for the same input. |
| `cast_lm_head_to_fp32` | bool | `False` |  | — | Whether to cast the language modeling head of the policy and reference models to float32. As recommended by the [ScaleRL](https://huggingface.co/papers/2510.13786) recipe. This flag is only supported when the model has untied word embeddin… |

**数据与提示 · Parameters that control the data preprocessing**

| 参数 | 类型 | 默认 | 关键 | 可选值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `remove_unused_columns` | bool | None | `False` |  | — | Whether to only keep the column 'prompt' in the dataset. If you use a custom reward function that requires any column other than 'prompts' and 'completions', you should keep this to `False`. |
| `num_generations` | int | None | `8` | ★ | — | Number of generations to sample. The effective batch size (num_processes * per_device_batch_size * gradient_accumulation_steps) must be evenly divisible by this value. |
| `num_generations_eval` | int | None | `None` |  | — | Number of generations to sample during evaluation. This allows using fewer generations during evaluation to save computation. If `None`, uses the value of `num_generations`. |
| `max_completion_length` | int | None | `512` | ★ | — | Maximum length of the generated completion. |
| `ds3_gather_for_generation` | bool | `True` |  | — | This setting applies to DeepSpeed ZeRO-3. If enabled, the policy model weights are gathered for generation, improving generation speed. However, disabling this option allows training models that exceed the VRAM capacity of a single GPU, al… |
| `shuffle_dataset` | bool | None | `True` |  | — | Whether to shuffle the training dataset. |
| `pad_to_multiple_of` | int | str | None | `auto` |  | — | If set, the prompts ids and completions ids will be padded to a multiple of this value. `'auto'` pads to a multiple of 128 on Neuron (AWS Trainium), which compiles once per input shape, and does not pad elsewhere. |

**采样 · Parameters that control generation**

| 参数 | 类型 | 默认 | 关键 | 可选值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `generation_batch_size` | int | None | `None` |  | — | Batch size to use for generation. If `None`, it defaults to the effective training batch size: `per_device_train_batch_size * num_processes * steps_per_generation`. |
| `steps_per_generation` | int | None | `None` |  | — | Number of steps per generation. If `None`, it defaults to `gradient_accumulation_steps`. |
| `temperature` | float | `1.0` |  | — | Temperature for sampling. The higher the temperature, the more random the completions. |
| `top_p` | float | `1.0` |  | — | Float that controls the cumulative probability of the top tokens to consider. Must be in (0, 1]. Set to 1.0 to consider all tokens. |
| `top_k` | int | `0` |  | — | Number of highest probability vocabulary tokens to keep for top-k-filtering. If `0`, top-k-filtering is disabled and all tokens are considered. |
| `min_p` | float | None | `None` |  | — | Minimum token probability, which will be scaled by the probability of the most likely token. It must be a value between 0.0 and 1.0. Typical values are in the 0.01-0.2 range. |
| `generation_kwargs` | dict | str | None | `None` |  | — | Additional keyword arguments to pass to `GenerationConfig` (if using transformers) or `SamplingParams` (if using vLLM) when sampling completions. This can be used to further customize the generation behavior, such as setting `suppress_toke… |
| `chat_template_kwargs` | dict | str | None | `None` |  | — | Additional keyword arguments to pass to the `apply_chat_template` function when generating completions. |
| `repetition_penalty` | float | `1.0` |  | — | Float that penalizes new tokens based on whether they appear in the prompt and the generated text so far. Values > 1.0 encourage the model to use new tokens, while values < 1.0 encourage the model to repeat tokens. |
| `cache_implementation` | str | None | `None` |  | — | Implementation of the cache method for faster generation when use_vllm is set to False. |

**环境与资源 · Parameters that control generation acceleration powered by vLLM**

| 参数 | 类型 | 默认 | 关键 | 可选值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `use_vllm` | bool | `False` |  | — | Whether to use vLLM for generating completions. If set to `True`, the trainer will use vLLM for generation instead of the default model.generate(). Requires `vllm` to be installed. |
| `vllm_mode` | str | `colocate` |  | `colocate`, `server` | Mode to use for vLLM integration when `use_vllm` is set to `True`. Must be one of `'server'` or `'colocate'`. `'server'`: The trainer will send generation requests to a separate vLLM server. Make sure a vLLM server is running (start with `… |
| `vllm_model_impl` | str | `vllm` |  | `vllm`, `transformers` | Model implementation to use for vLLM. Must be one of `transformers` or `vllm`. `transformers`: Use the `transformers` backend for model implementation. `vllm`: Use the `vllm` library for model implementation. |
| `vllm_enable_sleep_mode` | bool | `False` |  | — | Enable vLLM sleep mode to offload weights/cache during the optimizer step. Keeps GPU memory usage low, but waking the engine adds host–device transfer latency. |
| `vllm_structured_outputs_regex` | str | None | `None` |  | — | Regex for vLLM structured outputs. If `None` (default), structured outputs is disabled. |

**环境与资源 · Parameters that control the vLLM server (only used when `vllm_mode` is `"server"`)**

| 参数 | 类型 | 默认 | 关键 | 可选值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `vllm_server_base_url` | str | None | `None` |  | — | Base URL for the vLLM server (e.g., 'http://localhost:8000'). If provided, `vllm_server_host` and `vllm_server_port` are ignored. |
| `vllm_server_host` | str | `0.0.0.0` |  | — | Host of the vLLM server to connect to. Ignored if vllm_server_base_url is provided. |
| `vllm_server_port` | int | `8000` |  | — | Port of the vLLM server to connect to. Ignored if vllm_server_base_url is provided. |
| `vllm_server_timeout` | float | `240.0` |  | — | Total timeout duration in seconds to wait for the vLLM server to be up. If the server is not up after the timeout, a `ConnectionError` is raised. |
| `vllm_group_port` | int | `51216` |  | — | Port number for the weight update group. This is used to communicate with the vLLM server. Unless the port is occupied, there is no need to change it. |

**环境与资源 · Parameters that control colocated vLLM execution (only used when `vllm_mode` is `"colocate"`)**

| 参数 | 类型 | 默认 | 关键 | 可选值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `vllm_gpu_memory_utilization` | float | `0.3` |  | — | Control the GPU memory utilization for vLLM. This setting only applies when `vllm_mode` is set to `'colocate'`. If you are using `vllm_mode='server'`, this parameter must be passed separately when launching the vLLM server via the `--vllm_… |
| `vllm_max_model_length` | int | None | `None` |  | — | Context window for vLLM. Set it to at least the maximum prompt length in the dataset plus `max_completion_length`; if omitted, it is inferred from the model config. |
| `vllm_tensor_parallel_size` | int | `1` |  | — | Control the tensor parallel size for vLLM. This setting only applies when `vllm_mode` is set to `'colocate'`. If you are using `vllm_mode='server'`, this parameter must be passed separately when launching the vLLM server via the `--vllm_te… |

**损失与更新 · Parameters that control the training**

| 参数 | 类型 | 默认 | 关键 | 可选值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `beta` | float | `0.0` | ★ | — | KL coefficient. If `0.0` (default), the reference model is not loaded, reducing memory usage and improving training speed. [DeepSeek-R1 incentivizes reasoning in LLMs through reinforcement learning](https://huggingface.co/papers/2501.12948… |
| `num_iterations` | int | `1` |  | — | Number of iterations per batch (denoted as μ in the algorithm). |

**algorithm · 算法其他参数**

| 参数 | 类型 | 默认 | 关键 | 可选值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `epsilon` | float | `0.4` | ★ | — | Lower-bound clipping value, expressed in log space. The lower bound of the per-token importance ratio is exp(-epsilon). GMPO recommends 0.4. |

**损失与更新 · Parameters that control the training**

| 参数 | 类型 | 默认 | 关键 | 可选值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `delta` | float | None | `None` |  | — | Enables the upper clipping bound in two-sided GRPO loss when set to a float. If `None` (default), standard GRPO clipping is used. Recommended to be greater than `1 + ε` when enabled. This method is introduced in the [INTELLECT-2 tech repor… |
| `epsilon_high` | float | None | `None` |  | — | Upper-bound epsilon value for clipping. If not specified, it defaults to the same value as the lower-bound specified in argument `epsilon`. Paper DAPO recommends `0.28`. When used with `loss_type='cispo'`, this corresponds to the ε_max par… |
| `sapo_temperature_neg` | float | `1.05` |  | — | Temperature for tokens with non-positive advantage scores used in the `sapo` loss function. This parameter is introduced in the [Soft Adaptive Policy Optimization paper](https://huggingface.co/papers/2511.20347). |
| `sapo_temperature_pos` | float | `1.0` |  | — | Temperature for tokens with positive advantage scores used in the `sapo` loss function. This parameter is introduced in the [Soft Adaptive Policy Optimization paper](https://huggingface.co/papers/2511.20347). |
| `vespo_k_pos` | float | `2.0` |  | — | k parameter for positive advantages, it is the power exponent in the VESPO loss. Controls how aggressively we down-weight samples with low importance weights (when the importance sampling ratio < 1). |
| `vespo_lambda_pos` | float | `3.0` |  | — | lambda parameter for positive advantages, it is the decay factor in the VESPO loss. Controls how aggressively we down-weight samples with high importance weights (when the importance sampling ratio > 1). |
| `vespo_k_neg` | float | `3.0` |  | — | k parameter for negative advantages, it is the power exponent in the VESPO loss. Controls how aggressively we down-weight samples with low importance weights (when the importance sampling ratio < 1). |
| `vespo_lambda_neg` | float | `2.0` |  | — | lambda parameter for negative advantages, it is the exponential decay factor in the VESPO loss. Controls how aggressively we down-weight samples with high importance weights (when the importance sampling ratio > 1). |
| `importance_sampling_level` | str | `token` |  | `token`, `sequence` | Controls whether importance sampling ratios are computed at the `'token'` or `'sequence'` level. `'token'` keeps the raw per-token log-probability ratios (one weight per token). `'sequence'` averages the log-probability ratios across valid… |
| `reward_weights` | list[float] | None | `None` |  | — | Weights for each reward function. Must match the number of reward functions. If `None`, all rewards are weighted equally with weight `1.0`. |
| `multi_objective_aggregation` | str | `sum_then_normalize` |  | `sum_then_normalize`, `scale_rewards` | Method to aggregate multiple reward functions. Supported values are: `'sum_then_normalize'` (default): First sums the weighted rewards from each reward function, then applies reward scaling/normalization as specified by `scale_rewards` (se… |
| `scale_rewards` | str | `group` |  | `True`, `group` | Specifies the scaling strategy for rewards. Supported values are: `True` or `group'` (default): rewards are scaled by the standard deviation within each group, ensuring unit variance within a group. `'batch'`: rewards are scaled by the sta… |
| `loss_type` | str | `dapo` | ★ | `grpo`, `dr_grpo`, `dapo`, `bnpo`, `cispo`, `sapo`, `luspo`, `vespo` | Specifies the loss formulation to use. Supported values are 'grpo', 'dr_grpo', 'dapo', 'bnpo', 'cispo', 'sapo', 'luspo', and 'vespo'. 'grpo': Aggregates token-level losses by normalizing over sequence length. Not recommended due to length… |
| `mask_truncated_completions` | bool | `False` |  | — | When enabled, truncated completions are excluded from the loss calculation, preventing them from being incorrectly penalized and introducing noise during training. According to the DAPO paper, this is a good practice for training stability. |
| `sync_ref_model` | bool | `False` |  | — | Whether to synchronize the reference model with the active model every `ref_model_sync_steps` steps, using the `ref_model_mixup_alpha` parameter. |
| `ref_model_mixup_alpha` | float | `0.6` |  | — | α parameter from the TR-DPO paper, which controls the mix between the current policy and the previous reference policy during updates. The reference policy is updated according to the equation: `π_ref = α * π_θ + (1 - α) * π_ref_prev`. To… |
| `ref_model_sync_steps` | int | `512` |  | — | τ parameter from the TR-DPO paper, which determines how frequently the current policy is synchronized with the reference policy. To use this parameter, you must set `sync_ref_model=True`. |
| `top_entropy_quantile` | float | `1.0` |  | — | ρ parameter from Beyond the 80/20 Rule. Keeps in the policy loss term only the top-ρ quantile of tokens by entropy of the probability distribution at each sequence position, improving results. Range: [0.0-1.0]. A value of `0.0` masks all b… |
| `entropy_coef` | float | `0.0` |  | — | Coefficient of the entropy regularization term in the loss. A positive value adds an entropy bonus that encourages exploration. The bonus is always the mean per-token entropy regardless of `loss_type` (not rescaled to a loss type's policy… |
| `use_adaptive_entropy` | bool | `False` |  | — | Whether to use adaptive entropy control, introduced in Skywork-OR1 (https://huggingface.co/papers/2505.22312). When enabled, `entropy_coef` is incremented by `entropy_coef_delta` when entropy is below `entropy_target`, and decremented othe… |
| `entropy_coef_min` | float | `0.0` |  | — | Lower bound for the entropy coefficient when using adaptive entropy control. |
| `entropy_coef_max` | float | `1.0` |  | — | Upper bound for the entropy coefficient when using adaptive entropy control. |
| `entropy_coef_delta` | float | `0.005` |  | — | Step size for adjusting the entropy coefficient during adaptive entropy control. |
| `entropy_target` | float | `0.2` |  | — | Target mean per-token entropy (nats) for adaptive entropy control. The coefficient is only applied when current entropy is at or below this value. Measured over the same token set as the policy loss (all completion tokens, or the high-entr… |
| `max_tool_calling_iterations` | int | None | `None` |  | — | Maximum number of tool-calling turns when training an agent. If `None`, there is no limit and generation stops when the model generates a response turn with no tool calls or when the total response length reaches `max_model_length`. |
| `vllm_importance_sampling_correction` | bool | `True` |  | — | Whether to apply Importance Sampling (IS) to correct for the mismatch between vLLM completion logprobs and recomputed training logprobs. If set to `False`, no IS is applied regardless of `vllm_importance_sampling_mode`. When `True`, the se… |
| `vllm_importance_sampling_mode` | str | `sequence_mask` |  | — | Specifies how Importance Sampling (IS) is performed when vllm_importance_sampling_correction=True. Modes are defined along two orthogonal dimensions: (1) constraint, which determines how to handle ratios outside the bounds [C_min, C_max]—e… |
| `vllm_importance_sampling_clip_max` | float | None | `3.0` |  | — | Importance sampling upper bound C_max used by `vllm_importance_sampling_mode`. For '*_truncate' modes, ratios are clipped from above at C_max. For '*_mask' modes, ratios larger than C_max are set to zero. |
| `vllm_importance_sampling_clip_min` | float | None | `None` |  | — | Importance sampling lower bound C_min used by `vllm_importance_sampling_mode`. For `*_truncate` modes, ratios are clipped from below at C_min. For `*_mask` modes, ratios below C_min are set to zero. To strictly mask ratios below C_min with… |
| `off_policy_mask_threshold` | float | None | `None` |  | — | Threshold for off-policy sequence masking. If `None`, off-policy sequence masking is disabled. When set, sequences with negative advantages and high KL divergence are masked out to stabilize training. This parameter corresponds to the `del… |
| `use_bias_correction_kl` | bool | `True` |  | — | Whether to multiply the KL term by the importance sampling ratio, so that the KL gradient becomes the unbiased reverse-KL gradient, as described in the [DeepSeek-V3.2 paper](https://huggingface.co/papers/2512.02556). This changes the KL gr… |

**记录与产物 · Parameters that control the logging**

| 参数 | 类型 | 默认 | 关键 | 可选值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `log_completions` | bool | `False` |  | — | Whether to log a sample of (prompt, completion) pairs every `logging_steps` steps. If `rich` is installed, it prints the sample. If `wandb` logging is enabled, it logs it to `wandb`. |
| `log_multimodal` | bool | `True` |  | — | Whether to log multimodal content (images, videos, etc.) together with completions. Disable this to reduce log size when using high-resolution multimodal data. |
| `num_completions_to_print` | int | None | `None` |  | — | Number of completions to print with `rich`. If `None`, all completions are logged. |
| `log_unique_prompts` | bool | `False` |  | — | Whether to log unique prompts. If `True`, only unique prompts are logged. If `False`, all prompts are logged. |
| `log_completions_hub_repo` | str | None | `None` |  | — | Hugging Face Hub repository to save the completions. Should be a complete repository name like `'username/reponame'` or `'orgname/reponame'`, or just `'reponame'` in which case the repository will be created in the currently-logged-in Hugg… |

**环境与资源 · Parameters that control generation acceleration powered by transformers continuous batching**

| 参数 | 类型 | 默认 | 关键 | 可选值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `use_transformers_continuous_batching` | bool | `False` |  | — | Whether to use transformers' continuous batching engine for generating completions. Requires transformers>=5.8.0. |
| `transformers_continuous_batching_config` | dict | str | None | `None` |  | — | Keyword arguments for `transformers.generation.ContinuousBatchingConfig`. |

**已弃用 · 已弃用**

| 参数 | 类型 | 默认 | 关键 | 可选值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `use_transformers_paged` | bool | `False` | 弃用 | — | Deprecated. Use `use_transformers_continuous_batching` instead. |

**损失与更新 · vLLM 重要性采样**

| 参数 | 类型 | 默认 | 关键 | 可选值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `vllm_importance_sampling_cap` | float | None | `None` |  | — | Deprecated, use `vllm_importance_sampling_clip_max` instead. Importance sampling cap C used by `vllm_importance_sampling_mode`. For '*_truncate' modes, ratios are clipped from above at C. For '*_mask' modes, ratios larger than C are set to… |

### 取舍与坑

- ✅ 离群比值不放大，训练更稳
- ✅ 可以放宽裁剪区间，有效利用更多样本
- ⚠️ 几何平均需要对负数/零比值做保护，实现细节影响结果
- ⚠️ 实验命名空间，API 可能变动
- 🔻 参数面板大部分继承自 GRPOConfig，改 loss_type 前先确认与几何平均的假设是否冲突

## <a id="sdpo"></a>SDPO（`sdpo`）

- **全称**：Self-Distillation Policy Optimization（在线强化学习，实验）
- **配置类**：`SDPOConfig` @ `trl/experimental/sdpo/sdpo_config.py`（继承链：`SDPOConfig` → `_BaseConfig`）
- **训练器**：`SDPOTrainer` @ `trl/experimental/sdpo/sdpo_trainer.py`
- **文档**：`docs/source/sdpo_trainer.md` · 论文：<https://huggingface.co/papers/2601.20802>
- **启动**：`examples/sdpo_math/sdpo_math.py`
- **数据契约**：prompt；prompt（可选 feedback/solution 列）

> 在 on-policy 优化之外，把自己高奖励轨迹（或环境反馈）当作“自我教师”，把带反馈条件下的下一 token 预测蒸馏回策略，从而把稀疏的奖励信号变成密集的 token 级学习信号。


### 训练流程映射

| 阶段 | 该算法在此阶段做什么 |
| --- | --- |
| 模型与参考/教师 | 策略模型 + 自我教师（teacher_model_kind、teacher_update_rate、teacher_sync_steps）。 |
| 数据与提示 | prompt + 可选反馈文本。 |
| 监督信号 | reward_funcs 决定哪些轨迹算“成功”（success_reward_threshold）。 |
| 采样（rollout） | 采样 num_generations 条，可选重提示（max_reprompt_len、dont_reprompt_on_self_success）。 |
| 优势/偏好估计 | 奖励优势 + 自我蒸馏的 token 级权重（distillation_weight、distillation_topk）。 |
| 损失与更新 | PPO 式裁剪（epsilon/epsilon_high、importance_sampling_level）与蒸馏损失相加，beta 控制 KL。 |
| 记录与产物 | 同 GRPO，另有 flat-region 诊断指标。 |

### 参数


**algorithm · 算法其他参数**

| 参数 | 类型 | 默认 | 关键 | 可选值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `logging_steps` | float | `10` |  | — | Log every X updates steps. Should be an integer or a float in range `[0,1)`. If smaller than 1, will be interpreted as ratio of total training steps. |
| `gradient_checkpointing` | bool | `True` |  | — | Enable gradient checkpointing to trade compute for memory. Reduces memory at the cost of ~20%% slower training. |
| `bf16` | bool | None | `None` |  | — | Whether to use bf16 (mixed) precision instead of 32-bit. Requires Ampere or higher NVIDIA architecture or Intel XPU or using CPU (use_cpu) or Ascend NPU. If not set, it defaults to `True` if `fp16` is not set. |
| `lr_scheduler_kwargs` | dict | str | None | `None` |  | — | Additional parameters for the lr_scheduler, such as {'num_cycles': 1} for cosine with hard restarts. See the documentation of each scheduler for possible values. |
| `use_liger_kernel` | bool | `False` |  | — | Enable Liger Kernel optimizations: transformers patches the model's layers with Liger's Triton kernels. Deprecated in `SFTTrainer`, `DPOTrainer`, `KTOTrainer`, `GRPOTrainer` and `RLOOTrainer`, and will be removed in v2.0.0: use `model_init… |
| `torch_empty_cache_steps` | int | None | `None` |  | — | Number of steps to wait before calling `torch.<device>.empty_cache()`. Helps avoid CUDA OOM at a cost of ~10%% slower performance. If None, cache will not be emptied. |
| `model_init_kwargs` | dict[str, Any] | None | `None` |  | — | Keyword arguments for `transformers.AutoModelForCausalLM.from_pretrained`, used when the `model` argument of the `SDPOTrainer` is provided as a string. The `revision` value is also used when loading processing classes. |
| `trust_remote_code` | bool | `False` |  | — | Whether to allow loading models and tokenizers that ship custom Python code from the Hub. Forwarded to `AutoModelForCausalLM.from_pretrained` and `AutoProcessor.from_pretrained`. Also applied to reward-model and reward-tokenizer loads. |
| `disable_dropout` | bool | `True` |  | — | Whether to disable dropout in the model. This is useful for training with a reference model, as it prevents the model from generating different logprobs for the same input. |
| `remove_unused_columns` | bool | `False` |  | — | Whether to only keep the column 'prompt' in the dataset. If you use a custom reward function that requires any column other than 'prompts' and 'completions', you should keep this to `False`. |
| `max_prompt_length` | int | None | `512` |  | — | Maximum prompt length. Longer prompts are truncated from the left. |
| `num_generations` | int | `8` | ★ | — | Number of generations to sample. The effective batch size (num_processes * per_device_batch_size * gradient_accumulation_steps) must be evenly divisible by this value. |
| `num_generations_eval` | int | None | `None` |  | — | Number of generations to sample during evaluation. This allows using fewer generations during evaluation to save computation. If `None`, uses the value of `num_generations`. |
| `max_completion_length` | int | None | `256` | ★ | — | Maximum length of the generated completion. |
| `ds3_gather_for_generation` | bool | `True` |  | — | This setting applies to DeepSpeed ZeRO-3. If enabled, the policy model weights are gathered for generation, improving generation speed. However, disabling this option allows training models that exceed the VRAM capacity of a single GPU, al… |
| `shuffle_dataset` | bool | `True` |  | — | Whether to shuffle the training dataset. |
| `generation_batch_size` | int | None | `None` |  | — | Batch size to use for generation. If `None`, it defaults to the effective training batch size: `per_device_train_batch_size * num_processes * steps_per_generation`. |
| `steps_per_generation` | int | None | `None` |  | — | Number of steps per generation. If `None`, it defaults to `gradient_accumulation_steps`. |
| `temperature` | float | `1.0` |  | — | Temperature for sampling. The higher the temperature, the more random the completions. |
| `top_p` | float | `1.0` |  | — | Float that controls the cumulative probability of the top tokens to consider. Must be in (0, 1]. Set to 1.0 to consider all tokens. |
| `top_k` | int | `0` |  | — | Number of highest probability vocabulary tokens to keep for top-k-filtering. If `0`, top-k-filtering is disabled and all tokens are considered. |
| `min_p` | float | None | `None` |  | — | Minimum token probability, which will be scaled by the probability of the most likely token. It must be a value between 0.0 and 1.0. Typical values are in the 0.01-0.2 range. |
| `generation_kwargs` | dict[str, Any] | None | `None` |  | — | Additional keyword arguments to pass to `GenerationConfig` (if using transformers) or `SamplingParams` (if using vLLM) when sampling completions. This can be used to further customize the generation behavior, such as setting `suppress_toke… |
| `chat_template_kwargs` | dict[str, Any] | None | `None` |  | — | Additional keyword arguments to pass to the `apply_chat_template` function when generating completions. |
| `repetition_penalty` | float | `1.0` |  | — | Float that penalizes new tokens based on whether they appear in the prompt and the generated text so far. Values > 1.0 encourage the model to use new tokens, while values < 1.0 encourage the model to repeat tokens. |
| `cache_implementation` | str | None | `None` |  | — | Implementation of the cache method for faster generation when use_vllm is set to False. |
| `use_vllm` | bool | `False` |  | — | Whether to use vLLM for generating completions. If set to `True`, the trainer will use vLLM for generation instead of the default model.generate(). Requires `vllm` to be installed. |
| `use_teacher_server` | bool | `False` |  | — | Compute teacher logprobs from the running vLLM generation server instead of a local teacher forward. Only supported for `teacher_model_kind='live'` with `use_vllm=True`, `vllm_mode='server'`, `distillation_weight=1.0` (pure distillation),… |
| `vllm_mode` | str | `colocate` |  | `colocate`, `server` | Mode to use for vLLM integration when `use_vllm` is set to `True`. Must be one of `'server'` or `'colocate'`. `'server'`: The trainer will send generation requests to a separate vLLM server. Make sure a vLLM server is running (start with `… |
| `vllm_model_impl` | str | `vllm` |  | `vllm`, `transformers` | Model implementation to use for vLLM. Must be one of `transformers` or `vllm`. `transformers`: Use the `transformers` backend for model implementation. `vllm`: Use the `vllm` library for model implementation. |
| `vllm_enable_sleep_mode` | bool | `False` |  | — | Enable vLLM sleep mode to offload weights/cache during the optimizer step. Keeps GPU memory usage low, but waking the engine adds host–device transfer latency. |
| `vllm_server_base_url` | str | None | `None` |  | — | Base URL for the vLLM server (e.g., 'http://localhost:8000'). If provided, `vllm_server_host` and `vllm_server_port` are ignored. |
| `vllm_server_host` | str | `0.0.0.0` |  | — | Host of the vLLM server to connect to. Ignored if vllm_server_base_url is provided. |
| `vllm_server_port` | int | `8000` |  | — | Port of the vLLM server to connect to. Ignored if vllm_server_base_url is provided. |
| `vllm_group_port` | int | `51216` |  | — | Port number for the weight update group. This is used to communicate with the vLLM server. Unless the port is occupied, there is no need to change it. |
| `vllm_server_timeout` | float | `240.0` |  | — | Total timeout duration in seconds to wait for the vLLM server to be up. If the server is not up after the timeout, a `ConnectionError` is raised. |
| `vllm_tensor_parallel_size` | int | `1` |  | — | Control the tensor parallel size for vLLM. This setting only applies when `vllm_mode` is set to `'colocate'`. If you are using `vllm_mode='server'`, this parameter must be passed separately when launching the vLLM server via the `--vllm_te… |
| `vllm_gpu_memory_utilization` | float | `0.3` |  | — | Control the GPU memory utilization for vLLM. This setting only applies when `vllm_mode` is set to `'colocate'`. If you are using `vllm_mode='server'`, this parameter must be passed separately when launching the vLLM server via the `--vllm_… |
| `vllm_max_model_length` | int | None | `None` |  | — | Context window for vLLM. Set it to at least the maximum prompt length in the dataset plus `max_completion_length`; if omitted, it is inferred from the model config. |
| `num_iterations` | int | `1` |  | — | Number of iterations per batch (denoted as μ in the algorithm). |
| `loss_type` | str | `dapo` | ★ | `grpo`, `bnpo`, `dr_grpo`, `dapo` | Specifies the loss formulation to use. Supported values are 'grpo', 'bnpo', 'dr_grpo', and 'dapo'. 'grpo': Aggregates token-level losses by normalizing over sequence length. Not recommended due to length bias—this approach tends to prefer… |
| `mask_truncated_completions` | bool | `False` |  | — | When enabled, truncated completions are excluded from the loss calculation, preventing them from being incorrectly penalized and introducing noise during training. According to the DAPO paper, this is a good practice for training stability. |

**采样 · Parameters that control reprompting**

| 参数 | 类型 | 默认 | 关键 | 可选值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `dont_reprompt_on_self_success` | bool | `True` |  | — | Skip reprompting when model generates correct response. |

**损失与更新 · Parameters that control the online policy objective**

| 参数 | 类型 | 默认 | 关键 | 可选值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `beta` | float | `0.0` | ★ | — | KL coefficient. If `0.0` (default), the reference model is not loaded, reducing memory usage and improving training speed. [DeepSeek-R1 incentivizes reasoning in LLMs through reinforcement learning](https://huggingface.co/papers/2501.12948… |
| `epsilon` | float | `0.2` | ★ | — | Epsilon value for clipping. |
| `epsilon_high` | float | None | `None` |  | — | Upper-bound epsilon value for clipping. If not specified, it defaults to the same value as the lower-bound specified in argument `epsilon`. Paper DAPO recommends `0.28`. |
| `importance_sampling_level` | str | `token` |  | `token`, `sequence` | Controls whether importance sampling ratios are computed at the `'token'` or `'sequence'` level. `'token'` keeps the raw per-token log-probability ratios (one weight per token). `'sequence'` averages the log-probability ratios across valid… |
| `reward_weights` | list[float] | None | `None` |  | — | Weights for each reward function. Must match the number of reward functions. If `None`, all rewards are weighted equally with weight `1.0`. |
| `scale_rewards` | str | `group` |  | `True`, `group` | Specifies the scaling strategy for rewards. Supported values are: `True` or `'group'` (default): rewards are scaled by the standard deviation within each group, ensuring unit variance within a group. `'batch'`: rewards are scaled by the st… |

**损失与更新 · Parameters that control the SDPO loss**

| 参数 | 类型 | 默认 | 关键 | 可选值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `distillation_alpha` | float | `1.0` |  | — | Divergence interpolation coefficient. Sampled-token SDPO requires the official reverse-KL setting `distillation_alpha=1.0`. |
| `distillation_mode` | Literal['sampled_token', 'full_logits', 'topk_logits'] | `sampled_token` | ★ | — | Distillation objective mode. `sampled_token` is the default SDPO mode and requires `distillation_alpha=1.0`. |
| `distillation_topk` | int | None | `None` | ★ | — | Top-k approximation for logit-level SDPO. Must be set when `distillation_mode=topk_logits` and left unset otherwise. |
| `distillation_is_clip` | float | None | `2.0` |  | — | Clipping coefficient for importance sampling in self-distillation. `None` disables clipping. |
| `distillation_add_tail` | bool | `False` |  | — | Whether to add a tail bucket for non-top-k probability mass. |
| `distillation_weight` | float | `1.0` | ★ | — | Convex combination weight between the policy and self-distillation objectives. The loss is `(1 - distillation_weight) * policy_loss + distillation_weight * distillation_loss`. Must be in `[0, 1]`. `1.0` (default) trains purely on self-dist… |

**模型与参考/教师 · Parameters that control the teacher**

| 参数 | 类型 | 默认 | 关键 | 可选值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `teacher_model_kind` | str | `ema` | ★ | `base`, `live`, `ema` | Semantic teacher choice. `base` uses the initial student, `live` uses the current student, and `ema` uses an exponentially averaged teacher. |
| `teacher_update_rate` | float | `0.05` |  | — | Teacher update rate used for EMA teacher synchronization. |
| `teacher_sync_steps` | int | `1` | ★ | — | How often to synchronize the EMA teacher model. |

**采样 · Parameters that control reprompting**

| 参数 | 类型 | 默认 | 关键 | 可选值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `max_reprompt_len` | int | `10240` | ★ | — | Maximum length for reprompting in self-distillation. |
| `use_successful_as_teacher` | bool | `True` | ★ | — | Use successful rollouts as implicit feedback for self-distillation. |
| `success_reward_threshold` | float | `1.0` | ★ | — | Minimum reward for a rollout to be considered a successful demonstration. |
| `reprompt_template` | str | `{prompt}{solution}{feedback}

Correctly solve the original question.
` |  | — | Template for reprompting the teacher with a successful demonstration. |

**algorithm · 算法其他参数**

| 参数 | 类型 | 默认 | 关键 | 可选值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `solution_template` | str | `
Correct solution:

{successful_previous_attempt}

` |  | — | Template for formatting the successful demonstration text. |
| `feedback_template` | str | `
The following is feedback from your unsuccessful earlier attempt:

{feedback_raw}

` |  | — | Template for formatting environment feedback for reprompting. |
| `include_environment_feedback` | bool | `False` |  | — | Whether to include environment feedback in teacher reprompts when available. |
| `environment_feedback_only_without_solution` | bool | `False` |  | — | Whether to use feedback only when no successful solution is available. |
| `remove_thinking_from_demonstration` | bool | `False` |  | — | Whether to remove <think>...</think> blocks from the demonstration text. |
| `diagnostics_warning_interval` | int | `10` |  | — | Emit repeated trainer diagnostics every N consecutive degenerate steps. Set to 0 to disable. |
| `diagnostics_flat_tolerance` | float | `1e-08` |  | — | Tolerance used to decide whether reward variance or reprompt activity is effectively zero. |

### 取舍与坑

- ✅ 把稀疏奖励变成密集 token 信号，长 CoT 场景收敛更快
- ✅ 不需要外部教师模型
- ⚠️ 字段最多（63 个），配置面广、调参成本高
- ⚠️ 自我教师可能放大自己的错误
- 🔻 distillation_* 与 GRPO 的 loss_type 同时生效，两组超参要一起调
- 🔻 实验命名空间

## <a id="a2po"></a>A*-PO（`a2po`）

- **全称**：Optimal Advantage Regression Policy Optimization（在线强化学习，实验）
- **配置类**：`A2POConfig` @ `trl/experimental/a2po/a2po_config.py`（继承链：`A2POConfig` → `_BaseConfig`）
- **训练器**：`A2POTrainer` @ `trl/experimental/a2po/a2po_trainer.py`
- **文档**：`docs/source/a2po_trainer.md` · 论文：<https://huggingface.co/papers/2505.20686>
- **启动**：`库内构造 A2POTrainer`
- **数据契约**：prompt；prompt

> 两阶段：先用参考策略对每个 prompt 采 num_value_samples 条补全并打分，离线估计最优价值 V*（log-sum-exp 形式并缓存），训练时每个 prompt 只生成 1 条补全，回归到该价值——把在线采样成本压到最低。


### 训练流程映射

| 阶段 | 该算法在此阶段做什么 |
| --- | --- |
| 模型与参考/教师 | 策略模型 + 参考策略（估计 V* 用）。 |
| 数据与提示 | prompt；可 filter_all_incorrect 过滤全错样本。 |
| 监督信号 | reward_funcs。 |
| 采样（rollout） | 阶段一：每个 prompt 采 num_value_samples 条；阶段二：每个 prompt 1 条。 |
| 优势/偏好估计 | V*(x)=beta1·log(mean exp(r/beta1))，按 prompt 缓存。 |
| 损失与更新 | 隐式奖励 beta2·log(pi/pi_ref) 与 V* 的平方误差回归。 |
| 记录与产物 | 同标准 Trainer。 |

### 参数


**algorithm · 算法其他参数**

| 参数 | 类型 | 默认 | 关键 | 可选值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `logging_steps` | float | `10` |  | — | Log every X updates steps. Should be an integer or a float in range `[0,1)`. If smaller than 1, will be interpreted as ratio of total training steps. |
| `gradient_checkpointing` | bool | `True` |  | — | Enable gradient checkpointing to trade compute for memory. Reduces memory at the cost of ~20%% slower training. |
| `bf16` | bool | None | `None` |  | — | Whether to use bf16 (mixed) precision instead of 32-bit. Requires Ampere or higher NVIDIA architecture or Intel XPU or using CPU (use_cpu) or Ascend NPU. If not set, it defaults to `True` if `fp16` is not set. |
| `lr_scheduler_kwargs` | dict | str | None | `None` |  | — | Additional parameters for the lr_scheduler, such as {'num_cycles': 1} for cosine with hard restarts. See the documentation of each scheduler for possible values. |
| `use_liger_kernel` | bool | `False` |  | — | Enable Liger Kernel optimizations: transformers patches the model's layers with Liger's Triton kernels. Deprecated in `SFTTrainer`, `DPOTrainer`, `KTOTrainer`, `GRPOTrainer` and `RLOOTrainer`, and will be removed in v2.0.0: use `model_init… |
| `torch_empty_cache_steps` | int | None | `None` |  | — | Number of steps to wait before calling `torch.<device>.empty_cache()`. Helps avoid CUDA OOM at a cost of ~10%% slower performance. If None, cache will not be emptied. |

**模型与参考/教师 · Parameters that control the model and reference model**

| 参数 | 类型 | 默认 | 关键 | 可选值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `model_init_kwargs` | dict | None | `None` |  | — | Keyword arguments for `AutoModelForCausalLM.from_pretrained`, used when the `model` argument of the `A2POTrainer` is provided as a string. The `revision` value is also used when loading the processing class. |
| `trust_remote_code` | bool | `False` |  | — | Whether to allow loading models and tokenizers that ship custom Python code from the Hub. Forwarded to `AutoModelForCausalLM.from_pretrained` and `AutoTokenizer.from_pretrained`. |

**数据与提示 · Parameters that control the data preprocessing**

| 参数 | 类型 | 默认 | 关键 | 可选值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `remove_unused_columns` | bool | `False` |  | — | Whether to only keep the column 'prompt' in the dataset. If you use a custom reward function that requires any column other than 'prompts' and 'completions', you should keep this to `False`. |

**采样 · Parameters that control generation**

| 参数 | 类型 | 默认 | 关键 | 可选值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `max_prompt_length` | int | None | `512` | ★ | — | Maximum length of the prompt. If the prompt is longer than this, it is left-truncated. |
| `max_completion_length` | int | None | `256` | ★ | — | Maximum length of the generated completion. |
| `temperature` | float | `1.0` |  | — | Sampling temperature, used in both Stage 1 and Stage 2 generation. |
| `top_p` | float | `1.0` |  | — | Float that controls the cumulative probability of the top tokens to consider. Must be in (0, 1]. Set to `1.0` to consider all tokens. |
| `top_k` | int | None | `None` |  | — | Number of highest-probability vocabulary tokens to keep. If `None`, top-k filtering is disabled. |

**优势/估计 · Parameters that control Stage 1 (offline optimal value estimation)**

| 参数 | 类型 | 默认 | 关键 | 可选值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `num_value_samples` | int | `8` | ★ | — | Number of samples drawn from the reference policy per prompt to estimate `V*`. |
| `beta1` | float | `0.5` | ★ | — | KL temperature used to estimate `V*` in Stage 1. |
| `filter_all_incorrect` | bool | `True` | ★ | — | Whether to drop prompts for which all reference samples are incorrect. |

**损失与更新 · Parameters that control Stage 2 (on-policy regression)**

| 参数 | 类型 | 默认 | 关键 | 可选值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `beta2` | float | `0.001` | ★ | — | KL temperature used in the Stage 2 regression target. |
| `reward_weights` | list[float] | None | `None` |  | — | Weights for each reward function. Must match the number of reward functions. If `None`, all rewards are weighted equally with weight `1.0`. |

### 取舍与坑

- ✅ 训练阶段每题只生成 1 条补全，生成成本极低
- ✅ 离线价值估计可复用
- ⚠️ 需要先跑一遍全量采样，冷启动慢
- ⚠️ 两个 beta 分别控制价值平滑与隐式奖励，耦合较隐蔽
- 🔻 num_value_samples 越大 V* 越准，第一阶段耗时线性增长

## <a id="async_grpo"></a>Async GRPO（`async_grpo`）

- **全称**：Asynchronous GRPO（在线强化学习，实验）
- **配置类**：`AsyncGRPOConfig` @ `trl/experimental/async_grpo/async_grpo_config.py`（继承链：`AsyncGRPOConfig` → `_BaseConfig`）
- **训练器**：`AsyncGRPOTrainer` @ `trl/experimental/async_grpo/async_grpo_trainer.py`
- **文档**：`docs/source/async_grpo_trainer.md`
- **启动**：`需要外置 vLLM server 作为 rollout worker`
- **数据契约**：prompt；prompt（也用 token_budget 控制流水线投喂量）

> 与 GRPO 同一算法，但把生成与训练解耦：后台 worker 持续从 vLLM server 流式产出补全，训练循环只消费，生成与梯度更新重叠执行，吞吐显著提升；代价是数据可能过期（off-policy）。


### 训练流程映射

| 阶段 | 该算法在此阶段做什么 |
| --- | --- |
| 环境与资源 | 必须有一个可访问的 vLLM server（vllm_server_base_url）；token_budget/queue_maxsize/max_inflight_tasks 控制流水线水位。 |
| 模型与参考/教师 | 策略模型 + 权重同步（weight_sync_steps/weight_sync_timeout）。 |
| 数据与提示 | prompt 流式投喂。 |
| 监督信号 | reward_funcs。 |
| 采样（rollout） | 后台 worker 持续生成，max_staleness 限制数据新鲜度。 |
| 优势/偏好估计 | 同 GRPO 的组内相对优势。 |
| 损失与更新 | 裁剪（epsilon/epsilon_high）+ 可选 KL。 |
| 记录与产物 | 吞吐/水位/过期样本等异步指标。 |

### 参数


**algorithm · 算法其他参数**

| 参数 | 类型 | 默认 | 关键 | 可选值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `logging_steps` | float | `1` |  | — | Log every X update steps. Should be an integer or a float in range `[0,1)`. If smaller than 1, will be interpreted as ratio of total training steps. |
| `gradient_checkpointing` | bool | `True` |  | — | Enable gradient checkpointing to trade compute for memory. Reduces memory at the cost of ~20%% slower training. |
| `bf16` | bool | None | `None` |  | — | Whether to use bf16 (mixed) precision instead of 32-bit. Requires Ampere or higher NVIDIA architecture or Intel XPU or using CPU (use_cpu) or Ascend NPU. If not set, it defaults to `True` if `fp16` is not set. |
| `lr_scheduler_kwargs` | dict | str | None | `None` |  | — | Additional parameters for the lr_scheduler, such as {'num_cycles': 1} for cosine with hard restarts. See the documentation of each scheduler for possible values. |
| `use_liger_kernel` | bool | `False` |  | — | Enable Liger Kernel optimizations: transformers patches the model's layers with Liger's Triton kernels. Deprecated in `SFTTrainer`, `DPOTrainer`, `KTOTrainer`, `GRPOTrainer` and `RLOOTrainer`, and will be removed in v2.0.0: use `model_init… |
| `torch_empty_cache_steps` | int | None | `None` |  | — | Number of steps to wait before calling `torch.<device>.empty_cache()`. Helps avoid CUDA OOM at a cost of ~10%% slower performance. If None, cache will not be emptied. |

**模型与参考/教师 · Parameters that control the model**

| 参数 | 类型 | 默认 | 关键 | 可选值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `model_init_kwargs` | dict[str, Any] | str | None | `None` |  | — | Keyword arguments for `transformers.AutoModelForCausalLM.from_pretrained`, used when instantiating the model from a path. The `revision` value is also used when loading the processing class. |
| `dtype` | str | `float32` |  | `auto`, `bfloat16`, `float16`, `float32` | Dtype to load the model under. Defaults to 'float32', the precision the training-inference mismatch results this trainer is measured against were obtained at. A `dtype` in `model_init_kwargs` takes precedence. |
| `trust_remote_code` | bool | `False` |  | — | Whether to allow loading models and tokenizers that ship custom Python code from the Hub. Forwarded to `AutoModelForCausalLM.from_pretrained` and `AutoTokenizer.from_pretrained`. |
| `router_aux_loss_coef` | float | None | `None` |  | — | Coefficient of the load-balancing auxiliary loss for Mixture-of-Experts (MoE) models, added to the training loss with this weight. When not set, the value declared by the model config is used. Set to `0.0` to disable it. Fails when used wi… |

**损失与更新 · 训练超参**

| 参数 | 类型 | 默认 | 关键 | 可选值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `learning_rate` | float | `1e-06` |  | — | The initial learning rate for AdamW. |

**algorithm · 算法其他参数**

| 参数 | 类型 | 默认 | 关键 | 可选值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `lr_scheduler_type` | str | `constant` |  | — | Learning-rate schedule. Defaults to `constant`: when training is bounded by `num_train_epochs`, message-mode forks make the total step count unknown up front, so `max_steps` is only a safety ceiling and a decay horizon can't be calibrated.… |
| `ignore_data_skip` | bool | `True` |  | — | Always `True` for AsyncGRPO; the base Trainer's skip-and-replay loop does not apply to a live rollout queue. |

**采样 · Parameters that control generation**

| 参数 | 类型 | 默认 | 关键 | 可选值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `num_generations` | int | `8` | ★ | — | Number of generations per prompt to sample. |
| `max_completion_length` | int | `2048` | ★ | — | Maximum length of the generated completion. |
| `temperature` | float | `1.0` |  | — | Temperature for sampling. The higher the temperature, the more random the completions. |
| `top_p` | float | `1.0` |  | — | Float that controls the cumulative probability of the top tokens to consider. Must be in (0, 1]. Set to 1.0 to consider all tokens. |
| `top_k` | int | `0` |  | — | Number of highest probability vocabulary tokens to keep for top-k-filtering. If `0`, top-k-filtering is disabled and all tokens are considered. |
| `min_p` | float | None | `None` |  | — | Minimum token probability, which will be scaled by the probability of the most likely token. It must be a value between 0.0 and 1.0. Typical values are in the 0.01-0.2 range. |
| `repetition_penalty` | float | `1.0` |  | — | Float that penalizes new tokens based on whether they appear in the prompt and the generated text so far. Values > 1.0 encourage the model to use new tokens, while values < 1.0 encourage the model to repeat tokens. |
| `chat_template_kwargs` | dict | str | None | `None` |  | — | Additional keyword arguments to pass to the `apply_chat_template` function when generating completions. |
| `max_tool_calling_iterations` | int | None | `None` |  | — | Maximum number of tool-calling turns when training an agent. If `None`, there is no limit and generation stops when the model generates a response turn with no tool calls or when the total response length reaches `max_completion_length`. |
| `fork_threshold_tokens` | int | `1024` |  | — | A multi-turn conversation is reconciled into training rows by re-tokenizing the whole conversation every turn: a clean append stays one row, a rewrite forks a new row. A re-tokenization drift inside the last answer smaller than this many t… |

**环境与资源 · Parameters that control the vLLM server**

| 参数 | 类型 | 默认 | 关键 | 可选值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `vllm_server_base_url` | str | `http://localhost:8000` | ★ | — | Base URL of the vLLM server used for generation (e.g., 'http://localhost:8000'). |
| `vllm_server_timeout` | float | `240.0` |  | — | Total timeout duration in seconds to wait for the vLLM server to be ready. If the server is not up after the timeout, a `TimeoutError` is raised. |
| `request_timeout` | int | `600` |  | — | Timeout in seconds for individual HTTP requests to the vLLM server. |
| `weight_sync_timeout` | int | `1800` |  | — | Timeout in seconds for a weight transfer to the vLLM server. A transfer that does not complete within this time raises instead of hanging the run. |

**损失与更新 · Parameters that control the training**

| 参数 | 类型 | 默认 | 关键 | 可选值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `epsilon` | float | `0.2` | ★ | — | Epsilon value for clipping. |
| `epsilon_high` | float | None | `None` |  | — | Upper-bound epsilon value for clipping. If not specified, it defaults to the same value as the lower-bound specified in argument `epsilon`. Paper DAPO recommends `0.28`. |
| `token_budget` | int | None | `None` | ★ | — | Maximum number of tokens forwarded in a single row (one DP rank's forward) for dynamic token-budgeted micro-batching. When > 0, a `TokenBudgetBatcher` forms attention-cost-balanced micro-batches whose rows each stay within this budget, bou… |

**采样 · Parameters that control the async rollout pipeline**

| 参数 | 类型 | 默认 | 关键 | 可选值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `max_inflight_tasks` | int | `-1` | ★ | — | Maximum number of concurrent generation tasks sent to the vLLM server. Defaults to -1 (auto), which sets it to `max_staleness * per_device_train_batch_size * gradient_accumulation_steps * num_processes`. Generating more samples than this i… |
| `max_staleness` | int | `4` | ★ | — | Maximum number of weight update steps a rollout sample can lag behind the current model version before being discarded. |
| `queue_maxsize` | int | `1024` |  | — | Maximum number of rollout samples to buffer in the rollout queue. |
| `weight_sync_steps` | int | `1` | ★ | — | Number of training steps between weight synchronizations to the vLLM server. |
| `heartbeat_stale_after_s` | float | `300.0` |  | — | Seconds since the rollout worker's last heartbeat after which the trainer treats it as hung and aborts. |

**记录与产物 · Parameters that control the logging**

| 参数 | 类型 | 默认 | 关键 | 可选值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `log_completions` | bool | `False` |  | — | Whether to log a sample of (prompt, completion) pairs every `logging_steps` steps. If `rich` is installed, it prints the sample. If `wandb` logging is enabled, it logs it to `wandb`. |
| `num_completions_to_print` | int | None | `None` |  | — | Number of completions to print with `rich`. If `None`, all completions are logged. |

### 取舍与坑

- ✅ 生成与训练重叠，GPU 利用率高
- ✅ 适合长补全/大规模训练
- ⚠️ 必须自建 vLLM server，运维复杂度上升
- ⚠️ off-policy 程度由 max_staleness 决定，调不好会掉点
- 🔻 max_staleness 过大 → 训练数据与当前策略差距过大，梯度有偏

## <a id="dpo"></a>DPO（`dpo`）

- **全称**：Direct Preference Optimization（偏好对齐，稳定）
- **配置类**：`DPOConfig` @ `trl/trainer/dpo_config.py`（继承链：`DPOConfig` → `_BaseConfig`）
- **训练器**：`DPOTrainer` @ `trl/trainer/dpo_trainer.py`
- **文档**：`docs/source/dpo_trainer.md` · 论文：<https://huggingface.co/papers/2305.18290>
- **启动**：`trl dpo`
- **数据契约**：prompt、chosen、rejected；prompt + chosen + rejected（对话格式或纯文本；也支持隐式 prompt / 成对不带 prompt）

> 把 RLHF 的目标改写成对“chosen/rejected”成对样本的分类损失：奖励被隐含在策略与参考策略的对数比里，因此不需要采样、不需要奖励模型，训练稳定、成本低。


### 训练流程映射

| 阶段 | 该算法在此阶段做什么 |
| --- | --- |
| 模型与参考/教师 | 策略模型 + 参考模型（ref_model 或隐式 copy）；disable_dropout 默认 True。 |
| 数据与提示 | 成对偏好数据，max_length/truncation_mode 控制截断，padding_free 可省显存。 |
| 监督信号 | 人类/AI 偏好标签本身即监督信号；可用 loss_weights 组合多种损失。 |
| 采样（rollout） | 无（离线），因此训练速度取决于前向次数而非生成。 |
| 优势/偏好估计 | chosen/rejected 的隐式奖励差 beta·(log π/π_ref)，logits 由 _precompute_ref_log_probs 预计算缓存。 |
| 损失与更新 | 由 loss_type 决定（sigmoid/hinge/ipo/exo_pair/nca_pair/robust/bco_pair/sppo_hard/aot/apo_zero/apo_down/discopop/sft/sigmoid_norm），可选 label_smoothing、f_divergence 正则。 |
| 记录与产物 | rewards/chosen、rewards/rejected、rewards/margins、rewards/accuracies 等指标。 |

### 参数


**algorithm · 算法其他参数**

| 参数 | 类型 | 默认 | 关键 | 可选值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `logging_steps` | float | `10` |  | — | Log every X updates steps. Should be an integer or a float in range `[0,1)`. If smaller than 1, will be interpreted as ratio of total training steps. |
| `gradient_checkpointing` | bool | `True` |  | — | Enable gradient checkpointing to trade compute for memory. Reduces memory at the cost of ~20%% slower training. |
| `bf16` | bool | None | `None` |  | — | Whether to use bf16 (mixed) precision instead of 32-bit. Requires Ampere or higher NVIDIA architecture or Intel XPU or using CPU (use_cpu) or Ascend NPU. If not set, it defaults to `True` if `fp16` is not set. |
| `lr_scheduler_kwargs` | dict | str | None | `None` |  | — | Additional parameters for the lr_scheduler, such as {'num_cycles': 1} for cosine with hard restarts. See the documentation of each scheduler for possible values. |
| `use_liger_kernel` | bool | `False` |  | — | Enable Liger Kernel optimizations: transformers patches the model's layers with Liger's Triton kernels. Deprecated in `SFTTrainer`, `DPOTrainer`, `KTOTrainer`, `GRPOTrainer` and `RLOOTrainer`, and will be removed in v2.0.0: use `model_init… |
| `torch_empty_cache_steps` | int | None | `None` |  | — | Number of steps to wait before calling `torch.<device>.empty_cache()`. Helps avoid CUDA OOM at a cost of ~10%% slower performance. If None, cache will not be emptied. |

**损失与更新 · 训练超参**

| 参数 | 类型 | 默认 | 关键 | 可选值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `learning_rate` | float | `1e-06` |  | — | The initial learning rate for AdamW. |

**模型与参考/教师 · Parameters that control the model**

| 参数 | 类型 | 默认 | 关键 | 可选值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `model_init_kwargs` | dict[str, Any] | str | None | `None` |  | — | Keyword arguments for `AutoModelForCausalLM.from_pretrained`, used when the `model` argument of the `DPOTrainer` is provided as a string. The `revision` value is also used when loading the processing class. |
| `trust_remote_code` | bool | `False` |  | — | Whether to allow loading models and tokenizers that ship custom Python code from the Hub. Forwarded to `AutoModelForCausalLM.from_pretrained` and `AutoProcessor.from_pretrained`. |
| `router_aux_loss_coef` | float | None | `None` |  | — | Coefficient of the load-balancing auxiliary loss for Mixture-of-Experts (MoE) models, added to the training loss with this weight. When not set, the value declared by the model config is used. Set to `0.0` to disable it. Fails when used wi… |
| `disable_dropout` | bool | `True` |  | — | Whether to disable dropout in the model and reference model. |

**数据与提示 · Parameters that control the data preprocessing**

| 参数 | 类型 | 默认 | 关键 | 可选值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `dataset_num_proc` | int | None | `None` |  | — | Number of processes to use for processing the dataset. |
| `max_length` | int | None | `1024` | ★ | — | Maximum length of the tokenized sequence. Sequences longer than `max_length` are truncated from the left or right depending on the `truncation_mode`. If `None`, no truncation is applied. |
| `truncation_mode` | str | `keep_start` | ★ | `keep_start`, `keep_end` | Truncation mode to use when the sequence exceeds `max_length`. The only supported value is `'keep_start'`. The `'keep_end'` value is deprecated and will be removed in v2.0.0. |
| `padding_free` | bool | `False` | ★ | — | Whether to perform forward passes without padding by flattening all sequences in the batch into a single continuous sequence. This reduces memory usage by eliminating padding overhead. Currently, this is only supported with the FlashAttent… |
| `pad_to_multiple_of` | int | str | None | `auto` |  | — | If set, the sequences will be padded to a multiple of this value. `'auto'` pads to a multiple of 128 on Neuron (AWS Trainium), which compiles once per input shape, and does not pad elsewhere. |
| `precompute_ref_log_probs` | bool | `False` | ★ | — | Whether to precompute the reference model log probabilities for the entire training dataset before training. This allows to save memory during training, as the reference model does not need to be kept in memory. |
| `precompute_ref_batch_size` | int | None | `None` | ★ | — | Batch size to use when precomputing reference model log probabilities. This can be set higher than the training batch size to speed up preprocessing. If `None`, defaults to `per_device_train_batch_size` for training and `per_device_eval_ba… |

**损失与更新 · Parameters that control the training**

| 参数 | 类型 | 默认 | 关键 | 可选值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `loss_type` | list[str] | `['sigmoid']` | ★ | `sigmoid`, `hinge`, `ipo`, `exo_pair`, `nca_pair`, `robust`, `bco_pair`, `sppo_hard`, `aot`, `aot_unpaired`, `apo_zero`, `apo_down`, `discopop`, `sft`, `sigmoid_norm` | Type of loss to use. Possible values are: `'sigmoid'`, `'hinge'`, `'ipo'`, `'exo_pair'`, `'nca_pair'`, `'robust'`, `'bco_pair'`, `'sppo_hard'`, `'aot'`, `'aot_unpaired'`, `'apo_zero'`, `'apo_down'`, `'discopop'`, `'sft'`, `'sigmoid_norm'`.… |
| `loss_weights` | list[float] | None | — | ★ | — | List of loss weights for multi-loss combinations. Used when combining multiple loss types. Example: `[0.8, 0.2, 1.0]` for MPO. If not provided, defaults to equal weights (`1.0`) for all loss types. |
| `ld_alpha` | float | None | `None` | ★ | — | α parameter from the LD-DPO paper, which controls the weighting of the verbose token log-probabilities in responses. If `None`, no weighting is applied to the verbose part, and the loss is equivalent to the standard DPO loss. Must be in [0… |
| `f_divergence_type` | str | `reverse_kl` | ★ | `reverse_kl`, `forward_kl`, `js_divergence`, `alpha_divergence` | f-divergence regularizer between policy and reference (f-DPO paper). Possible values are: `reverse_kl` (default), `forward_kl`, `js_divergence`, `alpha_divergence`. Only the loss types built on the chosen-rejected reward difference support… |
| `f_alpha_divergence_coef` | float | `0.5` |  | — | α coefficient for the α-divergence u^-α regularizer, used only when `f_divergence_type='alpha_divergence'`. |
| `label_smoothing` | float | `0.0` | ★ | — | Label smoothing parameter used in Robust DPO and EXO. In Robust DPO, it is interpreted as the probability that a preference label is flipped and must lie in [0.0, 0.5); a typical value recommended by the Robust DPO paper is 0.1. In EXO, it… |
| `beta` | float | `0.1` | ★ | — | Parameter controlling the deviation from the reference model. Higher β means less deviation from the reference model. For the IPO loss (`loss_type='ipo'`), this value is the regularization parameter denoted by τ in the [paper](https://hugg… |
| `use_weighting` | bool | `False` | ★ | — | Whether to apply [WPO](https://huggingface.co/papers/2406.11827)-style weighting to preference pairs using the policy's length-normalized sequence probabilities. |
| `discopop_tau` | float | `0.05` |  | — | τ/temperature parameter from the DiscoPOP paper, which controls the shape of the log-ratio modulated loss when using `loss_type='discopop'`. The paper recommends the default value `discopop_tau=0.05`. |
| `activation_offloading` | bool | `False` |  | — | Whether to offload the activations to the CPU. |
| `sync_ref_model` | bool | `False` | ★ | — | Whether to synchronize the reference model with the active model every `ref_model_sync_steps` steps, using the `ref_model_mixup_alpha` parameter. This synchronization originates from the [TR-DPO](https://huggingface.co/papers/2404.09656) p… |
| `ref_model_mixup_alpha` | float | `0.6` |  | — | α parameter from the TR-DPO paper, which controls the mix between the current policy and the previous reference policy during updates. The reference policy is updated according to the equation: `π_ref = α * π_θ + (1 - α) * π_ref_prev`. To… |
| `ref_model_sync_steps` | int | `512` |  | — | τ parameter from the TR-DPO paper, which determines how frequently the current policy is synchronized with the reference policy. To use this parameter, you must set `sync_ref_model=True`. |

**数据与提示 · 数据与分词器**

| 参数 | 类型 | 默认 | 关键 | 可选值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `pad_token` | str | None | `None` |  | — | Deprecated. Set `tokenizer.pad_token` directly and pass it as `processing_class` to the trainer instead. |

### 同一配置内的算法变体

- `loss_type`：
  - sigmoid (默认)
  - hinge
  - ipo
  - exo_pair
  - nca_pair
  - robust
  - bco_pair
  - sppo_hard
  - aot
  - aot_unpaired
  - apo_zero
  - apo_down
  - discopop
  - sft
  - sigmoid_norm
- `f_divergence_type`：
  - reverse_kl (默认)
  - forward_kl
  - js_divergence
  - alpha_divergence

### 取舍与坑

- ✅ 不用采样、不用奖励模型，单卡也能跑
- ✅ 有大量可以复现的公开配方与数据
- ✅ 损失函数族覆盖 IPO/CPO/SimPO 思想等变体
- ⚠️ 离线偏好数据决定了上限，无法自我探索
- ⚠️ 对数据质量与分布极敏感，容易学到表面风格
- ⚠️ beta 与 loss_type 组合多，需要实验
- 🔻 beta 默认 0.1，改小会让模型偏离参考更远（更激进），改大则几乎不学习
- 🔻 precompute_ref_log_probs=True 可以大幅省显存但会多一遍全量前向
- 🔻 chosen/rejected 必须在同一 tokenizer 下编码，长度截断策略要与数据长度匹配

## <a id="kto"></a>KTO（`kto`）

- **全称**：Kahneman-Tversky Optimization（偏好对齐，稳定）
- **配置类**：`KTOConfig` @ `trl/trainer/kto_config.py`（继承链：`KTOConfig` → `_BaseConfig`）
- **训练器**：`KTOTrainer` @ `trl/trainer/kto_trainer.py`
- **文档**：`docs/source/kto_trainer.md` · 论文：<https://huggingface.co/papers/2402.01306>
- **启动**：`trl kto`
- **数据契约**：prompt、completion、label；prompt + completion + label(bool，True=desirable)

> 不需要成对偏好，只需每个样本“好/坏”的二元标签；用前景理论的效用函数直接最大化生成样本的效用，并用同 batch 内其他样本估计 KL 项。


### 训练流程映射

| 阶段 | 该算法在此阶段做什么 |
| --- | --- |
| 模型与参考/教师 | 策略模型 + 参考模型。 |
| 数据与提示 | desirable/undesirable 单样本；train_sampling_strategy 必须为 sequential 才能让 KL 配对成立。 |
| 监督信号 | 二元标签 + desirable_weight/undesirable_weight 调节偏重。 |
| 采样（rollout） | 无。 |
| 优势/偏好估计 | 前景理论效用 + 参考 KL（同 batch 配对估计）。 |
| 损失与更新 | KTO/apo_zero_unpaired 损失。 |
| 记录与产物 | rewards/kl、rewards/desirable、rewards/undesirable 等指标。 |

### 参数


**algorithm · 算法其他参数**

| 参数 | 类型 | 默认 | 关键 | 可选值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `logging_steps` | float | `10` |  | — | Log every X updates steps. Should be an integer or a float in range `[0,1)`. If smaller than 1, will be interpreted as ratio of total training steps. |
| `gradient_checkpointing` | bool | `True` |  | — | Enable gradient checkpointing to trade compute for memory. Reduces memory at the cost of ~20%% slower training. |
| `bf16` | bool | None | `None` |  | — | Whether to use bf16 (mixed) precision instead of 32-bit. Requires Ampere or higher NVIDIA architecture or Intel XPU or using CPU (use_cpu) or Ascend NPU. If not set, it defaults to `True` if `fp16` is not set. |
| `lr_scheduler_kwargs` | dict | str | None | `None` |  | — | Additional parameters for the lr_scheduler, such as {'num_cycles': 1} for cosine with hard restarts. See the documentation of each scheduler for possible values. |
| `use_liger_kernel` | bool | `False` |  | — | Enable Liger Kernel optimizations: transformers patches the model's layers with Liger's Triton kernels. Deprecated in `SFTTrainer`, `DPOTrainer`, `KTOTrainer`, `GRPOTrainer` and `RLOOTrainer`, and will be removed in v2.0.0: use `model_init… |
| `torch_empty_cache_steps` | int | None | `None` |  | — | Number of steps to wait before calling `torch.<device>.empty_cache()`. Helps avoid CUDA OOM at a cost of ~10%% slower performance. If None, cache will not be emptied. |

**损失与更新 · 训练超参**

| 参数 | 类型 | 默认 | 关键 | 可选值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `learning_rate` | float | `1e-06` |  | — | The initial learning rate for AdamW. |

**损失与更新 · 训练循环**

| 参数 | 类型 | 默认 | 关键 | 可选值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `train_sampling_strategy` | str | `sequential` | ★ | `random`, `sequential`, `group_by_length` | Sampler to use for the training dataloader. Loss types that estimate the KL divergence term (all except `'apo_zero_unpaired'`) require `'sequential'` because the KL completion for each example is precomputed against its neighbors in a fixe… |

**模型与参考/教师 · Parameters that control the model**

| 参数 | 类型 | 默认 | 关键 | 可选值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `model_init_kwargs` | dict[str, Any] | str | None | `None` |  | — | Keyword arguments for `AutoModelForCausalLM.from_pretrained`, used when the `model` argument of the `KTOTrainer` is provided as a string. The `revision` value is also used when loading the processing class. |
| `trust_remote_code` | bool | `False` |  | — | Whether to allow loading models and tokenizers that ship custom Python code from the Hub. Forwarded to `AutoModelForCausalLM.from_pretrained` and `AutoProcessor.from_pretrained`. |
| `router_aux_loss_coef` | float | None | `None` |  | — | Coefficient of the load-balancing auxiliary loss for Mixture-of-Experts (MoE) models, added to the training loss with this weight. When not set, the value declared by the model config is used. Set to `0.0` to disable it. Fails when used wi… |
| `disable_dropout` | bool | `True` |  | — | Whether to disable dropout in the model and reference model. |

**数据与提示 · Parameters that control the data preprocessing**

| 参数 | 类型 | 默认 | 关键 | 可选值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `dataset_num_proc` | int | None | `None` |  | — | Number of processes to use for processing the dataset. |
| `max_length` | int | None | `1024` | ★ | — | Maximum length of the tokenized sequence. Sequences longer than `max_length` are truncated from the right. If `None`, no truncation is applied. |
| `pad_to_multiple_of` | int | str | None | `auto` |  | — | If set, the sequences will be padded to a multiple of this value. `'auto'` pads to a multiple of 128 on Neuron (AWS Trainium), which compiles once per input shape, and does not pad elsewhere. |
| `precompute_ref_log_probs` | bool | `False` |  | — | Whether to precompute the reference model log probabilities for the entire training dataset before training. This allows to save memory during training, as the reference model does not need to be kept in memory. |
| `precompute_ref_batch_size` | int | None | `None` |  | — | Batch size to use when precomputing reference model log probabilities. This can be set higher than the training batch size to speed up preprocessing. If `None`, defaults to `per_device_train_batch_size` for training and `per_device_eval_ba… |

**损失与更新 · Parameters that control the training**

| 参数 | 类型 | 默认 | 关键 | 可选值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `loss_type` | str | `kto` | ★ | `kto`, `apo_zero_unpaired` | Type of loss to use. |
| `beta` | float | `0.1` | ★ | — | Parameter controlling the deviation from the reference model. Higher β means less deviation from the reference model. |
| `desirable_weight` | float | `1.0` | ★ | — | Desirable losses are weighed by this factor to counter unequal number of desirable and undesirable pairs. |
| `undesirable_weight` | float | `1.0` | ★ | — | Undesirable losses are weighed by this factor to counter unequal number of desirable and undesirable pairs. |
| `activation_offloading` | bool | `False` |  | — | Whether to offload the activations to the CPU. |
| `sync_ref_model` | bool | `False` |  | — | Whether to synchronize the reference model with the active model every `ref_model_sync_steps` steps, using the `ref_model_mixup_alpha` parameter. This synchronization originates from the [TR-DPO](https://huggingface.co/papers/2404.09656) p… |
| `ref_model_mixup_alpha` | float | `0.6` |  | — | α parameter from the TR-DPO paper, which controls the mix between the current policy and the previous reference policy during updates. The reference policy is updated according to the equation: `π_ref = α * π_θ + (1 - α) * π_ref_prev`. To… |
| `ref_model_sync_steps` | int | `512` |  | — | τ parameter from the TR-DPO paper, which determines how frequently the current policy is synchronized with the reference policy. To use this parameter, you must set `sync_ref_model=True`. |

### 取舍与坑

- ✅ 只要二元标注，数据获取成本最低
- ✅ 对正负样本不均衡可显式加权
- ⚠️ 没有成对对比，信号比 DPO 弱一档
- ⚠️ KL 估计依赖 batch 内配对，采样策略受限（必须 sequential）
- 🔻 改成 random/group_by_length 会让 KL 项失效（损失类型要不是 apo_zero_unpaired）
- 🔻 desirable_weight 与数据比例强相关，需要按标签比例调整

## <a id="orpo"></a>ORPO（`orpo`）

- **全称**：Odds Ratio Preference Optimization（偏好对齐，实验）
- **配置类**：`ORPOConfig` @ `trl/experimental/orpo/orpo_config.py`（继承链：`ORPOConfig` → `_BaseConfig`）
- **训练器**：`ORPOTrainer` @ `trl/experimental/orpo/orpo_trainer.py`
- **文档**：`docs/source/orpo_trainer.md` · 论文：<https://huggingface.co/papers/2403.07691>
- **启动**：`库内构造 ORPOTrainer`
- **数据契约**：prompt、chosen、rejected；成对偏好数据

> 把 SFT 损失与 odds-ratio 偏好项合成一个单步目标，不需要参考模型：chosen 走 SFT 交叉熵，rejected 只受 odds 比惩罚。


### 训练流程映射

| 阶段 | 该算法在此阶段做什么 |
| --- | --- |
| 模型与参考/教师 | 只有策略模型（无参考模型，显存最省）。 |
| 数据与提示 | 成对偏好数据。 |
| 监督信号 | 偏好标签。 |
| 采样（rollout） | 无。 |
| 优势/偏好估计 | chosen/rejected 的 odds 比。 |
| 损失与更新 | SFT 交叉熵 + beta 加权的 odds 比惩罚，一步到位。 |
| 记录与产物 | nll_loss、odds_ratio_loss、rewards/* 指标。 |

### 参数


**algorithm · 算法其他参数**

| 参数 | 类型 | 默认 | 关键 | 可选值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `logging_steps` | float | `10` |  | — | Log every X updates steps. Should be an integer or a float in range `[0,1)`. If smaller than 1, will be interpreted as ratio of total training steps. |
| `gradient_checkpointing` | bool | `True` |  | — | Enable gradient checkpointing to trade compute for memory. Reduces memory at the cost of ~20%% slower training. |
| `bf16` | bool | None | `None` |  | — | Whether to use bf16 (mixed) precision instead of 32-bit. Requires Ampere or higher NVIDIA architecture or Intel XPU or using CPU (use_cpu) or Ascend NPU. If not set, it defaults to `True` if `fp16` is not set. |
| `lr_scheduler_kwargs` | dict | str | None | `None` |  | — | Additional parameters for the lr_scheduler, such as {'num_cycles': 1} for cosine with hard restarts. See the documentation of each scheduler for possible values. |
| `use_liger_kernel` | bool | `False` |  | — | Enable Liger Kernel optimizations: transformers patches the model's layers with Liger's Triton kernels. Deprecated in `SFTTrainer`, `DPOTrainer`, `KTOTrainer`, `GRPOTrainer` and `RLOOTrainer`, and will be removed in v2.0.0: use `model_init… |
| `torch_empty_cache_steps` | int | None | `None` |  | — | Number of steps to wait before calling `torch.<device>.empty_cache()`. Helps avoid CUDA OOM at a cost of ~10%% slower performance. If None, cache will not be emptied. |

**损失与更新 · 训练超参**

| 参数 | 类型 | 默认 | 关键 | 可选值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `learning_rate` | float | `1e-06` |  | — | The initial learning rate for AdamW. |

**algorithm · 算法其他参数**

| 参数 | 类型 | 默认 | 关键 | 可选值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `max_length` | int | None | `1024` | ★ | — | Maximum length of the sequences (prompt + completion) in the batch. |
| `max_completion_length` | int | None | `None` | ★ | — | Maximum length of the completion. This argument is required if you want to use the default data collator and your model is an encoder-decoder. |
| `beta` | float | `0.1` | ★ | — | Parameter controlling the relative ratio loss weight in the ORPO loss. In the paper, it is denoted by λ. |
| `disable_dropout` | bool | `True` |  | — | Whether to disable dropout in the model. |
| `padding_value` | int | None | `None` |  | — | Padding value to use. If `None`, the padding value of the tokenizer is used. |
| `generate_during_eval` | bool | `False` | ★ | — | If `True`, generates and logs completions from the model to W&B during evaluation. |
| `is_encoder_decoder` | bool | None | `None` |  | — | When using the `model_init` argument (callable) to instantiate the model instead of the `model` argument, you need to specify if the model returned by the callable is an encoder-decoder model. |
| `model_init_kwargs` | dict[str, Any] | str | None | `None` |  | — | Keyword arguments to pass to `AutoModelForCausalLM.from_pretrained` when instantiating the model from a string. The `revision` value is also used when loading the processing class. |
| `trust_remote_code` | bool | `False` |  | — | Whether to allow loading models that ship custom Python code from the Hub. Forwarded to `AutoModelForCausalLM.from_pretrained`. |

**数据与提示 · 数据集加载**

| 参数 | 类型 | 默认 | 关键 | 可选值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `dataset_num_proc` | int | None | `None` |  | — | Number of processes to use for processing the dataset. |

### 取舍与坑

- ✅ 无需参考模型，显存占用最低
- ✅ 单步训练，省去 SFT→偏好两段式
- ⚠️ 对数据规模敏感，小数据容易过拟合偏好项
- ⚠️ 实验命名空间
- 🔻 beta 太小会退化成纯 SFT

## <a id="cpo"></a>CPO（`cpo`）

- **全称**：Contrastive Preference Optimization（偏好对齐，实验）
- **配置类**：`CPOConfig` @ `trl/experimental/cpo/cpo_config.py`（继承链：`CPOConfig` → `_BaseConfig`）
- **训练器**：`CPOTrainer` @ `trl/experimental/cpo/cpo_trainer.py`
- **文档**：`docs/source/cpo_trainer.md` · 论文：<https://huggingface.co/papers/2401.08417>
- **启动**：`库内构造 CPOTrainer`
- **数据契约**：prompt、chosen、rejected；成对偏好数据

> 面向机器翻译等场景的对比偏好优化：用成对样本 + 可选的 SFT 项（cpo_alpha）约束，避免模型在偏好数据上“拒绝”原本正确的翻译；也支持 SimPO 式长度归一化（simpo_gamma）。


### 训练流程映射

| 阶段 | 该算法在此阶段做什么 |
| --- | --- |
| 模型与参考/教师 | 只有策略模型。 |
| 数据与提示 | 成对偏好数据。 |
| 监督信号 | 偏好标签。 |
| 采样（rollout） | 无。 |
| 优势/偏好估计 | 对比项的 log-ratio。 |
| 损失与更新 | 对比损失 + cpo_alpha·SFT 项（可选 SimPO gamma 长度归一化），loss_type 决定具体形式。 |
| 记录与产物 | rewards/* 与 SFT 项指标。 |

### 参数


**algorithm · 算法其他参数**

| 参数 | 类型 | 默认 | 关键 | 可选值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `logging_steps` | float | `10` |  | — | Log every X updates steps. Should be an integer or a float in range `[0,1)`. If smaller than 1, will be interpreted as ratio of total training steps. |
| `gradient_checkpointing` | bool | `True` |  | — | Enable gradient checkpointing to trade compute for memory. Reduces memory at the cost of ~20%% slower training. |
| `bf16` | bool | None | `None` |  | — | Whether to use bf16 (mixed) precision instead of 32-bit. Requires Ampere or higher NVIDIA architecture or Intel XPU or using CPU (use_cpu) or Ascend NPU. If not set, it defaults to `True` if `fp16` is not set. |
| `lr_scheduler_kwargs` | dict | str | None | `None` |  | — | Additional parameters for the lr_scheduler, such as {'num_cycles': 1} for cosine with hard restarts. See the documentation of each scheduler for possible values. |
| `use_liger_kernel` | bool | `False` |  | — | Enable Liger Kernel optimizations: transformers patches the model's layers with Liger's Triton kernels. Deprecated in `SFTTrainer`, `DPOTrainer`, `KTOTrainer`, `GRPOTrainer` and `RLOOTrainer`, and will be removed in v2.0.0: use `model_init… |
| `torch_empty_cache_steps` | int | None | `None` |  | — | Number of steps to wait before calling `torch.<device>.empty_cache()`. Helps avoid CUDA OOM at a cost of ~10%% slower performance. If None, cache will not be emptied. |

**损失与更新 · 训练超参**

| 参数 | 类型 | 默认 | 关键 | 可选值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `learning_rate` | float | `1e-06` |  | — | The initial learning rate for AdamW. |

**algorithm · 算法其他参数**

| 参数 | 类型 | 默认 | 关键 | 可选值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `max_length` | int | None | `1024` | ★ | — | Maximum length of the sequences (prompt + completion) in the batch. |
| `max_completion_length` | int | None | `None` | ★ | — | Maximum length of the completion. This argument is required if you want to use the default data collator and your model is an encoder-decoder. |
| `beta` | float | `0.1` | ★ | — | Parameter controlling the deviation from the reference model. Higher β means less deviation from the reference model. |
| `label_smoothing` | float | `0.0` | ★ | — | Label smoothing factor. |
| `loss_type` | str | `sigmoid` | ★ | `sigmoid`, `hinge`, `ipo`, `simpo`, `alphapo` | Type of loss to use. |
| `disable_dropout` | bool | `True` |  | — | Whether to disable dropout in the model. |
| `cpo_alpha` | float | `1.0` | ★ | — | Weight of the BC regularizer in CPO training. |
| `simpo_gamma` | float | `0.5` | ★ | — | Target reward margin for the SimPO loss, used only when the `loss_type='simpo'`. |
| `alpha` | float | `0.0` | ★ | — | Alpha parameter that controls reward function shape across all loss types. When alpha=0 (default), uses standard log probability rewards. When `alpha != 0`, applies AlphaPO transformation: `r = (1 - p^(-alpha)) / alpha` from the AlphaPO pa… |
| `generate_during_eval` | bool | `False` |  | — | If `True`, generates and logs completions from the model to W&B during evaluation. |
| `is_encoder_decoder` | bool | None | `None` |  | — | Whether the model is an encoder-decoder model. |
| `model_init_kwargs` | dict[str, Any] | str | None | `None` |  | — | Keyword arguments to pass to `AutoModelForCausalLM.from_pretrained` when instantiating the model from a string. The `revision` value is also used when loading the processing class. |
| `trust_remote_code` | bool | `False` |  | — | Whether to allow loading models that ship custom Python code from the Hub. Forwarded to `AutoModelForCausalLM.from_pretrained`. |

**数据与提示 · 数据集加载**

| 参数 | 类型 | 默认 | 关键 | 可选值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `dataset_num_proc` | int | None | `None` |  | — | Number of processes to use for processing the dataset. |

### 取舍与坑

- ✅ 有无参考模型两种路径
- ✅ cpo_alpha 让模型不遗忘已有能力
- ⚠️ 参数之间耦合（loss_type 与 cpo_alpha/simpo_gamma 的可用组合）
- ⚠️ 实验命名空间
- 🔻 cpo_alpha 设 0 时变成纯偏好优化，需要确认数据量足够

## <a id="tpo"></a>TPO（`tpo`）

- **全称**：Triple Preference Optimization（偏好对齐，实验）
- **配置类**：`TPOConfig` @ `trl/experimental/tpo/tpo_config.py`（继承链：`TPOConfig` → `_BaseConfig`）
- **训练器**：`TPOTrainer` @ `trl/experimental/tpo/tpo_trainer.py`
- **文档**：`docs/source/tpo_trainer.md` · 论文：<https://huggingface.co/papers/2405.16681>
- **启动**：`库内构造 TPOTrainer`
- **数据契约**：prompt、chosen、rejected；成对偏好数据（第三路回答由模型自己生成）

> 在 DPO 的“chosen 优于 rejected”之外，再加“chosen 优于模型自己生成的 lower-quality 回答”这一项，三路对比提升指令跟随与推理能力。


### 训练流程映射

| 阶段 | 该算法在此阶段做什么 |
| --- | --- |
| 模型与参考/教师 | 策略模型（论文/实现中不依赖参考模型）。 |
| 数据与提示 | 成对偏好数据。 |
| 监督信号 | 偏好标签 + 自生成回答作为弱样本。 |
| 采样（rollout） | 训练中生成自回答。 |
| 优势/偏好估计 | 三项 log-ratio 对比。 |
| 损失与更新 | TPO 损失（tpo_alpha、tpo_l_gamma 控制三项权重）。 |
| 记录与产物 | 各类 rewards 指标。 |

### 参数


**algorithm · 算法其他参数**

| 参数 | 类型 | 默认 | 关键 | 可选值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `logging_steps` | float | `10` |  | — | Log every X updates steps. Should be an integer or a float in range `[0,1)`. If smaller than 1, will be interpreted as ratio of total training steps. |
| `gradient_checkpointing` | bool | `True` |  | — | Enable gradient checkpointing to trade compute for memory. Reduces memory at the cost of ~20%% slower training. |
| `bf16` | bool | None | `None` |  | — | Whether to use bf16 (mixed) precision instead of 32-bit. Requires Ampere or higher NVIDIA architecture or Intel XPU or using CPU (use_cpu) or Ascend NPU. If not set, it defaults to `True` if `fp16` is not set. |
| `lr_scheduler_kwargs` | dict | str | None | `None` |  | — | Additional parameters for the lr_scheduler, such as {'num_cycles': 1} for cosine with hard restarts. See the documentation of each scheduler for possible values. |
| `use_liger_kernel` | bool | `False` |  | — | Enable Liger Kernel optimizations: transformers patches the model's layers with Liger's Triton kernels. Deprecated in `SFTTrainer`, `DPOTrainer`, `KTOTrainer`, `GRPOTrainer` and `RLOOTrainer`, and will be removed in v2.0.0: use `model_init… |
| `torch_empty_cache_steps` | int | None | `None` |  | — | Number of steps to wait before calling `torch.<device>.empty_cache()`. Helps avoid CUDA OOM at a cost of ~10%% slower performance. If None, cache will not be emptied. |

**损失与更新 · 训练超参**

| 参数 | 类型 | 默认 | 关键 | 可选值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `learning_rate` | float | `5e-07` |  | — | The initial learning rate for AdamW. |

**模型与参考/教师 · Parameters that control the model**

| 参数 | 类型 | 默认 | 关键 | 可选值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `model_init_kwargs` | dict[str, Any] | str | None | `None` |  | — | Keyword arguments for `AutoModelForCausalLM.from_pretrained`, used when the `model` argument of the `TPOTrainer` is provided as a string. The `revision` value is also used when loading the processing class. |
| `trust_remote_code` | bool | `False` |  | — | Whether to allow loading models and tokenizers that ship custom Python code from the Hub. Forwarded to `AutoModelForCausalLM.from_pretrained` and `AutoProcessor.from_pretrained`. |
| `disable_dropout` | bool | `True` |  | — | Whether to disable dropout in the model. |

**数据与提示 · Parameters that control the data preprocessing**

| 参数 | 类型 | 默认 | 关键 | 可选值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `dataset_num_proc` | int | None | `None` |  | — | Number of processes to use for processing the dataset. |
| `max_length` | int | None | `1024` | ★ | — | Maximum length of the tokenized sequence. Sequences longer than `max_length` are truncated from the left or right depending on the `truncation_mode`. If `None`, no truncation is applied. |
| `truncation_mode` | str | `keep_start` |  | `keep_start`, `keep_end` | Truncation mode to use when the sequence exceeds `max_length`. |
| `pad_to_multiple_of` | int | None | `None` |  | — | If set, the sequences will be padded to a multiple of this value. |

**损失与更新 · Parameters that control the training**

| 参数 | 类型 | 默认 | 关键 | 可选值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `loss_type` | str | `sigmoid` | ★ | `sigmoid`, `hinge`, `ipo`, `tpo-l` | Type of loss to use. |
| `beta` | float | `0.01` | ★ | — | Parameter controlling the temperature of the TPO loss. For the IPO loss (`loss_type='ipo'`), this value is the regularization parameter denoted by τ in the IPO paper. |
| `label_smoothing` | float | `0.0` | ★ | — | Label smoothing factor. |
| `tpo_alpha` | float | `1.0` | ★ | — | Weight of the supervised NLL term computed on the gold (`reference`) response in TPO training. Setting `tpo_alpha=0.0` disables the NLL term and skips the corresponding forward pass. |
| `tpo_l_gamma` | float | `0.5` | ★ | — | Target reward margin γ for the TPO-L loss, used only when `loss_type='tpo-l'`. |

### 取舍与坑

- ✅ 三路对比比 DPO 的成对对比信息更多
- ⚠️ 需要在线生成，训练成本高于 DPO
- ⚠️ 实验命名空间
- 🔻 tpo_alpha 把三项权重连在一起，需整体调

## <a id="online_dpo"></a>Online DPO（`online_dpo`）

- **全称**：Online Direct Preference Optimization（偏好对齐，实验）
- **配置类**：`OnlineDPOConfig` @ `trl/experimental/online_dpo/online_dpo_config.py`（继承链：`OnlineDPOConfig` → `_BaseConfig`）
- **训练器**：`OnlineDPOTrainer` @ `trl/experimental/online_dpo/online_dpo_trainer.py`
- **文档**：`docs/source/online_dpo_trainer.md` · 论文：<https://huggingface.co/papers/2402.04792>
- **启动**：`库内构造 OnlineDPOTrainer`
- **数据契约**：prompt；prompt

> DPO 的在线版本：训练时对每个 prompt 采样两条补全，用奖励模型/AI 反馈在线标注偏好，再用 DPO 损失更新——把“离线偏好数据”换成“在线 AI 反馈”。


### 训练流程映射

| 阶段 | 该算法在此阶段做什么 |
| --- | --- |
| 模型与参考/教师 | 策略模型（+ 可选参考模型或 reward_model_path 指向的奖励模型）。 |
| 数据与提示 | prompt。 |
| 监督信号 | reward_funcs 或 reward_model_path。 |
| 采样（rollout） | 每个 prompt 生成两条补全，缺失 EOS 可用 missing_eos_penalty 惩罚。 |
| 优势/偏好估计 | 两条补全的奖励差 → 在线偏好标签。 |
| 损失与更新 | DPO 损失（loss_type 与 beta）。 |
| 记录与产物 | rewards/* 与偏好准确率。 |

### 参数


**algorithm · 算法其他参数**

| 参数 | 类型 | 默认 | 关键 | 可选值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `logging_steps` | float | `10` |  | — | Log every X updates steps. Should be an integer or a float in range `[0,1)`. If smaller than 1, will be interpreted as ratio of total training steps. |
| `gradient_checkpointing` | bool | `True` |  | — | Enable gradient checkpointing to trade compute for memory. Reduces memory at the cost of ~20%% slower training. |
| `bf16` | bool | None | `None` |  | — | Whether to use bf16 (mixed) precision instead of 32-bit. Requires Ampere or higher NVIDIA architecture or Intel XPU or using CPU (use_cpu) or Ascend NPU. If not set, it defaults to `True` if `fp16` is not set. |
| `lr_scheduler_kwargs` | dict | str | None | `None` |  | — | Additional parameters for the lr_scheduler, such as {'num_cycles': 1} for cosine with hard restarts. See the documentation of each scheduler for possible values. |
| `use_liger_kernel` | bool | `False` |  | — | Enable Liger Kernel optimizations: transformers patches the model's layers with Liger's Triton kernels. Deprecated in `SFTTrainer`, `DPOTrainer`, `KTOTrainer`, `GRPOTrainer` and `RLOOTrainer`, and will be removed in v2.0.0: use `model_init… |
| `torch_empty_cache_steps` | int | None | `None` |  | — | Number of steps to wait before calling `torch.<device>.empty_cache()`. Helps avoid CUDA OOM at a cost of ~10%% slower performance. If None, cache will not be emptied. |

**损失与更新 · 训练超参**

| 参数 | 类型 | 默认 | 关键 | 可选值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `learning_rate` | float | `5e-07` |  | — | The initial learning rate for AdamW. |

**algorithm · 算法其他参数**

| 参数 | 类型 | 默认 | 关键 | 可选值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `remove_unused_columns` | bool | `False` |  | — | Whether or not to automatically remove the columns unused by the model forward method. |
| `reward_model_path` | str | None | `None` | ★ | — | Path to the reward model. |
| `max_new_tokens` | int | `64` | ★ | — | Maximum number of tokens to generate per completion. |
| `max_length` | int | `512` |  | — | Maximum total length of the sequence (prompt + completion) used to compute log probabilities. If the sequence exceeds this limit, the leftmost tokens will be truncated to preserve as much of the completion as possible. Not applied when the… |
| `temperature` | float | `0.9` |  | — | Temperature for sampling. The higher the temperature, the more random the completions. |

**采样 · Parameters that control generation**

| 参数 | 类型 | 默认 | 关键 | 可选值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `top_p` | float | `1.0` |  | — | Float that controls the cumulative probability of the top tokens to consider. Must be in (0, 1]. Set to 1.0 to consider all tokens. |
| `top_k` | int | `0` |  | — | Number of highest probability vocabulary tokens to keep for top-k-filtering. If `0`, top-k-filtering is disabled and all tokens are considered. |
| `min_p` | float | None | `None` |  | — | Minimum token probability, which will be scaled by the probability of the most likely token. It must be a value between 0.0 and 1.0. Typical values are in the 0.01-0.2 range. |
| `repetition_penalty` | float | `1.0` |  | — | Float that penalizes new tokens based on whether they appear in the prompt and the generated text so far. Values > 1.0 encourage the model to use new tokens, while values < 1.0 encourage the model to repeat tokens. |
| `generation_kwargs` | dict | None | `None` |  | — | Additional keyword arguments to pass to `GenerationConfig` (if using transformers) or `SamplingParams` (if using vLLM) when sampling completions. This can be used to further customize the generation behavior, such as setting `suppress_toke… |
| `cache_implementation` | str | None | `None` |  | — | Implementation of the cache method for faster generation when use_vllm is set to False. |

**algorithm · 算法其他参数**

| 参数 | 类型 | 默认 | 关键 | 可选值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `missing_eos_penalty` | float | None | `None` | ★ | — | Penalty applied to the score when the model fails to generate an EOS token. This is useful to encourage to generate completions shorter than the maximum length (`max_new_tokens`). The penalty must be a positive value. |
| `beta` | list[float] | `<lambda: [0.1]>` | ★ | — | Parameter controlling the deviation from the reference model. Higher β means less deviation from the reference model. For the IPO loss (`loss_type='ipo'`), β is the regularization parameter denoted by τ in the [paper](https://huggingface.c… |
| `loss_type` | str | `sigmoid` | ★ | `sigmoid`, `ipo` | Type of loss to use. |
| `disable_dropout` | bool | `True` |  | — | Whether to disable dropout in the model. |

**环境与资源 · Parameters that control generation acceleration powered by vLLM**

| 参数 | 类型 | 默认 | 关键 | 可选值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `use_vllm` | bool | `False` | ★ | — | Whether to use vLLM for generating completions. Requires vLLM to be installed (`pip install trl[vllm]`). |
| `vllm_model_impl` | str | `vllm` |  | `vllm`, `transformers` | Model implementation to use for vLLM. Must be one of `transformers` or `vllm`. `transformers`: Use the `transformers` backend for model implementation. `vllm`: Use the `vllm` library for model implementation. |
| `vllm_structured_outputs_regex` | str | None | `None` |  | — | Regex for vLLM structured outputs. If `None` (default), structured outputs is disabled. |

**环境与资源 · Parameters that control colocated vLLM execution (only used when `vllm_mode` is `"colocate"`)**

| 参数 | 类型 | 默认 | 关键 | 可选值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `vllm_gpu_memory_utilization` | float | None | `0.55` |  | — | Control the GPU memory utilization for vLLM. This setting only applies when `vllm_mode` is set to `'colocate'`. If you are using `vllm_mode='server'`, this parameter must be passed separately when launching the vLLM server via the `--vllm_… |

**环境与资源 · Parameters that control generation acceleration powered by vLLM**

| 参数 | 类型 | 默认 | 关键 | 可选值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `vllm_mode` | str | `colocate` |  | `colocate`, `server` | Mode to use for vLLM integration when `use_vllm` is set to `True`. Must be one of `'server'` or `'colocate'`. `'server'`: The trainer will send generation requests to a separate vLLM server. Make sure a vLLM server is running (start with `… |

**环境与资源 · Parameters that control the vLLM server (only used when `vllm_mode` is `"server"`)**

| 参数 | 类型 | 默认 | 关键 | 可选值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `vllm_server_base_url` | str | None | `None` |  | — | Base URL for the vLLM server (e.g., 'http://localhost:8000'). If provided, `vllm_server_host` and `vllm_server_port` are ignored. |
| `vllm_server_host` | str | `0.0.0.0` |  | — | Host of the vLLM server to connect to. Ignored if vllm_server_base_url is provided. |
| `vllm_server_port` | int | `8000` |  | — | Port of the vLLM server to connect to. Ignored if vllm_server_base_url is provided. |
| `vllm_server_timeout` | float | `240.0` |  | — | Total timeout duration in seconds to wait for the vLLM server to be up. If the server is not up after the timeout, a `ConnectionError` is raised. |
| `vllm_group_port` | int | `51216` |  | — | Port number for the weight update group. This is used to communicate with the vLLM server. Unless the port is occupied, there is no need to change it. |

**环境与资源 · Parameters that control colocated vLLM execution (only used when `vllm_mode` is `"colocate"`)**

| 参数 | 类型 | 默认 | 关键 | 可选值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `vllm_tensor_parallel_size` | int | `1` |  | — | Control the tensor parallel size for vLLM. This setting only applies when `vllm_mode` is set to `'colocate'`. If you are using `vllm_mode='server'`, this parameter must be passed separately when launching the vLLM server via the `--vllm_te… |
| `vllm_enable_sleep_mode` | bool | `False` |  | — | Enable vLLM sleep mode to offload weights/cache during the optimizer step. Keeps GPU memory usage low, but waking the engine adds host–device transfer latency. |

**损失与更新 · Other parameters**

| 参数 | 类型 | 默认 | 关键 | 可选值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `ds3_gather_for_generation` | bool | `True` |  | — | This setting applies to DeepSpeed ZeRO-3. If enabled, the policy model weights are gathered for generation, improving generation speed. However, disabling this option allows training models that exceed the VRAM capacity of a single GPU, al… |
| `model_init_kwargs` | dict[str, Any] | str | None | `None` |  | — | Keyword arguments to pass to `AutoModelForCausalLM.from_pretrained` when instantiating the model from a string. The `revision` value is also used when loading reward processing classes. |
| `trust_remote_code` | bool | `False` |  | — | Whether to allow loading models that ship custom Python code from the Hub. Forwarded to `AutoModelForCausalLM.from_pretrained`. Also applied to reward-model and reward-tokenizer loads. |
| `reward_weights` | list[float] | None | `None` |  | — | Weights for combining multiple reward functions. Must match the number of reward functions. If None, all reward functions are equally weighted. |

### 取舍与坑

- ✅ 偏好在训练中持续更新，避免离线数据分布漂移
- ✅ 仍保留 DPO 的稳定损失形式
- ⚠️ 奖励模型质量直接决定上限
- ⚠️ 比离线 DPO 多了生成开销
- 🔻 奖励模型与策略模型共享显存，注意 vllm_gpu_memory_utilization

## <a id="distillation"></a>Distillation（`distillation`）

- **全称**：On-policy Knowledge Distillation（蒸馏，稳定）
- **配置类**：`DistillationConfig` @ `trl/trainer/distillation_config.py`（继承链：`DistillationConfig` → `_BaseConfig`）
- **训练器**：`DistillationTrainer` @ `trl/trainer/distillation_trainer.py`
- **文档**：`docs/source/distillation_trainer.md` · 论文：<https://huggingface.co/papers/2306.13649>
- **启动**：`trl distillation`
- **数据契约**：prompt；prompt（也支持 prompt+completion 的混合）

> 学生在自己的 on-policy 补全上学习教师的下一 token 分布，用广义 JSD（beta 在 forward KL 与 reverse KL 之间插值）作为目标，不需要奖励函数。


### 训练流程映射

| 阶段 | 该算法在此阶段做什么 |
| --- | --- |
| 模型与参考/教师 | 学生模型 + 教师模型（teacher_model_name_or_path，可本地或自定义 init 参数）。 |
| 数据与提示 | prompt。 |
| 监督信号 | 教师的下一 token 分布（分布级信号，而非标量奖励）。 |
| 采样（rollout） | 学生生成补全，max_completion_length 控制长度。 |
| 优势/偏好估计 | 学生/教师在该补全上的逐 token 分布差（JSD 插值）。 |
| 损失与更新 | 广义 JSD 损失，beta=0 前向 KL、beta=1 反向 KL。 |
| 记录与产物 | 蒸馏损失、perplexity 类指标与 checkpoint。 |

### 参数


**algorithm · 算法其他参数**

| 参数 | 类型 | 默认 | 关键 | 可选值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `logging_steps` | float | `10` |  | — | Log every X updates steps. Should be an integer or a float in range `[0,1)`. If smaller than 1, will be interpreted as ratio of total training steps. |
| `gradient_checkpointing` | bool | `True` |  | — | Enable gradient checkpointing to trade compute for memory. Reduces memory at the cost of ~20%% slower training. |
| `bf16` | bool | None | `None` |  | — | Whether to use bf16 (mixed) precision instead of 32-bit. Requires Ampere or higher NVIDIA architecture or Intel XPU or using CPU (use_cpu) or Ascend NPU. If not set, it defaults to `True` if `fp16` is not set. |
| `lr_scheduler_kwargs` | dict | str | None | `None` |  | — | Additional parameters for the lr_scheduler, such as {'num_cycles': 1} for cosine with hard restarts. See the documentation of each scheduler for possible values. |
| `use_liger_kernel` | bool | `False` |  | — | Enable Liger Kernel optimizations: transformers patches the model's layers with Liger's Triton kernels. Deprecated in `SFTTrainer`, `DPOTrainer`, `KTOTrainer`, `GRPOTrainer` and `RLOOTrainer`, and will be removed in v2.0.0: use `model_init… |
| `torch_empty_cache_steps` | int | None | `None` |  | — | Number of steps to wait before calling `torch.<device>.empty_cache()`. Helps avoid CUDA OOM at a cost of ~10%% slower performance. If None, cache will not be emptied. |

**损失与更新 · 训练超参**

| 参数 | 类型 | 默认 | 关键 | 可选值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `learning_rate` | float | `1e-06` |  | — | The initial learning rate for AdamW. |

**模型与参考/教师 · Parameters that control the model and the teacher model**

| 参数 | 类型 | 默认 | 关键 | 可选值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `model_init_kwargs` | dict[str, Any] | str | None | `None` |  | — | Keyword arguments for `AutoModelForCausalLM.from_pretrained`, used when the `model` argument of the trainer is provided as a string. The `revision` value is also used when loading the processing class. |
| `trust_remote_code` | bool | `False` |  | — | Whether to allow loading models and tokenizers that ship custom Python code from the Hub. Forwarded to `AutoModelForCausalLM.from_pretrained` and `AutoTokenizer.from_pretrained`, for both the student and teacher. |
| `teacher_model_name_or_path` | str | None | `None` | ★ | — | Model name or path for the teacher model. |
| `teacher_model_revision` | str | None | `None` |  | — | Model revision of the teacher model (e.g., branch name, tag, or commit hash). |
| `teacher_model_init_kwargs` | dict[str, Any] | str | None | `None` |  | — | Keyword arguments for `AutoModelForCausalLM.from_pretrained` when instantiating the teacher. |
| `disable_dropout` | bool | `False` |  | — | Whether to disable dropout in the student model during training. |

**数据与提示 · Parameters that control the data preprocessing**

| 参数 | 类型 | 默认 | 关键 | 可选值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `remove_unused_columns` | bool | None | `False` |  | — | Whether to only keep the column 'prompt' in the dataset. The trainer consumes the raw prompt column and generates completions on-policy, so it defaults to `False`. |
| `max_completion_length` | int | None | `512` | ★ | — | Maximum number of tokens to generate per completion. |
| `ds3_gather_for_generation` | bool | `True` |  | — | This setting applies to DeepSpeed ZeRO-3. If enabled, the policy model weights are gathered for generation, improving generation speed. However, disabling this option allows training models that exceed the VRAM capacity of a single GPU, al… |
| `shuffle_dataset` | bool | None | `True` |  | — | Whether to shuffle the training dataset. |
| `pad_to_multiple_of` | int | str | None | `auto` |  | — | If set, the prompts ids and completions ids will be padded to a multiple of this value. `'auto'` pads to a multiple of 128 on Neuron (AWS Trainium), which compiles once per input shape, and does not pad elsewhere. |

**采样 · Parameters that control generation**

| 参数 | 类型 | 默认 | 关键 | 可选值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `temperature` | float | `1.0` | ★ | — | Temperature for sampling and loss computation. Higher values produce softer distributions. |
| `top_p` | float | `1.0` |  | — | Top-p (nucleus) sampling parameter for on-policy generation. |
| `top_k` | int | `0` |  | — | Top-k sampling parameter for on-policy generation. 0 disables top-k filtering. |
| `min_p` | float | None | `None` |  | — | Minimum token probability, which will be scaled by the probability of the most likely token. It must be a value between 0.0 and 1.0. Typical values are in the 0.01-0.2 range. |
| `generation_kwargs` | dict | str | None | `None` |  | — | Additional keyword arguments to pass to `GenerationConfig` (if using transformers) or `SamplingParams` (if using vLLM) when sampling completions. This can be used to further customize the generation behavior, such as setting `suppress_toke… |
| `chat_template_kwargs` | dict | str | None | `None` |  | — | Additional keyword arguments to pass to the `apply_chat_template` function when generating completions. |
| `repetition_penalty` | float | `1.0` |  | — | Float that penalizes new tokens based on whether they appear in the prompt and the generated text so far. Values > 1.0 encourage the model to use new tokens, while values < 1.0 encourage the model to repeat tokens. |
| `cache_implementation` | str | None | `None` |  | — | Implementation of the cache method for faster generation when use_vllm is set to False. |

**环境与资源 · Parameters that control generation acceleration powered by vLLM**

| 参数 | 类型 | 默认 | 关键 | 可选值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `use_vllm` | bool | `False` | ★ | — | Whether to use vLLM for generating on-policy completions from the student model. |
| `vllm_mode` | str | `colocate` | ★ | `colocate`, `server` | Mode for student vLLM integration. Either "server" or "colocate". |
| `vllm_model_impl` | str | `vllm` |  | `vllm`, `transformers` | Model implementation backend for vLLM. Use "vllm" or "transformers". |
| `vllm_enable_sleep_mode` | bool | `False` |  | — | Enable vLLM sleep mode to offload student weights during the optimizer step. |
| `vllm_structured_outputs_regex` | str | None | `None` |  | — | Regex pattern for vLLM structured outputs. |

**环境与资源 · Parameters that control the vLLM server (only used when `vllm_mode` is `"server"`)**

| 参数 | 类型 | 默认 | 关键 | 可选值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `vllm_server_base_url` | str | None | `None` |  | — | Base URL for the student vLLM server. |
| `vllm_server_host` | str | `0.0.0.0` |  | — | Host of the student vLLM server. |
| `vllm_server_port` | int | `8000` |  | — | Port of the student vLLM server. |
| `vllm_server_timeout` | float | `240.0` |  | — | Timeout for connecting to the student vLLM server. |
| `vllm_group_port` | int | `51216` |  | — | Port for the vLLM weight-update group (NCCL communicator). |

**环境与资源 · Parameters that control colocated vLLM execution (only used when `vllm_mode` is `"colocate"`)**

| 参数 | 类型 | 默认 | 关键 | 可选值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `vllm_gpu_memory_utilization` | float | `0.3` |  | — | GPU memory utilization for the colocated student vLLM engine. |
| `vllm_max_model_length` | int | None | `None` |  | — | Maximum model sequence length for the colocated vLLM engine. |
| `vllm_tensor_parallel_size` | int | `1` |  | — | Tensor parallel size for the colocated student vLLM engine. |

**损失与更新 · Parameters that control the training**

| 参数 | 类型 | 默认 | 关键 | 可选值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `beta` | float | `1.0` | ★ | — | Interpolation coefficient for the Generalized JSD loss. 0.0 = forward KL, 0.5 = JSD, 1.0 = reverse KL. Unlike GRPO's `beta` (a KL-penalty coefficient against a reference model), here it selects the divergence itself; there is no reference-… |
| `max_tool_calling_iterations` | int | None | `None` |  | — | Maximum number of tool-calling turns when training an agent. If `None`, there is no limit and generation stops when the model generates a response turn with no tool calls or when the total response length reaches `max_model_length`. |

**记录与产物 · Parameters that control the logging**

| 参数 | 类型 | 默认 | 关键 | 可选值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `log_completions` | bool | `False` |  | — | Whether to log a sample of (prompt, completion) pairs every `logging_steps` steps. If `rich` is installed, it prints the sample. If `wandb` and/or `trackio` logging is enabled, it logs it to `wandb` and/or `trackio`. |
| `num_completions_to_print` | int | None | `None` |  | — | Number of completions to print with `rich`. If `None`, all completions are logged. |
| `log_unique_prompts` | bool | `False` |  | — | Whether to log unique prompts. If `True`, only unique prompts are logged. If `False`, all prompts are logged. |

### 取舍与坑

- ✅ 不需要奖励函数或偏好数据
- ✅ 分布级信号比标量奖励密集，蒸馏稳定
- ⚠️ 必须有教师模型，且要与学生共享 tokenizer（跨 tokenizer 见 GOLD）
- ⚠️ 生成开销与 RL 相当
- 🔻 教师模型的 chat template 与学生不同会让分布对齐错位

## <a id="gkd"></a>GKD（`gkd`）

- **全称**：Generalized Knowledge Distillation（蒸馏，实验）
- **配置类**：`GKDConfig` @ `trl/experimental/gkd/gkd_config.py`（继承链：`GKDConfig` → `SFTConfig` → `_BaseConfig`）
- **训练器**：`GKDTrainer` @ `trl/experimental/gkd/gkd_trainer.py`
- **文档**：`docs/source/gkd_trainer.md` · 论文：<https://huggingface.co/papers/2306.13649>
- **启动**：`库内构造 GKDTrainer`
- **数据契约**：prompt；prompt 或 prompt+completion（off-policy 部分）

> GKD 的经典实现：在 on-policy 与 off-policy（数据集自带补全）之间用 lmbda 插值，损失在序列级 KD 与广义 JSD 之间用 beta 插值，可选的 seq_kd 允许教师生成补全。


### 训练流程映射

| 阶段 | 该算法在此阶段做什么 |
| --- | --- |
| 模型与参考/教师 | 学生 + 教师（GKDConfig 继承 SFTConfig，因此沿用 SFT 的数据/打包参数）。 |
| 数据与提示 | prompt（可带补全做 off-policy 监督）。 |
| 监督信号 | 教师分布 + 数据集补全。 |
| 采样（rollout） | 按 lmbda 比例采样学生自己的补全。 |
| 优势/偏好估计 | 逐 token 分布差（beta 插值）。 |
| 损失与更新 | 广义 JSD / 序列级 KD 损失。 |
| 记录与产物 | 同 SFTTrainer 的指标与 checkpoint。 |

### 参数


**algorithm · 算法其他参数**

| 参数 | 类型 | 默认 | 关键 | 可选值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `logging_steps` | float | `10` |  | — | Log every X updates steps. Should be an integer or a float in range `[0,1)`. If smaller than 1, will be interpreted as ratio of total training steps. |
| `gradient_checkpointing` | bool | `True` |  | — | Enable gradient checkpointing to trade compute for memory. Reduces memory at the cost of ~20%% slower training. |
| `bf16` | bool | None | `None` |  | — | Whether to use bf16 (mixed) precision instead of 32-bit. Requires Ampere or higher NVIDIA architecture or Intel XPU or using CPU (use_cpu) or Ascend NPU. If not set, it defaults to `True` if `fp16` is not set. |
| `lr_scheduler_kwargs` | dict | str | None | `None` |  | — | Additional parameters for the lr_scheduler, such as {'num_cycles': 1} for cosine with hard restarts. See the documentation of each scheduler for possible values. |
| `use_liger_kernel` | bool | `False` |  | — | Enable Liger Kernel optimizations: transformers patches the model's layers with Liger's Triton kernels. Deprecated in `SFTTrainer`, `DPOTrainer`, `KTOTrainer`, `GRPOTrainer` and `RLOOTrainer`, and will be removed in v2.0.0: use `model_init… |
| `torch_empty_cache_steps` | int | None | `None` |  | — | Number of steps to wait before calling `torch.<device>.empty_cache()`. Helps avoid CUDA OOM at a cost of ~10%% slower performance. If None, cache will not be emptied. |

**损失与更新 · 训练超参**

| 参数 | 类型 | 默认 | 关键 | 可选值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `learning_rate` | float | `2e-05` |  | — | The initial learning rate for AdamW. |

**模型与参考/教师 · Parameters that control the model**

| 参数 | 类型 | 默认 | 关键 | 可选值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `model_init_kwargs` | dict[str, Any] | str | None | `None` |  | — | Keyword arguments for `AutoModelForCausalLM.from_pretrained`, used when the `model` argument of the `SFTTrainer` is provided as a string. The `revision` value is also used when loading the processing class. |
| `trust_remote_code` | bool | `False` |  | — | Whether to allow loading models and tokenizers that ship custom Python code from the Hub. Forwarded to `AutoModelForCausalLM.from_pretrained` and `AutoProcessor.from_pretrained`. |
| `router_aux_loss_coef` | float | None | `None` |  | — | Coefficient of the load-balancing auxiliary loss for Mixture-of-Experts (MoE) models, added to the training loss with this weight. When not set, the value declared by the model config is used. Set to `0.0` to disable it. Fails when used wi… |
| `chat_template_path` | str | None | `None` |  | — | If specified, sets the model's chat template. This can either be the path to a tokenizer (local directory or Hugging Face Hub model) or a direct path to a Jinja template file. When using a Jinja file, you must ensure that any special token… |

**数据与提示 · Parameters that control the data preprocessing**

| 参数 | 类型 | 默认 | 关键 | 可选值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `dataset_text_field` | str | `text` |  | — | Name of the column that contains text data in the dataset. |
| `dataset_kwargs` | dict[str, Any] | str | None | `None` |  | — | Dictionary of optional keyword arguments for the dataset preparation. The only supported key is `skip_prepare_dataset`. When the dataset contains images, `skip_prepare_dataset` is automatically treated as `True` regardless of the provided… |
| `dataset_num_proc` | int | None | `None` |  | — | Number of processes to use for processing the dataset. |
| `eos_token` | str | None | `None` |  | — | Token used to indicate the end of a turn or sequence. If `None`, it defaults to `processing_class.eos_token`. |
| `max_length` | int | None | `1024` |  | — | Maximum length of the tokenized sequence. Sequences longer than `max_length` are truncated from the left or right depending on the `truncation_mode`. If `None`, no truncation is applied. When packing is enabled, this value sets the sequenc… |
| `truncation_mode` | str | `keep_start` |  | `keep_start`, `keep_end` | Truncation mode to use when the sequence exceeds `max_length`. The only supported value is `'keep_start'`. The `'keep_end'` value is deprecated and will be removed in v2.0.0. |
| `shuffle_dataset` | bool | `False` |  | — | Whether to shuffle the dataset. |
| `packing` | bool | `False` |  | — | Whether to group multiple sequences into fixed-length blocks to improve computational efficiency and reduce padding. Uses `max_length` to define sequence length. |
| `packing_strategy` | str | `bfd` |  | `bfd`, `bfd_split`, `wrapped` | Strategy for packing sequences. Can be `'bfd'` (best-fit decreasing, truncates overflow), `'bfd_split'` (best-fit decreasing, splits overflow sequences), or `'wrapped'` (aggressive, cuts mid-sequence). |
| `padding_free` | bool | `False` |  | — | Whether to perform forward passes without padding by flattening all sequences in the batch into a single continuous sequence. This reduces memory usage by eliminating padding overhead. Currently, this is only supported with the FlashAttent… |
| `pad_to_multiple_of` | int | str | None | `auto` |  | — | If set, the sequences will be padded to a multiple of this value. `'auto'` pads to a multiple of 128 on Neuron (AWS Trainium), which compiles once per input shape, and does not pad elsewhere. |
| `eval_packing` | bool | None | `None` |  | — | Whether to pack the eval dataset. If `None`, uses the same value as `packing`. |

**损失与更新 · Parameters that control the training**

| 参数 | 类型 | 默认 | 关键 | 可选值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `completion_only_loss` | bool | None | `None` |  | — | Whether to compute loss only on the completion part of the sequence. If set to `True`, loss is computed only on the completion, which is supported only for prompt-completion datasets. If `False`, loss is computed on the entire sequence. If… |
| `assistant_only_loss` | bool | `False` |  | — | Whether to compute loss only on the assistant part of the sequence. If set to `True`, loss is computed only on the assistant responses, which is supported only for conversational datasets. If `False`, loss is computed on the entire sequenc… |
| `loss_type` | str | `nll` |  | `nll`, `dft`, `chunked_nll` | Type of loss to use. Possible values are `'nll'` (standard negative log-likelihood) and `'dft'` (Dynamic Fine-Tuning, https://huggingface.co/papers/2508.05629). |
| `activation_offloading` | bool | `False` |  | — | Whether to offload the activations to the CPU. |

**数据与提示 · 数据与分词器**

| 参数 | 类型 | 默认 | 关键 | 可选值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `pad_token` | str | None | `None` |  | — | Deprecated. Set `tokenizer.pad_token` directly and pass it as `processing_class` to the trainer instead. |

**algorithm · 算法其他参数**

| 参数 | 类型 | 默认 | 关键 | 可选值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `temperature` | float | `0.9` | ★ | — | Temperature for sampling. The higher the temperature, the more random the completions. |
| `lmbda` | float | `0.5` | ★ | — | Lambda parameter that controls the student data fraction (i.e., the proportion of on-policy student-generated outputs). |
| `beta` | float | `0.5` | ★ | — | Interpolation coefficient between `0.0` and `1.0` of the Generalized Jensen-Shannon Divergence loss. When beta is `0.0`, the loss is the KL divergence. When beta is `1.0`, the loss is the Inverse KL Divergence. |
| `max_new_tokens` | int | `128` | ★ | — | Maximum number of tokens to generate per completion. |

**模型与参考/教师 · 教师模型**

| 参数 | 类型 | 默认 | 关键 | 可选值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `teacher_model_name_or_path` | str | None | `None` | ★ | — | Model name or path of the teacher model. If `None`, the teacher model will be the same as the model being trained. |
| `teacher_model_init_kwargs` | dict[str, Any] | str | None | `None` |  | — | Keyword arguments to pass to `AutoModelForCausalLM.from_pretrained` when instantiating the teacher model from a string. |

**algorithm · 算法其他参数**

| 参数 | 类型 | 默认 | 关键 | 可选值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `disable_dropout` | bool | `True` |  | — | Whether to disable dropouts in `model`. |
| `seq_kd` | bool | `False` | ★ | — | Seq_kd parameter that controls whether to perform Sequence-Level KD (can be viewed as supervised FT on teacher-generated output). |

### 取舍与坑

- ✅ on/off-policy 可调，稳定；实现成熟
- ⚠️ 继承 SFTConfig，参数面板与 SFT 混在一起
- ⚠️ 需要教师模型
- 🔻 lmbda=0 退化为纯离线蒸馏（等价 SFT 式 KD），不再是 on-policy

## <a id="gold"></a>GOLD（`gold`）

- **全称**：General Online Logit Distillation（蒸馏，实验）
- **配置类**：`GOLDConfig` @ `trl/experimental/gold/gold_config.py`（继承链：`GOLDConfig` → `SFTConfig` → `_BaseConfig`）
- **训练器**：`GOLDTrainer` @ `trl/experimental/gold/gold_trainer.py`
- **文档**：`docs/source/gold_trainer.md`
- **启动**：`库内构造 GOLDTrainer`
- **数据契约**：prompt；prompt / ChatML 消息列表（子类 SFTTrainer）

> ULD（Universal Logit Distillation）的扩展：支持师生 tokenizer 不同，通过对齐可见文本片段、合并 logits 来构造损失；可选混合损失在词表匹配的 token 上用精确比对。


### 训练流程映射

| 阶段 | 该算法在此阶段做什么 |
| --- | --- |
| 模型与参考/教师 | 学生 + 教师 + 教师 tokenizer（use_uld_loss=True 时必填 teacher_tokenizer_name_or_path）。 |
| 数据与提示 | 同 SFT 的数据契约。 |
| 监督信号 | 教师的 logits（跨 tokenizer 对齐后）。 |
| 采样（rollout） | 学生 on-policy 生成。 |
| 优势/偏好估计 | BULK/跨 tokenizer 的 logit 对齐（uld_token_merge_strategy、use_extended_uld）。 |
| 损失与更新 | ULD 损失 + 交叉熵蒸馏项（uld_crossentropy_weight / uld_distillation_weight）。 |
| 记录与产物 | 同 SFT + 蒸馏指标。 |

### 参数


**algorithm · 算法其他参数**

| 参数 | 类型 | 默认 | 关键 | 可选值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `logging_steps` | float | `10` |  | — | Log every X updates steps. Should be an integer or a float in range `[0,1)`. If smaller than 1, will be interpreted as ratio of total training steps. |
| `gradient_checkpointing` | bool | `True` |  | — | Enable gradient checkpointing to trade compute for memory. Reduces memory at the cost of ~20%% slower training. |
| `bf16` | bool | None | `None` |  | — | Whether to use bf16 (mixed) precision instead of 32-bit. Requires Ampere or higher NVIDIA architecture or Intel XPU or using CPU (use_cpu) or Ascend NPU. If not set, it defaults to `True` if `fp16` is not set. |
| `lr_scheduler_kwargs` | dict | str | None | `None` |  | — | Additional parameters for the lr_scheduler, such as {'num_cycles': 1} for cosine with hard restarts. See the documentation of each scheduler for possible values. |
| `use_liger_kernel` | bool | `False` |  | — | Enable Liger Kernel optimizations: transformers patches the model's layers with Liger's Triton kernels. Deprecated in `SFTTrainer`, `DPOTrainer`, `KTOTrainer`, `GRPOTrainer` and `RLOOTrainer`, and will be removed in v2.0.0: use `model_init… |
| `torch_empty_cache_steps` | int | None | `None` |  | — | Number of steps to wait before calling `torch.<device>.empty_cache()`. Helps avoid CUDA OOM at a cost of ~10%% slower performance. If None, cache will not be emptied. |

**损失与更新 · 训练超参**

| 参数 | 类型 | 默认 | 关键 | 可选值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `learning_rate` | float | `1e-07` |  | — | The initial learning rate for AdamW. |

**模型与参考/教师 · Parameters that control the model**

| 参数 | 类型 | 默认 | 关键 | 可选值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `model_init_kwargs` | dict[str, Any] | str | None | `None` |  | — | Keyword arguments for `AutoModelForCausalLM.from_pretrained`, used when the `model` argument of the `SFTTrainer` is provided as a string. The `revision` value is also used when loading the processing class. |
| `trust_remote_code` | bool | `False` |  | — | Whether to allow loading models and tokenizers that ship custom Python code from the Hub. Forwarded to `AutoModelForCausalLM.from_pretrained` and `AutoProcessor.from_pretrained`. |
| `router_aux_loss_coef` | float | None | `None` |  | — | Coefficient of the load-balancing auxiliary loss for Mixture-of-Experts (MoE) models, added to the training loss with this weight. When not set, the value declared by the model config is used. Set to `0.0` to disable it. Fails when used wi… |
| `chat_template_path` | str | None | `None` |  | — | If specified, sets the model's chat template. This can either be the path to a tokenizer (local directory or Hugging Face Hub model) or a direct path to a Jinja template file. When using a Jinja file, you must ensure that any special token… |

**数据与提示 · Parameters that control the data preprocessing**

| 参数 | 类型 | 默认 | 关键 | 可选值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `dataset_text_field` | str | `text` |  | — | Name of the column that contains text data in the dataset. |
| `dataset_kwargs` | dict[str, Any] | str | None | `None` |  | — | Dictionary of optional keyword arguments for the dataset preparation. The only supported key is `skip_prepare_dataset`. When the dataset contains images, `skip_prepare_dataset` is automatically treated as `True` regardless of the provided… |
| `dataset_num_proc` | int | None | `None` |  | — | Number of processes to use for processing the dataset. |
| `eos_token` | str | None | `None` |  | — | Token used to indicate the end of a turn or sequence. If `None`, it defaults to `processing_class.eos_token`. |
| `max_length` | int | None | `1024` |  | — | Maximum length of the tokenized sequence. Sequences longer than `max_length` are truncated from the left or right depending on the `truncation_mode`. If `None`, no truncation is applied. When packing is enabled, this value sets the sequenc… |
| `truncation_mode` | str | `keep_start` |  | `keep_start`, `keep_end` | Truncation mode to use when the sequence exceeds `max_length`. The only supported value is `'keep_start'`. The `'keep_end'` value is deprecated and will be removed in v2.0.0. |
| `shuffle_dataset` | bool | `False` |  | — | Whether to shuffle the dataset. |
| `packing` | bool | `False` |  | — | Whether to group multiple sequences into fixed-length blocks to improve computational efficiency and reduce padding. Uses `max_length` to define sequence length. |
| `packing_strategy` | str | `bfd` |  | `bfd`, `bfd_split`, `wrapped` | Strategy for packing sequences. Can be `'bfd'` (best-fit decreasing, truncates overflow), `'bfd_split'` (best-fit decreasing, splits overflow sequences), or `'wrapped'` (aggressive, cuts mid-sequence). |
| `padding_free` | bool | `False` |  | — | Whether to perform forward passes without padding by flattening all sequences in the batch into a single continuous sequence. This reduces memory usage by eliminating padding overhead. Currently, this is only supported with the FlashAttent… |
| `pad_to_multiple_of` | int | str | None | `auto` |  | — | If set, the sequences will be padded to a multiple of this value. `'auto'` pads to a multiple of 128 on Neuron (AWS Trainium), which compiles once per input shape, and does not pad elsewhere. |
| `eval_packing` | bool | None | `None` |  | — | Whether to pack the eval dataset. If `None`, uses the same value as `packing`. |

**损失与更新 · Parameters that control the training**

| 参数 | 类型 | 默认 | 关键 | 可选值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `completion_only_loss` | bool | None | `None` |  | — | Whether to compute loss only on the completion part of the sequence. If set to `True`, loss is computed only on the completion, which is supported only for prompt-completion datasets. If `False`, loss is computed on the entire sequence. If… |
| `assistant_only_loss` | bool | `False` |  | — | Whether to compute loss only on the assistant part of the sequence. If set to `True`, loss is computed only on the assistant responses, which is supported only for conversational datasets. If `False`, loss is computed on the entire sequenc… |
| `loss_type` | str | `nll` |  | `nll`, `dft`, `chunked_nll` | Type of loss to use. Possible values are `'nll'` (standard negative log-likelihood) and `'dft'` (Dynamic Fine-Tuning, https://huggingface.co/papers/2508.05629). |
| `activation_offloading` | bool | `False` |  | — | Whether to offload the activations to the CPU. |

**数据与提示 · 数据与分词器**

| 参数 | 类型 | 默认 | 关键 | 可选值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `pad_token` | str | None | `None` |  | — | Deprecated. Set `tokenizer.pad_token` directly and pass it as `processing_class` to the trainer instead. |

**采样 · Parameters that control generation and the training loop**

| 参数 | 类型 | 默认 | 关键 | 可选值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `temperature` | float | `0.9` |  | — | Temperature for sampling. The higher the temperature, the more random the completions. |
| `top_p` | float | `0.95` |  | — | If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to `top_p` or higher are kept for generation. |
| `top_k` | int | `0` |  | — | Number of highest probability vocabulary tokens to keep for top-k-filtering. If `0`, top-k-filtering is disabled and all tokens are considered. |
| `lmbda` | float | `0.5` | ★ | — | Lambda parameter that controls the student data fraction (i.e., the proportion of on-policy student-generated outputs). |
| `beta` | float | `0.5` | ★ | — | Interpolation coefficient between `0.0` and `1.0` of the Generalized Jensen-Shannon Divergence loss. When beta is `0.0`, the loss is the KL divergence. When beta is `1.0`, the loss is the Inverse KL Divergence. |
| `max_completion_length` | int | `128` |  | — | Maximum number of tokens to generate per completion. |
| `teacher_model_name_or_path` | str | None | `None` |  | — | Model name or path of the teacher model. If `None`, the teacher model will be the same as the model being trained. |
| `teacher_model_revision` | str | None | `None` |  | — | Model revision of the teacher model (e.g., branch name, tag, or commit hash). If `None`, the default revision is used. |
| `teacher_model_init_kwargs` | dict[str, Any] | str | None | `None` |  | — | Keyword arguments to pass to `AutoModelForCausalLM.from_pretrained` when instantiating the teacher model from a string. The `revision` value is also used when loading the teacher processing class. |
| `teacher_tokenizer_name_or_path` | str | None | `None` | ★ | — | Tokenizer name or path for the teacher model. If None when using ULD loss, will use the same tokenizer as the student model (not recommended for cross-tokenizer distillation). |
| `disable_dropout` | bool | `True` |  | — | Whether to disable dropouts in `model`. |
| `seq_kd` | bool | `False` | ★ | — | Seq_kd parameter that controls whether to perform Sequence-Level KD (can be viewed as supervised FT on teacher-generated output). |
| `num_generations` | int | `1` |  | — | Number of generations per prompt. Increasing this will decrease the number of unique prompts per optimization step. |
| `generation_batch_size` | int | None | `None` |  | — | Number of unique prompts per worker per optimizer step. If None, computed from (per_device_train_batch_size * gradient_accumulation_steps) // num_generations. |

**损失与更新 · Parameters that control the ULD loss**

| 参数 | 类型 | 默认 | 关键 | 可选值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `use_uld_loss` | bool | `False` | ★ | — | Whether to use Universal Logit Distillation (ULD) loss instead of Generalized Jensen-Shannon Divergence loss. |
| `uld_token_merge_strategy` | str | `observed` | ★ | `observed`, `bayesian` | Strategy used to align answer logits and merge token probabilities in the ULD loss. With "observed", the answer logits are sliced at the answer positions and split tokens (when use_extended_uld=True) are merged by multiplying the marginal… |
| `use_extended_uld` | bool | `True` |  | — | Whether to enable extended ULD alignment that uses tokenizers to align and merge token probabilities across student and teacher tokenizations. When True, the trainer will compute token mappings and merge probabilities for split tokens; whe… |
| `uld_use_hybrid_loss` | bool | `False` | ★ | — | Whether to use a hybrid loss that combines ULD loss and JSD loss. When True, the final loss is a a combination of JSD for known token mappings and ULD for unknown token mappings. |
| `uld_hybrid_matched_weight` | float | None | `None` | ★ | — | Weight for the matched token loss component when using hybrid ULD + JSD loss. This weight scales the JSD loss computed over tokens that have a direct mapping between student and teacher tokenizations. If None, uses adaptive weighting based… |
| `uld_hybrid_unmatched_weight` | float | None | `None` | ★ | — | Weight for the unmatched token loss component when using hybrid ULD + JSD loss. This weight scales the ULD loss computed over tokens that do not have a direct mapping between student and teacher tokenizations. If None, uses adaptive weight… |
| `uld_crossentropy_weight` | float | `0.0` | ★ | — | Weight for the cross-entropy loss component in ULD loss. |
| `uld_distillation_weight` | float | `1.0` | ★ | — | Weight for the distillation loss component in ULD loss. |
| `uld_student_temperature` | float | `1.0` |  | — | Temperature for student logits in ULD loss computation. |
| `uld_teacher_temperature` | float | `1.0` |  | — | Temperature for teacher logits in ULD loss computation. |
| `uld_skip_student_eos` | bool | `True` |  | — | Whether to skip EOS token for student in ULD loss computation. |
| `uld_skip_teacher_eos` | bool | `True` |  | — | Whether to skip EOS token for teacher in ULD loss computation. |

**环境与资源 · Parameters that control vLLM integration**

| 参数 | 类型 | 默认 | 关键 | 可选值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `use_vllm` | bool | `False` |  | — | Whether to use vLLM for generating completions. Requires `vllm` to be installed. |
| `vllm_mode` | str | `colocate` |  | `colocate`, `server` | Mode for vLLM integration. Either "server" (connect to a running vLLM server) or "colocate" (run vLLM in the same process). |
| `vllm_server_base_url` | str | None | `None` |  | — | Base URL for the vLLM server (e.g., "http://localhost:8001"). If provided, vllm_server_host and vllm_server_port are ignored. |
| `vllm_server_host` | str | `0.0.0.0` |  | — | Host of the vLLM server when `vllm_mode="server"`. |
| `vllm_server_port` | int | `8001` |  | — | Port of the vLLM server when `vllm_mode="server"`. |
| `vllm_server_timeout` | float | `240.0` |  | — | Timeout (in seconds) for connecting to the vLLM server when `vllm_mode="server"`. |
| `vllm_group_port` | int | `51216` |  | — | Port for the vLLM weight-update group (NCCL communicator). |
| `vllm_gpu_memory_utilization` | float | `0.9` |  | — | GPU memory utilization for the colocated vLLM engine when `vllm_mode="colocate"`. Lower values reduce contention when sharing a device with the student/teacher models. |
| `vllm_tensor_parallel_size` | int | `1` |  | — | Tensor parallel size for the colocated vLLM engine when `vllm_mode="colocate"`. |
| `vllm_max_model_length` | int | None | `None` |  | — | Maximum model sequence length for the colocated vLLM engine when `vllm_mode="colocate"`. Defaults to the model's maximum context length. |
| `vllm_model_impl` | str | `vllm` |  | `vllm`, `transformers` | Model implementation backend to use in vLLM. Use "vllm" (default) or "transformers". |
| `vllm_structured_outputs_regex` | str | None | `None` |  | — | Regex pattern used for vLLM structured outputs (optional). |
| `vllm_sync_frequency` | int | `1` |  | — | Frequency (in training steps) to synchronize model weights to the vLLM engine. Set to 1 to sync after every step. |
| `vllm_enable_sleep_mode` | bool | `False` |  | — | Enable vLLM sleep mode to offload student weights/cache during the optimizer step. Keeps GPU memory usage low, but waking the engine adds host–device transfer latency. |

**记录与产物 · Parameters that control logging**

| 参数 | 类型 | 默认 | 关键 | 可选值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `log_completions` | bool | `False` |  | — | Whether to log a sample of (prompt, completion) pairs every `logging_steps` steps. If `rich` is installed, it prints the sample. If `wandb` logging is enabled, it logs it to `wandb`. |
| `log_completions_steps` | int | `100` |  | — | Number of steps between logging (prompt, completion) pairs. Only used if `log_completions` is set to `True`. |
| `num_completions_to_print` | int | None | `None` |  | — | Number of completions to print with `rich`. If `None`, all completions are logged. |
| `wandb_log_unique_prompts` | bool | `True` |  | — | Whether to log the unique prompts to wandb. This will create a new run for each unique prompt. |
| `callbacks` | list[str] | `<lambda: []>` |  | — | The callbacks to run during training. |

### 取舍与坑

- ✅ 唯一支持跨 tokenizer 蒸馏的路线
- ✅ 继承 GKD 的 on/off-policy 调度
- ⚠️ 参数最多之一（46 个），对齐策略需要理解成本
- ⚠️ 实验命名空间
- 🔻 教师 temperature 与学生 temperature 需分别设置（uld_student/teacher_temperature）

## <a id="sdft"></a>SDFT（`sdft`）

- **全称**：Self-Distilled Fine-Tuning（蒸馏，实验）
- **配置类**：`SDFTConfig` @ `trl/experimental/sdft/sdft_config.py`（继承链：`SDFTConfig` → `_BaseConfig`）
- **训练器**：`SDFTTrainer` @ `trl/experimental/sdft/sdft_trainer.py`
- **文档**：`docs/source/sdft_trainer.md` · 论文：<https://huggingface.co/papers/2601.19897>
- **启动**：`库内构造 SDFTTrainer`
- **数据契约**：prompt、privileged_context；prompt + 特权上下文（如标准答案/文档）

> 持续学习场景：学生从普通 prompt 生成，教师是“同一模型 + 特权上下文（privileged_context）”的条件版本，用它重打分来提供 token 级监督，从而在学新知识的同时少遗忘旧能力。


### 训练流程映射

| 阶段 | 该算法在此阶段做什么 |
| --- | --- |
| 模型与参考/教师 | 学生 + 教师副本（teacher_model_kind=base/live/ema）。 |
| 数据与提示 | prompt + privileged_context。 |
| 监督信号 | 教师（带特权上下文）的下一 token 分布。 |
| 采样（rollout） | 默认从普通 prompt 生成；generate_from_teacher 可改成教师条件采样。 |
| 优势/偏好估计 | 逐 token 分布差（distillation_mode=topk_logits / full）。 |
| 损失与更新 | 蒸馏损失（distillation_alpha、distillation_topk、num_loss_tokens_to_skip）。 |
| 记录与产物 | 同标准 Trainer。 |

### 参数


**algorithm · 算法其他参数**

| 参数 | 类型 | 默认 | 关键 | 可选值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `logging_steps` | float | `10` |  | — | Log every X updates steps. Should be an integer or a float in range `[0,1)`. If smaller than 1, will be interpreted as ratio of total training steps. |
| `gradient_checkpointing` | bool | `True` |  | — | Enable gradient checkpointing to trade compute for memory. Reduces memory at the cost of ~20%% slower training. |
| `bf16` | bool | None | `None` |  | — | Whether to use bf16 (mixed) precision instead of 32-bit. Requires Ampere or higher NVIDIA architecture or Intel XPU or using CPU (use_cpu) or Ascend NPU. If not set, it defaults to `True` if `fp16` is not set. |
| `lr_scheduler_kwargs` | dict | str | None | `None` |  | — | Additional parameters for the lr_scheduler, such as {'num_cycles': 1} for cosine with hard restarts. See the documentation of each scheduler for possible values. |
| `use_liger_kernel` | bool | `False` |  | — | Enable Liger Kernel optimizations: transformers patches the model's layers with Liger's Triton kernels. Deprecated in `SFTTrainer`, `DPOTrainer`, `KTOTrainer`, `GRPOTrainer` and `RLOOTrainer`, and will be removed in v2.0.0: use `model_init… |
| `torch_empty_cache_steps` | int | None | `None` |  | — | Number of steps to wait before calling `torch.<device>.empty_cache()`. Helps avoid CUDA OOM at a cost of ~10%% slower performance. If None, cache will not be emptied. |

**模型与参考/教师 · Parameters that control the model**

| 参数 | 类型 | 默认 | 关键 | 可选值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `model_init_kwargs` | dict[str, Any] | None | `None` |  | — | Keyword arguments for `transformers.AutoModelForCausalLM.from_pretrained`, used when the `model` argument of the `SDFTTrainer` is provided as a string. The `revision` value is also used when loading the processing class. |
| `trust_remote_code` | bool | `False` |  | — | Whether to allow loading models and tokenizers that ship custom Python code from the Hub. Forwarded to `AutoModelForCausalLM.from_pretrained` and `AutoProcessor.from_pretrained`, for both the student and teacher. |
| `disable_dropout` | bool | `True` |  | — | Whether to disable dropout in the student and teacher models. |

**数据与提示 · Parameters that control data preprocessing**

| 参数 | 类型 | 默认 | 关键 | 可选值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `remove_unused_columns` | bool | `False` |  | — | Whether to only keep the columns required by the trainer in the dataset. Keep this to `False` if you provide extra columns (such as `privileged_context`) that the trainer needs. |
| `max_prompt_length` | int | None | `512` |  | — | Maximum prompt length. Longer prompts are truncated from the left. |

**采样 · Parameters that control generation**

| 参数 | 类型 | 默认 | 关键 | 可选值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `num_generations` | int | `8` | ★ | — | Number of generations to sample. The effective batch size (num_processes * per_device_batch_size * gradient_accumulation_steps) must be evenly divisible by this value. |
| `num_generations_eval` | int | None | `None` |  | — | Number of generations to sample during evaluation. This allows using fewer generations during evaluation to save computation. If `None`, uses the value of `num_generations`. |
| `max_completion_length` | int | None | `256` |  | — | Maximum length of the generated completion. |
| `ds3_gather_for_generation` | bool | `True` |  | — | This setting applies to DeepSpeed ZeRO-3. If enabled, the policy model weights are gathered for generation, improving generation speed. However, disabling this option allows training models that exceed the VRAM capacity of a single GPU, al… |

**数据与提示 · Parameters that control data preprocessing**

| 参数 | 类型 | 默认 | 关键 | 可选值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `shuffle_dataset` | bool | `True` |  | — | Whether to shuffle the training dataset. |

**损失与更新 · Parameters that control the training**

| 参数 | 类型 | 默认 | 关键 | 可选值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `generation_batch_size` | int | None | `None` |  | — | Batch size to use for generation. If `None`, it defaults to the effective training batch size: `per_device_train_batch_size * num_processes * steps_per_generation`. |
| `steps_per_generation` | int | None | `None` |  | — | Number of steps per generation. If `None`, it defaults to `gradient_accumulation_steps`. |

**采样 · Parameters that control generation**

| 参数 | 类型 | 默认 | 关键 | 可选值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `temperature` | float | `1.0` |  | — | Temperature for sampling. The higher the temperature, the more random the completions. |
| `top_p` | float | `1.0` |  | — | Float that controls the cumulative probability of the top tokens to consider. Must be in (0, 1]. Set to 1.0 to consider all tokens. |
| `top_k` | int | `0` |  | — | Number of highest probability vocabulary tokens to keep for top-k-filtering. If `0`, top-k-filtering is disabled and all tokens are considered. |
| `min_p` | float | None | `None` |  | — | Minimum token probability, which will be scaled by the probability of the most likely token. It must be a value between 0.0 and 1.0. Typical values are in the 0.01-0.2 range. |
| `generation_kwargs` | dict[str, Any] | None | `None` |  | — | Additional keyword arguments to pass to `GenerationConfig` (if using transformers) or `SamplingParams` (if using vLLM) when sampling completions. This can be used to further customize the generation behavior, such as setting `suppress_toke… |
| `chat_template_kwargs` | dict[str, Any] | None | `None` |  | — | Additional keyword arguments to pass to the `apply_chat_template` function when generating completions. |
| `repetition_penalty` | float | `1.0` |  | — | Float that penalizes new tokens based on whether they appear in the prompt and the generated text so far. Values > 1.0 encourage the model to use new tokens, while values < 1.0 encourage the model to repeat tokens. |
| `cache_implementation` | str | None | `None` |  | — | Implementation of the cache method for faster generation when use_vllm is set to False. |

**环境与资源 · Parameters that control generation acceleration powered by vLLM**

| 参数 | 类型 | 默认 | 关键 | 可选值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `use_vllm` | bool | `False` |  | — | Whether to use vLLM for generating completions. If set to `True`, the trainer will use vLLM for generation instead of the default model.generate(). Requires `vllm` to be installed. |
| `use_teacher_server` | bool | `False` |  | — | Compute teacher logprobs from the running vLLM generation server instead of a local teacher forward. Only supported for `teacher_model_kind='live'` with `use_vllm=True` and `vllm_mode='server'`, and `distillation_mode` in {'sampled_token',… |
| `vllm_mode` | str | `colocate` |  | `colocate`, `server` | Mode to use for vLLM integration when `use_vllm` is set to `True`. Must be one of `'server'` or `'colocate'`. `'server'`: The trainer will send generation requests to a separate vLLM server. Make sure a vLLM server is running (start with `… |
| `vllm_model_impl` | str | `vllm` |  | `vllm`, `transformers` | Model implementation to use for vLLM. Must be one of `transformers` or `vllm`. `transformers`: Use the `transformers` backend for model implementation. `vllm`: Use the `vllm` library for model implementation. |
| `vllm_enable_sleep_mode` | bool | `False` |  | — | Enable vLLM sleep mode to offload weights/cache during the optimizer step. Keeps GPU memory usage low, but waking the engine adds host–device transfer latency. |
| `vllm_server_base_url` | str | None | `None` |  | — | Base URL for the vLLM server (e.g., 'http://localhost:8000'). If provided, `vllm_server_host` and `vllm_server_port` are ignored. |
| `vllm_server_host` | str | `0.0.0.0` |  | — | Host of the vLLM server to connect to. Ignored if vllm_server_base_url is provided. |
| `vllm_server_port` | int | `8000` |  | — | Port of the vLLM server to connect to. Ignored if vllm_server_base_url is provided. |
| `vllm_group_port` | int | `51216` |  | — | Port number for the weight update group. This is used to communicate with the vLLM server. Unless the port is occupied, there is no need to change it. |
| `vllm_server_timeout` | float | `240.0` |  | — | Total timeout duration in seconds to wait for the vLLM server to be up. If the server is not up after the timeout, a `ConnectionError` is raised. |
| `vllm_tensor_parallel_size` | int | `1` |  | — | Control the tensor parallel size for vLLM. This setting only applies when `vllm_mode` is set to `'colocate'`. If you are using `vllm_mode='server'`, this parameter must be passed separately when launching the vLLM server via the `--vllm_te… |
| `vllm_gpu_memory_utilization` | float | `0.3` |  | — | Control the GPU memory utilization for vLLM. This setting only applies when `vllm_mode` is set to `'colocate'`. If you are using `vllm_mode='server'`, this parameter must be passed separately when launching the vLLM server via the `--vllm_… |
| `vllm_max_model_length` | int | None | `None` |  | — | Context window for vLLM. Set it to at least the maximum prompt length in the dataset plus `max_completion_length`; if omitted, it is inferred from the model config. |

**损失与更新 · Parameters that control the training**

| 参数 | 类型 | 默认 | 关键 | 可选值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `num_iterations` | int | `1` |  | — | Number of iterations per batch (denoted as μ in the algorithm). |

**模型与参考/教师 · Parameters that control the teacher**

| 参数 | 类型 | 默认 | 关键 | 可选值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `teacher_model_kind` | str | `base` | ★ | `base`, `live`, `ema` | Semantic teacher choice for SDFT. `base` uses the initial student, `live` uses the current student, and `ema` uses an exponentially averaged teacher. |
| `teacher_update_rate` | float | `0.05` | ★ | — | EMA update rate used when `teacher_model_kind="ema"`. A value of `1.0` reduces the update to a hard overwrite, periodically resyncing the teacher to the current student weights. |
| `teacher_sync_steps` | int | `1` | ★ | — | Number of optimizer steps between teacher updates. |

**损失与更新 · Parameters that control the SDFT loss**

| 参数 | 类型 | 默认 | 关键 | 可选值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `distillation_alpha` | float | `0.5` | ★ | — | Divergence interpolation coefficient for SDFT top-k logit distillation. |
| `distillation_mode` | Literal['sampled_token', 'full_logits', 'topk_logits'] | `topk_logits` | ★ | — | Distillation objective mode. SDFT defaults to top-k logit distillation. |
| `distillation_topk` | int | None | `100` | ★ | — | Number of top tokens used by the default SDFT top-k logit objective. |
| `distillation_is_clip` | float | None | `2.0` |  | — | Clipping coefficient for importance sampling in self-distillation. `None` disables clipping. |
| `distillation_add_tail` | bool | `False` |  | — | Whether to add a tail bucket for non-top-k probability mass. |

**采样 · Parameters that control teacher-conditioned generation**

| 参数 | 类型 | 默认 | 关键 | 可选值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `generate_from_teacher` | bool | `False` | ★ | — | Whether on-policy generation should use the teacher-conditioned prompt instead of the student prompt. |
| `teacher_prompt_template` | str | `{prompt}

{privileged_context}` |  | — | Template used to combine the student prompt and privileged context into the teacher prompt. |

**损失与更新 · Parameters that control the SDFT loss**

| 参数 | 类型 | 默认 | 关键 | 可选值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `num_loss_tokens_to_skip` | int | `0` | ★ | — | Number of initial completion tokens to exclude from the distillation loss. |

### 取舍与坑

- ✅ 不需要奖励函数也不需要外部教师模型
- ✅ 面向持续学习，抗遗忘
- ⚠️ 需要构造 privileged_context 列
- ⚠️ 实验命名空间
- 🔻 teacher_model_kind=live 时没有滞后，可能自我强化；ema 需要调 teacher_update_rate

## <a id="iw_opd"></a>IW-OPD（`iw_opd`）

- **全称**：Importance-Weighted On-Policy Distillation（蒸馏，实验）
- **配置类**：`IWOPDConfig` @ `trl/experimental/iw_opd/iw_opd_config.py`（继承链：`IWOPDConfig` → `_BaseConfig`）
- **训练器**：`IWOPDTrainer` @ `trl/experimental/iw_opd/iw_opd_trainer.py`
- **文档**：`docs/source/iw_opd_trainer.md` · 论文：<https://huggingface.co/papers/2606.22600>
- **启动**：`库内构造 IWOPDTrainer；也可在 DistillationTrainer 里设 distillation_objective='iw_opd'`
- **数据契约**：prompt；prompt

> 按“累积的师生前缀差异”给蒸馏信号加权：越靠后的 token 监督越可能已经脱离分布，就压得越轻，从而缓解 on-policy 蒸馏的位置偏置。


### 训练流程映射

| 阶段 | 该算法在此阶段做什么 |
| --- | --- |
| 模型与参考/教师 | 学生 + 教师（可放远程 teacher server：use_teacher_server/teacher_model_server_url）。 |
| 数据与提示 | prompt。 |
| 监督信号 | 教师的 token 分布（可选 top-k 截断：loss_top_k）。 |
| 采样（rollout） | 学生 on-policy 生成。 |
| 优势/偏好估计 | 按前缀差异累积的重要性权重（iw_opd_gamma、iw_opd_epsilon、reverse_kl_top_1_mode）。 |
| 损失与更新 | 加权后的蒸馏/策略梯度损失。 |
| 记录与产物 | 同标准 Trainer。 |

### 参数


**algorithm · 算法其他参数**

| 参数 | 类型 | 默认 | 关键 | 可选值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `logging_steps` | float | `10` |  | — | Log every X updates steps. Should be an integer or a float in range `[0,1)`. If smaller than 1, will be interpreted as ratio of total training steps. |
| `gradient_checkpointing` | bool | `True` |  | — | Enable gradient checkpointing to trade compute for memory. Reduces memory at the cost of ~20%% slower training. |
| `bf16` | bool | None | `None` |  | — | Whether to use bf16 (mixed) precision instead of 32-bit. Requires Ampere or higher NVIDIA architecture or Intel XPU or using CPU (use_cpu) or Ascend NPU. If not set, it defaults to `True` if `fp16` is not set. |
| `lr_scheduler_kwargs` | dict | str | None | `None` |  | — | Additional parameters for the lr_scheduler, such as {'num_cycles': 1} for cosine with hard restarts. See the documentation of each scheduler for possible values. |
| `use_liger_kernel` | bool | `False` |  | — | Enable Liger Kernel optimizations: transformers patches the model's layers with Liger's Triton kernels. Deprecated in `SFTTrainer`, `DPOTrainer`, `KTOTrainer`, `GRPOTrainer` and `RLOOTrainer`, and will be removed in v2.0.0: use `model_init… |
| `torch_empty_cache_steps` | int | None | `None` |  | — | Number of steps to wait before calling `torch.<device>.empty_cache()`. Helps avoid CUDA OOM at a cost of ~10%% slower performance. If None, cache will not be emptied. |

**模型与参考/教师 · Parameters that control the model**

| 参数 | 类型 | 默认 | 关键 | 可选值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `model_init_kwargs` | dict[str, Any] | str | None | `None` |  | — | Keyword arguments for `AutoModelForCausalLM.from_pretrained`, used when the `model` argument of the trainer is provided as a string. The `revision` value is also used when loading the processing class. |
| `trust_remote_code` | bool | `False` |  | — | Whether to allow loading models and tokenizers that ship custom Python code from the Hub. Forwarded to `AutoModelForCausalLM.from_pretrained` and `AutoTokenizer.from_pretrained`, for both the student and teacher. |
| `max_length` | int | None | `1024` |  | — | Maximum total sequence length (prompt + completion) for tokenization and truncation. |

**损失与更新 · 训练超参**

| 参数 | 类型 | 默认 | 关键 | 可选值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `learning_rate` | float | `1e-06` |  | — | The initial learning rate for AdamW. |

**损失与更新 · Parameters that control the distillation**

| 参数 | 类型 | 默认 | 关键 | 可选值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `temperature` | float | `1.0` |  | — | Temperature for sampling and loss computation. Higher values produce softer distributions. |
| `lmbda` | float | `1.0` | ★ | — | Probability of using on-policy (student-generated) data per gradient accumulation slice. 0.0 = fully off-policy, 1.0 = fully on-policy. |
| `beta` | float | `1.0` | ★ | — | Interpolation coefficient for the Generalized JSD loss. 0.0 = forward KL, 0.5 = JSD, 1.0 = reverse KL. |
| `distillation_objective` | str | `iw_opd` | ★ | `iw_opd`, `jsd` | Objective to optimize. Use 'iw_opd' for sampled-token Importance-Weighted On-Policy Distillation or 'jsd' for the generalized JSD/KL objective. |
| `iw_opd_gamma` | float | `0.5` | ★ | — | Importance-weight amplification for distillation_objective='iw_opd'. |
| `iw_opd_epsilon` | float | `1e-08` | ★ | — | Stabilizer used when normalizing IW-OPD prefix weights. |
| `reverse_kl_top_1_mode` | str | `sampled` | ★ | — | Reverse-KL top-1 token selection when beta > 0 and loss_top_k == 1. Use 'sampled' for the actual completion token or 'argmax' for the student's top-1 token. The forward-KL support always uses the teacher's top-1 token. Ignored when beta ==… |
| `max_completion_length` | int | `512` |  | — | Maximum number of tokens to generate per completion. |
| `max_prompt_length` | int | None | `None` |  | — | Maximum number of tokens for the prompt. If None, auto-computed as max_length - max_completion_length. Prompts are truncated according to the tokenizer's truncation_side setting. |
| `disable_dropout` | bool | `True` |  | — | Whether to disable dropout in the student model during training. |

**模型与参考/教师 · Parameters that control the teacher model**

| 参数 | 类型 | 默认 | 关键 | 可选值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `teacher_model_name_or_path` | str | None | `None` |  | — | Model name or path for the teacher model. |
| `teacher_model_revision` | str | None | `None` |  | — | Model revision of the teacher model (e.g., branch name, tag, or commit hash). |
| `teacher_model_init_kwargs` | dict[str, Any] | str | None | `None` |  | — | Keyword arguments for `AutoModelForCausalLM.from_pretrained` when instantiating the teacher. |
| `use_teacher_server` | bool | `False` | ★ | — | Whether to use an external vLLM teacher server instead of a local teacher model. |
| `teacher_model_server_url` | str | None | `None` |  | — | Base URL of a vLLM server hosting the teacher model (e.g., "http://localhost:8000"). Required when use_teacher_server=True. |
| `loss_top_k` | int | `1` | ★ | — | Number of top tokens to use when computing the JSD/KL loss. Both student and teacher distributions are restricted to these K tokens (selected based on beta: teacher's top-k for forward KL, student's top-k for reverse KL, union of both for… |
| `loss_add_tail` | bool | `True` |  | — | Whether to append a tail bucket representing the remaining probability mass outside the selected top-k support. |

**采样 · Parameters that control on-policy generation**

| 参数 | 类型 | 默认 | 关键 | 可选值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `num_generations` | int | `1` |  | — | Number of completions to generate per prompt during on-policy generation. |
| `generation_batch_size` | int | None | `None` |  | — | Number of unique prompts per worker per optimizer step. If None, computed from (per_device_train_batch_size * gradient_accumulation_steps) // num_generations. |
| `top_p` | float | `0.95` |  | — | Top-p (nucleus) sampling parameter for on-policy generation. |
| `top_k` | int | `0` |  | — | Top-k sampling parameter for on-policy generation. 0 disables top-k filtering. |

**环境与资源 · Parameters that control vLLM for student generation**

| 参数 | 类型 | 默认 | 关键 | 可选值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `use_vllm` | bool | `False` |  | — | Whether to use vLLM for generating on-policy completions from the student model. |
| `vllm_mode` | str | `colocate` |  | `colocate`, `server` | Mode for student vLLM integration. Either "server" or "colocate". |
| `vllm_server_base_url` | str | None | `None` |  | — | Base URL for the student vLLM server. |
| `vllm_server_host` | str | `0.0.0.0` |  | — | Host of the student vLLM server. |
| `vllm_server_port` | int | `8001` |  | — | Port of the student vLLM server. |
| `vllm_server_timeout` | float | `240.0` |  | — | Timeout for connecting to the student vLLM server. |
| `vllm_group_port` | int | `51216` |  | — | Port for the vLLM weight-update group (NCCL communicator). |
| `vllm_gpu_memory_utilization` | float | `0.3` |  | — | GPU memory utilization for the colocated student vLLM engine. |
| `vllm_tensor_parallel_size` | int | `1` |  | — | Tensor parallel size for the colocated student vLLM engine. |
| `vllm_max_model_length` | int | None | `None` |  | — | Maximum model sequence length for the colocated vLLM engine. |
| `vllm_model_impl` | str | `vllm` |  | `vllm`, `transformers` | Model implementation backend for vLLM. Use "vllm" or "transformers". |
| `vllm_structured_outputs_regex` | str | None | `None` |  | — | Regex pattern for vLLM structured outputs. |
| `vllm_sync_frequency` | int | `1` |  | — | Frequency (in training steps) to synchronize student weights to the vLLM engine. |
| `vllm_enable_sleep_mode` | bool | `False` |  | — | Enable vLLM sleep mode to offload student weights during the optimizer step. |

**记录与产物 · Parameters that control logging**

| 参数 | 类型 | 默认 | 关键 | 可选值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `log_completions` | bool | `False` |  | — | Whether to log a sample of (prompt, completion) pairs every `log_completions_steps` steps. If `rich` is installed, it prints the sample. If `wandb` and/or `trackio` logging is enabled, it logs it to `wandb` and/or `trackio`. |
| `log_completions_steps` | int | `100` |  | — | Number of steps between logging completions. |
| `num_completions_to_print` | int | None | `None` |  | — | Number of completions to print. If None, all completions are logged. |

### 取舍与坑

- ✅ 缓解位置偏置，长序列蒸馏更稳
- ✅ 教师可放远程 server
- ⚠️ 该 trainer 是重构前 DistillationTrainer 的冻结快照，不再跟随主 trainer 演进
- ⚠️ 额外权重参数需要调
- 🔻 直接维护者标注为“不活跃维护”，新项目优先用 stable 的 DistillationTrainer + distillation_objective='iw_opd'

## <a id="async_distillation"></a>Async Distillation（`async_distillation`）

- **全称**：Asynchronous On-policy Distillation（蒸馏，实验）
- **配置类**：`AsyncDistillationConfig` @ `trl/experimental/async_distillation/async_distillation_config.py`（继承链：`AsyncDistillationConfig` → `_BaseConfig`）
- **训练器**：`AsyncDistillationTrainer` @ `trl/experimental/async_distillation/async_distillation_trainer.py`
- **文档**：`docs/source/async_distillation_trainer.md` · 论文：<https://huggingface.co/papers/2606.30406>
- **启动**：`需要外置 vLLM server（学生与教师各一）`
- **数据契约**：prompt；prompt

> 蒸馏的异步版：后台 worker 持续生成学生的 on-policy 补全并向教师 server 取 logprobs，训练循环只消费样本，生成与更新重叠。


### 训练流程映射

| 阶段 | 该算法在此阶段做什么 |
| --- | --- |
| 环境与资源 | vLLM server + teacher_server_urls，token_budget/queue_maxsize 控制水位。 |
| 模型与参考/教师 | 学生 + 教师（远程）。 |
| 数据与提示 | prompt 流式投喂。 |
| 监督信号 | 教师 top-k logprobs（teacher_temp/teacher_top_k）。 |
| 采样（rollout） | 后台 worker 持续生成。 |
| 优势/偏好估计 | 广义 JSD（beta 插值）。 |
| 损失与更新 | 蒸馏损失 + 权重同步。 |
| 记录与产物 | 吞吐与蒸馏损失指标。 |

### 参数


**algorithm · 算法其他参数**

| 参数 | 类型 | 默认 | 关键 | 可选值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `logging_steps` | float | `1` |  | — | Log every X update steps. Should be an integer or a float in range `[0,1)`. If smaller than 1, will be interpreted as ratio of total training steps. |
| `gradient_checkpointing` | bool | `True` |  | — | Enable gradient checkpointing to trade compute for memory. Reduces memory at the cost of ~20%% slower training. |
| `bf16` | bool | None | `None` |  | — | Whether to use bf16 (mixed) precision instead of 32-bit. Requires Ampere or higher NVIDIA architecture or Intel XPU or using CPU (use_cpu) or Ascend NPU. If not set, it defaults to `True` if `fp16` is not set. |
| `lr_scheduler_kwargs` | dict | str | None | `None` |  | — | Additional parameters for the lr_scheduler, such as {'num_cycles': 1} for cosine with hard restarts. See the documentation of each scheduler for possible values. |
| `use_liger_kernel` | bool | `False` |  | — | Enable Liger Kernel optimizations: transformers patches the model's layers with Liger's Triton kernels. Deprecated in `SFTTrainer`, `DPOTrainer`, `KTOTrainer`, `GRPOTrainer` and `RLOOTrainer`, and will be removed in v2.0.0: use `model_init… |
| `torch_empty_cache_steps` | int | None | `None` |  | — | Number of steps to wait before calling `torch.<device>.empty_cache()`. Helps avoid CUDA OOM at a cost of ~10%% slower performance. If None, cache will not be emptied. |

**模型与参考/教师 · Parameters that control the model**

| 参数 | 类型 | 默认 | 关键 | 可选值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `model_init_kwargs` | dict[str, Any] | str | None | `None` |  | — | Keyword arguments for `transformers.AutoModelForCausalLM.from_pretrained`, used when instantiating the student model from a path. The `revision` value is also used when loading the processing class. |
| `dtype` | str | `float32` |  | `auto`, `bfloat16`, `float16`, `float32` | Dtype to load the student model under. Defaults to 'float32', the precision the training-inference mismatch results the async trainers are measured against were obtained at. A `dtype` in `model_init_kwargs` takes precedence. |

**algorithm · 算法其他参数**

| 参数 | 类型 | 默认 | 关键 | 可选值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `ignore_data_skip` | bool | `True` |  | — | Always `True` for AsyncDistillation; the base Trainer's skip-and-replay loop does not apply to a live rollout queue. |

**模型与参考/教师 · Parameters that control the model**

| 参数 | 类型 | 默认 | 关键 | 可选值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `trust_remote_code` | bool | `False` |  | — | Whether to allow loading models and tokenizers that ship custom Python code from the Hub. Forwarded to `AutoModelForCausalLM.from_pretrained` and `AutoTokenizer.from_pretrained`. |

**损失与更新 · 训练超参**

| 参数 | 类型 | 默认 | 关键 | 可选值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `learning_rate` | float | `1e-06` |  | — | The initial learning rate for AdamW. |

**采样 · Parameters that control generation**

| 参数 | 类型 | 默认 | 关键 | 可选值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `max_completion_length` | int | `2048` |  | — | Maximum number of tokens to generate per completion. |
| `temperature` | float | `1.0` |  | — | Temperature for sampling the student's on-policy completions. |
| `top_p` | float | `1.0` |  | — | Top-p (nucleus) sampling parameter for on-policy generation. |
| `top_k` | int | `0` |  | — | Top-k sampling parameter for on-policy generation. 0 disables top-k filtering. |
| `min_p` | float | None | `None` |  | — | Minimum token probability, which will be scaled by the probability of the most likely token. It must be a value between 0.0 and 1.0. Typical values are in the 0.01-0.2 range. |
| `repetition_penalty` | float | `1.0` |  | — | Float that penalizes new tokens based on whether they appear in the prompt and the generated text so far. Values > 1.0 encourage the model to use new tokens, while values < 1.0 encourage the model to repeat tokens. |
| `chat_template_kwargs` | dict | str | None | `None` |  | — | Additional keyword arguments to pass to the `apply_chat_template` function when generating completions. |

**环境与资源 · Parameters that control the vLLM servers**

| 参数 | 类型 | 默认 | 关键 | 可选值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `vllm_server_base_url` | str | `http://localhost:8000` |  | — | Base URL of the student's vLLM server (e.g., 'http://localhost:8000'). |
| `vllm_server_timeout` | float | `240.0` |  | — | Total timeout duration in seconds to wait for the student's vLLM server to be ready. |
| `teacher_server_urls` | dict[str, str] | str | None | `None` | ★ | — | Teacher vLLM server(s), each started with `vllm serve <model> --logprobs-mode processed_logprobs --max-logprobs -1`; every teacher is static, no weights are ever streamed to any of them. Defaults to a single teacher at 'http://localhost:80… |
| `request_timeout` | int | `600` |  | — | Timeout in seconds for individual HTTP requests to any vLLM server. |
| `weight_sync_timeout` | int | `1800` |  | — | Timeout in seconds for a weight transfer to the student's vLLM server. A transfer that does not complete within this time raises instead of hanging the run. |

**损失与更新 · Parameters that control the distillation loss**

| 参数 | 类型 | 默认 | 关键 | 可选值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `beta` | float | `0.0` | ★ | — | Interpolation coefficient for the generalized JSD. At beta=0.0 (forward KL), the divergence is computed over the full teacher_top_k-wide teacher-reported support (plus tail bucket). At beta != 0.0, the support narrows to two candidates onl… |
| `teacher_temperature` | float | `1.0` |  | — | Softmax temperature of the divergence, applied to both the teacher's returned logprobs (server-side) and the student's own logits. Unrelated to `temperature`, which controls sampling. |
| `teacher_top_k` | int | `8` | ★ | — | Number of per-position candidate tokens requested from the teacher's `prompt_logprobs`, used to approximate its distribution without transmitting the full vocabulary over HTTP. Above 20, the teacher's server must have been started with `--… |
| `add_tail_bucket` | bool | `True` |  | — | Whether to append a tail bucket representing the remaining probability mass outside `teacher_top_k`, to avoid a trivially small divergence when `teacher_top_k` is small. |
| `token_budget` | int | None | `None` | ★ | — | Maximum number of real tokens packed into a single row (one DP rank's forward) for dynamic token-budgeted micro-batching. When > 0, a `TokenBudgetBatcher` forms Σ Lᵢ²-balanced micro-batches whose rows each stay within this budget. If None… |

**采样 · Parameters that control the async rollout pipeline**

| 参数 | 类型 | 默认 | 关键 | 可选值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `max_inflight_tasks` | int | `-1` | ★ | — | Maximum number of concurrent generation+scoring tasks in flight. Defaults to -1 (auto), which sets it to `max_staleness * per_device_train_batch_size * gradient_accumulation_steps * num_processes`. |
| `max_staleness` | int | `4` | ★ | — | Maximum number of weight update steps a rollout sample can lag behind the current model version before being discarded. |
| `queue_maxsize` | int | `1024` |  | — | Maximum number of rollout samples to buffer in the rollout queue. |
| `weight_sync_steps` | int | `1` | ★ | — | Number of training steps between weight synchronizations to the student's vLLM server. |
| `heartbeat_stale_after_s` | float | `300.0` |  | — | Seconds since the rollout worker's last heartbeat after which the trainer treats it as hung and aborts. |

**记录与产物 · Parameters that control the logging**

| 参数 | 类型 | 默认 | 关键 | 可选值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `log_completions` | bool | `False` |  | — | Whether to log a sample of (prompt, completion) pairs every `log_completions_steps` samples scored. |
| `log_completions_steps` | int | `100` |  | — | Number of scored samples between logging completions. |
| `num_completions_to_print` | int | None | `None` |  | — | Number of completions to print with `rich`. If `None`, all completions are logged. |

### 取舍与坑

- ✅ 长补全蒸馏的吞吐上限最高
- ⚠️ 需要自己拉两个 vLLM server
- ⚠️ off-policy 程度难控
- 🔻 max_staleness 与 weight_sync_steps 决定学生/生成模型的版本差

## <a id="server_distillation"></a>Server Distillation（`server_distillation`）

- **全称**：Teacher-server On-policy Distillation（蒸馏，实验）
- **配置类**：`ServerDistillationConfig` @ `trl/experimental/server_distillation/server_distillation_config.py`（继承链：`ServerDistillationConfig` → `DistillationConfig` → `_BaseConfig`）
- **训练器**：`ServerDistillationTrainer` @ `trl/experimental/server_distillation/server_distillation_trainer.py`
- **文档**：`docs/source/distillation_trainer.md`
- **启动**：`库内构造 ServerDistillationTrainer（teacher_model_server_url）`
- **数据契约**：prompt；prompt

> 教师放在外部 vLLM server 上：不跑本地教师前向，而是通过 VLLMClient 取回教师 top-k logprobs，因此散度只在稀疏支持上计算（beta>0 时 top-1，beta=0 时 top-k）。


### 训练流程映射

| 阶段 | 该算法在此阶段做什么 |
| --- | --- |
| 环境与资源 | 必须有教师 vLLM server（teacher_model_server_url）。 |
| 模型与参考/教师 | 只加载学生。 |
| 数据与提示 | prompt。 |
| 监督信号 | 教师 server 返回的 top-k logprob 稀疏分布。 |
| 采样（rollout） | 学生 on-policy 生成（继承 DistillationTrainer）。 |
| 优势/偏好估计 | 稀疏支持上的散度。 |
| 损失与更新 | 同 DistillationTrainer，省掉本地教师前向的显存与算力。 |
| 记录与产物 | 同 DistillationTrainer。 |

### 参数


**algorithm · 算法其他参数**

| 参数 | 类型 | 默认 | 关键 | 可选值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `logging_steps` | float | `10` |  | — | Log every X updates steps. Should be an integer or a float in range `[0,1)`. If smaller than 1, will be interpreted as ratio of total training steps. |
| `gradient_checkpointing` | bool | `True` |  | — | Enable gradient checkpointing to trade compute for memory. Reduces memory at the cost of ~20%% slower training. |
| `bf16` | bool | None | `None` |  | — | Whether to use bf16 (mixed) precision instead of 32-bit. Requires Ampere or higher NVIDIA architecture or Intel XPU or using CPU (use_cpu) or Ascend NPU. If not set, it defaults to `True` if `fp16` is not set. |
| `lr_scheduler_kwargs` | dict | str | None | `None` |  | — | Additional parameters for the lr_scheduler, such as {'num_cycles': 1} for cosine with hard restarts. See the documentation of each scheduler for possible values. |
| `use_liger_kernel` | bool | `False` |  | — | Enable Liger Kernel optimizations: transformers patches the model's layers with Liger's Triton kernels. Deprecated in `SFTTrainer`, `DPOTrainer`, `KTOTrainer`, `GRPOTrainer` and `RLOOTrainer`, and will be removed in v2.0.0: use `model_init… |
| `torch_empty_cache_steps` | int | None | `None` |  | — | Number of steps to wait before calling `torch.<device>.empty_cache()`. Helps avoid CUDA OOM at a cost of ~10%% slower performance. If None, cache will not be emptied. |

**损失与更新 · 训练超参**

| 参数 | 类型 | 默认 | 关键 | 可选值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `learning_rate` | float | `1e-06` |  | — | The initial learning rate for AdamW. |

**模型与参考/教师 · Parameters that control the model and the teacher model**

| 参数 | 类型 | 默认 | 关键 | 可选值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `model_init_kwargs` | dict[str, Any] | str | None | `None` |  | — | Keyword arguments for `AutoModelForCausalLM.from_pretrained`, used when the `model` argument of the trainer is provided as a string. The `revision` value is also used when loading the processing class. |
| `trust_remote_code` | bool | `False` |  | — | Whether to allow loading models and tokenizers that ship custom Python code from the Hub. Forwarded to `AutoModelForCausalLM.from_pretrained` and `AutoTokenizer.from_pretrained`, for both the student and teacher. |
| `teacher_model_name_or_path` | str | None | `None` |  | — | Model name or path for the teacher model. |
| `teacher_model_revision` | str | None | `None` |  | — | Model revision of the teacher model (e.g., branch name, tag, or commit hash). |
| `teacher_model_init_kwargs` | dict[str, Any] | str | None | `None` |  | — | Keyword arguments for `AutoModelForCausalLM.from_pretrained` when instantiating the teacher. |
| `disable_dropout` | bool | `False` |  | — | Whether to disable dropout in the student model during training. |

**数据与提示 · Parameters that control the data preprocessing**

| 参数 | 类型 | 默认 | 关键 | 可选值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `remove_unused_columns` | bool | None | `False` |  | — | Whether to only keep the column 'prompt' in the dataset. The trainer consumes the raw prompt column and generates completions on-policy, so it defaults to `False`. |
| `max_completion_length` | int | None | `512` |  | — | Maximum number of tokens to generate per completion. |
| `ds3_gather_for_generation` | bool | `True` |  | — | This setting applies to DeepSpeed ZeRO-3. If enabled, the policy model weights are gathered for generation, improving generation speed. However, disabling this option allows training models that exceed the VRAM capacity of a single GPU, al… |
| `shuffle_dataset` | bool | None | `True` |  | — | Whether to shuffle the training dataset. |
| `pad_to_multiple_of` | int | str | None | `auto` |  | — | If set, the prompts ids and completions ids will be padded to a multiple of this value. `'auto'` pads to a multiple of 128 on Neuron (AWS Trainium), which compiles once per input shape, and does not pad elsewhere. |

**采样 · Parameters that control generation**

| 参数 | 类型 | 默认 | 关键 | 可选值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `temperature` | float | `1.0` |  | — | Temperature for sampling and loss computation. Higher values produce softer distributions. |
| `top_p` | float | `1.0` |  | — | Top-p (nucleus) sampling parameter for on-policy generation. |
| `top_k` | int | `0` |  | — | Top-k sampling parameter for on-policy generation. 0 disables top-k filtering. |
| `min_p` | float | None | `None` |  | — | Minimum token probability, which will be scaled by the probability of the most likely token. It must be a value between 0.0 and 1.0. Typical values are in the 0.01-0.2 range. |
| `generation_kwargs` | dict | str | None | `None` |  | — | Additional keyword arguments to pass to `GenerationConfig` (if using transformers) or `SamplingParams` (if using vLLM) when sampling completions. This can be used to further customize the generation behavior, such as setting `suppress_toke… |
| `chat_template_kwargs` | dict | str | None | `None` |  | — | Additional keyword arguments to pass to the `apply_chat_template` function when generating completions. |
| `repetition_penalty` | float | `1.0` |  | — | Float that penalizes new tokens based on whether they appear in the prompt and the generated text so far. Values > 1.0 encourage the model to use new tokens, while values < 1.0 encourage the model to repeat tokens. |
| `cache_implementation` | str | None | `None` |  | — | Implementation of the cache method for faster generation when use_vllm is set to False. |

**环境与资源 · Parameters that control generation acceleration powered by vLLM**

| 参数 | 类型 | 默认 | 关键 | 可选值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `use_vllm` | bool | `False` |  | — | Whether to use vLLM for generating on-policy completions from the student model. |
| `vllm_mode` | str | `colocate` |  | `colocate`, `server` | Mode for student vLLM integration. Either "server" or "colocate". |
| `vllm_model_impl` | str | `vllm` |  | `vllm`, `transformers` | Model implementation backend for vLLM. Use "vllm" or "transformers". |
| `vllm_enable_sleep_mode` | bool | `False` |  | — | Enable vLLM sleep mode to offload student weights during the optimizer step. |
| `vllm_structured_outputs_regex` | str | None | `None` |  | — | Regex pattern for vLLM structured outputs. |

**环境与资源 · Parameters that control the vLLM server (only used when `vllm_mode` is `"server"`)**

| 参数 | 类型 | 默认 | 关键 | 可选值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `vllm_server_base_url` | str | None | `None` |  | — | Base URL for the student vLLM server. |
| `vllm_server_host` | str | `0.0.0.0` |  | — | Host of the student vLLM server. |

**algorithm · 算法其他参数**

| 参数 | 类型 | 默认 | 关键 | 可选值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `vllm_server_port` | int | `8001` | ★ | — | Port of the student vLLM server. Overrides the base default of 8000 to stay distinct from the teacher server, which commonly runs on 8000. |

**环境与资源 · Parameters that control the vLLM server (only used when `vllm_mode` is `"server"`)**

| 参数 | 类型 | 默认 | 关键 | 可选值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `vllm_server_timeout` | float | `240.0` |  | — | Timeout for connecting to the student vLLM server. |
| `vllm_group_port` | int | `51216` |  | — | Port for the vLLM weight-update group (NCCL communicator). |

**环境与资源 · Parameters that control colocated vLLM execution (only used when `vllm_mode` is `"colocate"`)**

| 参数 | 类型 | 默认 | 关键 | 可选值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `vllm_gpu_memory_utilization` | float | `0.3` |  | — | GPU memory utilization for the colocated student vLLM engine. |
| `vllm_max_model_length` | int | None | `None` |  | — | Maximum model sequence length for the colocated vLLM engine. |
| `vllm_tensor_parallel_size` | int | `1` |  | — | Tensor parallel size for the colocated student vLLM engine. |

**损失与更新 · Parameters that control the training**

| 参数 | 类型 | 默认 | 关键 | 可选值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `beta` | float | `1.0` | ★ | — | Interpolation coefficient for the Generalized JSD loss. 0.0 = forward KL, 0.5 = JSD, 1.0 = reverse KL. Unlike GRPO's `beta` (a KL-penalty coefficient against a reference model), here it selects the divergence itself; there is no reference-… |
| `max_tool_calling_iterations` | int | None | `None` |  | — | Maximum number of tool-calling turns when training an agent. If `None`, there is no limit and generation stops when the model generates a response turn with no tool calls or when the total response length reaches `max_model_length`. |

**记录与产物 · Parameters that control the logging**

| 参数 | 类型 | 默认 | 关键 | 可选值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `log_completions` | bool | `False` |  | — | Whether to log a sample of (prompt, completion) pairs every `logging_steps` steps. If `rich` is installed, it prints the sample. If `wandb` and/or `trackio` logging is enabled, it logs it to `wandb` and/or `trackio`. |
| `num_completions_to_print` | int | None | `None` |  | — | Number of completions to print with `rich`. If `None`, all completions are logged. |
| `log_unique_prompts` | bool | `False` |  | — | Whether to log unique prompts. If `True`, only unique prompts are logged. If `False`, all prompts are logged. |

**模型与参考/教师 · 教师模型**

| 参数 | 类型 | 默认 | 关键 | 可选值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `teacher_model_server_url` | str | None | `None` | ★ | — | Base URL of a vLLM server hosting the teacher model (e.g., "http://localhost:8000"). Required. |

**algorithm · 算法其他参数**

| 参数 | 类型 | 默认 | 关键 | 可选值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `loss_top_k` | int | `1` | ★ | — | Number of top tokens the teacher server returns and over which the divergence is computed. For beta == 0 this must be positive; for beta > 0 it must be 1. |
| `loss_add_tail` | bool | `True` | ★ | — | Whether to append a tail bucket representing the remaining probability mass outside the selected top-k support. |
| `reverse_kl_top_1_mode` | str | `sampled` | ★ | — | Reverse-KL top-1 token selection when beta > 0 and loss_top_k == 1. Use 'sampled' for the actual completion token or 'argmax' for the student's top-1 token. The forward-KL support always uses the teacher's top-1 token. |

### 取舍与坑

- ✅ 本地不用放教师模型，显存需求最低的蒸馏路线
- ✅ 适合教师远大于学生的场景
- ⚠️ 依赖教师 server 的可用性与网络带宽
- ⚠️ 只有 top-k 稀疏信号，散度估计有偏
- 🔻 loss_top_k 太小会丢掉教师分布的长尾信息

## <a id="reward"></a>Reward（`reward`）

- **全称**：Outcome Reward Model (ORM)（奖励模型，稳定）
- **配置类**：`RewardConfig` @ `trl/trainer/reward_config.py`（继承链：`RewardConfig` → `_BaseConfig`）
- **训练器**：`RewardTrainer` @ `trl/trainer/reward_trainer.py`
- **文档**：`docs/source/reward_trainer.md`
- **启动**：`trl reward`
- **数据契约**：prompt、chosen、rejected；成对偏好数据（chosen/rejected）

> 在成对偏好数据上训练一个给回复打分的 ORM，产物可以直接作为 GRPO/RLOO/Online DPO 的奖励来源。


### 训练流程映射

| 阶段 | 该算法在此阶段做什么 |
| --- | --- |
| 模型与参考/教师 | 序列分类模型（AutoModelForSequenceClassification，num_labels=1）。 |
| 数据与提示 | 成对偏好数据。 |
| 监督信号 | 偏好标签 → 打分排序目标。 |
| 采样（rollout） | 无。 |
| 优势/偏好估计 | chosen/rejected 的分数差。 |
| 损失与更新 | pairwise ranking loss（可加 center_rewards_coefficient 做正则）。 |
| 记录与产物 | accuracy、margin 等指标 + 可被 RL 调用的奖励模型。 |

### 参数


**algorithm · 算法其他参数**

| 参数 | 类型 | 默认 | 关键 | 可选值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `logging_steps` | float | `10` |  | — | Log every X updates steps. Should be an integer or a float in range `[0,1)`. If smaller than 1, will be interpreted as ratio of total training steps. |
| `gradient_checkpointing` | bool | `True` |  | — | Enable gradient checkpointing to trade compute for memory. Reduces memory at the cost of ~20%% slower training. |
| `bf16` | bool | None | `None` |  | — | Whether to use bf16 (mixed) precision instead of 32-bit. Requires Ampere or higher NVIDIA architecture or Intel XPU or using CPU (use_cpu) or Ascend NPU. If not set, it defaults to `True` if `fp16` is not set. |
| `lr_scheduler_kwargs` | dict | str | None | `None` |  | — | Additional parameters for the lr_scheduler, such as {'num_cycles': 1} for cosine with hard restarts. See the documentation of each scheduler for possible values. |
| `use_liger_kernel` | bool | `False` |  | — | Enable Liger Kernel optimizations: transformers patches the model's layers with Liger's Triton kernels. Deprecated in `SFTTrainer`, `DPOTrainer`, `KTOTrainer`, `GRPOTrainer` and `RLOOTrainer`, and will be removed in v2.0.0: use `model_init… |
| `torch_empty_cache_steps` | int | None | `None` |  | — | Number of steps to wait before calling `torch.<device>.empty_cache()`. Helps avoid CUDA OOM at a cost of ~10%% slower performance. If None, cache will not be emptied. |

**损失与更新 · 训练超参**

| 参数 | 类型 | 默认 | 关键 | 可选值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `learning_rate` | float | `0.0001` | ★ | — | The initial learning rate for AdamW. |

**模型与参考/教师 · Parameters that control the model**

| 参数 | 类型 | 默认 | 关键 | 可选值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `model_init_kwargs` | dict[str, Any] | str | None | `None` |  | — | Keyword arguments for `AutoModelForSequenceClassification.from_pretrained`, used when the `model` argument of the `RewardTrainer` is provided as a string. The `revision` value is also used when loading the tokenizer. |
| `trust_remote_code` | bool | `False` |  | — | Whether to allow loading models and tokenizers that ship custom Python code from the Hub. Forwarded to `AutoModelForSequenceClassification.from_pretrained` and `AutoTokenizer.from_pretrained`. |
| `chat_template_path` | str | None | `None` |  | — | If specified, sets the model's chat template. This can either be the path to a tokenizer (local directory or Hugging Face Hub model) or a direct path to a Jinja template file. When using a Jinja file, you must ensure that any special token… |
| `disable_dropout` | bool | `True` |  | — | Whether to disable dropout in the model. |

**数据与提示 · Parameters that control the data preprocessing**

| 参数 | 类型 | 默认 | 关键 | 可选值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `dataset_num_proc` | int | None | `None` |  | — | Number of processes to use for processing the dataset. |
| `eos_token` | str | None | `None` | ★ | — | Token used to indicate the end of a turn or sequence. If `None`, it defaults to `processing_class.eos_token`. |
| `max_length` | int | None | `1024` | ★ | — | Maximum length of the tokenized sequence. Samples are filtered out if either chosen or rejected sequence exceeds this value. If `None`, no filtering is applied. |
| `pad_to_multiple_of` | int | str | None | `auto` |  | — | If set, the sequences will be padded to a multiple of this value. `'auto'` pads to a multiple of 128 on Neuron (AWS Trainium), which compiles once per input shape, and does not pad elsewhere. |

**损失与更新 · Parameters that control the training**

| 参数 | 类型 | 默认 | 关键 | 可选值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `center_rewards_coefficient` | float | None | `None` | ★ | — | Coefficient to incentivize the reward model to output mean-zero rewards (proposed by [this paper](https://huggingface.co/papers/2312.09244), Eq. 2). Recommended value: `0.01`. |
| `activation_offloading` | bool | `False` |  | — | Whether to offload the activations to the CPU. |

**数据与提示 · 数据与分词器**

| 参数 | 类型 | 默认 | 关键 | 可选值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `pad_token` | str | None | `None` |  | — | Deprecated. Set `tokenizer.pad_token` directly and pass it as `processing_class` to the trainer instead. |

### 取舍与坑

- ✅ 一条命令得到可复用的打分器
- ✅ 与在线 RL trainer 的接口一致（reward_model_name_or_path）
- ⚠️ 只学排序，不保证分数尺度
- ⚠️ 奖励作弊（reward hacking）风险由下游 RL 承担
- 🔻 中心化系数 center_rewards_coefficient 能缓解分数漂移，但会让排序目标略有偏

## <a id="sft"></a>SFT（`sft`）

- **全称**：Supervised Fine-Tuning（监督微调，稳定）
- **配置类**：`SFTConfig` @ `trl/trainer/sft_config.py`（继承链：`SFTConfig` → `_BaseConfig`）
- **训练器**：`SFTTrainer` @ `trl/trainer/sft_trainer.py`
- **文档**：`docs/source/sft_trainer.md`
- **启动**：`trl sft`
- **数据契约**：messages；messages（ChatML）/ prompt+completion / 纯文本

> RL 之前的冷启动：在指令/补全数据上做监督微调，让模型先具备目标输出格式（如先想再答、带工具调用标记），之后 RL 才容易起量。


### 训练流程映射

| 阶段 | 该算法在此阶段做什么 |
| --- | --- |
| 模型与参考/教师 | 策略模型（可叠加 LoRA）。 |
| 数据与提示 | 指令数据；packing 可把多条样本打包成定长序列提升吞吐。 |
| 监督信号 | 监督标签（答案本身）。 |
| 采样（rollout） | 无。 |
| 优势/偏好估计 | 无。 |
| 损失与更新 | 交叉熵（loss_type=nll，可选 nll/chunked_nll/... 由配置决定）+ 可选 packing/padding_free。 |
| 记录与产物 | 同标准 Trainer。 |

### 参数


**algorithm · 算法其他参数**

| 参数 | 类型 | 默认 | 关键 | 可选值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `logging_steps` | float | `10` |  | — | Log every X updates steps. Should be an integer or a float in range `[0,1)`. If smaller than 1, will be interpreted as ratio of total training steps. |
| `gradient_checkpointing` | bool | `True` |  | — | Enable gradient checkpointing to trade compute for memory. Reduces memory at the cost of ~20%% slower training. |
| `bf16` | bool | None | `None` |  | — | Whether to use bf16 (mixed) precision instead of 32-bit. Requires Ampere or higher NVIDIA architecture or Intel XPU or using CPU (use_cpu) or Ascend NPU. If not set, it defaults to `True` if `fp16` is not set. |
| `lr_scheduler_kwargs` | dict | str | None | `None` |  | — | Additional parameters for the lr_scheduler, such as {'num_cycles': 1} for cosine with hard restarts. See the documentation of each scheduler for possible values. |
| `use_liger_kernel` | bool | `False` |  | — | Enable Liger Kernel optimizations: transformers patches the model's layers with Liger's Triton kernels. Deprecated in `SFTTrainer`, `DPOTrainer`, `KTOTrainer`, `GRPOTrainer` and `RLOOTrainer`, and will be removed in v2.0.0: use `model_init… |
| `torch_empty_cache_steps` | int | None | `None` |  | — | Number of steps to wait before calling `torch.<device>.empty_cache()`. Helps avoid CUDA OOM at a cost of ~10%% slower performance. If None, cache will not be emptied. |

**损失与更新 · 训练超参**

| 参数 | 类型 | 默认 | 关键 | 可选值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `learning_rate` | float | `2e-05` |  | — | The initial learning rate for AdamW. |

**模型与参考/教师 · Parameters that control the model**

| 参数 | 类型 | 默认 | 关键 | 可选值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `model_init_kwargs` | dict[str, Any] | str | None | `None` |  | — | Keyword arguments for `AutoModelForCausalLM.from_pretrained`, used when the `model` argument of the `SFTTrainer` is provided as a string. The `revision` value is also used when loading the processing class. |
| `trust_remote_code` | bool | `False` |  | — | Whether to allow loading models and tokenizers that ship custom Python code from the Hub. Forwarded to `AutoModelForCausalLM.from_pretrained` and `AutoProcessor.from_pretrained`. |
| `router_aux_loss_coef` | float | None | `None` |  | — | Coefficient of the load-balancing auxiliary loss for Mixture-of-Experts (MoE) models, added to the training loss with this weight. When not set, the value declared by the model config is used. Set to `0.0` to disable it. Fails when used wi… |
| `chat_template_path` | str | None | `None` |  | — | If specified, sets the model's chat template. This can either be the path to a tokenizer (local directory or Hugging Face Hub model) or a direct path to a Jinja template file. When using a Jinja file, you must ensure that any special token… |

**数据与提示 · Parameters that control the data preprocessing**

| 参数 | 类型 | 默认 | 关键 | 可选值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `dataset_text_field` | str | `text` | ★ | — | Name of the column that contains text data in the dataset. |
| `dataset_kwargs` | dict[str, Any] | str | None | `None` |  | — | Dictionary of optional keyword arguments for the dataset preparation. The only supported key is `skip_prepare_dataset`. When the dataset contains images, `skip_prepare_dataset` is automatically treated as `True` regardless of the provided… |
| `dataset_num_proc` | int | None | `None` |  | — | Number of processes to use for processing the dataset. |
| `eos_token` | str | None | `None` |  | — | Token used to indicate the end of a turn or sequence. If `None`, it defaults to `processing_class.eos_token`. |
| `max_length` | int | None | `1024` | ★ | — | Maximum length of the tokenized sequence. Sequences longer than `max_length` are truncated from the left or right depending on the `truncation_mode`. If `None`, no truncation is applied. When packing is enabled, this value sets the sequenc… |
| `truncation_mode` | str | `keep_start` |  | `keep_start`, `keep_end` | Truncation mode to use when the sequence exceeds `max_length`. The only supported value is `'keep_start'`. The `'keep_end'` value is deprecated and will be removed in v2.0.0. |
| `shuffle_dataset` | bool | `False` |  | — | Whether to shuffle the dataset. |
| `packing` | bool | `False` | ★ | — | Whether to group multiple sequences into fixed-length blocks to improve computational efficiency and reduce padding. Uses `max_length` to define sequence length. |
| `packing_strategy` | str | `bfd` | ★ | `bfd`, `bfd_split`, `wrapped` | Strategy for packing sequences. Can be `'bfd'` (best-fit decreasing, truncates overflow), `'bfd_split'` (best-fit decreasing, splits overflow sequences), or `'wrapped'` (aggressive, cuts mid-sequence). |
| `padding_free` | bool | `False` | ★ | — | Whether to perform forward passes without padding by flattening all sequences in the batch into a single continuous sequence. This reduces memory usage by eliminating padding overhead. Currently, this is only supported with the FlashAttent… |
| `pad_to_multiple_of` | int | str | None | `auto` |  | — | If set, the sequences will be padded to a multiple of this value. `'auto'` pads to a multiple of 128 on Neuron (AWS Trainium), which compiles once per input shape, and does not pad elsewhere. |
| `eval_packing` | bool | None | `None` |  | — | Whether to pack the eval dataset. If `None`, uses the same value as `packing`. |

**损失与更新 · Parameters that control the training**

| 参数 | 类型 | 默认 | 关键 | 可选值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `completion_only_loss` | bool | None | `None` | ★ | — | Whether to compute loss only on the completion part of the sequence. If set to `True`, loss is computed only on the completion, which is supported only for prompt-completion datasets. If `False`, loss is computed on the entire sequence. If… |
| `assistant_only_loss` | bool | `False` | ★ | — | Whether to compute loss only on the assistant part of the sequence. If set to `True`, loss is computed only on the assistant responses, which is supported only for conversational datasets. If `False`, loss is computed on the entire sequenc… |
| `loss_type` | str | `nll` | ★ | `nll`, `dft`, `chunked_nll` | Type of loss to use. Possible values are `'nll'` (standard negative log-likelihood) and `'dft'` (Dynamic Fine-Tuning, https://huggingface.co/papers/2508.05629). |
| `activation_offloading` | bool | `False` |  | — | Whether to offload the activations to the CPU. |

**数据与提示 · 数据与分词器**

| 参数 | 类型 | 默认 | 关键 | 可选值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `pad_token` | str | None | `None` |  | — | Deprecated. Set `tokenizer.pad_token` directly and pass it as `processing_class` to the trainer instead. |

### 取舍与坑

- ✅ 最稳的起点，几乎不会崩
- ✅ 为后续 RL 提供格式稳定的初版模型
- ⚠️ 纯模仿，无法超越数据质量
- ⚠️ 过久会损失探索能力（熵坍缩）
- 🔻 packing=True 时必须确认 attention 掩码正确，否则样本互相污染
