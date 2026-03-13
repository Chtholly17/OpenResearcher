# Multi-Turn RL 代码理解文档

本文档基于 `verl_rl/run_grpo_training_h200.sh` 和 verl 框架，系统性梳理 multi-turn GRPO 训练的代码实现。

---

## 一、文档计划 (Plan)

### 1. 整体架构
- [x] 1.1 训练入口与主循环
- [x] 1.2 数据流：DataProto、batch 结构
- [x] 1.3 组件关系：Actor / Rollout / Ref / Reward

### 2. Reward 计算
- [x] 2.1 custom_reward_function 加载
- [x] 2.2 NaiveRewardManager 调用链路
- [x] 2.3 compute_score 输入输出与 score 放置位置
- [x] 2.4 OpenResearcher 的 compute_score 逻辑（correctness、search、efficiency）

### 3. Advantage 计算
- [x] 3.1 GRPO vs GAE 区别
- [x] 3.2 compute_grpo_outcome_advantage 公式与实现
- [x] 3.3 token_level_rewards → advantages 的广播
- [x] 3.4 index 分组（同一 prompt 的 n 个 response）

### 4. 模型更新
- [x] 4.1 Actor update 入口与 mini-batch 划分
- [x] 4.2 Policy loss（vanilla PPO clip）
- [x] 4.3 KL loss（可选）与 entropy bonus
- [x] 4.4 response_mask 与 loss 聚合

### 5. Rollout 实现
- [x] 5.1 vLLM sync vs async，AgentLoopManager 触发条件
- [x] 5.2 ToolAgentLoop 状态机（PENDING → GENERATING → PROCESSING_TOOLS → INTERACTING → TERMINATED）
- [x] 5.3 工具调用：_call_tool、tool_response 拼接到 prompt
- [x] 5.4 Interaction（OpenResearcherInteraction）与 trajectory 反馈
- [x] 5.5 response_mask：1=模型 token，0=tool response token
- [x] 5.6 log_prob 计算与 old_log_prob 用途

### 6. 关键配置
- [x] 6.1 run_grpo_training_h200.sh 中的关键覆盖
- [x] 6.2 multi_turn、max_model_len、max_response_length 关系

---

## 二、整体架构

### 2.1 训练入口

**入口**（`verl/trainer/main_ppo.py`）：
- `@hydra.main(config_path="config", config_name="ppo_trainer")` 加载配置
- `ray.init()` 初始化 Ray
- `TaskRunner.run.remote(config)` 启动训练

**Worker 创建**（`main_ppo.py` 第 110–120 行）：
```python
if config.actor_rollout_ref.rollout.mode == "async":
    actor_rollout_cls = AsyncActorRolloutRefWorker  # 使用 AgentLoopManager
else:
    actor_rollout_cls = ActorRolloutRefWorker
```
- **async 模式**：`AsyncActorRolloutRefWorker` 内部创建 `AgentLoopManager`，支持 `ToolAgentLoop` 等多轮工具调用
- **sync 模式**：`ActorRolloutRefWorker` 使用 `vLLMRollout` 或 `SGLangRollout` 单轮生成

### 2.2 主循环与数据流（ray_trainer.py）

**主循环**（`fit()` 第 974–1140 行）：
1. **gen**：`generate_sequences(gen_batch)` 生成 responses（`gen_batch` 已 `repeat(n, interleave=True)`）
2. **reward**：`compute_reward(batch, reward_fn)` 计算每个 trajectory 的 score
3. **old_log_prob**：`actor_rollout_wg.compute_log_prob(batch)` 用当前 actor 重算 log_prob
4. **ref_log_prob**（可选）：`ref_policy_wg.compute_ref_log_prob(batch)` 用于 KL loss
5. **adv**：`compute_advantage(batch, adv_estimator=GRPO, ...)` 计算 advantages
6. **update_actor**：`actor_rollout_wg.update_actor(batch)` 更新策略
7. **update_critic**（若使用 GAE）：更新 value 网络

**DataProto 结构**（`verl/protocol.py`）：
- `batch`（TensorDict）：`prompts`、`responses`、`input_ids`、`attention_mask`、`position_ids`、`response_mask`、`old_log_probs`、`advantages`
- `non_tensor_batch`：`data_source`、`reward_model`、`extra_info`、`uid`、`__num_turns__` 等
- `repeat(repeat_times=n, interleave=True)` 使同一 prompt 的 n 个 response 共享同一 `uid`，供 GRPO 分组

**_get_gen_batch**（`ray_trainer.py` 第 501–516 行）：移除 `input_ids`、`attention_mask`、`position_ids`；保留 `data_source`、`reward_model`、`extra_info`、`uid`。async 模式下将 `batch.non_tensor_batch` 合并回 `gen_batch`，供 agent loop 使用 `extra_info.interaction_kwargs` 等。

### 2.3 组件关系

| 组件 | 作用 | 实现 |
|------|------|------|
| **Actor** | 可训练策略，PPO 更新 | `actor_module`（FSDP），`dp_actor.py` |
| **Rollout** | 生成 responses | vLLM/SGLang 或 `AgentLoopManager`（async） |
| **Ref** | 参考策略，计算 KL | `ref_module`，与 actor 共享基座 |
| **Reward** | 计算 trajectory score | `NaiveRewardManager` + `compute_score` |

---

## 三、Reward 计算

### 3.1 custom_reward_function 加载

**调用链**（`verl/trainer/main_ppo.py` → `verl/trainer/ppo/reward.py`）：
```
load_reward_manager(config, tokenizer, ...)
  → get_custom_reward_fn(config)
  → importlib.util.spec_from_file_location("custom_module", path)
  → getattr(module, "compute_score")
  → NaiveRewardManager(compute_score=compute_score)
```

**实现**（`reward.py` 第 41–93 行）：
- `config.custom_reward_function.path` 指定 `.py` 文件路径
- `config.custom_reward_function.name` 指定函数名（如 `compute_score`）
- 动态 import 后返回 `partial(_call_with_kwargs, raw_fn, reward_kwargs)`

配置（`openresearcher_multiturn_grpo.yaml`）：
```yaml
custom_reward_function:
  path: verl_rl/reward/openresearcher_reward.py
  name: compute_score
```

### 3.2 NaiveRewardManager 调用

**实现**（`verl/workers/reward_manager/naive.py` 第 46–126 行）：
```python
for i in range(len(data)):
    data_item = data[i]
    prompt_ids = data_item.batch["prompts"]
    valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
    valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
    valid_response_ids = response_ids[:valid_response_length]

    prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
    response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
    ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
    extra_info["num_turns"] = data_item.non_tensor_batch.get("__num_turns__", None)

    score = self.compute_score(data_source, solution_str, ground_truth, extra_info)
    reward_tensor[i, valid_response_length - 1] = reward  # 仅最后一 token 非零
```

**若已有 rm_scores**：Agent loop 若在 rollout 时已算 reward，会写 `batch["rm_scores"]`，则直接返回，不再调用 `compute_score`。

- `solution_str`：整条 trajectory 的文本（含 thinking、tool calls、tool responses、最终 answer）
- `reward_tensor` 为 token-level，shape `(bs, response_length)`，仅最后一个有效 token 非零

### 3.3 compute_score 输入输出与 score 放置位置

**输入**：`compute_score(data_source, solution_str, ground_truth, extra_info)`  
- `solution_str`：整条 trajectory 解码文本  
- `extra_info`：含 `num_turns`、`rollout_reward_scores` 等

**输出**：`float` 或 `dict`（含 `"score"` 键）。若为 dict，NaiveRewardManager 取 `reward = score["score"]`。

**放置**：`reward_tensor[i, valid_response_length - 1] = reward`，仅最后一个有效 token 位置非零。

### 3.4 OpenResearcher compute_score 逻辑（correctness、search、efficiency）

`verl_rl/reward/openresearcher_reward.py`：

1. **答案提取**：`extract_answer(solution_str)` → `(answer_text, is_explicit)`
   - `<answer>...</answer>`、`Exact Answer:`、`Final Answer:` 优先
   - 备选：从最后一个 `<think>` 中正则推断

2. **正确性**：`normalize_answer(pred) == normalize_answer(gt)` 或互相包含

3. **搜索**：`_count_search_calls(solution_str)` 统计 `browser.search`，`searched = n_searches >= 1`

4. **效率**：`_length_efficiency(num_turns) = max(0, 1 - num_turns/500)`，turn 越少 eff 越高

5. **得分表**（v0.5）：
   - Correct + ≥1 searches：`0.8 + 0.4 * eff` ∈ [0.8, 1.2]
   - Correct + 0 searches：0.3（memory recall）
   - Wrong + ≥1 searches：0.1
   - Wrong + 0 searches：0.0
   - No answer：0.0

### 3.5 token_level_scores / token_level_rewards

- `reward_tensor` 经 `batch.batch["token_level_scores"]` 传入
- 若 `use_kl_in_reward=False`：`token_level_rewards = token_level_scores`
- GRPO 使用 `token_level_rewards.sum(dim=-1)` 得到每个 response 的标量 reward

---

## 四、Advantage 计算

### 4.1 GRPO vs GAE

- **GAE**：需要 critic，用 value 和 TD 误差估计 advantage
- **GRPO**：无 critic，用同一 prompt 下多个 response 的 reward 做组内相对比较

### 4.2 compute_grpo_outcome_advantage

```python
# verl/trainer/ppo/core_algos.py
@register_adv_est(AdvantageEstimator.GRPO)
def compute_grpo_outcome_advantage(
    token_level_rewards,  # (bs, response_length)
    response_mask,
    index,  # 同一 prompt 的样本共享相同 index
    norm_adv_by_std_in_grpo=True,
    ...
):
    scores = token_level_rewards.sum(dim=-1)  # (bs,)
    for idx in id2score:
        scores_tensor = torch.stack(id2score[idx])
        id2mean[idx] = torch.mean(scores_tensor)
        id2std[idx] = torch.std(scores_tensor)
    for i in range(bsz):
        if norm_adv_by_std_in_grpo:
            scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
        else:
            scores[i] = scores[i] - id2mean[index[i]]
    scores = scores.unsqueeze(-1) * response_mask  # 广播到 token 维
    return scores, scores  # advantages, returns
```

- `index[i]`：第 i 个样本所属的 prompt 组
- 组内减去均值，可选除以标准差
- 结果广播到 `(bs, response_length)`，再乘 `response_mask`（仅对模型生成 token 有效）

### 4.3 token_level_rewards → advantages 的广播

- `scores = token_level_rewards.sum(dim=-1)`：得到 `(bs,)` 标量
- 组内标准化后：`scores.unsqueeze(-1) * response_mask` 广播到 `(bs, response_length)`
- `response_mask=0` 的位置（tool response token、padding）advantage 为 0

### 4.4 index 分组（同一 prompt 的 n 个 response）

**index 来源**：`index = data.non_tensor_batch["uid"]`（`ray_trainer.py` 第 237、251 行）

**uid 与 repeat**：
- 每个原始 sample 在 repeat 前分配一个 `uid`（`batch.non_tensor_batch["uid"] = [uuid4(), ...]`）
- `gen_batch.repeat(n, interleave=True)` 后，同一 prompt 的 n 个 response 共享同一 `uid`
- 例如 `n=8`：`uid = [u0,u0,u0,u0,u0,u0,u0,u0, u1,u1,...,u1, ...]`

**GRPO 分组**：`id2score[index[i]]` 按 `uid` 分组，同组内算 `mean`、`std`，得到相对 advantage。

### 4.5 调用处

```python
# ray_trainer.py 第 1115–1124 行
batch = compute_advantage(
    batch,
    adv_estimator=AdvantageEstimator.GRPO,
    num_repeat=config.actor_rollout_ref.rollout.n,
    norm_adv_by_std_in_grpo=config.algorithm.norm_adv_by_std_in_grpo,
)
# batch.batch["advantages"] 被更新
```

---

## 五、模型更新

### 5.1 Actor update 入口与 mini-batch 划分

**入口**（`ray_trainer.py` 第 1136–1138 行）：
```python
actor_output = self.actor_rollout_wg.update_actor(batch)
```

**实现**（`verl/workers/actor/dp_actor.py` 第 358–494 行）：
```python
data = data.select(batch_keys=["responses","response_mask","input_ids","attention_mask",
                               "position_ids","old_log_probs","advantages", ...])
mini_batches = data.split(self.config.ppo_mini_batch_size)  # 如 8
for _ in range(self.config.ppo_epochs):
    for mini_batch in mini_batches:
        micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)  # 如 1
        for micro_batch in micro_batches:
            entropy, log_prob = self._forward_micro_batch(...)
            pg_loss, ... = policy_loss_fn(old_log_prob, log_prob, advantages, response_mask, ...)
            loss.backward()
        self._optimizer_step()
```

### 5.2 Policy Loss（vanilla PPO clip）

**公式**（`core_algos.compute_policy_loss_vanilla`）：
```python
ratio = torch.exp(log_prob - old_log_prob)  # importance sampling ratio
pg_losses1 = -advantages * ratio
pg_losses2 = -advantages * torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio)
# 当 advantage<0 时取更保守的 clip，避免 ratio 过大
pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)
pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode="token-mean")
```

- `clip_ratio` 默认 0.2，对应 PPO 的 ε
- `response_mask=0` 的 token（tool response、padding）不参与 loss

### 5.3 KL Loss 与 Entropy

- **KL loss**（`use_kl_loss=True`）：`kld = kl_penalty(log_prob, ref_log_prob, kl_loss_type)`，`policy_loss += kl_loss * kl_loss_coef`（如 0.001）
- **Entropy bonus**（`entropy_coeff != 0`）：`policy_loss -= entropy * entropy_coeff`，鼓励探索

### 5.4 response_mask 与 loss 聚合

- `agg_loss(loss_mat, loss_mask, loss_agg_mode)`：只对 `loss_mask=1` 的 token 求平均（`token-mean`）或按配置聚合
- `response_mask`：1=模型生成 token，0=tool response token、padding

---

## 六、Rollout 实现

### 6.1 vLLM sync vs async，AgentLoopManager 触发条件

**触发条件**（`main_ppo.py` 第 119 行，`ray_trainer.py` 第 758–765 行）：
```python
if config.actor_rollout_ref.rollout.mode == "async":
    self.async_rollout_mode = True
    self.async_rollout_manager = AgentLoopManager(config, worker_group, rm_wg)
```

| 模式   | rollout.mode | 实现                | 多轮 + 工具 |
|--------|--------------|---------------------|-------------|
| sync   | "sync"       | vLLMRollout / SGLangRollout | SGLang 原生 multi_turn |
| async  | "async"      | AgentLoopManager + vLLM/SGLang 后端 | ToolAgentLoop |

**agent_name**（`agent_loop.py` 第 456–457 行）：若 `agent_name` 不在 `batch.non_tensor_batch` 中，默认 `"single_turn_agent"`。要使用 `tool_agent`，需在数据中提供 `agent_name` 或通过配置注入。OpenResearcher 通过 `default_agent_loop=tool_agent` 覆盖默认。

**注意**：`run_grpo_training_h200.sh` 未显式设置 `actor_rollout_ref.rollout.mode=async`。默认 `mode=sync` 时不会使用 `AgentLoopManager`，`tool_agent` 不会生效。multi-turn 工具调用需在配置或脚本中设置 `mode=async`。

### 6.2 ToolAgentLoop 状态机

**状态定义**（`tool_agent_loop.py` 第 35–40 行）：`PENDING`、`GENERATING`、`PROCESSING_TOOLS`、`INTERACTING`、`TERMINATED`。

**状态转换**（`run()` 第 147–161 行）：
```
PENDING --[apply_chat_template+tools]--> GENERATING
GENERATING --[有 tool_calls]--> PROCESSING_TOOLS
GENERATING --[有 interaction 且无 tool_calls]--> INTERACTING
GENERATING --[无 tool_calls 且无 interaction]--> TERMINATED
PROCESSING_TOOLS --[tool response 拼回]--> GENERATING
INTERACTING --[根据 should_terminate]--> TERMINATED 或 GENERATING
```

**终止条件**：`len(response_mask) >= response_length`、`assistant_turns >= max_assistant_turns`、`user_turns >= max_user_turns`、或 interaction 返回 `should_terminate=True`。

### 6.3 工具调用：_call_tool、tool_response 拼接到 prompt

**_handle_processing_tools_state**（第 258–358 行）：
```python
tasks = [self._call_tool(tc, agent_data.tools_kwargs) for tc in agent_data.tool_calls[:max_parallel_calls]]
responses = await asyncio.gather(*tasks)
for tool_response, tool_reward, _ in responses:
    add_messages.append({"role": "tool", "content": tool_response.text})
    agent_data.messages.extend(add_messages)
    # tokenize tool response，去掉 system prompt 前缀
    response_ids = tokenizer.apply_chat_template(add_messages, ...)[len(system_prompt):]
    agent_data.prompt_ids += response_ids
    agent_data.response_mask += [0] * len(response_ids)
```

**_call_tool**（第 411–456 行）：`tool.create()` → `tool.execute(instance_id, tool_args)` → `tool.release()`。`max_tool_response_length` 超长时按 `tool_response_truncate_side`（left/right/middle）截断。

### 6.4 Interaction（OpenResearcherInteraction）与 trajectory 反馈

**调用时机**：模型生成一轮后若无 tool calls，则进入 `INTERACTING`（`_handle_interacting_state`）。

**OpenResearcherInteraction**（`verl_rl/interactions/openresearcher_interaction.py`）：
- `generate_response(instance_id, messages)`：取最新 assistant message，用 `extract_answer` 解析 answer
- 若 answer 正确：返回 `(should_terminate=True, response="Correct!", reward=1.0)`
- 若 answer 错误：返回 `(should_terminate=False, response="", reward=0.0)`，模型可继续搜索
- `extra_info.interaction_kwargs` 含 `name: "openresearcher"`、`ground_truth`、`query`，来自 `preprocess_openresearcher.py` 的 `make_verl_record`

### 6.5 response_mask：1=模型 token，0=tool response token

- **模型生成**（`_handle_generating_state`）：`agent_data.response_mask += [1] * len(response_ids)`
- **Tool response**：`agent_data.response_mask += [0] * len(response_ids)`
- **Interaction 的 user 消息**：同样 `[0] * len(response_ids)`（视为环境/用户输入）
- 只有 `response_mask=1` 的 token 参与 advantage 与 policy loss

### 6.6 log_prob 计算与 old_log_prob 用途

- **Rollout**：`calculate_log_probs=True` 时，vLLM 返回每 token 的 log_prob，存入 `rollout_log_probs`（可选用于 TIS）
- **old_log_prob**：训练时由 `actor_rollout_wg.compute_log_prob(batch)` 用当前 actor 对 `input_ids` 重算得到
- **PPO**：`ratio = exp(log_prob - old_log_prob)` 做 importance sampling，`log_prob` 为本次 forward 的输出

---

## 七、关键配置

### 7.1 run_grpo_training_h200.sh 覆盖

| 配置 | 值 | 说明 |
|------|-----|------|
| data.train_batch_size | 16 | 每步 prompt 数 |
| data.max_prompt_length | 4096 | prompt 最大 token 数 |
| data.max_response_length | 98304 | response 最大 token 数（含 tool response） |
| data.filter_overlong_prompts | True | 过滤超长 prompt |
| data.return_raw_chat | True | 返回 messages 供 agent loop 使用 |
| actor_rollout_ref.rollout.n | 8 | 每个 prompt 生成 8 个 response |
| actor_rollout_ref.rollout.max_model_len | 102400 | vLLM 总 context 长度 |
| actor_rollout_ref.rollout.calculate_log_probs | True | 需要 old_log_prob 做 PPO |
| actor_rollout_ref.rollout.agent.default_agent_loop | tool_agent | 使用 ToolAgentLoop |
| actor_rollout_ref.rollout.multi_turn.max_assistant_turns | 250 | 最大 assistant 轮数 |
| actor_rollout_ref.rollout.multi_turn.max_user_turns | 250 | 最大 user/tool 轮数 |
| actor_rollout_ref.rollout.multi_turn.max_tool_response_length | 1024 | 单次 tool response 截断 |
| actor_rollout_ref.actor.use_kl_loss | True | 启用 KL loss |
| actor_rollout_ref.actor.kl_loss_coef | 0.001 | KL 系数 |
| custom_reward_function.path | verl_rl/reward/openresearcher_reward.py | 自定义 reward |
| algorithm.adv_estimator | grpo | GRPO advantage |
| algorithm.norm_adv_by_std_in_grpo | True | advantage 组内标准化 |

### 7.2 multi_turn、max_model_len、max_response_length 关系

- **max_model_len**：vLLM 单序列总长上限 = prompt + response（含 tool response）。超长会被 vLLM 拒绝。
- **max_response_length**：response 部分的 token 预算，包含模型生成 token 与 tool response token。通常 `max_response_length ≈ max_model_len - max_prompt_length`。
- **max_tool_response_length**：单次 tool 返回的截断长度（如 1024），超长按 middle/left/right 截断。
- **multi_turn.max_assistant_turns / max_user_turns**：控制多轮对话轮数，避免无限循环。

---

## 八、代码索引

| 功能 | 文件路径 |
|------|----------|
| 训练入口 | `verl/trainer/main_ppo.py` |
| 主循环 fit() | `verl/trainer/ppo/ray_trainer.py` |
| DataProto、repeat | `verl/protocol.py` |
| 数据加载、collate | `verl/utils/dataset/rl_dataset.py` |
| Reward 加载 | `verl/trainer/ppo/reward.py` |
| NaiveRewardManager | `verl/workers/reward_manager/naive.py` |
| OpenResearcher compute_score | `verl_rl/reward/openresearcher_reward.py` |
| compute_advantage、GRPO | `verl/trainer/ppo/ray_trainer.py`、`core_algos.py` |
| Actor update_policy | `verl/workers/actor/dp_actor.py` |
| Policy loss vanilla | `verl/trainer/ppo/core_algos.py` |
| AgentLoopManager | `verl/experimental/agent_loop/agent_loop.py` |
| ToolAgentLoop | `verl/experimental/agent_loop/tool_agent_loop.py` |
| OpenResearcherInteraction | `verl_rl/interactions/openresearcher_interaction.py` |
| 数据预处理 | `verl_rl/preprocess_openresearcher.py` |
| 配置 | `verl_rl/config/openresearcher_multiturn_grpo.yaml` |

---

## 九、数据流速览

```
DataLoader (parquet)
  → batch {prompts, raw_prompt, data_source, reward_model, extra_info, ...}
  → repeat(n, interleave) + uid
  → _get_gen_batch → gen_batch {prompts, raw_prompt, extra_info, ...}
  → generate_sequences (vLLM / AgentLoopManager)
      → ToolAgentLoop: PENDING→GENERATING→PROCESSING_TOOLS⇄GENERATING→INTERACTING→TERMINATED
      → output {prompt_ids, response_ids, response_mask, num_turns}
  → batch.union(gen_output)
  → compute_reward → token_level_scores
  → compute_log_prob (old), compute_ref_log_prob
  → compute_advantage (GRPO: 组内 r-mean, 广播)
  → update_actor (PPO clip loss, KL, entropy)
```
