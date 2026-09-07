# Proposal：环境锚定的反思式自动形式化 RL

- 日期：2026-09-07
- 状态：方法提案（尚未开训 grounded 目标）
- 底座：Qwen3.5-9B SFT（当前默认 v5；RL 建议先冻 v2 或 v5，不再改 schema）
- 相关实现：`ReForm/data/train/`（采集 / 单次 rollout）、`sft/`（LoRA / 评测）、`verl/examples/grpo_trainer/`（GRPO baseline）

---

## 1. 目标

下一阶段做 **强化学习**，目标是投顶会（ICLR / NeurIPS / ICML；备选 ACL / EMNLP）。

**不能**把「对现有 SFT 跑一层 GRPO」或「把 ReForm 的 PBSO 接到 9B + 工具上」写成主贡献。这两件事是必须先跑的 baseline。

论文要回答的问题：

> 在已经有反思格式（ReForm）和工具轨迹（本地 SFT）的前提下，如何让反思被 **检索 / Lean 编译器** 钉住，而不是只在语言里自说自话？

---

## 2. 现状（提案所依赖的事实）

### 2.1 任务

把自然语言数学 / 物理题译成 Lean 4 定理陈述（`:= by sorry`）。主指标 **BOTH** = Lean 编译 ∧ 外部 critic 语义忠实。

采集是外环（教师 gpt-4o + TokenHub critic + Kimina 门禁）。SFT / 评测 / 以后的 RL 学的是 **一条连续 assistant turn**：模型自己决定何时 `<tool_call>`、何时开下一轮、何时写 `</answer>`。

### 2.2 数据

三桶混合，入库 **10073** 条（`sft/data/stats.json`）：

| 桶 | 条数 | 说明 |
|---|---|---|
| 数学工具轨迹 | 3857 | leansearch + 可选 web + Lean 门禁 + critic |
| 物理多模态 | 2216 | 必须有图；physsearch + 可选 web |
| Physlib 平行 | 4000 | 无工具 / 无 critic，informal → 现成签名 |

造数阶段未对评测集去重；训练 / RL 必须按 `dataset_name` + `question_id` 去污染。

### 2.3 SFT 版本（同一批 assistant，只改 prompt）

| 版 | Prompt | 结果要点 |
|---|---|---|
| v1 | user = 指令 + 题目；全 Correct；未 sanitize | 会编造编译器原文；结尾几乎全 Correct |
| v2 | 仍无工具目录；sanitize + 15% Incorrect | **物理最好**（PhyX BOTH 39.1%）；数学也强 |
| v3 | 长工具目录贴在 user | AIME 升到约 42%；**物理崩**（PhyX 21.6%） |
| v4 | 短 schema 仍在 user | 物理部分回升，仍低于 v2 |
| v5 | schema 放到 system；user 只留题目 | 物理好于 v3/v4，仍低于 v2；AIME / ProofNet 下滑 |

v2–v5 说明：**改工具说明放哪换不回物理。** RL 应冻一套 prompt，不再开 v6 schema。

### 2.4 和 ReForm-8B 的数字差距（必须在论文里解释或追上）

| 集 | ReForm-8B sem | 本地 SFT-v2 BOTH |
|---|---|---|
| miniF2F | 87.7 | 86.1 |
| ProofNet | 65.6 | 59.1 |
| Putnam | 57.3 | 47.0 |
| AIME2025 | 46.7 | 34.3（10.3/30） |

对方无工具、无物理。SFT 在 Putnam / AIME 各差约 10–12 个点。RL 若只赢物理、数学再掉，审稿会写成「更弱的 ReForm + 图片」。

SFT 已暴露、且适合写成过程奖励的失败：

- 发明 Mathlib / Physlib 名字
- Lean 过、critic 不过（物理尤其明显）
- 加 catalog 后 `web_search` 从约 41% 掉到约 2%
- 自判几乎全是 Correct（critique 不受环境约束）

采集时 Lean 是 **门禁**（失败轮不进轨迹）。学生推理时看不见编译器。这是方法和采集协议的差，不是实现疏忽。

---

## 3. 相关工作：什么已经被占住

| 工作 | 已占住 | 没做 |
|---|---|---|
| ReForm + PBSO（ICLR 2026） | 单序列反思；`r_aux` + `r_task`；clip 折扣回报 | 无检索；编译器不进 token 流；只做数学 |
| ATF（2025-10） | 编译器 + LLM judge 当工具；外环改写；DPO / expert iteration | 不是位置异构 RL；无物理；不是库检索 |
| FormaRL（2025-08） | 无标注 NL 上 GRPO；奖励 = 编译 ∧ LLM 一致性 | 单遍；无工具；无反思 |
| MathForm（2026-08） | 检索 + 编译/语义造 FormalVerse（约 36.7 万）；再 SFT+RL | 检索用在造数；学生仍是单遍 |
| MMFormalizer（2026-01） | 「第一个」多模态物理形式化；PhyX-AF；提示式 recursive grounding | 无 RL |
| CriticLean（ACL 2026） | 训 critic；FineLeanCorpus | 不训带工具的反思 agent |
| ToRL / Search-R1 / TRACE | 工具 RL：mask 观察、turn-level credit | QA / 代码解释器，不是形式库 |

**禁止当 novelty 的句子：**

- 「我们给 ReForm 做了 RL」
- 「奖励 = 编译 + critic」
- 「模型能看编译错误再改」
- 「先检索 Mathlib 再训」
- 「第一个多模态自动形式化」
- 「工具 RL + GRPO」

---

## 4. 主主张（修订后）

**Format-preserving process RL：环境进奖励，不进（新的）观察。**

SFT / 评测协议已经固定：round 中间只有检索 `<tool_response>`；语义自检由模型自己写；Lean 与外部 critic 只在终点打分。采集时 Lean 是门禁，不进轨迹。

因此 **不要**把编译器原文拼进 RL 序列（那会和 SFT 不一致，也不是当前评测协议）。空档是：

> 保持与 SFT 相同的 token 形状，把检索利用、按轮编译、自判校准做成 **可自动算的过程奖励**，再按 span 做 credit assignment。

物理仍是 stress test，不是摘要第一句。

---

## 5. 方法（修订：P1 软化，P2 不进序列，P3 只打分）

### 5.1 三类 span（与 SFT 同形）

| Span | Loss | 即时奖励 \(r\)（事后算，不写回上下文） | 回报 |
|---|---|---|---|
| `<tool_call>…` | 开 | 软：后续 Lean **用到了**本次命中则 bonus；空搜 + 只用 prelude **不罚** | 从 compile / BOTH 折扣 |
| `<tool_response>` | **关** | 无 | 无 |
| Round critique / 自判 | 开 | 自判与终点 Lean/critic 是否一致（校准） | clip 折扣 |
| 当轮 / 最终 Lean | 开 | 该块是否编译；终点 BOTH | 终端 \(r_{\mathrm{task}}\) |

编译器可以在 **后台**对每个 ` ```lean4 ` 跑一遍，只产生标量 \(r_{\mathrm{compile}}\)，**不把报错文本插入序列**。

### 5.2 三个技术点（修订）

**P1′. 软检索利用（不再「名字必须来自检索」）**

硬规则会误伤空搜 + 正确的 `Finset`/`Real`/binder。改为：

- 用到命中中的声明名 → bonus
- 空搜或只用 prelude/binder → 0，不扣
- 可选：仅当 Lean 报 `unknown identifier X` 且 X 不在命中里 → 扣（与编译合取，独立贡献变小）

**P2′. 编译器只打分、不回灌**

默认不做 ATF 式「诊断进上下文」。与 SFT/评测对齐。若以后要做「看见报错再改」，必须先冷启动一小撮 compiler-in-the-loop SFT，作为单独消融，不当默认。

**P3′. 自判校准**

不要求 critique 引用编译器原文（轨迹里没有原文）。只比：模型写的 Correct/Incorrect 是否等于该轮/终点的 compile 或外部 critic。打的是「永远 Correct」那几个 token。

### 5.3 算法槽

同一政策、同一数据：

1. 终端 GRPO（0.5 compile + 0.5 semantic）
2. 原版 PBSO（语言 `r_aux` + `r_task`）
3. 过程 RL（P1′ + 后台 P2′ + P3′，序列与 SFT 同形）

三者 BOTH 分不开 → 没有方法论文。

---

## 6. 明确不做（或只做工程、不写进贡献）

| 想法 | 原因 | 是否还做 |
|---|---|---|
| 现有 verl GRPO | FormaRL 已是这个形状 | 做，当 Table 1 左列 |
| 官方 PBSO + 9B + 工具 | 审稿会写成 ReForm + tools | 做，当第二列 baseline |
| v6 tool schema | v2–v5 已是阴性 | **冻 prompt，停** |
| 硬 P1（名字必须 ⊆ 检索） | 空搜 + 正确 prelude 误伤 | 改为 P1′ 软 bonus |
| P2 把编译器拼进序列（无冷启动） | 与 SFT/评测协议不一致 | 默认不做；或先冷启动再单列消融 |
| 「第一个物理 AF 的 RL」当标题 | 应用向；MMFormalizer 占了多模态叙事 | 物理当第二节实验 |
| 只发 PhysicsConsistencyCheck | 数据集更像 ACL 配套 | 可做 200–400 条 human，绑在方法上 |

---

## 7. 实验与通过条件

### 7.1 必须有的消融

| 消融 | 通过 | 杀掉提案 |
|---|---|---|
| 终端 GRPO vs PBSO vs 过程 RL（P1′+P2′+P3′） | 过程 RL 在 **AIME + PhyX 的 BOTH** 上赢 | 三者在噪声内 |
| mask 检索 token vs 不 mask | mask 才能稳 | 不 mask 更好 → 故事错了 |
| P1′ / 后台 P2′ / P3′ 各拆开 | P1′ 降发明名且不伤空搜；P3′ 降「假 Correct」 | 只涨长度或工具次数 |
| （可选）冷启动后的编译器进序列 | 相对 P2′ 再降 Lean-only | 无冷启动就进序列 → 与 SFT 不一致，不当默认 |
| 数学 vs ReForm-8B syn/sem | Putnam / AIME sem 接到对方附近 | 数学再掉一大截 |
| 物理 vs SFT-v2；PhyX-AF vs MMFormalizer 编译率 | PhyX BOTH 稳定超过 39.1% | 物理 ≤ v2 |

评测协议：single-shot rollout、真工具、`EVAL_PROMPT_PATCH=0`、BOTH = critic ∧ Lean。数学用官方全集（`SPLIT=all`）。RL 训练数据必须去污染。

### 7.2 数字目标（投稿门槛）

- 数学：Putnam / AIME sem 接到 ReForm-8B 附近（约 57 / 47）。
- 物理：PhyX BOTH **稳定超过** SFT-v2 的 39.1%；SeePhys / PhysUni / PhyX-AF 相对 v2 同向提升。
- 只做到其中一项：更适合 workshop / 短文，不作为顶会主投。

---

## 8. 工作顺序

| 阶段 | 交付 | 论文角色 | 预估 |
|---|---|---|---|
| 0. Baseline | 现有 GRPO；v2 或 v5 prompt；工具开；奖励 = BOTH。日志：发明名率、`web_search` 率、critique–编译器一致率 | Table 1 左列 | 2–3 周 |
| 1. 方法 | P1′ 软 bonus + 后台按轮 \(r_{\mathrm{compile}}\) + P3′ 自判校准 + 检索 token mask。序列与 SFT 同形。冻一个 advantage 公式 | §3 | 方法核心 |
| 2. 数学表 | miniF2F / ProofNet / Putnam / AIME；对比 ReForm、Goedel-V2、ATF、FormaRL、MathForm（能跑则跑） | 没有这张表不投 ICLR/NeurIPS | |
| 3. 物理 stress test | PhyX / SeePhys / PhysUni / PhyX-AF；与 SFT-v2、MMFormalizer 数字对齐协议 | 证明不是数学-only PBSO | |
| 4. 可选配套 | 200–400 条物理语义 human；错误分类（发明名 / 空虚陈述 / 求错量） | appendix 或第二贡献 | |

---

## 9. 投稿定位

- **ICLR / NeurIPS / ICML**：主贡献必须是 **命名过的目标 + 消融表**。物理不是摘要第一句。
- **ACL / EMNLP**：方法可以薄一点，用物理错误类型和 human 集撑分析。

不要写 *we scale ReForm to physics with GRPO*。

---

## 10. 实现落点（提案对齐代码，不在此改）

- Rollout / 真工具：`ReForm/data/train/rollout_singleshot.py`，`verl/experimental/agent_loop/reform_tool_agent.py`
- 现有终端奖励：`verl/utils/reward_score/reform_rollout.py`（GRPO baseline）
- 官方 PBSO 参考：`ReForm/lean_plugins/reward.py`（`propagate_reward` + `r_aux` / `r_task`）
- 工具：`verl/tools/reform_search_tools.py`，采集侧 `leansearch_tool.py` / `physsearch_tool.py` / `web_search_tool.py`
- Lean：Kimina（`VERIFY_URL`），与评测同一套 header 约定（`VERIFY_CANONICAL_HEADER=1`）

阶段 1 需要动的三处：环境 token mask、按轮 compile 并拼接、advantage 按 span 而不是整条轨迹一个标量。

---

## 11. 参考文献（提案依据）

- Chen et al. ReForm, ICLR 2026. arXiv:2510.24592
- Guo et al. Autoformalizer with Tool Feedback (ATF). arXiv:2510.06857
- Huang et al. FormaRL. arXiv:2508.18914
- Pu et al. MathForm. arXiv:2608.14221
- Xiong et al. MMFormalizer. arXiv:2601.03017
- Peng et al. CriticLean. ACL 2026
- Jin et al. Search-R1. arXiv:2503.09516
- Li et al. ToRL. arXiv:2503.23383
- Zhang et al. TRACE. arXiv:2607.13988
