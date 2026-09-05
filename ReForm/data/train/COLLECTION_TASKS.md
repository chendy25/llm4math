# ReForm SFT 采集任务清单

> 决策（2026-07-27）：造数阶段**不做**评测去重/隔离；数学/物理先采满。  
> 训练时再按 `dataset_name` / `question_id` 拆 train/eval。  
> 目标规模：起步 **1 万** → 主推 **2 万** → 可扩到几万。  
> 配比：**平行 physlib ~20%** + **数学工具轨迹 ~55–60%** + **物理全流程多模态 ~20–25%**。

---

## 0. 前置条件（全部勾上再开大批量）

- [ ] LeanSearch serve 可用（`CORPORA=mathlib,physlib`），`POST /search?corpus=mathlib|physlib` 冒烟通过
- [ ] Kimina / Lean verify 可用（`VERIFY_URL` / `VERIFY_KEY`），Mathlib + Physlib 环境齐
- [ ] Auto + Critic LLM API 可用（TokenHub 或其它；物理 vision 需多模态模型，如 gpt-4o）
- [ ] Web search 可用（`SERPER_API_KEY` 或现网配置）；策略：**optional / corpus-first**
- [ ] 产出根目录约定，例如：  
  `ReForm/data/train/sft_collect/{parallel,math,physics,merged}/`
- [ ] 每条落盘保留：`dataset_name`, `question_id`, `retrieval_domain`, `is_parallel_physlib`, `tools_used`, `success`, `syntax_pass`, `num_rounds`, `num_tool_calls`；物理另保留是否含图

**统一过滤（入库主集）：** 优先 `success=true`；数学尽量 `syntax_pass=true`；物理尽量 `syntax_pass=true`（失败轨迹可另存 `failed/` 不进主配比）。

---

## 1. 桶 A — physlib 平行（~20%，无 tool / 无 critic）

| 项 | 内容 |
|---|---|
| 源 | `LeanSearch-v2/data/corpus/final/physlib_corpus.jsonl`（~12407） |
| 映射 | user ← `informal_name` + `informal_description`；assistant ← Lean `name` + `type`（± module） |
| 筛选 | 优先 `theorem/definition/abbrev/instance`；分层按 module 抽样；过长 type 截断或丢弃 |
| 配额 | 1 万档 **2k** / 2 万档 **4k** / 5 万档 **10k** |
| 输出 | `sft_collect/parallel/physlib_parallel_{N}.jsonl` |
| 标记 | `is_parallel_physlib=true`, `retrieval_domain=physics`（或 `physlib_lib`） |

### 任务

- [ ] **A1** 写转换脚本：corpus → SFT `messages`（单轮，无 `<tool_call>`）
- [ ] **A2** dry-run 100 条，人工抽查 10 条格式
- [ ] **A3** 导出 2k（起步）→ 需要时扩到 4k / 10k
- [ ] **A4** 统计 kind / module 分布，写入 `parallel/stats.json`

---

## 2. 桶 B — 数学工具轨迹（~55–60%）

| 项 | 内容 |
|---|---|
| 源（按序准备种子） | ① NuminaMath-1.5：`olympiads` / `amc_aime` / `aops_forum` / `olympiads_ref`<br>② AI-MO/aops<br>③（可选）Lean-Workbook 仅 NL<br>④ 本地：`aime2025` / `minif2f_testset` / `proofnet_testset` / `putnam_v4`（造数阶段可全用） |
| 管线 | `leansearch` + **可选 web** + Lean verify + critic（`max_rounds=3`） |
| 配额（成功轨迹） | 1 万档 **6k** / 2 万档 **12k** / 5 万档 **~30k** |
| 种子超采 | ×1.1–1.2（数学成功率高） |
| 输出 | `sft_collect/math/sft_math_*.jsonl` |

### 任务

- [ ] **B1** 下载/缓存 NuminaMath-1.5、AoPS；写种子导出脚本 → `seeds/math_numina.jsonl`, `seeds/math_aops.jsonl`
- [ ] **B2** 本地四套 test jsonl 转统一 schema（`informal_statement` + `dataset_name` + `question_id`）→ `seeds/math_local.jsonl`
- [ ] **B3**（可选）Workbook NL → `seeds/math_workbook_nl.jsonl`
- [ ] **B4** 合并种子清单 + 简单去重（可选；**不做评测隔离**）→ `seeds/math_all.jsonl`
- [ ] **B5** Smoke：`max-items=20`，确认 tool / web / verify 字段正常
- [ ] **B6** 小批 200 → 检查成功率、耗时、web 使用率
- [ ] **B7** 中批 2k → 再全量跑到配额（6k / 12k / …）
- [ ] **B8** 失败样本进 `math/failed/`，主集只留成功（+ 尽量 syntax_pass）

### 推荐命令骨架

```bash
python run_collect_sft_data.py \
  --input seeds/math_all.jsonl \
  --output sft_collect/math/sft_math_batch.jsonl \
  --use-leansearch --use-web-search \
  --joint-retrieval \          # 或仅 leansearch；数学路由到 leansearch
  --use-lean-verify \
  --leansearch-url "$LEANSEARCH_URL" \
  --max-rounds 3 --max-tool-calls 3 \
  --auto-model ... --critic-model ...
```

（具体 URL/模型以现网 `poc_phyx/run_*.sh` 为准。）

---

## 3. 桶 C — 物理全流程多模态（~20–25%）

| 项 | 内容 |
|---|---|
| 硬性 | **每条必须有图** + `--include-image` vision；禁止纯文本冒充本桶 |
| 源 | ① PhysUniBench（题题有图）<br>② SeePhys<br>③ PhyX（test_mini / test）<br>④ PhysicsEval **有图子集**（无图不进 C）<br>⑤ 自建有图题（若有） |
| 管线 | vision + `physsearch` + **可选 web** + Lean verify + critic |
| 配额（成功轨迹） | 1 万档 **2k** / 2 万档 **4k** / 5 万档 **~8–10k**（不够则降占比，不阻塞 A/B） |
| 种子超采 | ×2–3（覆盖差、失败多） |
| 输出 | `sft_collect/physics/sft_physics_mm_*.jsonl` |

### 任务

- [ ] **C1** 准备 PhysUniBench（json + `images/`）→ adapter → `seeds/physics_physunibench.jsonl`（含 image 路径/字节）
- [ ] **C2** 准备 SeePhys → `seeds/physics_seephys.jsonl`
- [ ] **C3** 准备 PhyX（HF 或本地）→ `seeds/physics_phyx.jsonl`（复用 `phyx_adapter`）
- [ ] **C4** PhysicsEval：筛有图/diagram 可加载子集 → `seeds/physics_physicseval_mm.jsonl`
- [ ] **C5** 合并 → `seeds/physics_mm_all.jsonl`；断言每条可加载 image
- [ ] **C6** Smoke：`max-items=10`，`--include-image --use-physsearch --use-web-search --use-lean-verify`；检查 SFT `messages` 含 `image_url`
- [ ] **C7** 小批 100 → 看成功率、physsearch/web 占比、vision 耗时
- [ ] **C8** 扩到 2k / 4k 成功轨迹；公开数据不够则记录缺口，用自建补或接受 C 占比下调
- [ ] **C9** 失败进 `physics/failed/`

### 推荐命令骨架

```bash
python run_collect_sft_data.py \
  --input seeds/physics_mm_all.jsonl \
  --output sft_collect/physics/sft_physics_mm_batch.jsonl \
  --include-image \
  --use-physsearch --use-web-search \
  --joint-retrieval \
  --use-lean-verify \
  --physsearch-url "$PHYSSEARCH_URL" \
  --max-rounds 3 --max-tool-calls 3 \
  --auto-model gpt-4o ...   # 必须多模态
```

---

## 4. 合并与配比（训练前可再跑）

- [ ] **M1** 合并 A+B+C 成功集 → `sft_collect/merged/sft_all_raw.jsonl`
- [ ] **M2** 按目标配比抽样（例 2 万：A 4k / B 12k / C 4k）→ `sft_all_balanced_20k.jsonl`
- [ ] **M3** 写出 `manifest.json`：各源条数、成功率、工具使用率、耗时、模型名、serve/verify 版本
- [ ] **M4**（训练时再做）按评测名单排除：miniF2F-test / ProofNet-test / Putnam / PhyX / SeePhys / LeanPhysBench 等

---

## 5. 里程碑时间盒（建议）

| 里程碑 | 交付 | 状态 |
|---|---|---|
| M0 | 前置冒烟（search / verify / vision / web） | [ ] |
| M1 | A 平行 2k + B smoke20 + C smoke10 | [ ] |
| M2 | 合成 **1 万**（A2k+B6k+C2k） | [ ] |
| M3 | 扩到 **2 万**（A4k+B12k+C4k） | [ ] |
| M4 | 视预算扩到几万；C 不足则文档记录实际占比 | [ ] |

---

## 6. 明确不做（造数阶段）

- [x] ~~数学对评测集去重~~（训练时再做）
- [x] ~~物理评测隔离~~（训练时再做）
- [ ] 不把无图物理题算进桶 C
- [ ] 不把平行 physlib 算进「全流程多模态物理」
- [ ] 不强制每题 web（optional / corpus-first）

---

## 7. 责任人 / 环境备忘（填空）

| 项 | 值 |
|---|---|
| LeanSearch URL | |
| VERIFY_URL | |
| Auto model | |
| Critic model | |
| Web 后端 | Serper / 其它 |
| 跑数节点 | |
| 产出 AFS 路径 | `.../ReForm/data/train/sft_collect/` |
