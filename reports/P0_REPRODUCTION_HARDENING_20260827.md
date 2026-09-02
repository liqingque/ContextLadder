# P0 — 复现核验加固（2026-08-27）

复赛规则规定晋级队须通过**代码复现核验，核查不通过不计成绩**。本阶段不是实验，
不涉及任何 gate，目标只有一个：**把"评审方拿到代码包能不能跑出同一份 prediction.csv"
从声明变成实测**。

## 方法

不在开发仓里"就地跑一次"——那样会被开发仓的既有状态污染。做法是把代码包
`Goai_TorchDragon/` 复制到一个与开发仓无关的干净目录（排除 `outputs/`、`prediction.csv`、
`SUBMISSION_VALIDATION.json`，即评审方运行前不该依赖的产物），按 `input/README.md`
放入官方数据，然后**按《代码与复现说明》第 3 节的命令原样执行**。

## 结果：逐字节可复现

| 核验项 | 结果 |
|---|---|
| 入口退出码 | 0 |
| 重训 + 推理耗时（单张 RTX 3090） | 121.7 s / 124.6 s（两次独立执行） |
| 生成 `prediction.csv` SHA-256 | `a7cff2acd719bd6d7bb09e307eae96a4a81b037e16a667c2d114d113aaaf4d43` |
| 随包 `prediction.csv` SHA-256 | **同一值**，228,979,792 字节 |
| 契约 | `submission_contract: PASS`，`fit_rows: 5920`，`held_out_val_rows: 3038` |
| `scripts/verify_submission.py` | **13/13 PASS** |
| 分析侧五个脚本从包根 `--help` | 全部导入成功 |
| 环境版本 vs `configs/env-spec.yaml` | 逐条一致 |

`verify_submission.py` 的 13 项覆盖：行数 4454、样本 ID 唯一、ID 顺序与测试元数据一致、
首列为 `sample_ID`、蛋白列数 5243、列名与顺序与契约完全一致、全部有限值、输出为 log2(raw)
（均值 20.5856）、无 0.0 占位列、拟合行数等于 train 划分（5920）、验证集 3038 行留出、
入口脚本的 `fit_indices` 限定在 train 划分、入口脚本从不引用 test 蛋白组。

**这份代码包在 12 天后、在一个不同的目录路径下、由完整重训重新生成的预测与已提交文件逐字节相同。**
确定性来自 `run_hcce.py: seed_everything`（`random` / `numpy` / `torch` / `torch.cuda` 四处种子 +
`cudnn.deterministic = True` + `cudnn.benchmark = False`）与 DataLoader 的固定 generator 种子。

## 发现并修掉的四个缺陷

### D1 `requirements_submission.txt` 缺 pyarrow（会让分析侧脚本装完即崩）

包内 6 个脚本读写 parquet，`src/mosaic/contracts.py` 在**模块层** `import pyarrow`，
而依赖清单只列了 torch / numpy / pandas / scikit-learn / joblib / matplotlib / PyYAML。
按该清单 `pip install` 的评审方，跑最终提交入口没问题，但一跑
`evaluate_official_modules.py` / `compound_invariance_check.py` / `gate_w_summary.py` / `dcb_*`
就会因缺 pyarrow 失败。

已补 `pyarrow==17.0.0` 与 `scipy==1.10.1`，并把 `rdkit` 作为**可选**依赖注释说明
（只有 `dcb_loco_harness.py` 的 Tanimoto 对照通道用到，且是函数内延迟导入，不装不影响提交路径）。
文件头的"初赛代码包"也改为"复赛"。

### D2 `代码与复现说明.md` 的文件清单已过期

该文档 §1 列 9 个脚本 + 4 个 config；代码包实际有 15 个脚本、9 个 config、
`external_data/`、9 份报告。README 已更新到复赛版本，这份文档没跟上——
评审方会看到一堆文档里没提的文件。

已重写 §1，按**提交路径 / 备选基线 / 分析证伪侧 / 依赖记录**四类重新组织，
并明确标注哪些脚本**不在最终预测路径上**。

### D3 打包目录与工作区再次出现漂移（同一类缺陷第三次出现）

逐文件比对 `scripts/` 与 `Goai_TorchDragon/scripts/`：`a2b_train_variants.py` 相差 45 行——
工作区有本轮 P3 新增的 `--ema-decay`，代码包没有。也就是说**代码包无法复现 P3 的 EMA 实验**。
`ensemble_eval.py`（P3 重建 3-seed 集成所必需）、`configs/p3_*.json` 与 P3 报告也都不在包内。

已同步 `a2b_train_variants.py`、新增 `ensemble_eval.py`、`configs/p3_gates.json`、
`configs/p3_candidates.json`、`reports/P3_CAPACITY_EMA_BLEND_20260827.md`，
并把这批改动按项目规则回同步到 `submission_package/` 与工作区 `scripts/`。
三个树在代码与文档面上现已完全一致（除各自固有的 `__pycache__`、`input/`、`outputs/`、
`prediction.csv` 与 `PACKAGING.md`）。

**同步之后重跑了一次完整复现**：输出与第一次、与随包文件仍然逐字节相同
（同一 SHA-256），证明 `--ema-decay`（默认 0.0）的加入不改变提交路径的任何行为。
这一步是必须做的——改了训练脚本就必须重新证明提交物不变，而不是假设它不变。

### D4 开发仓根部的过时 `SUBMISSION_VALIDATION.json` 是一个已经引爆过一次的陷阱

`/data/LXM/VC/SUBMISSION_VALIDATION.json` 是 2026-08-11 FiLM 时期的记录，
`fit_rows: 8958`（train_val 全量）。它从未被提交，但字面上直接与合规声明冲突。
2026-08-26 就是因为读了工作区的过时副本，写出一份"提交包泄漏了验证集"的自我举报文档，
随后被对已提交 ZIP 的逐值比对推翻并回滚。

未删除该历史文件（保持记录完整），改为在其旁新增
`SUBMISSION_VALIDATION.README.md`，写明它不是提交物、真源是 `Goai_TorchDragon/`，
并指向 08-26 的 CORRECTION 条目。

## 诚实边界：哪些东西评审方复核不了

这一条写进了《代码与复现说明》新增的第 7 节，也记在这里：

- **可以独立复核**：最终提交路径的完整重训与推理（逐字节）；提交契约与训练边界
  （`verify_submission.py`，无需重跑训练）；评估器、不变性检查、LOCO harness 的**实现**；
  各阶段**冻结判据**（`configs/*_gates.json`，均带冻结时点与 SHA-256）。
- **不能在不重建候选的前提下复核**：`reports/` 里的模块分与 ΔW 数值。这些脚本的输入是
  开发仓 `outputs/` 下的候选预测 parquet，按数据边界不随包分发。评审方要完整复核，
  必须先用包内入口重建对应候选的验证集预测。

**我们不宣称这些数值可以被本包单独复核。** 与其让评审方自己撞上这堵墙，不如写明。

## 未做的事

- 没有在全新 venv 里跑一次 `pip install -r requirements_submission.txt` 的端到端安装验证
  （本机 `tl` 环境的版本已逐条比对一致，但"清单能装出这套环境"这件事本身没有实测）。
  这是本阶段唯一留下的未验证声明。
- 没有做容器化（Dockerfile）。官方目前未要求，且会引入新的未验证声明。
- 没有改动 `prediction.csv`、模型、任何冻结判据或已提交的结论。

## 评审方复核命令

```bash
# 1. 放数据：把官方四个文件按原名放入 input/（见 input/README.md）
# 2. 装依赖
pip install -r requirements_submission.txt      # 或 conda activate tl
# 3. 不重跑训练，直接核对提交物（约 1 分钟）
python scripts/verify_submission.py
# 4. 完整重训与推理（RTX 3090 约 2 分钟），并与随包文件比对
python scripts/make_mask_compound_3seed_submission.py
sha256sum prediction.csv
# 期望：a7cff2acd719bd6d7bb09e307eae96a4a81b037e16a667c2d114d113aaaf4d43
```
