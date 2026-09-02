# P3 — 容量 / EMA / 跨架构混合（2026-08-27）

## 冻结记录

- 判据 [configs/p3_gates.json](/data/LXM/VC/configs/p3_gates.json) — SHA-256 `c5676706b36a00fbcdc24dfadadb62b21fc79bfa599bf3dbcdb9a39d4af0f03d`，在任何 P3 数据进入管线之前冻结，执行期未改动（跑完后复核哈希一致）。
- 执行计划 [refine-logs/EXECUTION_PLAN_P3_20260827.md](/data/LXM/VC/refine-logs/EXECUTION_PLAN_P3_20260827.md) — SHA-256 `8e8c647af6a64926b0cef768c1b9da346024994d54340dd37b81c80c83928570`。
- 候选清单 [configs/p3_candidates.json](/data/LXM/VC/configs/p3_candidates.json) — SHA-256 `be531f00061dc586384b89607963e3128b5aff8e706243ebdf70f6c880f8533f`，评估前冻结。
- Append-only 更正 [refine-logs/P3_APPEND_ONLY_NOTES.md](/data/LXM/VC/refine-logs/P3_APPEND_ONLY_NOTES.md) 条目 P3-C1（Gate-Z 的 ULP 口径，写于评估器运行之前）。
- Gate-W **原样沿用 P1**，未动任何阈值。`+0.010 = 已冻结 S1 效应量地板 0.053 × 官方权重 0.20`。
- 官方评估命令：
  `python scripts/evaluate_official_modules.py --candidates configs/p3_candidates.json --baseline mask_compound_3seed_base --out outputs/p3_official_plate --workers 4 --bootstrap 2000`
- `no_test_truth = true`。test 真值、test 蛋白组、test 元数据全程未读。评估器未被修改。

## 结局

**(ii) INFORMATIVE NEGATIVE，且其"informative"的部分随即被 Gate-Z 抵消。**

十个候选全部 NOT ADOPTED。**ΔW 的最大值是 +0.0050，恰好是采纳地板 +0.010 的一半**；三个互相独立的杠杆家族没有一个能越过该划分自身的检测门槛。唯一两个 CI 排除 0 的模块改善都是 S1，而两者都被判定为化合物零信息。

## 执行记录

| 阶段 | 内容 | 运行数 | 耗时 |
|---|---|---:|---|
| R3A | 容量/训练预算网格（epochs / dim / mask_p 单轴 + COMBO） | 24 次训练 | ~19 min（GPU 0-2） |
| R3B | EMA decay ∈ {0.99, 0.995, 0.999} | 9 次训练 | 8.5 min（GPU 3） |
| R3C | 跨架构混合（无训练） | 5 候选构造 | ~8 min（CPU） |

全部 exit 0，无 NaN、无 OOM、无 FAILED。

### harness 改动与回归核验（执行实验之前完成）

1. `scripts/a2b_train_variants.py` 新增 `--ema-decay`（默认 0，影子副本不回流优化器，与 `--swa-last-n` 互斥）。
   **`--ema-decay 0` 在 seed 20260810 上复现 `rmse = 0.42166375888106966`，与 `outputs/p1_multiseed/seed20260810/metrics.json` 逐位相同（delta = 0.0）。**
2. `scripts/ensemble_eval.py` 改为从 checkpoint 的 `emb.*.weight` 形状推断 embedding 维（原先硬编码 64，会让 dim≠64 静默载错）。
   **3-seed top-3 集成复现 `0.4189243130772146`，与 P1 报告值一致。**

本机管线经实测是确定性的（同配置复跑逐位一致），因此报告中的任何数值差异都是真实差异。

## 官方六模块结果（`official_plate`，val 匹配行 2,806 = seen 139 / S1 1,065 / S2 1,333 / S3 269）

| 候选 | abs PCC | RMSE | FC all | S1 | S2 | S3 FC | DEP | W | ΔW | ΔW_noS2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mask_compound_3seed_base | 0.988598 | 0.420259 | 0.491355 | 0.203387 | 0.407705 | 0.344605 | 0.891936 | 0.5218 | — | — |
| R3A_ep60 | 0.988688 | 0.419291 | 0.498316 | 0.214941 | 0.410855 | 0.345332 | 0.893863 | 0.5267 | +0.0049 | +0.0053 |
| R3A_ep120 | 0.988664 | 0.419949 | **0.500426** | 0.212391 | 0.414187 | 0.340677 | 0.891966 | **0.5268** | **+0.0050** | +0.0046 |
| R3A_mp040 | 0.988605 | 0.420090 | 0.490667 | 0.206567 | 0.406000 | 0.346356 | 0.891659 | 0.5221 | +0.0003 | +0.0008 |
| R3A_combo | 0.988656 | 0.419580 | 0.497012 | 0.216012 | 0.408779 | 0.345616 | 0.892800 | 0.5261 | +0.0043 | +0.0051 |
| R3B_ema_d99 | 0.988672 | 0.418716 | 0.492585 | 0.207877 | 0.409567 | 0.345475 | 0.891742 | 0.5235 | +0.0017 | +0.0016 |
| R3C_C0 | 0.988641 | 0.419479 | 0.494967 | 0.218435 | 0.407705 | 0.344605 | 0.893163 | 0.5258 | +0.0040 | +0.0050 |
| R3C_C1_w25 | 0.988683 | 0.418582 | 0.494278 | 0.213086 | 0.407525 | 0.346336 | 0.892288 | 0.5247 | +0.0028 | +0.0036 |
| R3C_C1_w50（主候选） | 0.988686 | 0.418561 | 0.495023 | 0.216693 | 0.407757 | 0.345860 | 0.892718 | 0.5256 | +0.0038 | +0.0047 |
| R3C_C1_w75 | 0.988672 | 0.418860 | 0.495266 | **0.218547** | 0.407818 | 0.345283 | 0.893230 | 0.5260 | +0.0042 | +0.0052 |
| R3C_C2_rowscoped | 0.988699 | **0.418227** | 0.495442 | 0.216693 | 0.407123 | 0.346708 | 0.892520 | 0.5257 | +0.0038 | +0.0049 |

机器可读版本：`outputs/p3_official_plate/gate_w_table.md` 与 `gate_w_summary.json`。

## 逐候选裁决

判据五条（§0.5）+ 两条否决（§0.6）。完整机器可读版本：`outputs/p3_official_plate/P3_ADJUDICATION.json`。

| 候选 | ΔRMSE | c5 (≤−0.001) | ΔW | c2 (≥+0.010) | ΔS1 | S1 占 ΔW | S1 CI 排除 0 | 化合物不变 | Gate-Z | 裁决 |
|---|---:|:--:|---:|:--:|---:|---:|:--:|:--:|:--:|---|
| R3A_ep60 | −0.000968 | ✗ | +0.0049 | ✗ | +0.011554 | 47% | 否 | 是 | — | NOT ADOPTED |
| R3A_ep120 | −0.000310 | ✗ | +0.0050 | ✗ | +0.009004 | 36% | 否 | 是 | — | NOT ADOPTED |
| R3A_mp040 | −0.000168 | ✗ | +0.0003 | ✗ | +0.003180 | 223% | 否 | 是 | 触发 | NOT ADOPTED |
| R3A_combo | −0.000679 | ✗ | +0.0043 | ✗ | +0.012625 | 59% | **是** | 是 | 触发 | NOT ADOPTED |
| R3B_ema_d99 | −0.001543 | ✓ | +0.0017 | ✗ | +0.004490 | 54% | 否 | 是 | 触发 | NOT ADOPTED |
| R3C_C0 | −0.000780 | ✗ | +0.0040 | ✗ | +0.015048 | 76% | 否 | 是 | 触发 | NOT ADOPTED |
| R3C_C1_w25 | −0.001677 | ✓ | +0.0028 | ✗ | +0.009699 | 68% | **是** | 是 | 触发 | NOT ADOPTED |
| R3C_C1_w50 | −0.001698 | ✓ | +0.0038 | ✗ | +0.013306 | 71% | 否 | 是 | 触发 | NOT ADOPTED |
| R3C_C1_w75 | −0.001399 | ✓ | +0.0042 | ✗ | +0.015161 | 73% | 否 | 是 | 触发 | NOT ADOPTED |
| R3C_C2_rowscoped | −0.002032 | ✓ | +0.0038 | ✗ | +0.013306 | 70% | 否 | 是 | 触发 | NOT ADOPTED |

**没有任何候选通过 Gate-W 的 c2 或 c4。** 提交线不变，提交包未被触碰。

判据可判定性的限制，如实记录：官方评估器的配对 cluster bootstrap 只覆盖 S1（四种粒度）与 S2，**不产出 ΔW 的置信区间，也不产出 absPCC / FC / S3 / DEP 的区间**。因此条件 1 与条件 3 在本轮不可判定。这不影响裁决——条件 2 与条件 4 的点估计失败对每个候选都已是决定性的（与 P1 同）。

## 三条杠杆各自的读数

### R3A 容量与训练预算 — 整条轴大概率是平的

| config | epochs/dim/mask_p | 3-seed 集成 RMSE | Δ |
|---|---|---:|---:|
| ep60 | 60/64/0.25 | 0.419290878 | −0.000968 |
| combo | 60/64/0.40 | 0.419579679 | −0.000679 |
| ep120 | 120/64/0.25 | 0.419948842 | −0.000310 |
| mp040 | 40/64/0.40 | 0.420090395 | −0.000168 |
| base_ref | 40/64/0.25 | 0.420258877 | −0.000000 |
| ep80 | 80/64/0.25 | 0.420353129 | +0.000094 |
| mp015 | 40/64/0.15 | 0.421538029 | +0.001279 |
| dim128 | 40/128/0.25 | 0.423748989 | +0.003490 |
| dim96 | 40/96/0.25 | 0.434773469 | +0.014515 |

三条读法：

1. **epochs 轴非单调**：40 → 0.42026、60 → 0.41929、80 → 0.42035、120 → 0.41995。60 最好，但 80 反弹到比基线还差。四个水平全距 0.0011，与被声称的效应同量级。**非单调 + 效应≈噪声，最合理的解释是这条轴没有真实信号**，ep60 的第一名是集成层面的抽样起伏。
2. **COMBO 反证可加性**：规则先于结果地选出 60/64/0.40，实测 0.419580，**比 ep60 单独更差**。若 epochs 与 mask_p 的增益是真实的，二者本应叠加。
3. **dim 轴明确有害且非单调**：dim96（+0.0145）比 dim128（+0.0035）更差。方向上复现了 2026-08-11 旧口径 probe 的"dim 64 最好"，但非单调说明它不是"越大越差"这么简单。

`base_ref` 由既有 `p1_multiseed` 三个 seed 以相同 `ensemble_eval.py` 调用构造（**无新训练**），复现 0.420258877，与冻结基线 0.420258878 差 1e-9。这顺带确认 `mask_compound_3seed_base` 就是那三个 seed 的 3-seed 集成，每条轴的比较都是同构造对同构造。

**口径提醒**：`ensemble_eval.py` 的 `fc_pcc`（base_ref 0.418227，覆盖率 0.586）**不是官方 FC 模块**（官方 FC_all 基线 0.491355，匹配集合不同），只能在 R3A 内部横向比较，不得当官方读数引用。

### R3B EMA — 比 SWA 强，但仍在同一个带里

| decay | 3-seed 集成 RMSE | Δ | 残留初始化权重 `decay^1880` |
|---|---:|---:|---:|
| 0.99 | 0.418716232 | −0.001543 | 6.2e-9 |
| 0.995 | 0.418768082 | −0.001491 | 8.1e-5 |
| 0.999 | 0.495316281 | +0.075057 | **0.1524** |

同一基线下 P1 的 SWA：N=5 −0.000794、N=10 −0.001282；top-3 seed 选择 −0.001335。
**EMA d99 的 −0.001543 是本项目至今所有单杠杆里最好的 RMSE 改善**，比 SWA N=10 再好 −0.000261。但幅度仍在 −0.001~−0.0015 带内，换算到 ΔW 只有 +0.0017。

**d999 的崩塌是本轮实现的缺陷，不是 EMA 的性质。** 本轮新加的 EMA 影子副本从模型的**初始** `state_dict` 起步，没有 bias correction、没有 warmup；40 epoch × 47 step = 1,880 步后残留在影子里的随机初始化权重是 `0.999^1880 = 0.1524`，即约 15%。三个 seed 一致退化（0.495/0.500/0.497），是构造性欠收敛。
**作用域限制**：d999 测的是"无 bias correction 的长窗口 EMA"，不能读成"长窗口 EMA 无效"。按预注册停止条件，本轮**未**补 bias correction、未加长预算、未改网格。若要重测长窗口，必须作为一次新的预注册。

### R3C 跨架构混合 — 身份核验推翻了 P1 的候选，但结论不变

**C-000 身份核验（实测，非假设）**：每个 parquet 与 `metrics.json` 的一个变体在 1e-9 容差下唯一对应，三个 seed 映射一致。

| 文件 | 变体 | seed42 实测 RMSE |
|---|---|---:|
| `prediction_val_retrieval_baseline.parquet` | `retrieval_baseline` | 0.737219305 |
| `prediction_val_rap_r3.parquet` | `rap_r3` | 0.448807196 |
| `prediction_val_rap_hybrid.parquet` | **`rap_r4_hybrid`（r1-free）** | 0.419562907 |
| `prediction_val_rap_r4.parquet` | `rap_r4_hybrid_r1_mix` | 0.420851306 |

`rap_r4_raw_r1_mix` 在盘上没有 parquet，故 5 个变体对 4 个文件无歧义。
**P1 评估的 RAP-R4 候选是含 R1 凸混合（`r1.weight = 0.30`）的那一版**，按 Gate-R1F 不得作为 P3 分量；r1-free 的 `rap_r4_hybrid` 此前从未过官方评估器，本轮补上（候选 C0）。

**RAP-R4 hybrid 不是独立完整模型，必须写明**：`run_config.yaml` 记录 `stability.fallback_prediction` 指向基线 parquet、`l1_rows = 1222`、`l1_blend = 0.25`。实测它在 3,038 行中只有 **1,222 行（40.2%）**与基线不同，其余 **1,816 行就是基线**（差 ≤ 3.55e-15，而差异行 ≥ 0.71，两组间隔 14 个数量级，计数与阈值无关）。C0 的全部 −0.00078 来自那 40% 的行。

C0 单独（−0.000780）**比分量 A（top-3，0.418924313）还差 +0.000555**，它只有作为混合分量的价值。

**C2_rowscoped 是 RMSE 赢家（−0.002032）但不是预注册主候选**，主候选是 C1_w50。看到数字之后改主候选属于决策准则漂移，未做。C2 照报不采纳。

## 本轮最重要的科学结果：候选数从 8 扩到 11，化合物零信息依旧

对全部 11 个候选（含基线）在官方粒度下做不变性检查（`outputs/p3_official_plate/invariance/compound_invariance.json`）：

| 候选组 | S1 跨化合物行对 | 逐位相同 | max\|Δ\| |
|---|---:|---:|---:|
| 基线 + R3A 四个 + R3B 一个（**新训练的模型**） | 991 | 991 | **0.000e+00** |
| R3C 五个混合候选 | 991 | 990 | 1.907e-06 |

R3C 那一对 `1.9073486328125e-06` **恰好等于 `2^-19`，即 log2 尺度量级 16 上的一个 float32 ULP**；主控独立复核 `C1_w50` 与 `0.5·A + 0.5·B` 的最大差同为该值，dump 前的 float64 构造下为 1.536e-07。两个分量都不含未见化合物信息，凸组合造不出任一分量都没有的信息。按 append-only 条目 P3-C1 的口径（≤1 ULP 视为不变，**收紧**而非放宽否决），全部 11 个候选判为化合物不变。

**这把此前的结论从"8 个候选"推进到"11 个候选"，且关键在于新增的五个里有四个是本轮从头训练的模型**（不同 epoch 预算、不同嵌入维、不同掩码率、加权平均），而不只是后处理混合。它们的 S1 在 0.2034 → 0.2185 之间移动了最多 +0.0152，**而携带的化合物信息恰好为零**。

这正是 `reports/FUSAI_PROPOSAL_DRAFT_20260826.md` §一/§二 那条判断的直接加强证据：S1 上的分差完全由共享上下文响应与对照锚定的预测质量决定，与"化合物建模能力"无关。本轮还给出了它的推论形式——**十个候选的 ΔS1 与 ΔW 高度同向，而 S1 平均贡献了 ΔW 的 47%~76%**，即当前所有优化侧改动"抬高 20% 权重模块"的机制都是同一个：把共享分量预测得更准。

## 与 P1 的一致性：单看 RMSE 仍然会错排

RMSE 最优的是 C2_rowscoped（0.418227），但 ΔW 最优的是 ep120（+0.0050），而 ep120 的 RMSE 只改善 −0.00031（排名倒数第三）。ep120 的 FC_all 是全场最高的 0.500426（+0.0091），S3 FC 却是全场最低的 0.340677（−0.0039）。
**M1 在 2026-08-16 得到的"只看单一 RMSE 会错排相关性模块"在 P3 上再次复现。** 任何只报 RMSE 的选择都会选错。

## 本轮暴露的 harness 债（记录，不在本轮修）

1. **`scripts/compound_invariance_check.py` 的逐位判定对混合候选过严**：由多个 float32 parquet 混合而成的候选会因 dump 精度被报成 `compound_invariant: false`。下次在混合候选上使用时，应在脚本层加显式 ULP 容差参数并预注册其取值。
2. **`scripts/a2b_train_variants.py` 的 EMA 缺 bias correction**：影子从初始权重起步，长窗口 decay 在短预算下必然欠收敛。若重开长窗口 EMA，须先补 bias correction 并作为新预注册。
3. **官方评估器不产出 ΔW 的置信区间**，也不产出除 S1/S2 外各模块的区间，导致 Gate-W 的条件 1 与条件 3 长期不可判定。若未来仍以 ΔW 为采纳量，应补上 ΔW 层面的配对 bootstrap。

## 结论与建议

三个互相独立的优化侧杠杆家族，十个候选，**ΔW 上限 +0.0050，恰为采纳地板的一半**。这不是"差一点"，而是量化了一件事：**当前提交线在优化侧的可实现幅度，整体低于该划分本身的检测能力**。加上此前已封闭的家族（逐蛋白后处理 4 次不可迁移、外部数据 3 次证伪、稀疏机制先验数学上关闭、P2 稠密先验功效不足），性能侧可动的空间已经很薄。

- **提交线维持 `mask_compound_3seed_base` 不变。**
- **不下调任何阈值**，不把 C2 提为主候选，不为 ep60 差 3e-5 开口子。
- 剩余时间的边际收益在 55% 的非性能分（科学意义 30% + 创新性 20% + 开源 5%）与复现核验加固上；本轮新增的"11 个候选化合物零信息、其中四个是新训练模型"是可直接写进方案说明文档 §一/§二 的增量证据。

## 可复现性

```bash
source /home/lxm/anaconda3/etc/profile.d/conda.sh && conda activate tl
export MPLCONFIGDIR=/tmp/goai-mpl
# R3A / R3B 的逐条训练命令见 outputs/p3_capacity/GRID_SUMMARY.json 与 outputs/p3_ema/EMA_SUMMARY.json
python scripts/evaluate_official_modules.py --candidates configs/p3_candidates.json \
  --baseline mask_compound_3seed_base --out outputs/p3_official_plate --workers 4 --bootstrap 2000
python scripts/gate_w_summary.py --metrics outputs/p3_official_plate/module_metrics.json \
  --baseline mask_compound_3seed_base --granularity official_plate --out outputs/p3_official_plate
python scripts/compound_invariance_check.py --candidates configs/p3_candidates.json \
  --out outputs/p3_official_plate/invariance --granularity official_plate
```

产物：`outputs/p3_capacity/`、`outputs/p3_ema/`、`outputs/p3_blend/`、`outputs/p3_official_plate/`
（`module_metrics.json`、`gate_w_summary.json`、`gate_w_table.md`、`P3_ADJUDICATION.json`、`invariance/`）。

本轮 `test_truth_read = false`；未修改评估器、未修改任何已冻结 gate 配置、未修改 `Goai_TorchDragon/` 与 `submission_package/`。
