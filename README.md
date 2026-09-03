# 上下文阶梯 ContextLadder

<p align="center">
<img src="对名含义.png" alt="TorchDragon · 以智能之火，照亮未知" width="100%"/>
</p>

<p align="center"><strong>TorchDragon — Illuminate the Unknown.<br>烛龙——以智能之火，照亮未知。</strong></p>

**GOAI 赛道三 · 虚拟细胞方向 · 复赛提交包**
作品全称：上下文阶梯——面向未见扰动与未见菌株的虚拟酵母蛋白组预测
技术代号：HCCE-Proteome ｜ 团队：TorchDragon

在本项目检查的候选集合、冻结验证划分与既定评估器下，S1 场景内的预测未体现可辨别的化合物特异差异。据此我们把「如何优雅地不知道」当作一等设计目标——让模型对实验条件的依赖组织成可以逐级收缩的层级，宁可退到更粗的一级，也不在没有证据的地方假装知道。（结论范围限于本项目候选与本次数据划分，不外推为所有可能模型。）

本目录面向复赛（2026-09-03 截止）的「最终代码 + 完整实验结果」与代码复现审核，由 2026-08-15 的初赛提交包演进而来。不包含决赛海报或路演文件。


> ## 关于本仓库
>
> 这里是 GOAI 赛道三 · 虚拟细胞方向复赛作品 **上下文阶梯 ContextLadder** 的公开代码仓库，
> 内容与提交给组委会的代码材料包同源。与提交包相比只有三处**刻意的**差异：
>
> | 项 | 提交包 | 本仓库 | 原因 |
> |---|---|---|---|
> | `prediction.csv`（193 MB） | 有 | **无** | 超过 GitHub 单文件 100 MB 上限。其 SHA-256 为 `59f99dc431aa5bd6dc5abb46a5390c64072fda505097cb9523b77198a502b17e`，用下面命令 3 可在本地完整重建并逐字节复现 |
> | 负责人手机号 | 有（README 首页与 manifest） | **无** | 不在公开仓库披露个人联系方式 |
> | `PACKAGING.md` | 有 | 无 | 只与打包流程有关，对复现无用 |
>
> 其余全部一致：源码、`configs/final.yaml`、三个成员 checkpoint（各 28.6 MB，随仓库分发）、
> 四份说明文档与合并版、`reports/`、`tests/`、外部数据披露、清单四件套。
> **官方竞赛数据属于非公开资料，不在本仓库，也不会以任何派生形式发布**；按 `input/README.md`
> 自行放置后即可运行下面的四条命令。
>
> 许可 **CC BY-NC 4.0**（见 `LICENSE` 与 `LICENSES/`）。



## 提交信息

| 项 | 值 |
|---|---|
| 作品标识 | 组委会未下发作品编号，本作品以**队伍名 + 作品名称**标识；提交文件名按组委会给定的 `AI4R_AIVC_队伍名_作品名称_{代码材料,非代码材料}.zip` |
| 作品名称 | 上下文阶梯 ContextLadder |
| 团队 | TorchDragon |
| 最终模型 | `ContextLadder_HCCE_mask_compound_3seed` |
| **prediction.csv SHA-256** | `59f99dc431aa5bd6dc5abb46a5390c64072fda505097cb9523b77198a502b17e` |
| prediction 规格 | 4,454 行 × (`sample_ID` + **4,422** 蛋白列)，`prediction_scale: log2` |
| 配置 hash | `5a0503e25f411f7754709d14ce586f2e275be17cb0065278c3f3899cf36f8f6b`（= `configs/final.yaml` 的 SHA-256，与 `run_manifest.json` / `prediction_manifest.json` / `REPRODUCIBILITY_MANIFEST.json` 四处一致） |
| 代码版本 | 见 `REPRODUCIBILITY_MANIFEST.json` 的 `code_entry_points`（六个入口逐个 SHA-256）。本作品不以 git release/tag 冻结版本——代码包以**逐文件 SHA-256** 冻结，粒度更细且不依赖仓库可达性；任何一个入口被改动都会与本清单不符。 |
| 负责人 | 李晓蒙（方法与实现）、徐逸飞（生物建模） |
| 负责人联系方式 | 随赛事提交材料提供，未在公开仓库披露 |

**已知限制**（完整列表见 `REPRODUCIBILITY_MANIFEST.json`）：

1. 复赛提交说明写明标准建模空间为 4,422 个蛋白，按其规则（train 划分 5,920 行上缺失率 < 0.80）
   实算**恰好得到 4,422**，口径一致。未下发的是 feature contract **文件本身**，因此列顺序取自官方
   `train_val` 矩阵的原始列序（剔除被删列后保持相对顺序，首列 `1-Oct`、末列 `ZWF1`）；列表与
   SHA-256 落盘于 `runs/final/artifacts/protein_list.txt` 供核对。
   （附注：初赛期一份讲解材料提到 4,232，该数字已被复赛说明取代；我们逐一测试了阈值、分母、
   缺失定义与 QC 处理等变体，均无法从发放数据还原出 4,232。反过来，含 QC 行统计得 4,422、
   去掉 QC 行得 4,428——官方的 4,422 确认了前者。）
2. **数值容差**：随机种子、cuDNN deterministic 与 `benchmark=False` 均已固定，**同硬件重跑逐字节一致**
   （已实测两次）；但 GPU 归约不保证跨硬件位确定，不同显卡上末位可能有差异。
3. 20% 权重的上下文均值残差模块上，本模型不提供化合物特异信号——这是实测结论而非假设，
   见《方案说明文档》第四、七节。

## 名字的含义

<p align="center">
<img src="队徽.png" alt="TorchDragon 队徽" width="180"/>
</p>

**TorchDragon — Illuminate the Unknown.**
**烛龙——以智能之火，照亮未知。**

TorchDragon 源于中国神话「烛龙」。传说烛龙以目照彻幽暗，而在人工智能中，我们希望以算法为目、以计算为炬，从数据中洞察规律。Torch 同时呼应 PyTorch，Dragon 则承载中国神话意象。TorchDragon 因此代表「以智能之火，照亮未知」。

「阶梯」指本方案的核心设计原则：把模型对实验条件的依赖组织成一道**可以逐级收缩的层级**。测量上下文中，板号被建模为仪器与来源之上的**分层收缩残差**（0.25 缩放 ＋ L2 收缩）；化学扰动上，训练时随机掩码 25% 的化合物 token，让模型学出一个**训练过的共享未知态**。两处是同一原则的两次实例化——**宁可退到更粗的一级，也不要在没有证据的地方假装知道**。这正对应赛题的四个泛化场景（未见化合物、未见菌株、双重未知、时间插值）。

> 措辞说明：板号分支是**收缩**而非严格归零，未见板的板级残差是否应当严格关闭，正由 E1（`configs/e1_unknown_fallback.yaml`）与 E2 的分组伪 OOD 协议验证中；在该实验给出结论前，本文不声称「未见板自动、严格退回上一级」。

## 提交内容

本目录整体打包为**单个 zip** 提交，官方要求的提交物（方案说明文档 + 训练与推理源代码 + 复现说明 + 外部数据来源）全部在内：

| 提交物 | 对应文件 |
|---|---|
| 方案说明文档 | `方案说明文档.md` |
| 技术路线概述 | `技术路线概述.md` |
| 代码与复现说明 | `代码与复现说明.md` |
| 数据与开源计划说明 | `数据与开源计划说明.md` |
| 以上四份的单文件合并版 | `提交材料_合并版.md`（评审如只看一份文档，看这份即可） |
| 训练与推理源代码 | `scripts/`、`src/`、`configs/`（清单见下） |
| 预测结果 | `prediction.csv`（4,454 行 × `sample_ID` + 4,422 蛋白列，log2） |
| 训练产物与权重 | `runs/final/`（三个成员 checkpoint + 预处理契约 + run manifest） |
| 清单与校验 | `REPRODUCIBILITY_MANIFEST.json`、`prediction_manifest.json`、`validation_report.json` |
| 外部数据来源披露 | `external_data/source_manifest.json`、`entity_mapping.csv`、`RAW_SOURCES.md`、`reports/` 两份报告 |
| 许可 | `LICENSE`、`LICENSES/` |
| 测试 | `tests/`（免数据契约测试，`pytest tests/` 应为 12 passed） |
| 静态合规审查辅助 | `TEST_TRUTH_ACCESS.md`（包内测试真值路径的全部出现位置及其性质，逐条列出） |

## 源代码清单

```text
scripts/build_embeddings.py                     # 命令 1：外部特征构建（本方案无需执行）
scripts/train.py                                # 命令 2：从头训练，落盘三个成员 checkpoint
scripts/predict.py                              # 命令 3：冻结模型推理，生成 prediction.csv
scripts/validate_submission.py                  # 命令 4：提交格式与数据边界校验
scripts/a2b_train_variants.py                   # A2 掩码训练变体
scripts/run_hcce.py                             # HCCE-Proteome 基础模型与训练流程
scripts/make_mask_compound_3seed_submission.py  # 初赛历史入口（合体，产出 5,243 列，口径已弃用）
scripts/make_hcce_submission.py                 # 历史 HCCE 入口（5,243 列，口径已弃用）
scripts/make_submission.py                      # 历史 FiLM 入口（5,243 列，口径已弃用）
scripts/run_p5_interactions.py
scripts/run_baselines.py
scripts/p0_audit_data.py
scripts/evaluate_official_modules.py            # 官方口径六模块评估器（对照池 / 分层 / 置换 null / 配对 bootstrap）
scripts/compound_invariance_check.py            # 化合物不变性恒等式检查
scripts/gate_w_summary.py                       # Gate-W 加权分与 ΔW 汇总
scripts/dcb_build_cache.py                      # 嵌套 LOCO 共享缓存构建
scripts/dcb_loco_harness.py                     # 37 折嵌套无泄漏 LOCO harness
scripts/build_entity_mapping.py                 # 生成 external_data/entity_mapping.csv
scripts/build_source_manifest.py                # 生成 external_data/source_manifest.json
scripts/build_reproducibility_manifest.py       # 生成 REPRODUCIBILITY_MANIFEST.json
src/
configs/data_paths.yaml
configs/field_mapping.yaml
configs/protein_feature_contract.json
configs/env-spec.yaml
configs/final.yaml                              # 冻结的最终配置
configs/dcb_gates.json                          # 冻结判据（DCB-40）
configs/p05_gates.json                          # 冻结判据（P0.5 Gate-W）
configs/p1_gates.json                           # 冻结判据（P1 Gate-W）
configs/m1_candidates.json                      # 冻结候选清单
configs/p05_candidates.json                     # 冻结候选清单
external_data/RAW_SOURCES.md                    # 外部资源来源/版本/覆盖率/弃权规则
external_data/processed/                        # 分析侧派生外部特征（不进入预测模型）
reports/DATA_PROVENANCE_SEARCH_20260811.md      # 外部资源检索与来源/版本披露
reports/ACCESSION_MAPPING_AUDIT.md              # strain_id → accession 映射审计
runs/final/                                     # 训练产物：checkpoints/ + artifacts/ + run_manifest.json
tests/                                          # 免数据契约测试
LICENSES/                                       # 项目许可 + 第三方资源许可
REPRODUCIBILITY_MANIFEST.json                   # 总清单（命令/配置/权重/预测/边界/已知限制）
TEST_TRUTH_ACCESS.md                            # 测试真值路径的全部出现位置与性质（对应验收流程第 2 步）
prediction_manifest.json / validation_report.json
requirements.txt
requirements_submission.txt
requirements_tl_snapshot.txt
```

> 官方推荐结构把 `checkpoints/` 与 `artifacts/` 画在顶层；本包按「语义等价」把它们放在
> `runs/final/` 下——那正是 `train.py --output-dir runs/final` 跑出来的结构，不额外维护第二份副本。

官方原始数据不随公开代码包重新分发。数据放置和授权边界见 `数据与开源计划说明.md`。

## 当前方案状态

- 最终候选模型：`HCCE-Proteome + A2 化合物掩码训练`（传统 FiLM 专家 + HCCE FiLM 专家，固定 50/50 融合），提交输出为三 seed（20260810、3407、42）等权集成。
- 最终入口：`scripts/train.py`（训练）+ `scripts/predict.py`（推理），两者分离；推理脚本不含任何蛋白组读取路径。
- 训练轮数：40；embedding 维度：64；化合物掩码率：0.25。
- **最终训练策略**：在冻结验证划分上完成模型选型（结构、轮数、掩码率、种子集合）后，按本包 `configs/final.yaml` 的冻结配置**仅在 `split_final == "train"` 的 5,920 行上从头重训**。重训不借用任何验证或测试标签；最终 `prediction.csv` 由这一次重训的三个成员按固定权重组合生成，组合代码即 `scripts/predict.py`。
- **训练范围：仅 `split_final == "train"` 的 5,920 行**；验证划分 3,038 行不参与训练，也不用于估计任何统计量，符合官方「数据使用约束与最终评测」条款。
- 蛋白目标：train-only 缺失率 < 0.8 的 **4,422** 个蛋白，即复赛提交说明规定的标准建模空间；提交产物只含这 4,422 列。（初赛口径要求覆盖 5,243 列、其余 821 列以 train 均值填充，该步骤在复赛口径下不再出现。）
- 目标空间：`log2(raw)`，值域 10.85–34.50。
- 最终候选模型不使用任何外部数据，也不使用未确认的 `strain_id → accession` 映射。
- `prediction.csv` 为最终产物，`validate_submission.py` 判 `PASS`（九项硬检查全过）。
- 三个成员 checkpoint 随包分发于 `runs/final/checkpoints/`，SHA-256 入 manifest；`predict.py` 加载前逐个校验。
- **实测资源（单张 RTX 3090）**：训练 97.6~107.7 s（三个种子）、推理 16.2~18.9 s、**峰值显存 923 MiB**；磁盘约 280 MB（其中 prediction.csv 193 MB、三个 checkpoint 共 86 MB）。
- 训练不使用早停：轮数固定为 40，由冻结验证集在模型选择阶段一次性选定后冻结，因此复现运行不会因停在不同 epoch 而产生分歧。
- 不提供断点续训：单成员训练约 30~40 s、三个成员全程约 100 s，失败恢复方式是直接重跑训练命令；`train.py` 每次运行完整重写 `runs/final/` 下的 checkpoint 与 artifacts，不读取也不依赖任何中间状态，重跑无残留风险。

复核提交产物（无需重跑训练）：

```bash
python scripts/validate_submission.py --prediction prediction.csv --run-dir runs/final
```

## 运行位置

本代码包为自包含目录：解压后，代码包根目录即项目根目录。所有路径均在包内使用相对路径（`configs/data_paths.yaml` 中的输入文件指向包内的 `input/` 目录，官方原始数据按 `数据与开源计划说明.md` 的授权边界由参赛队自行放置，不随代码包分发）。运行命令：

```bash
cd <代码包根目录>
conda activate tl   # 或 pip install -r requirements.txt；环境配置见 configs/env-spec.yaml

# 1) 外部特征构建 —— 本方案不使用任何外部特征，无需执行
python scripts/build_embeddings.py --output artifacts/embeddings

# 2) 从头训练最终模型（约 100 秒）
python scripts/train.py \
  --metadata "input/WAYB_WAYC_metadata_train_val(1).csv" \
  --proteome input/WAYB_WAYC_proteome_raw_train_val.csv \
  --config configs/final.yaml --output-dir runs/final

# 3) 冻结模型推理，生成 prediction.csv（约 16 秒）
python scripts/predict.py \
  --metadata "input/WAYB_WAYC_metadata_test(1).csv" \
  --run-dir runs/final --output prediction.csv

# 4) 校验提交格式与数据边界
python scripts/validate_submission.py --prediction prediction.csv --run-dir runs/final
```

不带 `--metadata` / `--proteome` 时按 `configs/data_paths.yaml` 取默认路径，效果相同。
详细说明、产物清单与复现核验记录见《代码与复现说明》。
