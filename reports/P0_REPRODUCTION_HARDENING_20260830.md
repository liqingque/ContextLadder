# 复现核验（2026-08-30）：P4 后的复赛口径管线

> **已被取代**：见 `reports/P0_REPRODUCTION_HARDENING_20260901.md`。本文验证的是 `configs/final.yaml`
> 补齐规范 IV.4.1 必填项之前的产物；那次订正改变了配置哈希与三个 checkpoint 哈希，因此本文的
> checkpoint 校验值已不适用于当前提交物。`prediction.csv` 的 SHA-256 两轮相同。保留不删——它真实验证过当时的管线。


本文取代 `reports/P0_REPRODUCTION_HARDENING_20260829.md`。08-29 那次核验真实发生且全部通过，
但它验证的是 P4 之前的代码；此后 `scripts/train.py` 增加了 `ema_decay` 透传、
`configs/final.yaml` 增加了 `rejected_alternatives` 记录。**模型与提交产物未变。**

## 方法

把 `Goai_TorchDragon/` 复制到与开发仓无关的目录，排除 `runs/`、`prediction.csv`、各 manifest 与 `logs/`，
按 `input/README.md` 放入官方数据——**只放 train_val 的 metadata/proteome 与 test 的 metadata；
`WAYB_WAYC_proteome_raw_test.csv` 全程未放入**。然后按 README 四条主命令原样执行。

## 结果

| 核验项 | 结果 |
|---|---|
| 命令 1 `build_embeddings.py` | exit 0；如实报告本方案无需外部特征 |
| 命令 2 `train.py`（三个种子，40 轮） | exit 0，**97.57 s** |
| 命令 3 `predict.py` | exit 0，**18.68 s** |
| 命令 4 `validate_submission.py` | **PASS**，九项硬检查全过 |
| 峰值显存（GPU 0，逐秒采样） | **923 MiB** |
| **`prediction.csv` SHA-256** | `59f99dc4…2b17e`，**与随包文件同一值** |
| **三个 checkpoint SHA-256** | **逐个与随包 `run_manifest.json` 一致** |
| 包根 `pytest tests/` | **12 passed** |
| 分析侧五个脚本从包根 `--help` | 全部导入成功 |

**测试蛋白组从未放入，四条命令仍全部成功**——`predict.py` 中不存在任何蛋白组读取路径，
`tests/test_submission_contract.py` 对此有静态断言。

## 本轮修掉的一处同步缺口

首次复现时 checkpoint SHA-256 与随包不一致。排查发现**包内 `scripts/train.py` 是旧版**——
`ema_decay` 透传只改了开发仓副本，未同步进代码包。由于当前配置 `ema_decay: 0.0`，两版行为相同，
`prediction.csv` 仍逐字节一致，所以这个缺口不会被产物比对发现，只会被 checkpoint 字段比对发现。

同步后重跑，prediction 与三个 checkpoint **全部一致**。顺带做了全量 `scripts/` 与 `configs/` 的
逐文件比对：其余脚本一致；`data_paths.yaml` 与 `env-spec.yaml` 的差异经核对是**有意的**
（包内版去掉开发仓专属引用、GPU 数按复现场景写 1）。

**教训**：产物级比对（prediction SHA-256）不足以发现代码同步缺口——当改动在当前配置下是行为等价的时候，
产物会一致而代码不同。checkpoint 内嵌的配置字段是更敏感的探针。

## 确定性

固定项在 `run_hcce.py: seed_everything`：`random.seed` / `np.random.seed` / `torch.manual_seed` /
`torch.cuda.manual_seed_all`，以及 `cudnn.deterministic = True`、`cudnn.benchmark = False`；
DataLoader shuffle 用 `torch.Generator().manual_seed(seed)`。
**跨硬件不作保证**：上述一致性均在同型号 RTX 3090 上取得。
