> **已被 `reports/P0_REPRODUCTION_HARDENING_20260830.md` 取代**（P4 之后代码有更新；模型与提交产物未变）。本文记录的核验真实发生且全部通过，保留备查。

# 复现核验（2026-08-29）：复赛口径管线的干净目录实测

本文取代 `reports/P0_REPRODUCTION_HARDENING_20260827.md`。08-27 那次核验真实发生且全部通过，
但它验证的是**初赛口径管线**（训练+推理合体入口、`sample_ID` + 5,243 列产物）。该管线已被
复赛改造取代：入口拆分为 `train.py` / `predict.py`，提交产物改为 `sample_ID` + 4,422 列。

## 方法

不在开发仓就地跑——那会被开发仓既有状态污染。做法是把 `Goai_TorchDragon/` 复制到与开发仓
无关的目录，并**排除评审方运行前不该依赖的一切产物**：

```
--exclude 'runs/'                    # 训练产物（含三个 checkpoint）
--exclude 'prediction.csv'           # 提交产物
--exclude 'prediction_manifest.json' --exclude 'validation_report.json'
--exclude 'outputs/'
```

然后按 `input/README.md` 放入官方数据。**只放三个文件**：train_val 的 metadata 与 proteome、
test 的 metadata。**`WAYB_WAYC_proteome_raw_test.csv` 全程未放入**——如果推理路径存在任何对
测试蛋白组的隐式依赖，这一步就会直接失败。

随后按 README 的四条主命令原样执行。

## 结果：逐字节可复现

| 核验项 | 结果 |
|---|---|
| 命令 1 `build_embeddings.py` | exit 0；如实报告"本方案无需执行"，写出 embedding manifest |
| 命令 2 `train.py`（三个种子） | exit 0，**102.61 s** wall（单张 RTX 3090） |
| 命令 3 `predict.py` | exit 0，**18.92 s** wall |
| 命令 4 `validate_submission.py` | **verdict: PASS**，九项硬检查全过，零项失败 |
| **三个 checkpoint SHA-256** | 与随包 `run_manifest.json` **完全一致**（`b347a477…` / `f378368b…` / `11fbe918…`） |
| **`prediction.csv` SHA-256** | `59f99dc431aa5bd6dc5abb46a5390c64072fda505097cb9523b77198a502b17e` —— **与随包文件同一值** |
| 峰值显存（GPU 0） | **923 MiB** |
| 包根 `pytest tests/` | **12 passed** |
| 分析侧五个脚本从包根 `--help` | 全部导入成功 |
| 环境版本 vs `configs/env-spec.yaml` | 逐条一致（Python 3.8.18 / torch 2.1.0 / numpy 1.24.3 / pandas 2.0.3 / sklearn 1.2.2） |

**测试蛋白组从未放入干净目录，四条命令全部成功。** 这不是声明，是构造性证据：
`predict.py` 里不存在任何蛋白组读取路径（`tests/test_submission_contract.py` 对此有静态断言）。

## 确定性的三重证据

1. **checkpoint 跨重跑相同**：开发机上两次独立训练，三个成员 SHA-256 完全相同。
2. **checkpoint 跨环境相同**：干净目录训练出的三个 checkpoint 与随包文件 SHA-256 相同。
3. **prediction 跨环境相同**：干净目录生成的 `prediction.csv` 与随包文件同一 SHA-256。

固定项在 `run_hcce.py: seed_everything`：`random.seed` / `np.random.seed` / `torch.manual_seed` /
`torch.cuda.manual_seed_all`，以及 `cudnn.deterministic = True`、`cudnn.benchmark = False`；
DataLoader shuffle 用 `torch.Generator().manual_seed(seed)`。

**跨硬件不作保证**：GPU 归约顺序随架构变化，不同显卡上末位可能有差异。上述三条均在同型号
RTX 3090 上取得。该容差已写入 `README.md` 的已知限制与 `configs/final.yaml` 的 `determinism` 段。

## 与改造前的一致性

新拆出的管线**没有改变模型**：在 4,422 个模型预测列上，新产物与初赛口径产物
`max|Δ| = 0.000000e+00`、`corr = 1.00000000`（4,454 × 4,422 全量比较）。两者的差异只在于
初赛口径额外输出 821 个以 train 均值填充的列，而复赛提交说明不再要求它们。

## 复现命令

```bash
cd <干净目录>/pkg
conda activate tl   # 或 pip install -r requirements.txt
python scripts/build_embeddings.py --output artifacts/embeddings
python scripts/train.py --metadata "input/WAYB_WAYC_metadata_train_val(1).csv" \
    --proteome input/WAYB_WAYC_proteome_raw_train_val.csv \
    --config configs/final.yaml --output-dir runs/final
python scripts/predict.py --metadata "input/WAYB_WAYC_metadata_test(1).csv" \
    --run-dir runs/final --output prediction.csv
python scripts/validate_submission.py --prediction prediction.csv --run-dir runs/final
```
