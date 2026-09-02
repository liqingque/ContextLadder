# P0 — 复现核验（2026-09-01）：按方向说明原文重做，并补上唯一未验证的声明

> 本文取代 `P0_REPRODUCTION_HARDENING_20260830.md`。08-30 版验证的是 `configs/final.yaml`
> 订正之前的产物；本轮因补齐规范第 IV.4.1 条明列的配置项而重跑了全链，checkpoint 哈希随之变化，
> 因此必须重新验证，不能沿用旧记录。旧版保留不删——它真实验证过当时的管线。

## 为什么重跑

《虚拟细胞方向材料提交说明》第 IV.4.1 条要求最终配置写明网络维度、层数、heads、dropout、
batch size、学习率调度、梯度裁剪、断点续训与日志恢复方式。原 `final.yaml` 缺这一批。补齐后
配置文件的 SHA-256 改变，而**配置 hash 内嵌在每个 checkpoint 里**，所以三个权重全部变了。

**关键验证：`prediction.csv` 的 SHA-256 完全没变**（`59f99dc4…2b17e`）。新增的四段是纯文档，
训练代码不读取它们，模型行为一字未改。这一点是本轮唯一需要证明的事，已经证明。

## 干净目录复现

做法与前几轮一致：把代码包复制到与开发仓无关的目录，**排除 `runs/`、`prediction.csv` 与三份清单**
（评审方运行前不该依赖的产物），只放入 train_val 的 metadata / proteome 与 test 的 **metadata**
（**测试蛋白组从未放入**），按 README 四条主命令原样执行。

| 核验项 | 结果 |
|---|---|
| 命令 2 `train.py`（三个种子，40 轮） | exit 0，**97.97 s** |
| 命令 3 `predict.py` | exit 0，**17.45 s** |
| 命令 4 `validate_submission.py` | **verdict PASS**，`failed_checks: []` |
| **`prediction.csv` SHA-256** | `59f99dc4…2b17e`，**与随包文件逐字节相同** |
| **三个 checkpoint SHA-256** | `4e56da3c…` / `3faed25d…` / `00bdec0b…`，**三个全部与随包一致** |
| 测试蛋白组 | 目录中不存在，四条命令仍全部成功 |

## 补上：全新 venv 的端到端安装验证

这是 08-27 以来一直挂着的**唯一未验证声明**——"按 `requirements_submission.txt` 能装出这套环境"
此前只做过版本逐条比对，从未实测。规范的验收流程第 3 步明确写着「在干净环境安装依赖，
执行单元测试或 smoke test」，所以这一项不能再留着。

```bash
python -m venv venv && venv/bin/pip install -r requirements_submission.txt
```

| 核验项 | 结果 |
|---|---|
| 全新 venv（Python 3.8.18）安装 | **exit 0** |
| 实际装出的关键版本 | torch 2.1.0、numpy 1.24.3、pandas 2.0.3、scikit-learn 1.2.2、pyarrow 17.0.0、scipy 1.10.1 —— 与 `configs/env-spec.yaml` 一致 |
| `pytest tests/`（新 venv） | **12 passed** |
| 五个分析侧脚本从包根 `--help` | 全部导入成功 |

### 顺带修掉的一个缺陷：清单里没有 pytest

新 venv 里跑 `pytest tests/` 报 `No module named pytest`——README 让评审方跑这条命令，
依赖清单却没列它。按我们自己的说明操作会直接失败。已补 `pytest==7.4.4` 并注明它只服务于
验收流程第 3 步，不装不影响四条主命令。

## 本轮修掉的三个不一致（详见 `FUSAI_SUBMISSION_REQUIREMENTS_20260827.md` 附录）

1. **README 的配置 hash 停在旧值**，与其余四处不符——规范把"文档与实际不同版本"列为不接受的提交形式。
2. **`prediction_manifest.json` 指向包内不存在的路径**（`runs/final/prediction.csv`）。
3. **`runs/final/` 下有两份重复清单**，干净复现根本不会产生它们。已移出，现在该目录恰好等于
   `train.py --output-dir runs/final` 的产出。

第 1 条的根因是耦合没被写下来：`final.yaml` 的哈希同时出现在 README、三份清单和每个 checkpoint 里。
已在 `PACKAGING.md` 新增「一处必须一起改的耦合」一节，把这条规则固化。

## 诚实边界（不变）

- **可以独立复核**：提交路径的完整重训与推理（逐字节）、提交契约与训练边界、评估器与不变性检查的
  实现、各阶段冻结判据。
- **不能在不重建候选的前提下复核**：`reports/` 里的模块分与 ΔW 数值——它们的输入是开发仓 `outputs/`
  下的候选预测，按数据边界不随包分发。我们不宣称这些数值可由本包单独复核。
- GPU 归约不保证跨硬件位确定，上述一致性均在同型号 RTX 3090 上取得。
