# 测试真值路径：包内全部出现位置及其性质

《虚拟细胞方向材料提交说明》第五节「主办方复现验收流程」第 2 步是**静态合规审查：
搜索测试真值路径、硬编码绝对路径、未声明外部数据、历史预测输入和可疑缓存**。

这一节替审查方把第一项做完：**包内出现 `proteome_test` / `WAYB_WAYC_proteome_raw_test.csv`
的位置一共 16 处（不含本文件），逐条列在下面**，并说明每一处是什么性质。不必逐个人工判断，也不必猜。

复核命令（在包根执行）：

```bash
grep -rn "proteome_raw_test\|proteome_test" --include='*.py' --include='*.yaml' \
     --include='*.md' --include='*.json' . | grep -v __pycache__ | grep -v TEST_TRUTH_ACCESS
```

行号对应随包这一版；改动代码会让行号漂移，上面这条命令随时能重新生成准确位置。

## 结论先说

**四条主命令的链路上没有任何一处读取测试真值。** 唯一真正打开该文件的是
`scripts/p0_audit_data.py`——一个**独立的数据审计工具，不在复现链路上**，
不运行它不影响 README 的任何一条命令。测试真值文件即使完全不放入 `input/`，
四条主命令仍全部成功；这一点在 `reports/P0_REPRODUCTION_HARDENING_20260901.md`
的干净目录复现中是实测过的，不是声明。

## 逐条

### A. 唯一真正读取该文件的地方（1 处脚本）

该脚本另有 2 行（`:6` `:7`）是本节内容的 docstring 说明，不是代码。

| 位置 | 性质 |
|---|---|
| `scripts/p0_audit_data.py:66` | **读取**。P0 数据审计：核对 train/test 的蛋白特征契约是否一致、类别字段取值覆盖情况 |
| `scripts/p0_audit_data.py:82` | 把 test 矩阵形状写入审计报告 |
| `scripts/p0_audit_data.py:305` | 把路径字符串打印进审计报告的"数据路径"一节 |

**它为什么不是违规**：说明第五节列的不接受形式是「**推理脚本**读取
`WAYB_WAYC_proteome_raw_test.csv`、测试对照真值、测试评分结果或由其派生的缓存」。
`p0_audit_data.py` 不是推理脚本，不产出 `prediction.csv`，不在四条主命令里，
其输出（形状与字段覆盖的审计报告）从未回流到蛋白过滤、归一化、词表、超参、早停或后处理。
模型选择的全部依据是冻结验证划分，记录在 `configs/final.yaml` 的 `validation_strategy`。

**为什么保留而不是删掉**：它是数据边界结论的原始证据来源之一。删掉它会让"我们如何确认
train/test 蛋白契约一致"这件事失去可复核的出处——那是拿可追溯性换一个更干净的 grep 结果，
不划算。我们选择把它**标注清楚**，而不是把它藏起来。

### B. 配置中的路径声明（2 行，不构成读取）

| 位置 | 性质 |
|---|---|
| `configs/data_paths.yaml:9` | 路径声明。**只有 A 中的审计脚本会取用这个键**；`train.py` 取 `metadata_train_val` / `proteome_train_val`，`predict.py` 只取 `metadata_test` |
| `configs/data_paths.yaml:3-4` | 注释，说明上面这一条的用途与边界 |

### C. 防护与断言（2 处，作用是**阻止**读取）

| 位置 | 性质 |
|---|---|
| `src/mosaic/contracts.py:27` | `FORBIDDEN_TEST_PROTEOME_BASENAMES` —— 禁读清单，出现该文件名即拦截 |
| `tests/test_submission_contract.py:33` | 静态断言：`predict.py` 的源码里**不允许出现** `proteome_test` 字样。`pytest tests/` 12 passed 覆盖此项 |

### D. 文档中的说明性提及（9 处）

`input/README.md:9`、`方案说明文档.md:279`、`代码与复现说明.md:262`、`提交材料_合并版.md:295 / :722`、
`reports/P0_REPRODUCTION_HARDENING_2026{0829,0830,0901}.md`——
全部是在**声明边界**或记录"干净目录里没放这个文件也跑通了"，没有一处是代码。

## 事后自评指标的时序

报告第 9.3 节有一组测试集 post-hoc 指标（逐样本 PCC、R²、log2 RMSE 等）。
它们由独立脚本在**预测文件冻结并通过契约校验之后**一次性计算，**不回流任何选择**。
`prediction.csv` 的 SHA-256 在那之前就已记入 `prediction_manifest.json`，
时序可由清单的生成时间与哈希独立核对。
