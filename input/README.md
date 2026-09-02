# 官方数据放置目录

官方竞赛数据属于非公开资料，不随本代码包分发。评审方或复现方请将组委会邮件发放的四个文件放入本目录（文件名保持原样）：

```text
input/WAYB_WAYC_metadata_train_val(1).csv     # 8,958 行样本元数据
input/WAYB_WAYC_proteome_raw_train_val.csv    # 8,958 × 5,243 原始强度矩阵
input/WAYB_WAYC_metadata_test(1).csv          # 4,454 行测试元数据
input/WAYB_WAYC_proteome_raw_test.csv         # ★ 测试真值：四条主命令全都不读它，可以不放
```

> **★ 那一行不是必需的。** 复现四条主命令只需要前三个文件。测试真值仅供包内唯一的独立
> 审计工具 `scripts/p0_audit_data.py` 使用，该工具不在复现链路上。**不放这个文件，
> README 的四条命令依然全部成功**——这一点在干净目录里实测过
> （`reports/P0_REPRODUCTION_HARDENING_20260901.md`）。包内测试真值路径的全部出现位置
> 及其性质，逐条列在包根的 `TEST_TRUTH_ACCESS.md`。

路径在 `configs/data_paths.yaml` 中配置。放好后即可运行：

```bash
python scripts/train.py --metadata "input/WAYB_WAYC_metadata_train_val(1).csv" \
    --proteome input/WAYB_WAYC_proteome_raw_train_val.csv \
    --config configs/final.yaml --output-dir runs/final
python scripts/predict.py --metadata "input/WAYB_WAYC_metadata_test(1).csv" \
    --run-dir runs/final --output prediction.csv
```
