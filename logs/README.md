# 运行日志

干净目录复现（与开发仓无关的目录，按 `input/README.md` 放入官方数据后按 README 四条主命令原样执行）
的真实控制台输出，对应复赛提交说明 G 项「运行日志」。

| 文件 | 内容 |
|---|---|
| `01_build_embeddings.log` | 命令 1：如实报告本方案无需外部特征 |
| `02_train.log` | 命令 2：三个种子训练，含 run manifest 全文 |
| `03_predict.log` | 命令 3：逐成员 checkpoint 校验与推理，含 prediction manifest 全文 |
| `04_validate.log` | 命令 4：提交格式与数据边界校验（`verdict: PASS`） |
| `05_pytest.log` | 免数据契约测试（12 passed） |
| `gpu_memory.log` | 全程逐秒 GPU 显存采样，**峰值 923 MiB** |

本轮生成的 `prediction.csv` 与随包文件 SHA-256 相同
（`59f99dc431aa5bd6dc5abb46a5390c64072fda505097cb9523b77198a502b17e`），
**三个 checkpoint 的 SHA-256 亦与 `runs/final/run_manifest.json` 逐个一致**。
训练 97.6 s、推理 18.7 s、峰值显存 923 MiB（单张 RTX 3090）。
