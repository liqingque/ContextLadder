# P2 稠密化合物先验（R2-000）— 2026-08-26

## 结论

P2 在 R2-000 预注册功效门停止，冻结结局为 **UNDERPOWERED**。固定
`sigma_between=0.1571`、37 个聚类、δ 网格
`[0.05, 0.075, 0.10, 0.125, 0.15, 0.175, 0.20]` 下，最保守的 3×σ 情景中，
δ=0.20 的 `P(REP GAP | rho=0)=0.740`，低于 0.80；硬帽内没有任何 δ 达标。
因此没有进入 R2-001 覆盖审计、没有选择建模来源、没有构造 φ_DENSE，也没有运行
缓存、LOCO、Gate-0 或 Gate-1。该结果不是贡献性结论。

## 冻结判据与更正

- Gate-0：`rho >= 0.20`。
- Gate-1：`Delta S1 >= 0.053`，且按冻结官方口径做配对 cluster bootstrap。
- δ：固定上述网格，硬帽 0.20；不因结果选取或移动阈值。
- E1 唯一配置淘汰：condition number >100 或 effective rank < q。
  split-half 稳定性仅作 R2-000 预检诊断，不作为淘汰门；历史谱在 k=1…36
  均未达到原 0.99/35-of-37 规则，因此在看任何新 rho 前登记为不可达。
- oracle 上限若执行时只能使用完整 `k=36` 张成；本轮未执行。
- P2 冻结计划要求 φ_TXT、Tanimoto、random 均为控制且
  `gate1_eligible=false`。配置中的继承性转录值已在任何 R2-001 来源/特征访问前
  以 append-only correction `R2-000-C1` 更正；φ_DENSE 才是唯一预声明的主通道槽位。

## 可复现性

环境：`conda activate tl`，Python 3.8.18；运行目录 `/data/LXM/VC`。

```bash
source /home/lxm/anaconda3/etc/profile.d/conda.sh
conda activate tl
MPLCONFIGDIR=/tmp/goai-mpl python scripts/dcb_power_sim.py
```

产物与哈希：

- [configs/p2_gates.json](/data/LXM/VC/configs/p2_gates.json) — SHA-256
  `14c39c7fb124874d0a8392de8c25e4dc029a2e626314ad3f86d09658a8a84bc4`
- [power_sim.json](/data/LXM/VC/outputs/p2_dense/power_sim/power_sim.json) — SHA-256
  `60525267020880f14ce5851ce37124a565f06e83e000633bb4f0fe32e8592a6c`
- [dcb_power_sim.py](/data/LXM/VC/scripts/dcb_power_sim.py) — SHA-256
  `3d2a5e2ff8bc4bf45a6f8534f5606795527ca66950ab5d14d0cab03ae8a960c9`
- 环境快照：[ENV_SNAPSHOT.txt](/data/LXM/VC/outputs/p2_dense/power_sim/ENV_SNAPSHOT.txt)
- 门冻结记录：[R2-000_GATE_FREEZE_RECORD.json](/data/LXM/VC/outputs/p2_dense/R2-000_GATE_FREEZE_RECORD.json)

本轮 `new_source_accessed=false`、`coverage_audit_run=false`、
`dcb_build_cache_run=false`、`dcb_loco_run=false`、`test_truth_read=false`；未修改
shared evaluator、a2b、P1 或提交产物。

## 外部数据约束

本轮停止发生在覆盖率审计之前，因此没有对公开 HIP/HOP 或 LLM 稠密来源作
Gate-eligible 判断；不存在可宣称的覆盖率结论。下一次若要重开，必须由用户明确
授权新的 R2-000 预注册（不能沿用本次 UNDERPOWERED 之后继续建模）。
