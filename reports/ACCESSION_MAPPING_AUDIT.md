# Accession and Independent-Data Audit

日期：2026-08-11

> **编者按（随初赛提交包附上）**
>
> 本文是团队开发过程中的**历史内部审计记录**，随包提交仅为满足官方「须列出全部外部数据来源」的披露要求，供评审核查我们对外部资源的处理过程。阅读时请注意三点：
>
> 1. **对本次提交的约束性结论**：最终提交模型（`scripts/make_mask_compound_3seed_submission.py`）**不使用任何外部数据**，预测输入只有官方发放的样本元数据字段。本文讨论的 accession 映射、基因组特征与外部菌株数据**均未进入最终模型**。
> 2. 文中出现的 `external_data/`、`outputs/p4_strain/` 等路径属于团队**开发仓内部路径**，不在本代码包内，也不影响本包的复现。
> 3. 文中提到的 `BioCal-MoE` 是同期另一条探索性技术路线的内部代号，**不是**本次提交的模型；本次提交的模型为 HCCE-Proteome（见《方案说明文档》）。
>
> 下方原文按记录时间保留，未作追溯性修改。

> **状态更新（2026-08-13）**：本文记录的是外部公开交叉表下载前的历史审计，
> 不再代表当时的 mapping 状态。1011 Yeast Genomes 论文的公开 supplementary
> table（非竞赛组委会发布）提供了候选的 competition-ID 对齐：BAH=SX3、
> BAI=BJ6、CEK=JCM_2985-4B、CGD=UCD_09-448、CRD=FIMA_3；DHY210 仍仅使用
> NCBI/SGD S288C proxy。该候选映射用于开发期的探索性实验，其合同、文件
> hash 和禁止字段记录在开发仓的 `external_data/` 下（不在本代码包内）。
> **该映射及其派生的任何外部特征均未进入最终提交模型。** 旧的“不猜
> accession”结论对当时缺少交叉表的时间点仍然成立。

## 结论

当前数据包不能支持完整的 genome-aware 新模型实验。现有训练集只有 4 个菌株标签（BAH、CEK、CGD、DHY210），而本地包没有菌株全名、BioSample、assembly、read accession、基因组 FASTA 或注释表。BAI 和 CRD 只出现在 validation/test 角色中，不能把它们的 test/validation 标签当成新增训练数据。

外部检索只得到一个可复核候选：公开菌株表将 BAH 标准化为 SX3；NCBI BioProject 为 SX3 提供 `GCA_003277085.1`、`SAMN07436827` 和 `PRJNA396809`。但当前数据集没有 provenance 链接证明 BAH 就是该 SX3 样本，因此此映射被标记为 `CANDIDATE_DATASET_IDENTITY_UNVERIFIED`，本轮不把它作为训练特征。

详表：开发仓 `outputs/p4_strain/accession_mapping_audit.csv`（不在本代码包内）。

## 独立 plate 审计

官方 train 有 5,920 行、144 个 plate、4 个 data source；WAYB、WAYB_rep1、WAYB_rep2、WAYC 分别覆盖 24、24、24、72 个 plate。官方 validation 与 train 共享全部 144 个 plate ID，因此它不是独立 plate 外部验证集。已有实验用 train-only 的 5-fold plate-group OOF 作为严格的内部 plate 鲁棒性代理，但这不等价于新增独立实验批次。

## 运行门控

- `BAH` 候选映射：只允许进入“待确认候选”审计，不进入 genome feature fitting。
- `BAI/CEK/CGD/CRD/DHY210`：保持 unresolved，不猜测 accession，不下载并拼接不确定的 genome feature。
- genome-aware 新模型 GPU 训练：暂缓，直到拿到可追溯的 mapping table 和至少一个与当前蛋白测量矩阵对齐的独立 strain/plate 数据集。
- 当前可复现实验结论仍是 metadata-only 路线（当时的内部代号 BioCal-MoE，非本次提交模型）；此前 gate-only 三 seed 的 plate cluster CI 均跨 0，不能宣称稳健增益。

## 证据链接

- BAH/SX3 标准化名称与 `ERP014555`：<https://enviromicro-journals.onlinelibrary.wiley.com/doi/full/10.1111/1751-7915.70337>
- SX3 的 NCBI assembly、BioSample 和 BioProject：<https://www.ncbi.nlm.nih.gov/bioproject/PRJNA396809>
- 1002 Yeast Genomes 数据资源与 read archive 说明：<https://www.nature.com/articles/s41586-018-0030-5>

## 下一步输入要求

请提供以下任一形式后再启动 genome-aware GPU 实验：

1. `internal_strain_id -> standardized_name -> BioSample/assembly/accession` 的映射 CSV，并注明来源；或
2. 每个训练菌株对应的 genome FASTA/GFF（文件名或 checksum 能回指内部标签），以及新增独立 strain/plate 的样本 metadata 和蛋白目标矩阵。

拿到数据后，模型将使用“基因组摘要专家 + Concat/FiLM 专家 + group-aware shrinkage gate”，只在 mapping confidence 通过时激活 genome expert，并用 leave-one-strain-out 与 leave-one-plate-group-out、3 seeds、冻结 train-only folds 重新评估。
