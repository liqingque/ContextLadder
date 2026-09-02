# Competition Dataset Provenance Search

日期：2026-08-11

> **编者按（随初赛提交包附上）**
>
> 本文是团队开发过程中的历史内部记录，随包提交仅为满足官方「须列出全部外部数据来源」的披露要求。检索目的是判断能否为菌株构建**实体特征**（官方规则允许外部公开资源用于实体特征构建），结论是找不到可信的样本级 crosswalk，因此**未采用任何外部资源**；本文列出的所有公开资料均未进入最终提交模型的预测输入，也从未被用于获取或推断测试集真值。下方原文按记录时间保留。

## 结论

当前最可信的判断是：比赛数据很可能来自或截取自“虚拟酵母”项目的数据采集体系，但目前没有证据证明它就是某篇已经公开发布的完整论文数据集，也没有找到可以把比赛内部 strain 标签直接映射到公共 accession 的官方 crosswalk。

## 本地数据证据

- 官方压缩包只有四个文件：train/validation metadata、train/validation proteome、test metadata、test proteome。
- 当前矩阵为 train_val 8,958 行、test 4,454 行、5,243 个蛋白列；metadata 中出现 `WAYB`、`WAYB_rep1`、`WAYB_rep2`、`WAYC` 四个内部 data source。
- 当前 metadata 的 strain 标签为 `BAH`、`BAI`、`CEK`、`CGD`、`CRD`、`DHY210`；没有论文 DOI、BioProject、BioSample、assembly、SRA/ENA 或 PRIDE 字段。
- 蛋白列使用酵母基因名（例如 `AAC1`、`AAD10`），而不是样本级公共蛋白 accession；这只能说明物种/注释体系，不能证明数据来源。

## 公开来源检索结果

### 高相似项目线索

2026 年 Nature 的“Towards the construction of a virtual yeast”描述了虚拟酵母计划；同一项目的公开报道进一步描述了 969 个天然酵母菌株、超过 1.5 万份时间分辨蛋白组，以及碳氮源、温度和化学胁迫等维度。这与本比赛的任务结构高度相似，但公开页面没有给出本比赛 `WAYB/WAYC`、内部 strain 标签与样本 accession 的逐行对应表。

因此，当前结论是 `LIKELY_RELATED_PROJECT_BUT_NOT_PROVEN_SAME_DATASET`，不是 `PUBLIC_DATASET_CONFIRMED`。

### 标签重合但不能直接映射

2025 年 UPR 研究论文中确实出现 `CGD`、`BAH`、`BAI`、`CEK` 等野生背景标签，但该论文研究的是 UPR 基因缺失、WGS 和相关蛋白组，不包含当前比赛的 `WAYB/WAYC` 采集标识、56 种化合物设计或本矩阵的样本级 crosswalk；论文中也没有 `DHY210`。因此只能作为“标签曾被使用过”的旁证，不能作为当前比赛数据的 accession 映射。

### 1002/1011 Yeast Genomes

1002/1011 Yeast Genomes 项目公开了大量酵母基因组、SNP/CNV、ORF presence/absence、距离矩阵和部分表型数据，是后续 genome feature 的潜在资源；但当前尚未找到 `BAH/BAI/CEK/CGD/CRD/DHY210 → 1002 isolate` 的官方对应关系。

## 可执行判断

1. 可以继续使用公开 1002/1011 Yeast Genomes 或 NCBI/ENA genome 资源，但必须先获得当前比赛 strain 标签的准确 crosswalk。
2. 不能把公开报道中的 969 个菌株名单、相似标签或 `BAH→SX3` 候选直接当成当前样本身份。
3. 目前不能证明有可直接下载、与本比赛样本一一对应的公开蛋白目标矩阵；外部蛋白目标数据只能先作为旁路验证/特征资源，不并入主榜监督训练。
4. 最有效的下一步是向官方索取 `internal strain ID → source strain name → accession`，或索取该数据采集项目的 DOI/BioProject/PRIDE 编号。

## 证据链接

- GOAI AI for Research 官方赛道页：<https://www.goaihz.com/en/tracks?track=ai4s>
- Nature 虚拟酵母文章：<https://www.nature.com/articles/s41586-026-10574-9>
- 虚拟酵母项目公开报道（次级来源）：<https://m.thepaper.cn/newsDetail_forward_33510450>
- UPR 论文：<https://static1.squarespace.com/static/53dfd288e4b0c0da377c6bb5/t/68f948243fe8fb065d301481/1761167396308/Genes%2BDev.-2025-Bartolutti-gad.352490.124.pdf>
- 1002 Yeast Genomes：<https://www.nature.com/articles/s41586-018-0030-5>
