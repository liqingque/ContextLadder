# 第三方资源许可

本代码包自身以 CC BY-NC 4.0 发布（见 `PROJECT-CC-BY-NC-4.0.txt`）。

**最终模型不使用任何第三方数据或权重**——它只消费官方发放的样本元数据字段。
下列资源仅用于《方案说明文档》第七节的证伪实验，其派生产物随包分发以便结论可被复核，
逐条来源、版本与校验和见 `../external_data/source_manifest.json`。

| 资源 | 用途 | 许可 |
|---|---|---|
| 1011 Yeast Genomes（Peter et al., Nature 2018） | 菌株基因组特征证伪 | 期刊开放获取条款，保留出处署名 |
| NCBI RefSeq GCF_000146045.2 (R64) | 蛋白序列查找（ESM2 实验） | NCBI 公共数据 / 署名 |
| PubChem PUG REST | 化合物结构证伪 | PubChem 公共数据条款 |
| NeuML/pubmedbert-base-embeddings | 机理文本嵌入通道 | 开放权重，依模型卡条款 |
| 结构化机理先验（本项目生成） | 稀疏先验证伪 | 随本包发布 |

**官方竞赛数据为非公开资料，不随本包再分发。** 运行依赖的开源软件版本见 `../requirements.txt`。
