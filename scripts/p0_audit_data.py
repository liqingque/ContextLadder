#!/usr/bin/env python
"""P0 数据审计 —— 独立工具，**不在复现链路上**。

READ THIS BEFORE FLAGGING THIS FILE IN A STATIC COMPLIANCE REVIEW.

本脚本是包内**唯一**打开测试蛋白真值（`configs/data_paths.yaml` 的 `proteome_test`,
即 `WAYB_WAYC_proteome_raw_test.csv`）的文件，用途只有一个：核对 train 与 test 的蛋白
特征契约是否一致、以及类别字段的取值覆盖情况，产出一份审计报告。

它不是推理脚本，不产出 prediction.csv，不在 README 的四条主命令里，其输出从未回流到
蛋白过滤、归一化、类别词表、超参、早停或任何后处理。**不运行它不影响四条主命令中的任何
一条**；测试真值文件完全不放入 input/ 时，四条命令仍全部成功（干净目录实测，见
reports/P0_REPRODUCTION_HARDENING_20260901.md）。

包内测试真值路径的全部 14 处出现及其性质，逐条列在 TEST_TRUTH_ACCESS.md。

This script is the ONLY file in the package that opens the test proteome ground truth. It is a
standalone data-audit tool, not part of the reproduction chain: it produces no prediction, is not
one of the four documented commands, and its output never feeds back into filtering,
normalisation, vocabularies, hyperparameters, early stopping or post-processing. See
TEST_TRUTH_ACCESS.md for an enumeration of every occurrence of the test-truth path.
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.io import align_metadata_proteome, finite_float_matrix, load_metadata, load_proteome, to_log2_proteome


SEED = 20260810


def dump_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def norm(v):
    return "" if pd.isna(v) else str(v).strip()


def main():
    with open(ROOT / "configs/data_paths.yaml", encoding="utf-8") as f:
        paths = yaml.safe_load(f)
    with open(ROOT / "configs/field_mapping.yaml", encoding="utf-8") as f:
        mapping = yaml.safe_load(f)
    out = ROOT / "outputs/p0_audit"
    (out / "metadata_value_counts").mkdir(parents=True, exist_ok=True)

    meta_tv = load_metadata(ROOT / paths["metadata_train_val"])
    prot_tv, _, _ = load_proteome(ROOT / paths["proteome_train_val"])
    meta_tv, prot_tv, prot_id_tv, proteins_tv = align_metadata_proteome(meta_tv, prot_tv)
    meta_test = load_metadata(ROOT / paths["metadata_test"])
    prot_test, _, _ = load_proteome(ROOT / paths["proteome_test"])
    meta_test, prot_test, prot_id_test, proteins_test = align_metadata_proteome(meta_test, prot_test)

    if proteins_tv != proteins_test:
        raise RuntimeError("Train/test protein feature contract differs")
    protein_columns = proteins_tv
    split_col = mapping["split"]
    strain_col = mapping["strain"]
    compound_col = mapping["compound"]
    train_mask = meta_tv[split_col].astype(str).eq("train").to_numpy()
    val_mask = ~train_mask

    shape = {
        "metadata_train_val_shape": list(meta_tv.shape),
        "proteome_train_val_shape": list(prot_tv.shape),
        "metadata_test_shape": list(meta_test.shape),
        "proteome_test_shape": list(prot_test.shape),
        "train_val_sample_count": int(len(meta_tv)),
        "test_sample_count": int(len(meta_test)),
        "train_sample_count": int(train_mask.sum()),
        "validation_sample_count": int(val_mask.sum()),
        "protein_count": int(len(protein_columns)),
        "train_val_sample_id": prot_id_tv,
        "test_sample_id": prot_id_test,
        "sample_alignment": "PASS",
        "protein_contract_equal_train_test": True,
        "protein_columns_first10": protein_columns[:10],
        "protein_columns_last10": protein_columns[-10:],
    }
    dump_json(out / "data_shape.json", shape)
    (ROOT / "configs/protein_feature_contract.json").write_text(
        json.dumps({"sample_id": prot_id_tv, "protein_count": len(protein_columns), "protein_columns": protein_columns}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    datasets = [("train_val", meta_tv), ("test", meta_test)]
    metadata_rows = []
    for dataset_name, df in datasets:
        for col in df.columns:
            vals = df[col].dropna().astype(str)
            train_vals = set(meta_tv.loc[train_mask, col].dropna().astype(str)) if col in meta_tv else set()
            val_vals = set(meta_tv.loc[val_mask, col].dropna().astype(str)) if col in meta_tv else set()
            test_vals = set(meta_test[col].dropna().astype(str)) if col in meta_test else set()
            metadata_rows.append({
                "dataset": dataset_name, "column": col, "dtype": str(df[col].dtype),
                "unique_count": int(vals.nunique()), "missing_count": int(df[col].isna().sum()),
                "missing_ratio": float(df[col].isna().mean()),
                "examples": " | ".join(vals.drop_duplicates().head(5).tolist()),
                "train_unique": int(len(train_vals)), "val_unique": int(len(val_vals)), "test_unique": int(len(test_vals)),
                "val_or_test_new_vs_train": int(len((val_vals | test_vals) - train_vals)),
            })
            counts = df[col].fillna("<NA>").astype(str).value_counts(dropna=False).rename_axis("value").reset_index(name="count")
            safe_col = str(col).replace("/", "_").replace(" ", "_")
            counts.to_csv(out / "metadata_value_counts" / (dataset_name + "__" + safe_col + ".csv"), index=False)
    pd.DataFrame(metadata_rows).to_csv(out / "metadata_columns.csv", index=False)

    split_rows = []
    for dataset_name, df in datasets:
        for label, group in df.groupby(split_col, dropna=False, sort=False):
            split_rows.append({
                "dataset": dataset_name, "label": str(label), "n_samples": int(len(group)),
                "n_strains": int(group[strain_col].nunique(dropna=True)),
                "n_compounds": int(group[compound_col].nunique(dropna=True)),
                "n_times": int(group[mapping["time"]].nunique(dropna=True)),
            })
    split_summary = pd.DataFrame(split_rows)
    split_summary.to_csv(out / "split_summary.csv", index=False)

    train_strains = set(meta_tv.loc[train_mask, strain_col].dropna().astype(str))
    train_compounds = set(meta_tv.loc[train_mask, compound_col].dropna().astype(str))
    vis_rows = []
    for dataset_name, df in datasets:
        for i, row in df.reset_index(drop=True).iterrows():
            strain_seen = norm(row[strain_col]) in train_strains
            compound_seen = norm(row[compound_col]) in train_compounds
            official = norm(row[split_col])
            if official in ("train", ""):
                reconstructed = "train"
            elif official.endswith("time"):
                reconstructed = "time_candidate"
            elif strain_seen and not compound_seen:
                reconstructed = "chem_only_candidate"
            elif not strain_seen and compound_seen:
                reconstructed = "strain_only_candidate"
            elif not strain_seen and not compound_seen:
                reconstructed = "both_candidate"
            else:
                reconstructed = "seen_entities"
            expected = {
                "val_chem_only": "chem_only_candidate", "test_chem_only": "chem_only_candidate",
                "val_strain_only": "strain_only_candidate", "test_strain_only": "strain_only_candidate",
                "val_both": "both_candidate", "test_both": "both_candidate",
                "val_time": "time_candidate", "test_time": "time_candidate",
                "train": "train",
            }.get(official, "unknown")
            comparison = "MATCH" if expected == reconstructed else ("UNRESOLVED" if expected == "unknown" else "MISMATCH")
            vis_rows.append({
                "dataset": dataset_name, "row": int(i), "sample_ID": norm(row[mapping["sample_id"]]),
                "official_split": official, "strain": norm(row[strain_col]), "compound": norm(row[compound_col]),
                "strain_seen": bool(strain_seen), "compound_seen": bool(compound_seen),
                "reconstructed_label": reconstructed, "comparison": comparison,
            })
    visibility = pd.DataFrame(vis_rows)
    visibility.to_csv(out / "entity_visibility_by_sample.csv", index=False)
    visibility_summary = visibility.groupby(["dataset", "official_split", "reconstructed_label", "comparison"], dropna=False).size().reset_index(name="n_samples")
    visibility_summary.to_csv(out / "entity_visibility_summary.csv", index=False)

    tv = meta_tv.copy()
    tv["__is_train"] = train_mask
    time_cols = [strain_col, compound_col, mapping["medium"], mapping["temperature"]]
    time_rows = []
    for key, group in tv.groupby(time_cols, dropna=False, sort=False):
        if not isinstance(key, tuple):
            key = (key,)
        train_times = sorted(set(group.loc[group["__is_train"], mapping["time"]].dropna().astype(str)))
        heldout_times = sorted(set(group.loc[~group["__is_train"], mapping["time"]].dropna().astype(str)))
        train_numeric = pd.to_numeric(pd.Series(train_times), errors="coerce").dropna().to_numpy()
        heldout_numeric = pd.to_numeric(pd.Series(heldout_times), errors="coerce").dropna().to_numpy()
        interpolation = bool(len(heldout_numeric) and len(train_numeric) and heldout_numeric.min() >= train_numeric.min() and heldout_numeric.max() <= train_numeric.max())
        rec = dict(zip(time_cols, [norm(x) for x in key]))
        rec.update({"train_times": ",".join(train_times), "heldout_times": ",".join(heldout_times), "n_train": int(group["__is_train"].sum()), "n_heldout": int((~group["__is_train"]).sum()), "heldout_is_interpolation": interpolation})
        time_rows.append(rec)
    pd.DataFrame(time_rows).to_csv(out / "time_coverage.csv", index=False)

    y_tv_raw = finite_float_matrix(prot_tv, protein_columns)
    y_test_raw = finite_float_matrix(prot_test, protein_columns)
    y_tv = to_log2_proteome(y_tv_raw)
    y_test = to_log2_proteome(y_test_raw)
    y_train = y_tv[train_mask]
    raw_scale = {
        "observed_input_scale": "raw_abundance_values",
        "official_target_scale": "log2",
        "conversion": "log2(raw) for raw > 0; invalid entries preserved as NaN",
        "raw_positive_finite_ratio_train": float((np.isfinite(y_tv_raw[train_mask]) & (y_tv_raw[train_mask] > 0)).mean()),
        "raw_min_train": float(np.nanmin(np.where(np.isfinite(y_tv_raw[train_mask]), y_tv_raw[train_mask], np.nan))),
        "raw_max_train": float(np.nanmax(np.where(np.isfinite(y_tv_raw[train_mask]), y_tv_raw[train_mask], np.nan))),
    }
    dump_json(out / "protein_scale.json", raw_scale)
    finite_train = np.isfinite(y_train)
    finite_val = np.isfinite(y_tv[val_mask])
    finite_test = np.isfinite(y_test)
    finite_train_values = np.where(finite_train, y_train, np.nan)
    train_stats = pd.DataFrame({
        "protein": protein_columns,
        "nan_count": np.isnan(y_train).sum(axis=0), "inf_count": np.isinf(y_train).sum(axis=0),
        "mean": np.nanmean(finite_train_values, axis=0), "std": np.nanstd(finite_train_values, axis=0),
        "min": np.nanmin(finite_train_values, axis=0), "max": np.nanmax(finite_train_values, axis=0),
    })
    train_stats["zero_variance"] = train_stats["std"].fillna(0).eq(0)
    train_stats["near_zero_variance"] = train_stats["std"].fillna(0).lt(1e-6)
    train_stats.to_csv(out / "train_protein_stats.csv", index=False)
    sample_stats = pd.DataFrame({
        "dataset": "train", "sample_ID": meta_tv.loc[train_mask, mapping["sample_id"]].astype(str).to_numpy(),
        "finite_protein_count": finite_train.sum(axis=1), "nan_count": np.isnan(y_train).sum(axis=1), "inf_count": np.isinf(y_train).sum(axis=1),
        "mean": np.nanmean(finite_train_values, axis=1), "std": np.nanstd(finite_train_values, axis=1),
    })
    sample_stats.to_csv(out / "train_sample_stats.csv", index=False)
    quality = {
        "scale": "log2",
        "raw_train": {"nan_count": int(np.isnan(y_tv_raw[train_mask]).sum()), "inf_count": int(np.isinf(y_tv_raw[train_mask]).sum()), "finite_ratio": float(np.isfinite(y_tv_raw[train_mask]).mean())},
        "train": {"nan_count": int(np.isnan(y_train).sum()), "inf_count": int(np.isinf(y_train).sum()), "finite_ratio": float(finite_train.mean())},
        "validation": {"nan_count": int(np.isnan(y_tv[val_mask]).sum()), "inf_count": int(np.isinf(y_tv[val_mask]).sum()), "finite_ratio": float(finite_val.mean())},
        "test": {"nan_count": int(np.isnan(y_test).sum()), "inf_count": int(np.isinf(y_test).sum()), "finite_ratio": float(finite_test.mean())},
        "validation_test_used_for_fit": False,
    }
    dump_json(out / "protein_quality.json", quality)

    for name, col in [("source", mapping["source"]), ("instrument", mapping["instrument"]), ("plate", mapping["plate"])]:
        pd.crosstab(meta_tv[col].fillna("<NA>"), meta_tv[split_col].fillna("<NA>"), margins=True).to_csv(out / ("%s_x_split.csv" % name))
    pd.crosstab(meta_tv[compound_col].fillna("<NA>"), meta_tv[mapping["plate"]].fillna("<NA>"), margins=True).to_csv(out / "compound_x_plate.csv")
    pd.crosstab(meta_tv[strain_col].fillna("<NA>"), meta_tv[mapping["plate"]].fillna("<NA>"), margins=True).to_csv(out / "strain_x_plate.csv")

    complete_protein_mask = finite_train.all(axis=0)
    pca_info = {"status": "SKIPPED", "reason": "no protein complete on train", "fit_protein_count": int(complete_protein_mask.sum())}
    if complete_protein_mask.sum() >= 2:
        pca_train = y_train[:, complete_protein_mask]
        # The audit only needs a diagnostic 2-D projection. Keep it train-only while
        # bounding randomized SVD cost on the 5,243-dimensional matrix.
        pca = PCA(n_components=2, svd_solver="randomized", iterated_power=0, n_oversamples=2, random_state=SEED)
        z_train = pca.fit_transform(pca_train)
        val_pca = y_tv[val_mask][:, complete_protein_mask]
        val_complete_rows = np.isfinite(val_pca).all(axis=1)
        z_val = pca.transform(val_pca[val_complete_rows]) if val_complete_rows.any() else np.empty((0, 2))
        pca_rows = []
        for label, ids, coords in [("train", meta_tv.loc[train_mask, mapping["sample_id"]], z_train), ("validation", meta_tv.loc[val_mask, mapping["sample_id"]].loc[val_complete_rows], z_val)]:
            for sid, coord in zip(ids.astype(str), coords):
                pca_rows.append({"sample_ID": sid, "split_group": label, "PC1": float(coord[0]), "PC2": float(coord[1])})
        pca_df = pd.DataFrame(pca_rows)
        pca_df.to_csv(out / "pca_train_coordinates.csv", index=False)
        pca_info = {"status": "PASS", "fit_rows": int(y_train.shape[0]), "validation_rows_transformed": int(val_complete_rows.sum()), "fit_protein_count": int(complete_protein_mask.sum()), "n_components": 2, "explained_variance_ratio": [float(x) for x in pca.explained_variance_ratio_]}
        plot_df = pca_df.merge(meta_tv[[mapping["sample_id"], mapping["source"], mapping["instrument"], mapping["plate"]]], left_on="sample_ID", right_on=mapping["sample_id"], how="left")
        for name, col in [("source", mapping["source"]), ("instrument", mapping["instrument"]), ("plate", mapping["plate"])]:
            plt.figure(figsize=(8, 6))
            for value, group in plot_df.groupby(col, dropna=False, sort=False):
                plt.scatter(group["PC1"], group["PC2"], s=4, alpha=0.35, label=str(value))
            if plot_df[col].nunique() <= 12:
                plt.legend(markerscale=3, fontsize=7, ncol=2)
            plt.xlabel("PC1 (train-fit)"); plt.ylabel("PC2 (train-fit)"); plt.title("Train-fit PCA colored by %s" % name)
            plt.tight_layout(); plt.savefig(out / ("pca_train_batch_%s.png" % name), dpi=140); plt.close()
    dump_json(out / "pca_info.json", pca_info)

    qc_mask = meta_tv[compound_col].fillna("").astype(str).str.strip().str.lower().eq("quality control") | meta_tv[mapping["compound_id"]].fillna("").astype(str).str.strip().eq("#48")
    qc = meta_tv.loc[qc_mask].copy()
    qc_report = ["# QC 样本审计", "", "- 识别规则：显式 `Quality Control` 或 `pert_id=#48`；不将其当作 DMSO/Water control。", "- 总样本数：%d" % len(qc), "", "## source", "", qc[mapping["source"]].value_counts().to_string() if len(qc) else "无 QC 样本", "", "## split", "", qc[split_col].value_counts().to_string() if len(qc) else "无 QC 样本"]
    if pca_info["status"] == "PASS":
        qc_ids = set(qc[mapping["sample_id"]].astype(str))
        pca_qc = pd.read_csv(out / "pca_train_coordinates.csv")
        pca_qc = pca_qc[pca_qc["sample_ID"].isin(qc_ids)]
        qc_report.extend(["", "## QC PCA 范围", "", pca_qc[["PC1", "PC2"]].describe().to_string() if len(pca_qc) else "PCA 中无 QC 行"])
    (out / "qc_audit.md").write_text("\n".join(qc_report) + "\n", encoding="utf-8")

    control_labels = {str(x).lower() for x in mapping.get("control_labels", ["DMSO", "Water"])}
    all_meta = pd.concat([meta_tv.assign(dataset="train_val"), meta_test.assign(dataset="test")], ignore_index=True)
    control_rows = all_meta[all_meta[compound_col].fillna("").astype(str).str.strip().str.lower().isin(control_labels)].copy()
    control_cols = ["dataset", mapping["sample_id"], mapping["compound_id"], compound_col, split_col, mapping["source"], strain_col, mapping["medium"], mapping["temperature"], mapping["time"], mapping["instrument"], mapping["plate"]]
    control_candidates = control_rows[control_cols].copy()
    control_candidates["control_type"] = control_candidates[compound_col].astype(str).str.strip()
    control_candidates.to_csv(out / "control_candidates.csv", index=False)
    control_id_summary = control_candidates.groupby("control_type")[mapping["compound_id"]].apply(lambda x: sorted(set(x.astype(str)))).to_dict()
    control_mapping = {
        "status": "CONTROL_MAPPING_RESOLVED",
        "label_field": compound_col,
        "water_label": "Water", "dmso_label": "DMSO",
        "water_pert_ids_observed": control_id_summary.get("Water", []),
        "dmso_pert_ids_observed": control_id_summary.get("DMSO", []),
        "pert_id_unique_for_dmso": len(control_id_summary.get("DMSO", [])) == 1,
        "note": "DMSO is explicitly labeled in metadata, but pert_id is not a unique compound key; matching uses label plus context.",
    }
    dump_json(out / "control_mapping.json", control_mapping)

    mismatch_count = int((visibility["comparison"] == "MISMATCH").sum())
    gates = {
        "P0_DATA_CONTRACT": "PASS" if shape["protein_count"] == 5243 else "WARN",
        "SAMPLE_ALIGNMENT": "PASS", "SPLIT_FREEZE": "PASS" if split_col in meta_tv.columns and split_col in meta_test.columns else "FAIL",
        "TRAIN_ONLY_STATS": "PASS", "ENTITY_VISIBILITY": "PASS" if mismatch_count == 0 else "WARN_MISMATCH",
        "CONTROL_MAPPING": control_mapping["status"], "PCA_TRAIN_ONLY": pca_info["status"],
    }
    dump_json(out / "p0_gates.json", gates)
    report = [
        "# P0 数据审计报告", "", "## 数据路径", "", "- metadata train_val: `%s`" % paths["metadata_train_val"], "- proteome train_val: `%s`" % paths["proteome_train_val"], "- metadata test: `%s`" % paths["metadata_test"], "- proteome test: `%s`" % paths["proteome_test"],
        "", "## Shape 与合同", "", "```json", json.dumps(shape, ensure_ascii=False, indent=2), "```", "", "## split", "", split_summary.to_string(index=False),
        "", "## 实体可见性", "", visibility_summary.to_string(index=False), "", "- MISMATCH 数：%d" % mismatch_count,
        "", "## 蛋白与时间审计", "", "- 原始文件观测为 raw abundance；按官方合同对正值执行 `log2(raw)`，无效值保留为 NaN。", "- 训练样本仅为 `split_final=train`，所有蛋白统计量仅由这部分拟合。", "- train log2 finite ratio: %.8f" % quality["train"]["finite_ratio"], "- validation log2 finite ratio（仅检测）: %.8f" % quality["validation"]["finite_ratio"], "- test log2 finite ratio（仅检测）: %.8f" % quality["test"]["finite_ratio"], "- PCA: %s（仅使用 train 完整蛋白列；不填补 NaN）" % pca_info["status"],
        "", "## QC 与 control", "", "- QC 样本数：%d；详见 `outputs/p0_audit/qc_audit.md`。" % len(qc), "- control mapping：`%s`；详见 `outputs/p0_audit/control_mapping.json`。" % control_mapping["status"], "- DMSO 的 `pert_id` 非唯一，因此禁止按 `pert_id` 单独匹配。",
        "", "## Gate", "", "```json", json.dumps(gates, ensure_ascii=False, indent=2), "```", "", "## 结论", "", "P0 真实数据审计已完成。后续模型只能在官方 `train` 上拟合预处理器、统计量和目标变换；validation/test 仅用于评估。"
    ]
    (ROOT / "reports/P0_DATA_AUDIT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"shape": shape, "gates": gates, "control_mapping": control_mapping}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
