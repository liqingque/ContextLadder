# P1 Baseline Results

- 状态：`P1_COMPLETE`
- evaluator：`SPEC_APPROX`（官方手册给出评分规格，但当前目录未发现官方 evaluator 脚本）
- control mapping：`CONTROL_MAPPING_RESOLVED`（按 `perturbation_no_concentration` 的显式 `DMSO`/`Water` 标签；`pert_id` 不作为唯一化合物键）
- seed：`20260810`
- fit split：`split_final=train`；validation 共 3,038 个样本，仅用于选择 alpha 和评估。

## Target / missing-value protocol

- 原始文件名为 `proteome_raw`，数值范围也显示为 raw abundance；按官方 log2 合同对正值执行 `log2(raw)`。
- train finite ratio：0.72323713；validation/test 仅检测，不拟合统计量。
- Ridge 目标缺失处理：每个蛋白仅用 train 观测值的均值填充 train 目标缺口；all-missing train 蛋白设为 0；该策略已写入每个 run config。
- 评估按每个 sample 的有限 protein 交集计算，未对 validation/test 缺失值做填充。

## Overall validation comparison

            Model  Abs PCC   Abs R2     RMSE  FC PCC (spec approx)  FC coverage
        TrainMean 0.953882 0.883633 0.942392              0.239450     0.585887
 Ridge_biological 0.962347 0.899921 0.874128              0.279281     0.585887
Ridge_measurement 0.982272 0.963154 0.525593              0.348326     0.585887
       Ridge_full 0.984998 0.968519 0.486450              0.379957     0.585887
  ExtraTreesPCA64 0.983379 0.965188 0.511380              0.358375     0.585887

## Metrics by frozen validation split

            model           split  abs_pcc   abs_r2     rmse
        TrainMean        val_both 0.950411 0.869852 0.999782
        TrainMean   val_chem_only 0.950322 0.868375 1.008047
        TrainMean val_strain_only 0.956765 0.895287 0.890050
        TrainMean        val_time 0.955571 0.895909 0.885506
 Ridge_biological        val_both 0.951204 0.871998 0.992021
 Ridge_biological   val_chem_only 0.954981 0.876972 0.975101
 Ridge_biological val_strain_only 0.968371 0.917746 0.788204
 Ridge_biological        val_time 0.972057 0.927795 0.737527
Ridge_measurement        val_both 0.981388 0.961405 0.541708
Ridge_measurement   val_chem_only 0.984330 0.967181 0.502552
Ridge_measurement val_strain_only 0.980921 0.960584 0.539982
Ridge_measurement        val_time 0.983147 0.964158 0.504917
       Ridge_full        val_both 0.982252 0.963310 0.528234
       Ridge_full   val_chem_only 0.988894 0.976230 0.427768
       Ridge_full val_strain_only 0.982463 0.963489 0.520991
       Ridge_full        val_time 0.988265 0.974700 0.426270
  ExtraTreesPCA64        val_both 0.980788 0.960033 0.551583
  ExtraTreesPCA64   val_chem_only 0.987151 0.972573 0.459411
  ExtraTreesPCA64 val_strain_only 0.981050 0.960692 0.540385
  ExtraTreesPCA64        val_time 0.985184 0.968225 0.476630

## Selected alpha

- Ridge_biological: alpha=100.0；选择依据为 validation overall abs-PCC。
     alpha  runtime_sec  n_samples  n_proteins  valid_pcc_samples  valid_r2_samples  abs_pcc  abs_pcc_median  abs_pcc_p25  abs_pcc_p75   abs_r2     rmse      mae   fc_pcc  fc_coverage
  0.100000    19.697026       3038        5243               3038              3038 0.961936        0.969375     0.942879     0.977709 0.899045 0.878234 0.647375 0.292635     0.585887
  1.000000    21.346217       3038        5243               3038              3038 0.961951        0.969328     0.942992     0.977690 0.899075 0.878098 0.647352 0.292497     0.585887
 10.000000    18.141640       3038        5243               3038              3038 0.962072        0.968987     0.943853     0.977633 0.899333 0.876939 0.647258 0.291104     0.585887
100.000000    16.134929       3038        5243               3038              3038 0.962347        0.966426     0.946061     0.977003 0.899921 0.874128 0.651030 0.279281     0.585887

- Ridge_measurement: alpha=10.0；选择依据为 validation overall abs-PCC。
     alpha  runtime_sec  n_samples  n_proteins  valid_pcc_samples  valid_r2_samples  abs_pcc  abs_pcc_median  abs_pcc_p25  abs_pcc_p75   abs_r2     rmse      mae   fc_pcc  fc_coverage
  0.100000    18.375719       3038        5243               3038              3038 0.982111        0.983147     0.980372     0.985652 0.962975 0.526758 0.345680 0.338772     0.585887
  1.000000    18.086874       3038        5243               3038              3038 0.982168        0.983189     0.980501     0.985651 0.963072 0.526074 0.345381 0.340227     0.585887
 10.000000    17.545639       3038        5243               3038              3038 0.982272        0.983178     0.980548     0.985581 0.963154 0.525593 0.347289 0.348326     0.585887
100.000000    14.684380       3038        5243               3038              3038 0.979286        0.980549     0.976767     0.983131 0.956698 0.570818 0.389478 0.338176     0.585887

- Ridge_full: alpha=10.0；选择依据为 validation overall abs-PCC。
     alpha  runtime_sec  n_samples  n_proteins  valid_pcc_samples  valid_r2_samples  abs_pcc  abs_pcc_median  abs_pcc_p25  abs_pcc_p75   abs_r2     rmse      mae   fc_pcc  fc_coverage
  0.100000    38.400640       3038        5243               3038              3038 0.984620        0.985323     0.982119     0.989293 0.967815 0.491800 0.318766 0.371955     0.585887
  1.000000    38.492708       3038        5243               3038              3038 0.984695        0.985375     0.982236     0.989298 0.967961 0.490685 0.318053 0.373304     0.585887
 10.000000    34.603698       3038        5243               3038              3038 0.984998        0.985580     0.982608     0.989201 0.968519 0.486450 0.316250 0.379957     0.585887
100.000000    24.341397       3038        5243               3038              3038 0.983718        0.984582     0.981480     0.987285 0.965448 0.509902 0.339843 0.367677     0.585887

## Matched-control coverage

- treatment rows considered：2806
- matched rows：1644
- coverage：0.585887
- FC 是按赛题说明实现的规格近似，不能称为官方最终分数。四个 baseline 的 coverage 相同；差异只来自预测。

## Gate interpretation

1. Train Mean 是绝对预测地板：abs-PCC=0.953882。
2. biological-only Ridge 达到 abs-PCC=0.962347；相对 Mean 提升 0.008465。
3. measurement-only Ridge 达到 abs-PCC=0.982272，明显高于 biological-only，说明 source/instrument/plate 对当前 validation 具有很强解释力，batch shortcut 风险必须保留。
4. full Ridge 达到 abs-PCC=0.984998、FC PCC（规格近似）=0.379957；相对 measurement-only 的绝对 PCC 增益为 0.002726。
5. full Ridge 最低的 split 是 val_both（abs-PCC=0.982252），其次是 val_strain_only（0.982463）；chem-only 与 time 相对更高，分别为 0.988894 与 0.988265。
6. control 标签已解决，可以进入 P2；但首要动作应是检查 matched-control coverage 仅 58.6%，并把匹配键/未匹配原因作为 P2 的独立审计。

## P1 Gate

- `BEST_ABSOLUTE_BASELINE = Ridge_full_alpha10`
- `BEST_FC_BASELINE = Ridge_full_alpha10`（在当前 SPEC_APPROX evaluator 下）
- `BATCH_SHORTCUT_WARNING = TRUE`
- `P2_ALLOWED = TRUE`，前提是继续使用已冻结的 control label+context 规则。

## Artifacts

- `reports/P0_DATA_AUDIT.md`
- `outputs/p0_audit/`
- `outputs/p1_baselines/train_mean/`
- `outputs/p1_baselines/ridge_biological/`
- `outputs/p1_baselines/ridge_measurement/`
- `outputs/p1_baselines/ridge_full/`
- `outputs/LEADERBOARD_LOCAL.csv`
- `logs/p0_audit_success_20260810.log`
- `logs/p1_baselines_20260810.log`
