#!/usr/bin/env python
"""Generate the figures embedded in the semi-final slide deck.

Kept as a script rather than a notebook so the deck's numbers are traceable to
the artifacts they came from. Every value here is quoted from a report or a
JSON artifact under reports/ and outputs/.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from pathlib import Path

CJK = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
fm.fontManager.addfont(CJK)
CJK_NAME = fm.FontProperties(fname=CJK).get_name()
plt.rcParams.update({"font.family": CJK_NAME, "font.size": 11, "axes.unicode_minus": False,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 200, "savefig.bbox": "tight"})
BLUE, GREY, RED = "#2B6CB0", "#A0AEC0", "#C53030"
OUT = Path("outputs/slides_assets"); OUT.mkdir(parents=True, exist_ok=True)

# 1) official six modules
mods = ["绝对保真度\n20%", "匹配对照FC\n25%", "S1残差\n20%", "S2残差\n20%", "时间插值\n10%", "DEP\n5%"]
vals = [0.9886, 0.4914, 0.2034, 0.4077, 0.6717, 0.8919]
fig, ax = plt.subplots(figsize=(7.6, 3.1))
ax.bar(range(6), [1]*6, color=GREY, alpha=.18)
ax.bar(range(6), vals, color=[BLUE, BLUE, RED, BLUE, BLUE, BLUE])
for i, v in enumerate(vals): ax.text(i, v+.03, f"{v:.3f}", ha="center", fontsize=10)
ax.set_xticks(range(6)); ax.set_xticklabels(mods)
ax.set_ylim(0, 1.12); ax.set_ylabel("模块得分")
ax.set_title("官方六模块实测（红色 = 模型贡献为零的模块）")
fig.savefig(OUT/"modules.png"); plt.close(fig)

# 2) capacity vs content
fig, ax = plt.subplots(figsize=(6.6, 3.2))
names = ["oracle\n全张成 k=36", "φ_TXT\n机理文本", "φ_SSPS\n机理先验", "Tanimoto\n化学相似", "随机\n负对照"]
rho = [0.2671, 0.0268, -0.0325, -0.0148, -0.0250]
ax.bar(range(5), rho, color=[BLUE]+[GREY]*4)
ax.axhline(0.20, color=RED, ls="--", lw=1.4)
ax.text(4.4, 0.208, "Gate 门槛 0.20", color=RED, ha="right", fontsize=9)
ax.axhline(0, color="k", lw=.8)
for i, v in enumerate(rho): ax.text(i, v + (.013 if v >= 0 else -.030), f"{v:+.4f}", ha="center", fontsize=9)
ax.set_xticks(range(5)); ax.set_xticklabels(names, fontsize=9)
ax.set_ylabel("ρ 与目标残差的对齐度"); ax.set_ylim(-0.075, 0.32)
ax.set_title("容量够，内容不够：近十倍的差距")
fig.savefig(OUT/"capacity_content.png"); plt.close(fig)

# 3) Gate-W vs the official baseline family
fig, ax = plt.subplots(figsize=(6.6, 2.9))
lbl = ["本方案", "分场景\n低秩统计", "分场景\n梯度提升", "全局GBDT\nPCA-128", "全局GBDT\nPCA-64"]
W = [0.5218, 0.4471, 0.4424, 0.4186, 0.4181]
ax.bar(range(5), W, color=[BLUE]+[GREY]*4)
for i, v in enumerate(W):
    ax.text(i, v+.010, f"{v:.4f}", ha="center", fontsize=9)
    if i: ax.text(i, v/2, f"ΔW\n{v-W[0]:+.4f}", ha="center", color="white", fontsize=9)
ax.set_xticks(range(5)); ax.set_xticklabels(lbl, fontsize=9)
ax.set_ylabel("官方加权分 W"); ax.set_ylim(0, 0.62)
ax.set_title("与官方基线家族对照（采纳门槛 ΔW ≥ +0.010）")
fig.savefig(OUT/"gate_w.png"); plt.close(fig)

print("wrote:", ", ".join(sorted(p.name for p in OUT.glob("*.png"))))
