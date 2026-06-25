# -*- coding: utf-8 -*-
"""
真实疲劳数据 · Virkler 裂纹扩展数据集 (探索)
=============================================
数据来源: WarrRich/Virkler-Data (从 Bogdanoff & Kozin 1985 图4.5.3 数字化, 近似版)
底层实验: Virkler et al. 1977, 2024-T3 铝, 中心裂纹, 等幅载荷, 68 试件
格式: V1=试件编号, V2=循环数(千次), V3=裂纹半长(mm)
输出 -> results/virkler/
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/mnt/c/Users/Liu/desktop/裂纹分析预测"
OUT  = f"{ROOT}/results/virkler"; os.makedirs(OUT, exist_ok=True)

df = pd.read_csv(f"{ROOT}/data/virkler/VirklerData.csv")
df.columns = ["specimen", "kcycles", "a_mm"]
specs = sorted(df["specimen"].unique())
print(f"试件数: {len(specs)}", flush=True)
print(f"每试件点数: {df.groupby('specimen').size().unique()}", flush=True)
print(f"裂纹长范围: {df['a_mm'].min():.2f} ~ {df['a_mm'].max():.2f} mm", flush=True)
print(f"循环数范围: {df['kcycles'].min():.0f} ~ {df['kcycles'].max():.0f} (千次)", flush=True)

# 画全部试件 a-N 曲线 (展示真实疲劳散布)
fig, ax = plt.subplots(figsize=(7, 5))
for s in specs:
    g = df[df["specimen"] == s]
    ax.plot(g["kcycles"], g["a_mm"], "-", lw=0.6, alpha=0.6)
ax.set_xlabel("cycles (×1000)"); ax.set_ylabel("crack length a (mm)")
ax.set_title(f"Virkler real data: {len(specs)} specimens (a-N scatter)")
ax.grid(True)
fig.tight_layout(); fig.savefig(f"{OUT}/virkler_aN_all.png", dpi=120); plt.close(fig)

# 同一裂纹长下寿命的离散性 (真实疲劳散布有多大)
target = 30.0
lives = []
for s in specs:
    g = df[df["specimen"] == s].sort_values("a_mm")
    if g["a_mm"].max() >= target >= g["a_mm"].min():
        lives.append(np.interp(target, g["a_mm"], g["kcycles"]))
lives = np.array(lives)
if len(lives):
    print(f"\n到达 a={target}mm 的寿命: 均值 {lives.mean():.0f}k, "
          f"范围 {lives.min():.0f}~{lives.max():.0f}k, "
          f"最大/最小 = {lives.max()/lives.min():.2f}倍 (真实疲劳散布!)", flush=True)
print("完成. 图在", OUT, flush=True)
