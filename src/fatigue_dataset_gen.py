# -*- coding: utf-8 -*-
"""
阶段3 数据生成: 用 Paris 模型批量造疲劳数据集 (供神经网络训练)
============================================================
扫多种 循环应力 σmax × 初始裂纹 a0, 每条 a-N 曲线上每一点都是一个样本:
  输入特征: a_mm(当前裂纹长), sigma_max, dK(应力强度幅)
  预测目标: RUL(剩余寿命=总寿命-当前循环数)
输出 -> data/fatigue_ml_dataset.csv
"""
import os, csv
import numpy as np

OUT = "/mnt/c/Users/Liu/desktop/裂纹分析预测/data"
os.makedirs(OUT, exist_ok=True)

C, m = 1.0e-11, 3.0      # Paris 系数
W, K_IC, dKth, R = 0.05, 50.0, 3.0, 0.1

def Y(aw):
    return 1.12 - 0.231*aw + 10.55*aw**2 - 21.72*aw**3 + 30.39*aw**4

def simulate(sigma_max, a0):
    dsig = sigma_max*(1.0 - R)
    a, N = a0, 0.0
    A, NN, DK = [a], [N], [Y(a/W)*dsig*np.sqrt(np.pi*a)]
    da = (W - a0)/4000.0
    while a < 0.95*W:
        aw = a/W
        dK = Y(aw)*dsig*np.sqrt(np.pi*a)
        if Y(aw)*sigma_max*np.sqrt(np.pi*a) >= K_IC or dK <= dKth:
            break
        a += da; N += da/(C*dK**m)
        A.append(a); NN.append(N); DK.append(dK)
    return np.array(A), np.array(NN), np.array(DK)

# 扫描条件: 多应力水平 × 多初始裂纹
sigma_levels = np.arange(70.0, 161.0, 5.0)     # 70..160 MPa, 19 档
a0_levels    = [0.003, 0.005, 0.008]           # 3 mm/5 mm/8 mm
rows = []
for s in sigma_levels:
    for a0 in a0_levels:
        A, NN, DK = simulate(s, a0)
        if len(A) < 10:
            continue
        Nf = NN[-1]                            # 该条件总寿命
        for i in range(len(A)):
            rows.append([s, a0*1e3, A[i]*1e3, NN[i], Nf - NN[i], DK[i]])

with open(f"{OUT}/fatigue_ml_dataset.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["sigma_max_MPa", "a0_mm", "a_mm", "N_cycles", "RUL_cycles", "dK"])
    w.writerows(rows)

rows = np.array(rows)
print(f"生成 {len(rows)} 个样本, {len(sigma_levels)} 应力档 × {len(a0_levels)} 初始裂纹", flush=True)
print(f"RUL 范围: {rows[:,4].min():.0f} ~ {rows[:,4].max():.0f} 循环", flush=True)
print("保存 ->", f"{OUT}/fatigue_ml_dataset.csv", flush=True)
