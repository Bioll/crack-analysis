# -*- coding: utf-8 -*-
"""
疲劳裂纹扩展 (Paris 公式)  ——  快速版 da/dN
============================================
模型: 单边裂纹试件 (SENT), 远场循环拉应力, 经典 Paris 律
      da/dN = C (ΔK)^m       ΔK = Y(a/W)·Δσ·√(πa)
积分裂纹长 a 从初始 a0 增长, 直到 K_max 达到断裂韧性 K_IC (失稳断裂)。

输出 -> results/fatigue_paris/
  - crack_growth.png    a-N 曲线 + Paris(da/dN vs ΔK) 直线
  - fatigue_dataset.csv  多应力水平的 (Δσ, a, N, ΔK, dadN) , 供阶段三神经网络
单位: 应力 MPa, 长度 m, ΔK 单位 MPa·√m
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = "/mnt/c/Users/Liu/desktop/裂纹分析预测/results/fatigue_paris"
os.makedirs(OUT_DIR, exist_ok=True)

# ---- 材料 / 几何 / 加载参数 (钢, 典型值) ----
C    = 1.0e-11      # Paris 系数 (da/dN[m/cyc], ΔK[MPa√m])
m    = 3.0          # Paris 指数
W    = 0.05         # 试件宽度 50 mm
a0   = 0.005        # 初始裂纹 5 mm
K_IC = 50.0         # 断裂韧性 MPa√m
dKth = 3.0          # 扩展门槛, 低于此不扩展
R    = 0.1          # 应力比 σ_min/σ_max

def Y(aw):          # SENT 几何因子
    return (1.12 - 0.231*aw + 10.55*aw**2 - 21.72*aw**3 + 30.39*aw**4)

def simulate(sigma_max):
    """给定循环最大应力, 积分 Paris 律, 返回 (a, N, dK, dadN) 序列。"""
    dsigma = sigma_max * (1.0 - R)        # 应力范围 Δσ
    a, N = a0, 0.0
    A, NN, DK, RATE = [a], [N], [], []
    da = (W - a0) / 4000.0                # 裂纹长步进
    while a < 0.95 * W:
        aw = a / W
        dK = Y(aw) * dsigma * np.sqrt(np.pi * a)      # ΔK
        Kmax = Y(aw) * sigma_max * np.sqrt(np.pi * a)
        if Kmax >= K_IC:                  # 达到断裂韧性 -> 失稳断裂, 寿命终止
            break
        if dK <= dKth:                    # 低于门槛不扩展
            break
        dadN = C * dK**m                  # da/dN
        dN = da / dadN                    # 走 da 需要的循环数
        a += da; N += dN
        A.append(a); NN.append(N); DK.append(dK); RATE.append(dadN)
    return (np.array(A), np.array(NN), np.array(DK), np.array(RATE))

# ---- 跑多个应力水平 ----
sigma_levels = [80.0, 100.0, 120.0, 140.0]    # MPa
results = {}
print(f"{'σmax(MPa)':>10} {'寿命N_f(cyc)':>14} {'终裂纹a(mm)':>12}", flush=True)
for s in sigma_levels:
    A, NN, DK, RATE = simulate(s)
    results[s] = (A, NN, DK, RATE)
    print(f"{s:>10.0f} {NN[-1]:>14.3e} {A[-1]*1e3:>12.2f}", flush=True)

# ---- 出图 ----
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
for s in sigma_levels:
    A, NN, DK, RATE = results[s]
    ax1.plot(NN, A*1e3, label=f"σmax={s:.0f} MPa")
    ax2.loglog(DK, RATE, label=f"σmax={s:.0f} MPa")
ax1.set_xlabel("cycles  N"); ax1.set_ylabel("crack length a (mm)")
ax1.set_title("crack growth  a-N"); ax1.legend(); ax1.grid(True)
ax2.set_xlabel("ΔK (MPa·√m)"); ax2.set_ylabel("da/dN (m/cycle)")
ax2.set_title(f"Paris law (slope m={m:.0f})"); ax2.legend(); ax2.grid(True, which="both")
fig.tight_layout(); fig.savefig(f"{OUT_DIR}/crack_growth.png", dpi=120); plt.close(fig)

# ---- 存数据集 (供阶段三神经网络) ----
import csv
with open(f"{OUT_DIR}/fatigue_dataset.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["sigma_max_MPa", "a_mm", "N_cycles", "dK_MPa_sqrt_m", "dadN_m_per_cyc"])
    for s in sigma_levels:
        A, NN, DK, RATE = results[s]
        for i in range(len(DK)):
            w.writerow([f"{s:.0f}", f"{A[i]*1e3:.5f}", f"{NN[i]:.3f}",
                        f"{DK[i]:.4f}", f"{RATE[i]:.6e}"])
print("完成. 结果在", OUT_DIR, flush=True)
