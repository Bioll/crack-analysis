# -*- coding: utf-8 -*-
"""
阶段2 · 分析裂纹 (完整版)
=========================
读取阶段1的 fields.npz, 提取几何特征 + 力学量 + 分类标签。

几何: 裂纹长度 / 宽度 / 曲折度tortuosity / 分叉数 / 扩展方向
力学: 应力强度因子 K_I (SENT手册公式) / 裂纹尖端位置 / 扩展速率 da/d(位移)
       (注: 单调加载, 给的是 da/d位移; 真正的 da/dN 需循环加载仿真)
标签: 是否萌生 / 裂纹等级(轻/中/重) / 是否失稳扩展

输出 -> results/sent/analysis/
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
from scipy.ndimage import convolve, label

SENT_DIR = "/mnt/c/Users/Liu/desktop/裂纹分析预测/results/sent"
OUT_DIR  = f"{SENT_DIR}/analysis"
os.makedirs(OUT_DIR, exist_ok=True)

# ---- 1. 读取 ----
data = np.load(f"{SENT_DIR}/fields.npz")
coords, d_all = data["coords"], data["d"]
disp, force   = data["disp"], data["force"]
cmod          = data["cmod"]                 # 裂纹嘴张开位移 (修正2)
ell, N        = float(data["ell"]), int(data["N"])
lam, mu, Gc   = float(data["E_lambda"]), float(data["E_mu"]), float(data["Gc"])
nstep, h, ng  = d_all.shape[0], 1.0 / int(data["N"]), int(data["N"]) + 1
um = disp * 1e3

# 【修正3·方法B】材料弹性常数 + 相场自洽的断裂韧性 K_c = sqrt(E'·Gc)
E_young = mu * (3*lam + 2*mu) / (lam + mu)          # 杨氏模量
nu      = lam / (2 * (lam + mu))                    # 泊松比
E_prime = E_young / (1 - nu**2)                     # 平面应变有效模量
K_c     = np.sqrt(E_prime * Gc)                     # 标定基准: 扩展时 G=Gc -> K=K_c
print(f"读入 {nstep} 步, N={N}, ell={ell}", flush=True)
print(f"[方法B] E={E_young/1e3:.1f}GPa, nu={nu:.3f}, E'={E_prime/1e3:.1f}GPa, "
      f"Gc={Gc}N/mm -> K_c=sqrt(E'·Gc)={K_c:.1f} MPa·mm^0.5", flush=True)

# ---- 2. 节点损伤 -> 规则网格 D[iy,ix] ----
ix = np.round(coords[:, 0] * N).astype(int)
iy = np.round(coords[:, 1] * N).astype(int)
xg = np.linspace(0, 1, ng)
def to_grid(v):
    G = np.zeros((ng, ng)); G[iy, ix] = v; return G

# 3x3 邻居核 (数骨架邻居)
K8 = np.ones((3, 3)); K8[1, 1] = 0

# ---- 3. 逐步提取特征 ----
crack_len  = np.zeros(nstep)   # 表面密度积分长度
width      = np.zeros(nstep)   # 有效带宽
tort       = np.ones(nstep)    # 曲折度
n_branch   = np.zeros(nstep, int)  # 分叉点数
n_segment  = np.zeros(nstep, int)  # 独立裂纹段数
tip_x      = np.zeros(nstep)
tip_y      = np.full(nstep, 0.5)
direction  = np.zeros(nstep)   # 扩展方向角(度), 0=水平, 负=向下

for k in range(nstep):
    D = to_grid(d_all[k])
    dDdy, dDdx = np.gradient(D, h, h)
    gamma = D**2 / (2*ell) + 0.5*ell*(dDdx**2 + dDdy**2)
    crack_len[k] = gamma.sum() * h * h

    mask = D > 0.5
    area = mask.sum() * h * h
    width[k] = area / crack_len[k] if crack_len[k] > 1e-9 else 0.0
    n_segment[k] = label(mask)[1]

    cols = np.where(mask.any(axis=0))[0]
    if cols.size:
        jt = cols.max()                       # 最右列 = 裂纹尖端
        tip_x[k] = xg[jt]
        col = D[:, jt]
        tip_y[k] = np.sum(xg * col) / np.sum(col)
        direction[k] = np.degrees(np.arctan2(tip_y[k] - 0.5, tip_x[k] - 0.5)) \
                        if tip_x[k] > 0.5 else 0.0

    # 骨架化 -> 曲折度 + 分叉数
    if mask.sum() > 5:
        skel = skeletonize(mask)
        if skel.sum() > 2:
            nb = convolve(skel.astype(int), K8, mode="constant")
            n_branch[k] = label((skel & (nb >= 3)))[1]
            skel_len = skel.sum() * h
            straight = np.hypot(tip_x[k] - 0.0, tip_y[k] - 0.5)  # 左边缘到尖端
            tort[k] = skel_len / straight if straight > 1e-6 else 1.0

# ---- 4. 力学量 ----
net_len     = crack_len - crack_len[0]
growth_rate = np.gradient(crack_len, disp)        # da/d(位移)
a = np.clip(tip_x, 1e-6, 0.999)                   # 边裂纹长度 (从左边缘)
aw = a / 1.0                                      # a/W, W=1
Y = 1.12 - 0.231*aw + 10.55*aw**2 - 21.72*aw**3 + 30.39*aw**4   # SENT几何因子
sigma = force / 1.0                               # 名义应力 (W=B=1)
K_I = sigma * np.sqrt(np.pi * a) * Y              # MPa·mm^0.5 (线弹性估计)

# ---- 5. 分类标签 ----
INIT_THR = 0.02
initiated = net_len > INIT_THR                    # 是否萌生
init_disp = um[initiated][0] if initiated.any() else None

peak_idx  = int(np.argmax(force))
peak_force, peak_disp = force[peak_idx], um[peak_idx]
# 失稳: 峰值后 & 扩展速率显著为正 (裂纹快速冲)
gr_thr = 0.2 * np.nanmax(growth_rate)
unstable = (np.arange(nstep) >= peak_idx) & (growth_rate > gr_thr)
unstable_disp = um[unstable][0] if unstable.any() else None

# 裂纹等级: 按尖端推进占剩余韧带(0.5)的比例
adv_frac = np.clip((tip_x - 0.5) / 0.5, 0, 1)
severity = np.where(~initiated, "none",
            np.where(adv_frac < 0.2, "light",
            np.where(adv_frac < 0.7, "medium", "heavy")))
final_sev = severity[-1]

# ---- 6. 打印小结 ----
print("="*52, flush=True)
print(f"是否萌生        : {'是' if initiated.any() else '否'}"
      + (f" (起裂位移 {init_disp:.3f} um)" if init_disp else ""), flush=True)
print(f"峰值载荷        : {peak_force:.1f} N  @ {peak_disp:.3f} um", flush=True)
print(f"失稳扩展onset   : "
      + (f"{unstable_disp:.3f} um" if unstable_disp else "未检测到"), flush=True)
print(f"初始预制裂纹长  : {crack_len[0]:.3f}  (修正1后应≈理论值0.5)", flush=True)
print(f"最终裂纹长度    : {crack_len[-1]:.3f}", flush=True)
print(f"最终CMOD张开量  : {cmod[-1]*1e3:.3f} um  (修正2: 物理张开位移, 非带宽)", flush=True)
print(f"断裂韧性K_c     : {K_c:.1f} MPa·mm^0.5  (修正3方法B: 扩展时G=Gc)", flush=True)
print(f"最终曲折度      : {tort[-1]:.3f}", flush=True)
print(f"最终分叉点数    : {n_branch[-1]}   独立裂纹段: {n_segment[-1]}", flush=True)
print(f"最终扩展方向角  : {direction[-1]:.1f}° (0=水平,负=向下)", flush=True)
print(f"裂纹等级(最终)  : {final_sev}", flush=True)
print("="*52, flush=True)

# ---- 7. 出图 A: 特征时间序列 ----
fig, ax = plt.subplots(2, 3, figsize=(13, 7.5))
ax[0,0].plot(um, crack_len, label="total"); ax[0,0].plot(um, net_len, "--", label="net")
ax[0,0].set_title("crack length"); ax[0,0].set_xlabel("disp (um)"); ax[0,0].legend(); ax[0,0].grid(True)
ax[0,1].plot(um, tip_x, "C3"); ax[0,1].set_title("crack tip x"); ax[0,1].set_xlabel("disp (um)"); ax[0,1].grid(True)
ax[0,2].plot(um, growth_rate, "C2"); ax[0,2].set_title("growth rate dL/d(disp)"); ax[0,2].set_xlabel("disp (um)"); ax[0,2].grid(True)
ax[1,0].plot(um, K_I, "C4", label="K_I (LEFM trend)")
ax[1,0].axhline(K_c, ls="--", c="k", label=f"K_c=√(E'Gc)={K_c:.0f}")
ax[1,0].axvline(peak_disp, ls=":", c="r"); ax[1,0].set_title("stress intensity K (method B)")
ax[1,0].set_xlabel("disp (um)"); ax[1,0].legend(fontsize=8); ax[1,0].grid(True)
ax[1,1].plot(um, cmod*1e3, "C5"); ax[1,1].set_title("crack opening disp CMOD")
ax[1,1].set_xlabel("disp (um)"); ax[1,1].set_ylabel("CMOD (um)"); ax[1,1].grid(True)
ax[1,2].plot(um, tort, "C6"); ax[1,2].set_title("tortuosity"); ax[1,2].set_xlabel("disp (um)"); ax[1,2].grid(True)
fig.tight_layout(); fig.savefig(f"{OUT_DIR}/crack_features.png", dpi=120); plt.close(fig)

# ---- 8. 出图 B: 路径 + 带标签的载荷曲线 ----
Dfin = to_grid(d_all[-1])
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.5))
a1.contourf(xg, xg, Dfin, levels=np.linspace(0,1,21), cmap="inferno")
a1.set_aspect("equal"); a1.set_title(f"final crack  (severity={final_sev}, dir={direction[-1]:.0f}deg)")
a1.set_xlabel("x"); a1.set_ylabel("y")
a2.plot(um, force, "-")
a2.plot(peak_disp, peak_force, "r*", ms=12, label=f"peak {peak_force:.0f}N")
if init_disp: a2.axvline(init_disp, ls="--", c="g", label=f"initiation {init_disp:.2f}um")
if unstable_disp: a2.axvline(unstable_disp, ls=":", c="r", label=f"unstable {unstable_disp:.2f}um")
a2.set_xlabel("disp (um)"); a2.set_ylabel("force (N)"); a2.set_title("labeled load-displacement")
a2.legend(); a2.grid(True)
fig.tight_layout(); fig.savefig(f"{OUT_DIR}/crack_path_labels.png", dpi=120); plt.close(fig)

# ---- 9. 存全特征 CSV (每步一行, 可直接喂阶段三神经网络) ----
import csv
with open(f"{OUT_DIR}/crack_features.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["disp_um","force_N","crack_length","net_growth","cmod_um","tortuosity",
                "n_branch","n_segment","tip_x","tip_y","direction_deg","K_I","K_c",
                "growth_rate","initiated","severity","unstable"])
    for k in range(nstep):
        w.writerow([f"{um[k]:.4f}", f"{force[k]:.4f}", f"{crack_len[k]:.5f}",
                    f"{net_len[k]:.5f}", f"{cmod[k]*1e3:.5f}", f"{tort[k]:.4f}",
                    n_branch[k], n_segment[k], f"{tip_x[k]:.4f}", f"{tip_y[k]:.4f}",
                    f"{direction[k]:.2f}", f"{K_I[k]:.3f}", f"{K_c:.3f}", f"{growth_rate[k]:.4f}",
                    int(initiated[k]), severity[k], int(unstable[k])])
print("分析完成. 结果在", OUT_DIR, flush=True)
