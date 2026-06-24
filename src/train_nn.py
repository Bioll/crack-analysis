# -*- coding: utf-8 -*-
"""
阶段3 · 神经网络: 预测疲劳剩余寿命 RUL
=====================================
输入: a_mm(当前裂纹长), sigma_max(循环应力), dK(应力强度幅)
输出: RUL (剩余循环数)  —— 目标取 log1p (跨度大)
划分: 按应力档留出 {85,110,135} MPa 不训练, 测对"没见过载荷"的泛化
模型: MLP (PyTorch, CPU)
输出 -> results/nn/  +  models/rul_mlp.pt
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

ROOT = "/mnt/c/Users/Liu/desktop/裂纹分析预测"
OUT  = f"{ROOT}/results/nn"; os.makedirs(OUT, exist_ok=True)
os.makedirs(f"{ROOT}/models", exist_ok=True)
torch.manual_seed(0); np.random.seed(0)

# ---- 1. 读数据 ----
df = pd.read_csv(f"{ROOT}/data/fatigue_ml_dataset.csv")
HELDOUT = [85.0, 110.0, 135.0]                       # 测试集: 这几档不参与训练
test_mask = df["sigma_max_MPa"].isin(HELDOUT)
feat = ["a_mm", "sigma_max_MPa", "dK"]

Xtr = df.loc[~test_mask, feat].values.astype(np.float32)
Xte = df.loc[ test_mask, feat].values.astype(np.float32)
ytr = np.log1p(df.loc[~test_mask, "RUL_cycles"].values.astype(np.float32))
yte = np.log1p(df.loc[ test_mask, "RUL_cycles"].values.astype(np.float32))
print(f"训练 {len(Xtr)} 样本, 测试 {len(Xte)} 样本 (留出应力档 {HELDOUT})", flush=True)

# ---- 2. 标准化 (用训练集统计量) ----
xm, xs = Xtr.mean(0), Xtr.std(0) + 1e-8
ym, ys = ytr.mean(), ytr.std() + 1e-8
def nx(X): return (X - xm) / xs
def ny(y): return (y - ym) / ys
Xtr_t = torch.tensor(nx(Xtr)); ytr_t = torch.tensor(ny(ytr)).reshape(-1, 1)
Xte_t = torch.tensor(nx(Xte))

# ---- 3. MLP ----
model = nn.Sequential(
    nn.Linear(3, 64), nn.ReLU(),
    nn.Linear(64, 64), nn.ReLU(),
    nn.Linear(64, 32), nn.ReLU(),
    nn.Linear(32, 1))
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
lossf = nn.MSELoss()

# ---- 4. 训练 ----
n_epoch, bs = 300, 512
N = len(Xtr_t); hist = []
for ep in range(n_epoch):
    perm = torch.randperm(N)
    tot = 0.0
    for i in range(0, N, bs):
        idx = perm[i:i+bs]
        opt.zero_grad()
        loss = lossf(model(Xtr_t[idx]), ytr_t[idx])
        loss.backward(); opt.step()
        tot += loss.item() * len(idx)
    hist.append(tot / N)
    if (ep+1) % 50 == 0:
        print(f"epoch {ep+1:3d}  train MSE(std) {hist[-1]:.4f}", flush=True)

# ---- 5. 评估 (反标准化回真实 RUL) ----
model.eval()
with torch.no_grad():
    pred_te = model(Xte_t).numpy().flatten() * ys + ym
pred_rul = np.expm1(pred_te)                          # log1p 逆变换
true_rul = np.expm1(yte)

r2  = 1 - np.sum((pred_rul-true_rul)**2) / np.sum((true_rul-true_rul.mean())**2)
mae = np.mean(np.abs(pred_rul - true_rul))
# 相对误差只在寿命较长(>1000循环)处算, 且取中位数 (临近断裂RUL->0会让相对误差失真)
sel = true_rul > 1000
mre = np.median(np.abs(pred_rul[sel] - true_rul[sel]) / true_rul[sel]) * 100
print(f"[测试集·未见载荷]  R²={r2:.4f}  MAE={mae:.0f} cyc  中位相对误差(RUL>1000)={mre:.1f}%", flush=True)

# ---- 6. 出图 ----
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.5))
a1.semilogy(hist); a1.set_xlabel("epoch"); a1.set_ylabel("train MSE (std)")
a1.set_title("training curve"); a1.grid(True, which="both")
lim = [0, true_rul.max()*1.05]
a2.scatter(true_rul, pred_rul, s=4, alpha=0.3)
a2.plot(lim, lim, "r--", label="ideal")
a2.set_xlabel("true RUL (cycles)"); a2.set_ylabel("predicted RUL")
a2.set_title(f"held-out loads  R²={r2:.3f}"); a2.legend(); a2.grid(True)
fig.tight_layout(); fig.savefig(f"{OUT}/rul_prediction.png", dpi=120); plt.close(fig)

# 单条曲线: 取一个留出应力档, 看 RUL 随裂纹长的预测 vs 真实
s0 = HELDOUT[1]
sub = df[(df["sigma_max_MPa"] == s0) & (df["a0_mm"] == 5.0)].sort_values("a_mm")
if len(sub):
    Xs = torch.tensor(nx(sub[feat].values.astype(np.float32)))
    with torch.no_grad():
        ps = np.expm1(model(Xs).numpy().flatten()*ys + ym)
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.plot(sub["a_mm"], sub["RUL_cycles"], "-", label="true")
    ax.plot(sub["a_mm"], ps, "--", label="NN predict")
    ax.set_xlabel("crack length a (mm)"); ax.set_ylabel("RUL (cycles)")
    ax.set_title(f"RUL vs crack @ σmax={s0:.0f}MPa (unseen)"); ax.legend(); ax.grid(True)
    fig.tight_layout(); fig.savefig(f"{OUT}/rul_curve_example.png", dpi=120); plt.close(fig)

torch.save(model.state_dict(), f"{ROOT}/models/rul_mlp.pt")
print("完成. 图在", OUT, " 模型在 models/rul_mlp.pt", flush=True)
