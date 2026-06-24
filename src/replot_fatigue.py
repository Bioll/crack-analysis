# -*- coding: utf-8 -*-
"""用已存的 CSV 重画相场疲劳曲线 (不重跑仿真)。
   da/dN 改用连续裂纹长度 L (而非网格量化的 tip_x), 并轻度平滑, 去掉栅栏假象。"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = "/mnt/c/Users/Liu/desktop/裂纹分析预测/results/fatigue_phasefield"
arr = np.genfromtxt(f"{D}/fatigue_phasefield.csv", delimiter=",", names=True)
N, tip, L = arr["cycle"], arr["tip_x"], arr["crack_length"]

# 用连续 L 算 da/dN, 滑动平均平滑
dLdN = np.gradient(L, N)
def smooth(y, w=15):
    k = np.ones(w)/w
    return np.convolve(y, k, mode="same")
dLdN_s = smooth(np.maximum(dLdN, 0))

fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.5))
a1.plot(N, tip, "-", label="tip x")
a1.plot(N, L, "--", label="crack length L")
a1.set_xlabel("cycles N"); a1.set_ylabel("crack size")
a1.set_title("phase-field fatigue  a-N"); a1.legend(); a1.grid(True)

m = dLdN_s > 1e-6
a2.semilogy(L[m], dLdN_s[m], "-")
a2.set_xlabel("crack length L"); a2.set_ylabel("dL/dN (smoothed)")
a2.set_title("growth rate vs crack length"); a2.grid(True, which="both")
fig.tight_layout(); fig.savefig(f"{D}/fatigue_aN.png", dpi=120); plt.close(fig)
print("重画完成:", f"{D}/fatigue_aN.png")
