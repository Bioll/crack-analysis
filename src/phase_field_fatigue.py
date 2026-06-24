# -*- coding: utf-8 -*-
"""
相场疲劳 (高保真 da/dN)  ——  Carrara/Alessi/De Lorenzis 2020 框架
================================================================
机理: 循环加载下, 疲劳累积变量 ᾱ 不断累加"活性应变能";
      疲劳退化函数 f(ᾱ) 削弱断裂韧性 Gc -> 裂纹逐圈缓慢扩展。
加速: 每个循环只在峰值载荷算一次, 每圈累加一份峰值活性能 (R=0, 谷值能≈0)。

输出 -> results/fatigue_phasefield/
单位: 应力 MPa, 长度 mm
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpi4py import MPI
from petsc4py import PETSc
import ufl
from dolfinx import fem, mesh
from dolfinx.fem.petsc import LinearProblem

OUT_DIR = "/mnt/c/Users/Liu/desktop/裂纹分析预测/results/fatigue_phasefield"
os.makedirs(OUT_DIR, exist_ok=True)

# ---- 材料 / 相场参数 (与单调 SENT 一致) ----
E_lambda, E_mu = 121.15e3, 80.77e3
K_bulk = E_lambda + E_mu
Gc, ell, k_res = 2.7, 0.02, 1.0e-6

# ---- 疲劳 / 循环参数 (可调) ----
N_mesh    = 100          # 网格
u_max     = 3.8e-3       # 循环峰值位移 mm (低于单调峰值~5.25um, 才是疲劳)
alpha_T   = 2000.0       # 疲劳阈值 (标定: 每圈累加~18, 设几千 -> 百圈级潜伏)
n_cycles  = 2000         # 循环数
max_stag  = 4

# ---- 网格 / 空间 ----
domain = mesh.create_unit_square(MPI.COMM_WORLD, N_mesh, N_mesh, mesh.CellType.quadrilateral)
V_u = fem.functionspace(domain, ("Lagrange", 1, (2,)))
V_d = fem.functionspace(domain, ("Lagrange", 1))
u, d = fem.Function(V_u), fem.Function(V_d)
d_old, d_lb, H = fem.Function(V_d), fem.Function(V_d), fem.Function(V_d)
alpha_bar = fem.Function(V_d)               # 疲劳累积变量 ᾱ
f_fat = fem.Function(V_d); f_fat.x.array[:] = 1.0   # 疲劳退化函数 f(ᾱ)

def eps(v): return ufl.sym(ufl.grad(v))
def g(dm): return (1.0 - dm)**2 + k_res
def pos(x): return 0.5*(x + abs(x))
def sigma(v, dm):
    e = eps(v)
    return g(dm)*(E_lambda*ufl.tr(e)*ufl.Identity(2) + 2.0*E_mu*e)
def psi_pos(v):
    e = eps(v); tr = ufl.tr(e); ed = e - 0.5*tr*ufl.Identity(2)
    return 0.5*K_bulk*pos(tr)**2 + E_mu*ufl.inner(ed, ed)

# ---- 边界 ----
def bottom(x): return np.isclose(x[1], 0.0)
def top(x):    return np.isclose(x[1], 1.0)
dofs_bot = fem.locate_dofs_geometrical(V_u, bottom)
dofs_top = fem.locate_dofs_geometrical(V_u, top)
u_bot = fem.Constant(domain, np.array([0.0, 0.0], dtype=PETSc.ScalarType))
u_top = fem.Constant(domain, np.array([0.0, 0.0], dtype=PETSc.ScalarType))
bc_u = [fem.dirichletbc(u_bot, dofs_bot, V_u), fem.dirichletbc(u_top, dofs_top, V_u)]

# 预制裂纹: H 场播种 (自然窄轮廓)
def precrack_line(x):
    return np.logical_and(np.abs(x[1]-0.5) < 0.75/N_mesh, x[0] <= 0.5+1e-12)
H.x.array[fem.locate_dofs_geometrical(V_d, precrack_line)] = 1.0e4

# ---- 子问题 ----
du, vu = ufl.TrialFunction(V_u), ufl.TestFunction(V_u)
a_u = ufl.inner(sigma(du, d), eps(vu))*ufl.dx
L_u = ufl.inner(fem.Constant(domain, np.array([0.0, 0.0], dtype=PETSc.ScalarType)), vu)*ufl.dx
prob_u = LinearProblem(a_u, L_u, bcs=bc_u, u=u, petsc_options_prefix="u_",
                       petsc_options={"ksp_type": "preonly", "pc_type": "lu"})

# 损伤子问题: 断裂韧性被 f_fat 削弱 -> f·Gc
dd, qd = ufl.TrialFunction(V_d), ufl.TestFunction(V_d)
a_d = ((2.0*H + f_fat*Gc/ell)*dd*qd + f_fat*Gc*ell*ufl.dot(ufl.grad(dd), ufl.grad(qd)))*ufl.dx
L_d = 2.0*H*qd*ufl.dx
prob_d = LinearProblem(a_d, L_d, bcs=[], u=d, petsc_options_prefix="d_",
                       petsc_options={"ksp_type": "preonly", "pc_type": "lu"})

psi_pos_expr = fem.Expression(psi_pos(u), V_d.element.interpolation_points)          # 无退化 -> 驱动 H
psi_fat_expr = fem.Expression(g(d)*psi_pos(u), V_d.element.interpolation_points)     # 退化 -> 疲劳累积
psi_func = fem.Function(V_d)

# 裂纹长度 (表面密度积分) 用网格重排
coords = V_d.tabulate_dof_coordinates()
ix = np.round(coords[:, 0]*N_mesh).astype(int)
iy = np.round(coords[:, 1]*N_mesh).astype(int)
ng, hh = N_mesh+1, 1.0/N_mesh
def crack_length():
    G = np.zeros((ng, ng)); G[iy, ix] = d.x.array
    gy, gx = np.gradient(G, hh, hh)
    return (np.sum(G**2/(2*ell) + 0.5*ell*(gx**2+gy**2)))*hh*hh
def tip_x():
    G = np.zeros((ng, ng)); G[iy, ix] = d.x.array
    cols = np.where((G > 0.9).any(axis=0))[0]
    return cols.max()/N_mesh if cols.size else 0.0

def update_f():
    ab = alpha_bar.x.array
    f_fat.x.array[:] = np.where(ab <= alpha_T, 1.0, (2*alpha_T/(ab+alpha_T))**2)

# ---- 循环加载 ----
u_top.value[1] = u_max
cyc, a_hist, L_hist = [], [], []
print(f"开始相场疲劳: u_max={u_max*1e3}um, alpha_T={alpha_T}, {n_cycles}圈", flush=True)
for n in range(1, n_cycles+1):
    update_f()                              # 用当前 ᾱ 更新 f
    for it in range(max_stag):
        d_old.x.array[:] = d.x.array
        prob_u.solve()
        psi_func.interpolate(psi_pos_expr)
        H.x.array[:] = np.maximum(H.x.array, psi_func.x.array)
        prob_d.solve()
        d.x.array[:] = np.maximum(d.x.array, d_lb.x.array)
        if np.max(np.abs(d.x.array - d_old.x.array)) < 1e-3:
            break
    d_lb.x.array[:] = d.x.array
    # 疲劳累积: 每圈加一份峰值活性能
    psi_func.interpolate(psi_fat_expr)
    alpha_bar.x.array[:] += psi_func.x.array

    L, ax = crack_length(), tip_x()
    cyc.append(n); L_hist.append(L); a_hist.append(ax)
    if n % 25 == 0 or n == 1:
        print(f"cycle {n:5d}  a_tip={ax:.3f}  L={L:.3f}  "
              f"alpha_max={alpha_bar.x.array.max():.1f}  f_min={f_fat.x.array.min():.3f}", flush=True)
    if ax > 0.95:
        print(f"裂纹贯穿 @ cycle {n}", flush=True); break

# ---- 出图 ----
cyc, a_hist, L_hist = np.array(cyc), np.array(a_hist), np.array(L_hist)
dadN = np.gradient(a_hist, cyc)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
ax1.plot(cyc, a_hist, "-"); ax1.set_xlabel("cycles N"); ax1.set_ylabel("crack tip x")
ax1.set_title("phase-field fatigue  a-N"); ax1.grid(True)
ax2.semilogy(a_hist, np.maximum(dadN, 1e-12), "-"); ax2.set_xlabel("crack length a")
ax2.set_ylabel("da/dN"); ax2.set_title("growth rate vs crack length"); ax2.grid(True, which="both")
fig.tight_layout(); fig.savefig(f"{OUT_DIR}/fatigue_aN.png", dpi=120); plt.close(fig)

import csv
with open(f"{OUT_DIR}/fatigue_phasefield.csv", "w", newline="") as fcsv:
    w = csv.writer(fcsv); w.writerow(["cycle", "tip_x", "crack_length", "dadN"])
    for i in range(len(cyc)):
        w.writerow([cyc[i], f"{a_hist[i]:.4f}", f"{L_hist[i]:.4f}", f"{dadN[i]:.6e}"])
print("完成. 结果在", OUT_DIR, flush=True)
