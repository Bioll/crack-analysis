# -*- coding: utf-8 -*-
"""
复刻 Bourdin/Francfort/Marigo 变分相场 · 1D 杆 (AT2)
====================================================
变分相场断裂最基础的两个结果, 用 1D 杆 (位移控制拉伸) 复刻:
  (1) 应力-应变【软化曲线】: 弹性上升 -> 峰值 -> 软化 (含 AT2 齐次解析解对照)
  (2) 裂纹的【最优损伤轮廓】 d(x) = exp(-|x-x0|/ell)  (AT2 的 Γ-收敛裂纹剖面)
输出 -> results/bourdin1d/
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

OUT_DIR = "/mnt/c/Users/Liu/desktop/裂纹分析预测/results/bourdin1d"
os.makedirs(OUT_DIR, exist_ok=True)

E, Gc, ell, k_res = 210.0e3, 2.7, 0.05, 1.0e-6
L, nx = 1.0, 500

domain = mesh.create_interval(MPI.COMM_WORLD, nx, [0.0, L])
V_u = fem.functionspace(domain, ("Lagrange", 1))
V_d = fem.functionspace(domain, ("Lagrange", 1))
u, d = fem.Function(V_u), fem.Function(V_d)
d_old, d_lb, H = fem.Function(V_d), fem.Function(V_d), fem.Function(V_d)

def g(dm): return (1.0-dm)**2 + k_res

# 中心弱化区 (降低 Gc 10%) 触发裂纹在中点定位
xc = V_d.tabulate_dof_coordinates()[:, 0]
Gc_field = fem.Function(V_d)
Gc_field.x.array[:] = np.where(np.abs(xc-0.5) < 0.02, 0.9*Gc, Gc)

def left(x):  return np.isclose(x[0], 0.0)
def right(x): return np.isclose(x[0], L)
dofs_l = fem.locate_dofs_geometrical(V_u, left)
dofs_r = fem.locate_dofs_geometrical(V_u, right)
u_l = fem.Constant(domain, PETSc.ScalarType(0.0))
u_r = fem.Constant(domain, PETSc.ScalarType(0.0))
bc_u = [fem.dirichletbc(u_l, dofs_l, V_u), fem.dirichletbc(u_r, dofs_r, V_u)]

# 位移子问题: ((1-d)^2 E u') v' = 0
uu, vv = ufl.TrialFunction(V_u), ufl.TestFunction(V_u)
a_u = g(d)*E*ufl.grad(uu)[0]*ufl.grad(vv)[0]*ufl.dx
L_u = fem.Constant(domain, PETSc.ScalarType(0.0))*vv*ufl.dx
prob_u = LinearProblem(a_u, L_u, bcs=bc_u, u=u, petsc_options_prefix="u_",
                       petsc_options={"ksp_type": "preonly", "pc_type": "lu"})

# 损伤子问题 (AT2): ((2H + Gc/ell) d q + Gc ell d' q') = 2H q
dd, qd = ufl.TrialFunction(V_d), ufl.TestFunction(V_d)
a_d = ((2.0*H + Gc_field/ell)*dd*qd + Gc_field*ell*ufl.grad(dd)[0]*ufl.grad(qd)[0])*ufl.dx
L_d = 2.0*H*qd*ufl.dx
prob_d = LinearProblem(a_d, L_d, bcs=[], u=d, petsc_options_prefix="d_",
                       petsc_options={"ksp_type": "preonly", "pc_type": "lu"})

# 弹性能密度 (驱动 H)
psi_expr = fem.Expression(0.5*E*ufl.grad(u)[0]**2, V_d.element.interpolation_points)
psi_func = fem.Function(V_d)
# 平均应力 form
stress_form = fem.form(g(d)*E*ufl.grad(u)[0]*ufl.dx)   # ∫σ dx, 除以 L 得平均

n_steps, deps = 240, 5.0e-5     # 应变增量
eps_hist, sig_hist = [], []
for step in range(1, n_steps+1):
    u_r.value = step*deps*L
    for it in range(8):
        d_old.x.array[:] = d.x.array
        prob_u.solve()
        psi_func.interpolate(psi_expr)
        H.x.array[:] = np.maximum(H.x.array, psi_func.x.array)
        prob_d.solve()
        d.x.array[:] = np.maximum(d.x.array, d_lb.x.array)
        if np.max(np.abs(d.x.array-d_old.x.array)) < 1e-4:
            break
    d_lb.x.array[:] = d.x.array
    sig = MPI.COMM_WORLD.allreduce(fem.assemble_scalar(stress_form), op=MPI.SUM)/L
    eps_hist.append(step*deps); sig_hist.append(sig)

eps_hist, sig_hist = np.array(eps_hist), np.array(sig_hist)

# AT2 齐次解析解: d_h = E e^2 /(E e^2 + Gc/ell), sig = (1-d_h)^2 E e
e = eps_hist
dh = E*e**2/(E*e**2 + Gc/ell)
sig_h = (1.0-dh)**2 * E * e

# ---- 图1: 应力-应变软化 ----
fig, ax = plt.subplots(figsize=(5.5, 4))
ax.plot(eps_hist, sig_hist, "-", label="numerical (1D bar)")
ax.plot(e, sig_h, "--", label="AT2 homogeneous (analytic)")
ax.set_xlabel("strain"); ax.set_ylabel("stress (MPa)")
ax.set_title("Bourdin/AT2 1D bar: stress-strain softening"); ax.legend(); ax.grid(True)
fig.tight_layout(); fig.savefig(f"{OUT_DIR}/stress_strain.png", dpi=120); plt.close(fig)

# ---- 图2: 最终损伤轮廓 vs 解析 exp(-|x-0.5|/ell) ----
xs = V_d.tabulate_dof_coordinates()[:, 0]
order = np.argsort(xs)
fig, ax = plt.subplots(figsize=(5.5, 4))
ax.plot(xs[order], d.x.array[order], "-", label="numerical d(x)")
ax.plot(xs[order], np.exp(-np.abs(xs[order]-0.5)/ell), "--", label="exp(-|x-0.5|/ell)")
ax.set_xlabel("x"); ax.set_ylabel("damage d"); ax.set_xlim(0.2, 0.8)
ax.set_title("Bourdin/AT2 1D: optimal crack damage profile"); ax.legend(); ax.grid(True)
fig.tight_layout(); fig.savefig(f"{OUT_DIR}/damage_profile.png", dpi=120); plt.close(fig)

print(f"峰值应力 数值={sig_hist.max():.1f} MPa, 解析={sig_h.max():.1f} MPa", flush=True)
print("完成. 结果在", OUT_DIR, flush=True)
