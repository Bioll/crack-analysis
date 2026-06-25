# -*- coding: utf-8 -*-
"""
复刻 Amor, Marigo, Maurini (2009) · 拉/压不对称
================================================
要点: 没有能量分裂时, 材料在【受压】下也会出现虚假裂纹;
      用 Amor 体积-偏量分裂(只让拉伸驱动开裂)后, 受压开裂被显著抑制。
做法: 同一预制裂纹板, 对比三种情形的损伤:
  (A) 无分裂 + 受压   -> 虚假开裂 (坏)
  (B) Amor分裂 + 受压 -> 抑制开裂 (好)
  (C) Amor分裂 + 受拉 -> 正常开裂 (参照)
输出 -> results/amor/
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

OUT_DIR = "/mnt/c/Users/Liu/desktop/裂纹分析预测/results/amor"
os.makedirs(OUT_DIR, exist_ok=True)

E_lambda, E_mu = 121.15e3, 80.77e3
K_bulk = E_lambda + E_mu
Gc, ell, k_res = 2.7, 0.02, 1.0e-6
N = 100

def run(use_amor, sign, label):
    """use_amor: True=Amor分裂, False=无分裂; sign: +1拉 / -1压"""
    domain = mesh.create_unit_square(MPI.COMM_WORLD, N, N, mesh.CellType.quadrilateral)
    V_u = fem.functionspace(domain, ("Lagrange", 1, (2,)))
    V_d = fem.functionspace(domain, ("Lagrange", 1))
    u, d = fem.Function(V_u), fem.Function(V_d)
    d_old, d_lb, H = fem.Function(V_d), fem.Function(V_d), fem.Function(V_d)

    def eps(v): return ufl.sym(ufl.grad(v))
    def g(dm): return (1.0-dm)**2 + k_res
    def pos(x): return 0.5*(x+abs(x))
    def sigma(v, dm):
        e = eps(v)
        return g(dm)*(E_lambda*ufl.tr(e)*ufl.Identity(2) + 2.0*E_mu*e)
    def psi_drive(v):
        e = eps(v); tr = ufl.tr(e)
        if use_amor:                                  # Amor: 只取拉伸(正体积)+偏量
            ed = e - 0.5*tr*ufl.Identity(2)
            return 0.5*K_bulk*pos(tr)**2 + E_mu*ufl.inner(ed, ed)
        else:                                         # 无分裂: 全部弹性能都驱动
            return 0.5*E_lambda*tr**2 + E_mu*ufl.inner(e, e)

    def bottom(x): return np.isclose(x[1], 0.0)
    def top(x):    return np.isclose(x[1], 1.0)
    dofs_bot = fem.locate_dofs_geometrical(V_u, bottom)
    dofs_top = fem.locate_dofs_geometrical(V_u, top)
    u_bot = fem.Constant(domain, np.array([0.0, 0.0], dtype=PETSc.ScalarType))
    u_top = fem.Constant(domain, np.array([0.0, 0.0], dtype=PETSc.ScalarType))
    bc_u = [fem.dirichletbc(u_bot, dofs_bot, V_u), fem.dirichletbc(u_top, dofs_top, V_u)]

    def precrack_line(x):
        return np.logical_and(np.abs(x[1]-0.5) < 0.75/N, x[0] <= 0.5+1e-12)
    H.x.array[fem.locate_dofs_geometrical(V_d, precrack_line)] = 1.0e4

    du_, vu = ufl.TrialFunction(V_u), ufl.TestFunction(V_u)
    a_u = ufl.inner(sigma(du_, d), eps(vu))*ufl.dx
    L_u = ufl.inner(fem.Constant(domain, np.array([0.0, 0.0], dtype=PETSc.ScalarType)), vu)*ufl.dx
    prob_u = LinearProblem(a_u, L_u, bcs=bc_u, u=u, petsc_options_prefix="u_",
                           petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    dd, qd = ufl.TrialFunction(V_d), ufl.TestFunction(V_d)
    a_d = ((2.0*H + Gc/ell)*dd*qd + Gc*ell*ufl.dot(ufl.grad(dd), ufl.grad(qd)))*ufl.dx
    L_d = 2.0*H*qd*ufl.dx
    prob_d = LinearProblem(a_d, L_d, bcs=[], u=d, petsc_options_prefix="d_",
                           petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    psi_expr = fem.Expression(psi_drive(u), V_d.element.interpolation_points)
    psi_func = fem.Function(V_d)

    # 加载到固定位移幅值 (拉或压)
    n_steps, du_step = 70, 1.0e-4
    for step in range(1, n_steps+1):
        u_top.value[1] = sign*step*du_step
        for it in range(5):
            d_old.x.array[:] = d.x.array
            prob_u.solve()
            psi_func.interpolate(psi_expr)
            H.x.array[:] = np.maximum(H.x.array, psi_func.x.array)
            prob_d.solve()
            d.x.array[:] = np.maximum(d.x.array, d_lb.x.array)
            if np.max(np.abs(d.x.array-d_old.x.array)) < 1e-3:
                break
        d_lb.x.array[:] = d.x.array

    coords = V_d.tabulate_dof_coordinates()
    # 裂纹长度(表面密度)
    ix = np.round(coords[:,0]*N).astype(int); iy = np.round(coords[:,1]*N).astype(int)
    G = np.zeros((N+1, N+1)); G[iy, ix] = d.x.array
    gy, gx = np.gradient(G, 1.0/N, 1.0/N)
    L = np.sum(G**2/(2*ell) + 0.5*ell*(gx**2+gy**2))*(1.0/N)**2
    print(f"[{label}]  d_max={d.x.array.max():.3f}  裂纹长度={L:.3f}", flush=True)
    return coords, d.x.array.copy(), d.x.array.max(), L

cases = [(False, -1, "A: no-split + compression"),
         (True,  -1, "B: Amor-split + compression"),
         (True,  +1, "C: Amor-split + tension (ref)")]
results = [run(*c) for c in cases]

fig, axs = plt.subplots(1, 3, figsize=(14, 4.3))
for ax, (coords, dvals, dmax, L), (_, _, lab) in zip(axs, results, cases):
    ax.tricontourf(coords[:,0], coords[:,1], dvals, levels=np.linspace(0,1,21), cmap="inferno")
    ax.set_aspect("equal"); ax.set_title(f"{lab}\nLcrack={L:.2f}")
    ax.set_xlabel("x"); ax.set_ylabel("y")
fig.suptitle("Amor 2009 replication: split suppresses spurious cracking under compression")
fig.tight_layout(); fig.savefig(f"{OUT_DIR}/amor_compare.png", dpi=120)
print("完成. 结果在", OUT_DIR, flush=True)
