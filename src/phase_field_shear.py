# -*- coding: utf-8 -*-
"""
复刻 Miehe 2010 · mode-II 剪切 SENT 基准
========================================
和 phase_field_sent.py 同一套相场模型, 仅边界条件改为【水平剪切】:
  底边固定; 顶边施加水平位移 u_x (u_y=0)。
预期: 预制裂纹尖端起裂, 裂纹【斜向右下约 60-70°】扩展 (Miehe 2010 标志性结果)。
输出 -> results/shear/
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
import dolfinx.mesh as dmesh

OUT_DIR = "/mnt/c/Users/Liu/desktop/裂纹分析预测/results/shear"
os.makedirs(OUT_DIR, exist_ok=True)

# 材料/相场参数 (同 Miehe 基准)
E_lambda, E_mu = 121.15e3, 80.77e3
K_bulk = E_lambda + E_mu
Gc, ell, k_res = 2.7, 0.02, 1.0e-6

N = 150
domain = mesh.create_unit_square(MPI.COMM_WORLD, N, N, mesh.CellType.quadrilateral)
V_u = fem.functionspace(domain, ("Lagrange", 1, (2,)))
V_d = fem.functionspace(domain, ("Lagrange", 1))
u, d = fem.Function(V_u), fem.Function(V_d)
d_old, d_lb, H = fem.Function(V_d), fem.Function(V_d), fem.Function(V_d)

def eps(v): return ufl.sym(ufl.grad(v))
def g(dm): return (1.0 - dm)**2 + k_res
def pos(x): return 0.5*(x + abs(x))
def sigma(v, dm):                                    # hybrid 各向同性退化
    e = eps(v)
    return g(dm)*(E_lambda*ufl.tr(e)*ufl.Identity(2) + 2.0*E_mu*e)
def psi_pos(v):                                      # Amor 拉伸能 -> 驱动开裂
    e = eps(v); tr = ufl.tr(e); ed = e - 0.5*tr*ufl.Identity(2)
    return 0.5*K_bulk*pos(tr)**2 + E_mu*ufl.inner(ed, ed)

# ---- 边界条件: 剪切 ----
def bottom(x): return np.isclose(x[1], 0.0)
def top(x):    return np.isclose(x[1], 1.0)
dofs_bot = fem.locate_dofs_geometrical(V_u, bottom)
dofs_top = fem.locate_dofs_geometrical(V_u, top)
u_bot = fem.Constant(domain, np.array([0.0, 0.0], dtype=PETSc.ScalarType))
u_top = fem.Constant(domain, np.array([0.0, 0.0], dtype=PETSc.ScalarType))  # [ux, uy], ux每步更新, uy=0
bc_u = [fem.dirichletbc(u_bot, dofs_bot, V_u), fem.dirichletbc(u_top, dofs_top, V_u)]

# 预制裂纹: H 场播种 (细线)
def precrack_line(x):
    return np.logical_and(np.abs(x[1]-0.5) < 0.75/N, x[0] <= 0.5+1e-12)
H.x.array[fem.locate_dofs_geometrical(V_d, precrack_line)] = 1.0e4

# ---- 子问题 ----
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
psi_expr = fem.Expression(psi_pos(u), V_d.element.interpolation_points)
psi_func = fem.Function(V_d)

# 顶边水平反力 = ∫_top sigma_xy ds
fdim = domain.topology.dim - 1
top_facets = dmesh.locate_entities_boundary(domain, fdim, top)
mt = dmesh.meshtags(domain, fdim, np.sort(top_facets), np.full(len(top_facets), 1, dtype=np.int32))
ds_top = ufl.Measure("ds", domain=domain, subdomain_data=mt)(1)
reaction_form = fem.form(sigma(u, d)[0, 1] * ds_top)

coords = V_d.tabulate_dof_coordinates()
xs, ys = coords[:, 0], coords[:, 1]
def save_damage(step, disp):
    fig, ax = plt.subplots(figsize=(5, 4.5))
    tcf = ax.tricontourf(xs, ys, d.x.array, levels=np.linspace(0, 1, 21), cmap="inferno")
    fig.colorbar(tcf, ax=ax, label="damage d")
    ax.set_aspect("equal"); ax.set_title(f"shear step {step:03d}  u_x={disp*1e3:.2f}um")
    ax.set_xlabel("x"); ax.set_ylabel("y"); fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/shear_{step:03d}.png", dpi=120); plt.close(fig)

# ---- 加载 (水平剪切位移) ----
n_steps, du_step, max_stag = 200, 1.0e-4, 6
disp_hist, force_hist = [], []
print(f"剪切加载: {n_steps}步, du={du_step}mm", flush=True)
for step in range(1, n_steps+1):
    disp = step*du_step
    u_top.value[0] = disp                         # 顶边水平位移 (uy=0)
    for it in range(max_stag):
        d_old.x.array[:] = d.x.array
        prob_u.solve()
        psi_func.interpolate(psi_expr)
        H.x.array[:] = np.maximum(H.x.array, psi_func.x.array)
        prob_d.solve()
        d.x.array[:] = np.maximum(d.x.array, d_lb.x.array)
        if np.max(np.abs(d.x.array - d_old.x.array)) < 1e-3:
            break
    d_lb.x.array[:] = d.x.array
    fy = MPI.COMM_WORLD.allreduce(fem.assemble_scalar(reaction_form), op=MPI.SUM)
    disp_hist.append(disp); force_hist.append(fy)
    if step % 20 == 0 or step == 1:
        print(f"step {step:3d}  ux={disp*1e3:5.2f}um  Fx={fy:9.3f}N  d_max={d.x.array.max():.3f}", flush=True)
    if step % 10 == 0:
        save_damage(step, disp)

fig, ax = plt.subplots(figsize=(5, 4))
ax.plot(np.array(disp_hist)*1e3, force_hist, "-")
ax.set_xlabel("shear disp u_x (um)"); ax.set_ylabel("shear reaction Fx (N)")
ax.set_title("mode-II shear load-displacement"); ax.grid(True)
fig.tight_layout(); fig.savefig(f"{OUT_DIR}/shear_load_disp.png", dpi=120)
print("完成. 结果在", OUT_DIR, flush=True)
