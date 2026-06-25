# -*- coding: utf-8 -*-
"""
高难案例 #3 · 动态裂纹分叉 (Borden et al. 2012)
===============================================
动力学相场: 动量方程加惯性项 rho*u_tt = div(sigma); 显式中心差分 + 集中质量。
预制裂纹板, 突加恒定拉伸 -> 裂纹加速 -> 达极限速度【分叉成 Y 形】。
单位: mm, N, MPa, tonne, s
输出 -> results/dynamic/
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
from dolfinx.fem.petsc import LinearProblem, assemble_vector
import dolfinx.mesh as dmesh

OUT_DIR = "/mnt/c/Users/Liu/desktop/裂纹分析预测/results/dynamic"
os.makedirs(OUT_DIR, exist_ok=True)

# ---- 材料 (Borden 2012, 玻璃/PMMA 类脆性) ----
E, nu = 32.0e3, 0.2                       # MPa
lam = E*nu/((1+nu)*(1-2*nu)); mu = E/(2*(1+nu))
rho = 2.45e-9                             # tonne/mm^3 (2450 kg/m^3)
Gc, ell, k_res = 3.0e-3, 1.0, 1.0e-6      # Gc=3 J/m^2=0.003 N/mm; ell=1mm
Lx, Ly, a0 = 100.0, 40.0, 50.0            # 板 100x40, 预制裂纹 x in [0,50] at y=20
sigma_load = 2.0                          # MPa 突加拉应力 (提高 -> 裂纹更快, 更易分叉)

nx, ny = 200, 80                          # h=0.5mm, ell/h=2
domain = mesh.create_rectangle(MPI.COMM_WORLD, [[0, 0], [Lx, Ly]], [nx, ny],
                               mesh.CellType.quadrilateral)
V_u = fem.functionspace(domain, ("Lagrange", 1, (2,)))
V_d = fem.functionspace(domain, ("Lagrange", 1))
u, u_old, u_new = fem.Function(V_u), fem.Function(V_u), fem.Function(V_u)
d, d_lb, H = fem.Function(V_d), fem.Function(V_d), fem.Function(V_d)

def eps(v): return ufl.sym(ufl.grad(v))
def g(dm): return (1.0-dm)**2 + k_res
def pos(x): return 0.5*(x+abs(x))
def sigma(v, dm):
    e = eps(v)
    return g(dm)*(lam*ufl.tr(e)*ufl.Identity(2) + 2.0*mu*e)
def psi_pos(v):
    e = eps(v); tr = ufl.tr(e); ed = e - 0.5*tr*ufl.Identity(2)
    return 0.5*(lam+mu)*pos(tr)**2 + mu*ufl.inner(ed, ed)

# 预制裂纹: H 场播种 (沿 y=Ly/2, x<=a0)
hY = Ly/ny
def precrack(x): return np.logical_and(np.abs(x[1]-Ly/2) < 0.75*hY, x[0] <= a0+1e-9)
H.x.array[fem.locate_dofs_geometrical(V_d, precrack)] = 1.0e3

# ---- 显式动力学: 集中质量 + 内/外力 ----
vv = ufl.TestFunction(V_u)
ones = fem.Function(V_u); ones.x.array[:] = 1.0
M_lump = assemble_vector(fem.form(rho*ufl.inner(ones, vv)*ufl.dx))
M_lump.assemble(); Mvals = M_lump.array.copy()

# 外力: 上边 +sigma(向上), 下边 -sigma(向下) -> mode I
def top(x):    return np.isclose(x[1], Ly)
def bot(x):    return np.isclose(x[1], 0.0)
fdim = domain.topology.dim - 1
ft = dmesh.locate_entities_boundary(domain, fdim, top)
fb = dmesh.locate_entities_boundary(domain, fdim, bot)
facets = np.concatenate([ft, fb]); marks = np.concatenate([np.full(len(ft), 1), np.full(len(fb), 2)]).astype(np.int32)
order = np.argsort(facets)
mt = dmesh.meshtags(domain, fdim, facets[order], marks[order])
ds = ufl.Measure("ds", domain=domain, subdomain_data=mt)
trac_top = fem.Constant(domain, np.array([0.0, sigma_load], dtype=PETSc.ScalarType))
trac_bot = fem.Constant(domain, np.array([0.0, -sigma_load], dtype=PETSc.ScalarType))
Fext = assemble_vector(fem.form(ufl.inner(trac_top, vv)*ds(1) + ufl.inner(trac_bot, vv)*ds(2)))
Fext.assemble(); Fext_v = Fext.array.copy()

fint_form = fem.form(ufl.inner(sigma(u, d), eps(vv))*ufl.dx)

# 损伤子问题 (AT2, 线性, CG 求解)
dd, qd = ufl.TrialFunction(V_d), ufl.TestFunction(V_d)
a_d = ((2.0*H + Gc/ell)*dd*qd + Gc*ell*ufl.dot(ufl.grad(dd), ufl.grad(qd)))*ufl.dx
L_d = 2.0*H*qd*ufl.dx
prob_d = LinearProblem(a_d, L_d, bcs=[], u=d, petsc_options_prefix="d_",
                       petsc_options={"ksp_type": "cg", "pc_type": "jacobi", "ksp_rtol": "1e-8"})
psi_expr = fem.Expression(psi_pos(u), V_d.element.interpolation_points)
psi_func = fem.Function(V_d)

coords = V_d.tabulate_dof_coordinates()
xs, ys = coords[:, 0], coords[:, 1]
def save_damage(step, t):
    fig, ax = plt.subplots(figsize=(8, 3.6))
    tcf = ax.tricontourf(xs, ys, d.x.array, levels=np.linspace(0, 1, 21), cmap="inferno")
    fig.colorbar(tcf, ax=ax, label="d")
    ax.set_aspect("equal"); ax.set_title(f"dynamic step {step}  t={t*1e6:.1f}us")
    ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)"); fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/dyn_{step:04d}.png", dpi=110); plt.close(fig)

# ---- 时间推进 (显式中心差分) ----
c_d = np.sqrt((lam+2*mu)/rho)
dt = 0.3*(Lx/nx)/c_d                       # CFL 安全系数 0.3
n_steps = 1600
print(f"动态: c_d={c_d:.3e} mm/s, dt={dt:.3e}s, {n_steps}步, 总时{n_steps*dt*1e6:.1f}us", flush=True)
u.x.array[:] = 0.0; u_old.x.array[:] = 0.0
for step in range(1, n_steps+1):
    # 位移显式更新: u_new = 2u - u_old + dt^2 * (Fext - Fint)/M
    b = assemble_vector(fint_form); b.assemble()
    acc = (Fext_v - b.array) / Mvals
    u_new.x.array[:] = 2.0*u.x.array - u_old.x.array + dt*dt*acc
    u_old.x.array[:] = u.x.array
    u.x.array[:] = u_new.x.array
    # 相场更新 (用当前 u)
    psi_func.interpolate(psi_expr)
    H.x.array[:] = np.maximum(H.x.array, psi_func.x.array)
    prob_d.solve()
    d.x.array[:] = np.maximum(d.x.array, d_lb.x.array)
    d_lb.x.array[:] = d.x.array
    if step % 100 == 0:
        tipx = xs[d.x.array > 0.9].max() if np.any(d.x.array > 0.9) else 0.0
        print(f"step {step:4d}  t={step*dt*1e6:5.1f}us  d_max={d.x.array.max():.3f}  tip_x={tipx:.1f}", flush=True)
        save_damage(step, step*dt)
print("完成. 结果在", OUT_DIR, flush=True)
