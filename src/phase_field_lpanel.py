# -*- coding: utf-8 -*-
"""
高难案例 #1 · 复刻 L 形板 (L-shaped panel, Winkler 2001 型)
=========================================================
无预制裂纹: 裂纹从 L 的【凹角 (re-entrant corner)】应力奇异处【自行萌生】并扩展。
几何: 单位方板挖去右上 1/4 -> L 形, 凹角在 (0.5, 0.5)。
载荷: 左脚底固定; 右臂端底部向上拉 -> 凹角受拉开裂。
网格: gmsh 生成, 裂纹带(y≈0.5)加密。
输出 -> results/lpanel/
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpi4py import MPI
from petsc4py import PETSc
import ufl
import gmsh
from dolfinx import fem
from dolfinx.io import gmsh as dgmsh        # dolfinx 0.11: 接口在 dolfinx.io.gmsh (非 gmshio)
from dolfinx.fem.petsc import LinearProblem

OUT_DIR = "/mnt/c/Users/Liu/desktop/裂纹分析预测/results/lpanel"
os.makedirs(OUT_DIR, exist_ok=True)

# ---- 1. gmsh 生成 L 形网格 ----
gmsh.initialize()
gmsh.option.setNumber("General.Terminal", 0)
gmsh.model.add("Lpanel")
lc = 0.03
pts_xy = [(0, 0), (1, 0), (1, 0.5), (0.5, 0.5), (0.5, 1), (0, 1)]   # L 多边形 (凹角(0.5,0.5))
pts = [gmsh.model.geo.addPoint(x, y, 0, lc) for x, y in pts_xy]
lines = [gmsh.model.geo.addLine(pts[i], pts[(i+1) % len(pts)]) for i in range(len(pts))]
cl = gmsh.model.geo.addCurveLoop(lines)
surf = gmsh.model.geo.addPlaneSurface([cl])
gmsh.model.geo.synchronize()
# 裂纹带 y∈[0.38,0.62] 加密
fld = gmsh.model.mesh.field.add("Box")
gmsh.model.mesh.field.setNumber(fld, "VIn", 0.008)
gmsh.model.mesh.field.setNumber(fld, "VOut", 0.03)
gmsh.model.mesh.field.setNumber(fld, "XMin", 0.0); gmsh.model.mesh.field.setNumber(fld, "XMax", 1.0)
gmsh.model.mesh.field.setNumber(fld, "YMin", 0.38); gmsh.model.mesh.field.setNumber(fld, "YMax", 0.62)
gmsh.model.mesh.field.setAsBackgroundMesh(fld)
gmsh.model.addPhysicalGroup(2, [surf], 1)
gmsh.model.mesh.generate(2)
md = dgmsh.model_to_mesh(gmsh.model, MPI.COMM_WORLD, 0, gdim=2)
domain = md.mesh if hasattr(md, "mesh") else md[0]
gmsh.finalize()
print(f"L形网格: {domain.topology.index_map(2).size_local} 单元", flush=True)

# ---- 2. 相场模型 (同 hybrid 设置) ----
E_lambda, E_mu = 121.15e3, 80.77e3
Gc, ell, k_res = 2.7, 0.02, 1.0e-6
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
def psi_pos(v):
    e = eps(v); tr = ufl.tr(e); ed = e - 0.5*tr*ufl.Identity(2)
    return 0.5*(E_lambda+E_mu)*pos(tr)**2 + E_mu*ufl.inner(ed, ed)

# ---- 3. 边界条件 ----
# 整个底边固定; 拉右臂的右竖边(x=1, y<=0.5)向上 -> 剪弯臂部, 应力集中到凹角(远离固定区)
def foot(x):  return np.isclose(x[1], 0.0)
def loadp(x): return np.logical_and(np.isclose(x[0], 1.0), x[1] <= 0.5+1e-9)
dofs_foot = fem.locate_dofs_geometrical(V_u, foot)
dofs_load = fem.locate_dofs_geometrical(V_u, loadp)
u_fix = fem.Constant(domain, np.array([0.0, 0.0], dtype=PETSc.ScalarType))
u_pull = fem.Constant(domain, np.array([0.0, 0.0], dtype=PETSc.ScalarType))   # uy 每步更新
bc_u = [fem.dirichletbc(u_fix, dofs_foot, V_u), fem.dirichletbc(u_pull, dofs_load, V_u)]

# ---- 4. 子问题 ----
du_, vu = ufl.TrialFunction(V_u), ufl.TestFunction(V_u)
a_u = ufl.inner(sigma(du_, d), eps(vu))*ufl.dx
L_u = ufl.inner(fem.Constant(domain, np.array([0.0, 0.0], dtype=PETSc.ScalarType)), vu)*ufl.dx
prob_u = LinearProblem(a_u, L_u, bcs=bc_u, u=u, petsc_options_prefix="u_",
                       petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
# 禁止加载边附近开裂(强制 d=0), 否则被加载的边自身先裂; 迫使裂纹从凹角萌生
def nocrack(x): return x[0] >= 0.88
dofs_d0 = fem.locate_dofs_geometrical(V_d, nocrack)
zero_d = fem.Function(V_d)
bc_d = [fem.dirichletbc(zero_d, dofs_d0)]

dd, qd = ufl.TrialFunction(V_d), ufl.TestFunction(V_d)
a_d = ((2.0*H + Gc/ell)*dd*qd + Gc*ell*ufl.dot(ufl.grad(dd), ufl.grad(qd)))*ufl.dx
L_d = 2.0*H*qd*ufl.dx
prob_d = LinearProblem(a_d, L_d, bcs=bc_d, u=d, petsc_options_prefix="d_",
                       petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
psi_expr = fem.Expression(psi_pos(u), V_d.element.interpolation_points)
psi_func = fem.Function(V_d)

import matplotlib.tri as mtri
coords = V_d.tabulate_dof_coordinates()
xs, ys = coords[:, 0], coords[:, 1]
# 屏蔽挖掉的右上区域(凹形域): 质心落在 [0.5,1]x[0.5,1] 的三角形不画
_tri = mtri.Triangulation(xs, ys)
_cx = xs[_tri.triangles].mean(axis=1); _cy = ys[_tri.triangles].mean(axis=1)
_tri.set_mask(np.logical_and(_cx > 0.5, _cy > 0.5))
def save_damage(step, disp):
    fig, ax = plt.subplots(figsize=(5, 4.5))
    tcf = ax.tricontourf(_tri, d.x.array, levels=np.linspace(0, 1, 21), cmap="inferno")
    fig.colorbar(tcf, ax=ax, label="damage d")
    ax.set_aspect("equal"); ax.set_title(f"L-panel step {step:03d}  uy={disp*1e3:.1f}um")
    ax.set_xlabel("x"); ax.set_ylabel("y"); fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/lpanel_{step:03d}.png", dpi=120); plt.close(fig)

# ---- 5. 加载 ----
n_steps, du_step, max_stag = 250, 5.0e-5, 6
print(f"L板加载: {n_steps}步, du={du_step}mm", flush=True)
for step in range(1, n_steps+1):
    u_pull.value[1] = step*du_step
    for it in range(max_stag):
        d_old.x.array[:] = d.x.array
        prob_u.solve()
        psi_func.interpolate(psi_expr)
        H.x.array[:] = np.maximum(H.x.array, psi_func.x.array)
        prob_d.solve()
        d.x.array[:] = np.maximum(d.x.array, d_lb.x.array)
        if np.max(np.abs(d.x.array-d_old.x.array)) < 1e-3:
            break
    d_lb.x.array[:] = d.x.array
    if step % 20 == 0 or step == 1:
        print(f"step {step:3d}  uy={step*du_step*1e3:5.2f}um  d_max={d.x.array.max():.3f}", flush=True)
        save_damage(step, step*du_step)
print("完成. 结果在", OUT_DIR, flush=True)
