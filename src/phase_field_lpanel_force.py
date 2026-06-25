# -*- coding: utf-8 -*-
"""
高难案例 #1(力控版) · L 形板 (Winkler) —— 力控加载
===================================================
改进: 用【Neumann 牵引力】加载(不在边界硬压位移, 避免加载边自裂);
      固定左脚底, 在右臂端施加【向下】牵引力 -> 臂部下弯 -> 凹角顶纤维受拉 -> 凹角起裂。
力控临近失稳会发散, 故 d 接近 1 即停 (捕捉萌生+初始扩展)。
输出 -> results/lpanel_force/
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from mpi4py import MPI
from petsc4py import PETSc
import ufl
import gmsh
from dolfinx import fem
from dolfinx.io import gmsh as dgmsh
from dolfinx.fem.petsc import LinearProblem
import dolfinx.mesh as dmesh

OUT_DIR = "/mnt/c/Users/Liu/desktop/裂纹分析预测/results/lpanel_force"
os.makedirs(OUT_DIR, exist_ok=True)

# ---- gmsh L 形网格 ----
gmsh.initialize(); gmsh.option.setNumber("General.Terminal", 0)
gmsh.model.add("Lf")
lc = 0.03
pts_xy = [(0, 0), (1, 0), (1, 0.5), (0.5, 0.5), (0.5, 1), (0, 1)]
pts = [gmsh.model.geo.addPoint(x, y, 0, lc) for x, y in pts_xy]
lines = [gmsh.model.geo.addLine(pts[i], pts[(i+1) % len(pts)]) for i in range(len(pts))]
surf = gmsh.model.geo.addPlaneSurface([gmsh.model.geo.addCurveLoop(lines)])
gmsh.model.geo.synchronize()
fld = gmsh.model.mesh.field.add("Box")
gmsh.model.mesh.field.setNumber(fld, "VIn", 0.008); gmsh.model.mesh.field.setNumber(fld, "VOut", 0.03)
gmsh.model.mesh.field.setNumber(fld, "XMin", 0.3); gmsh.model.mesh.field.setNumber(fld, "XMax", 1.0)
gmsh.model.mesh.field.setNumber(fld, "YMin", 0.3); gmsh.model.mesh.field.setNumber(fld, "YMax", 0.62)
gmsh.model.mesh.field.setAsBackgroundMesh(fld)
gmsh.model.addPhysicalGroup(2, [surf], 1)
gmsh.model.mesh.generate(2)
md = dgmsh.model_to_mesh(gmsh.model, MPI.COMM_WORLD, 0, gdim=2)
domain = md.mesh if hasattr(md, "mesh") else md[0]
gmsh.finalize()
print(f"L形网格: {domain.topology.index_map(2).size_local} 单元", flush=True)

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

# 固定左脚底 (x<=0.5, y=0)
def foot(x): return np.logical_and(np.isclose(x[1], 0.0), x[0] <= 0.5+1e-9)
dofs_foot = fem.locate_dofs_geometrical(V_u, foot)
u_fix = fem.Constant(domain, np.array([0.0, 0.0], dtype=PETSc.ScalarType))
bc_u = [fem.dirichletbc(u_fix, dofs_foot, V_u)]

# 右臂端窄条 (x>=0.94, y=0) 施加向下牵引力 (Neumann)
fdim = domain.topology.dim - 1
def loadstrip(x): return np.logical_and(np.isclose(x[1], 0.0), x[0] >= 0.94-1e-9)
load_facets = dmesh.locate_entities_boundary(domain, fdim, loadstrip)
mt = dmesh.meshtags(domain, fdim, np.sort(load_facets), np.full(len(load_facets), 1, dtype=np.int32))
ds_load = ufl.Measure("ds", domain=domain, subdomain_data=mt)(1)
T = fem.Constant(domain, PETSc.ScalarType(0.0))           # 牵引力大小, 每步增大
trac = ufl.as_vector([0.0, -T])                           # 向下

du_, vu = ufl.TrialFunction(V_u), ufl.TestFunction(V_u)
a_u = ufl.inner(sigma(du_, d), eps(vu))*ufl.dx
L_u = ufl.dot(trac, vu)*ds_load                           # 力控: 表面牵引
prob_u = LinearProblem(a_u, L_u, bcs=bc_u, u=u, petsc_options_prefix="u_",
                       petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
# 加载尖端禁裂(d=0), 迫使裂纹只能从凹角(0.5,0.5)萌生
def nocrack(x): return x[0] >= 0.8
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

coords = V_d.tabulate_dof_coordinates()
xs, ys = coords[:, 0], coords[:, 1]
_tri = mtri.Triangulation(xs, ys)
_cx = xs[_tri.triangles].mean(axis=1); _cy = ys[_tri.triangles].mean(axis=1)
_tri.set_mask(np.logical_and(_cx > 0.5, _cy > 0.5))
def save_damage(step, T_):
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.tricontourf(_tri, d.x.array, levels=np.linspace(0, 1, 21), cmap="inferno")
    ax.set_aspect("equal"); ax.set_title(f"L-panel(force) step {step:03d}  T={T_:.0f}")
    ax.set_xlabel("x"); ax.set_ylabel("y"); fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/lpf_{step:03d}.png", dpi=120); plt.close(fig)

# ---- 力控加载: 逐步增大牵引力 T ----
n_steps, dT, max_stag = 400, 20.0, 6
print(f"L板力控: 最多{n_steps}步, dT={dT}", flush=True)
for step in range(1, n_steps+1):
    T.value = step*dT
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
    dmax = d.x.array.max()
    if step % 20 == 0 or step == 1 or dmax > 0.9:
        print(f"step {step:3d}  T={T.value:7.0f}  d_max={dmax:.3f}", flush=True)
        save_damage(step, float(T.value))
    if dmax > 0.98:                       # 凹角已开裂, 力控将失稳 -> 停
        print(f"凹角开裂 @ step {step}, T={T.value:.0f}; 停止(力控失稳)", flush=True)
        break
print("完成. 结果在", OUT_DIR, flush=True)
