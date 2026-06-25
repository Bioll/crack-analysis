# -*- coding: utf-8 -*-
"""
高难案例 #2 · 带孔板裂纹萌生 (plate with a hole)
================================================
无预制裂纹: 方板中央圆孔, 顶边上拉/底边固定。
裂纹从【孔的左右两侧】(应力集中~3倍处)【自行萌生】并水平向外扩展。
(载荷在顶/底边为均匀位移, 无突跳; 孔边集中度最高 -> 孔先裂, 避免加载边自裂)
网格: gmsh OCC 布尔挖孔, 裂纹带(y≈0.5)加密。
输出 -> results/holeplate/
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

OUT_DIR = "/mnt/c/Users/Liu/desktop/裂纹分析预测/results/holeplate"
os.makedirs(OUT_DIR, exist_ok=True)
R = 0.15   # 孔半径

# ---- gmsh OCC: 方板挖圆孔 ----
gmsh.initialize()
gmsh.option.setNumber("General.Terminal", 0)
gmsh.model.add("holeplate")
rect = gmsh.model.occ.addRectangle(0, 0, 0, 1, 1)
disk = gmsh.model.occ.addDisk(0.5, 0.5, 0, R, R)
gmsh.model.occ.cut([(2, rect)], [(2, disk)])
gmsh.model.occ.synchronize()
surfs = [t for (dim, t) in gmsh.model.occ.getEntities(2)]
fld = gmsh.model.mesh.field.add("Box")
gmsh.model.mesh.field.setNumber(fld, "VIn", 0.008)
gmsh.model.mesh.field.setNumber(fld, "VOut", 0.03)
gmsh.model.mesh.field.setNumber(fld, "XMin", 0.0); gmsh.model.mesh.field.setNumber(fld, "XMax", 1.0)
gmsh.model.mesh.field.setNumber(fld, "YMin", 0.35); gmsh.model.mesh.field.setNumber(fld, "YMax", 0.65)
gmsh.model.mesh.field.setAsBackgroundMesh(fld)
gmsh.model.addPhysicalGroup(2, surfs, 1)
gmsh.model.mesh.generate(2)
md = dgmsh.model_to_mesh(gmsh.model, MPI.COMM_WORLD, 0, gdim=2)
domain = md.mesh if hasattr(md, "mesh") else md[0]
gmsh.finalize()
print(f"带孔网格: {domain.topology.index_map(2).size_local} 单元", flush=True)

# ---- 相场模型 ----
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

def bottom(x): return np.isclose(x[1], 0.0)
def top(x):    return np.isclose(x[1], 1.0)
dofs_bot = fem.locate_dofs_geometrical(V_u, bottom)
dofs_top = fem.locate_dofs_geometrical(V_u, top)
u_bot = fem.Constant(domain, np.array([0.0, 0.0], dtype=PETSc.ScalarType))
u_top = fem.Constant(domain, np.array([0.0, 0.0], dtype=PETSc.ScalarType))
bc_u = [fem.dirichletbc(u_bot, dofs_bot, V_u), fem.dirichletbc(u_top, dofs_top, V_u)]

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

fdim = domain.topology.dim - 1
top_facets = dmesh.locate_entities_boundary(domain, fdim, top)
mt = dmesh.meshtags(domain, fdim, np.sort(top_facets), np.full(len(top_facets), 1, dtype=np.int32))
ds_top = ufl.Measure("ds", domain=domain, subdomain_data=mt)(1)
reaction_form = fem.form(sigma(u, d)[1, 1]*ds_top)

coords = V_d.tabulate_dof_coordinates()
xs, ys = coords[:, 0], coords[:, 1]
_tri = mtri.Triangulation(xs, ys)
_cx = xs[_tri.triangles].mean(axis=1); _cy = ys[_tri.triangles].mean(axis=1)
_tri.set_mask((_cx-0.5)**2 + (_cy-0.5)**2 < R**2)   # 屏蔽孔内
def save_damage(step, disp):
    fig, ax = plt.subplots(figsize=(5, 4.5))
    tcf = ax.tricontourf(_tri, d.x.array, levels=np.linspace(0, 1, 21), cmap="inferno")
    fig.colorbar(tcf, ax=ax, label="damage d")
    ax.set_aspect("equal"); ax.set_title(f"hole-plate step {step:03d}  uy={disp*1e3:.2f}um")
    ax.set_xlabel("x"); ax.set_ylabel("y"); fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/hole_{step:03d}.png", dpi=120); plt.close(fig)

n_steps, du_step, max_stag = 150, 1.0e-4, 6
disp_hist, force_hist = [], []
print(f"带孔板加载: {n_steps}步, du={du_step}mm", flush=True)
for step in range(1, n_steps+1):
    u_top.value[1] = step*du_step
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
    fy = MPI.COMM_WORLD.allreduce(fem.assemble_scalar(reaction_form), op=MPI.SUM)
    disp_hist.append(step*du_step); force_hist.append(fy)
    if step % 10 == 0 or step == 1:
        print(f"step {step:3d}  uy={step*du_step*1e3:5.2f}um  Fy={fy:8.2f}N  d_max={d.x.array.max():.3f}", flush=True)
        save_damage(step, step*du_step)

fig, ax = plt.subplots(figsize=(5, 4))
ax.plot(np.array(disp_hist)*1e3, force_hist, "-")
ax.set_xlabel("disp uy (um)"); ax.set_ylabel("reaction Fy (N)")
ax.set_title("hole-plate load-displacement"); ax.grid(True)
fig.tight_layout(); fig.savefig(f"{OUT_DIR}/hole_load_disp.png", dpi=120)
print("完成. 结果在", OUT_DIR, flush=True)
