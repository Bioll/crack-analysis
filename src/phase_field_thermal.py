# -*- coding: utf-8 -*-
"""
高难案例 #4 · 热震裂纹阵列 (Bourdin et al. 2014)
================================================
多物理耦合: 瞬态温度场(热扩散) + 热弹性(热应变) + 相场断裂。
顶边骤冷 -> 冷层受拉(平行冷边) -> 垂直冷边萌生【周期裂纹阵列】向内生长。
(无量纲参数; 微小随机 Gc 扰动以触发阵列自组织)
输出 -> results/thermal/
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

OUT_DIR = "/mnt/c/Users/Liu/desktop/裂纹分析预测/results/thermal"
os.makedirs(OUT_DIR, exist_ok=True)
np.random.seed(0)

# 无量纲材料参数
E, nu = 1.0, 0.2
lam = E*nu/((1+nu)*(1-2*nu)); mu = E/(2*(1+nu))
alpha = 1.0                              # 热膨胀系数
beta = (3*lam+2*mu)*alpha                # 热应力系数
Gc0, ell, k_res = 0.01, 0.04, 1.0e-6
Dth = 1.0                                # 热扩散系数
Lx, Ly = 4.0, 1.0                        # 宽板, 冷顶边 y=Ly
nx, ny = 200, 50                         # h=0.02, ell/h=2

domain = mesh.create_rectangle(MPI.COMM_WORLD, [[0, 0], [Lx, Ly]], [nx, ny],
                               mesh.CellType.quadrilateral)
V_u = fem.functionspace(domain, ("Lagrange", 1, (2,)))
V_d = fem.functionspace(domain, ("Lagrange", 1))
V_T = fem.functionspace(domain, ("Lagrange", 1))
u, d = fem.Function(V_u), fem.Function(V_d)
d_old, d_lb, H = fem.Function(V_d), fem.Function(V_d), fem.Function(V_d)
T, T_n = fem.Function(V_T), fem.Function(V_T)     # 温度(ΔT), 当前/上一步

# 微小随机 Gc 扰动(±3%) 触发裂纹阵列自组织
Gc_f = fem.Function(V_d)
Gc_f.x.array[:] = Gc0*(1.0 + 0.03*(np.random.rand(Gc_f.x.array.size)-0.5)*2)

def eps(v): return ufl.sym(ufl.grad(v))
def g(dm): return (1.0-dm)**2 + k_res
def pos(x): return 0.5*(x+abs(x))
def eps_e(v):                            # 弹性应变 = 总应变 - 热应变
    return eps(v) - alpha*T*ufl.Identity(2)
def sigma(v, dm):
    e = eps_e(v)
    return g(dm)*(lam*ufl.tr(e)*ufl.Identity(2) + 2.0*mu*e)
def psi_pos(v):                          # 拉伸弹性能(Amor) 驱动开裂
    e = eps_e(v); tr = ufl.tr(e); ed = e - 0.5*tr*ufl.Identity(2)
    return 0.5*(lam+mu)*pos(tr)**2 + mu*ufl.inner(ed, ed)

# ---- 温度子问题: 瞬态热扩散 (隐式欧拉) ----
dt = 2.0e-3
Tt, Tq = ufl.TrialFunction(V_T), ufl.TestFunction(V_T)
a_T = (Tt*Tq + dt*Dth*ufl.dot(ufl.grad(Tt), ufl.grad(Tq)))*ufl.dx
L_T = T_n*Tq*ufl.dx
def topcool(x): return np.isclose(x[1], Ly)
dofs_Ttop = fem.locate_dofs_geometrical(V_T, topcool)
T_cold = fem.Constant(domain, PETSc.ScalarType(-1.0))    # 顶边骤冷到 -1
bc_T = [fem.dirichletbc(T_cold, dofs_Ttop, V_T)]
prob_T = LinearProblem(a_T, L_T, bcs=bc_T, u=T, petsc_options_prefix="T_",
                       petsc_options={"ksp_type": "cg", "pc_type": "hypre"})

# ---- 位移子问题 (热弹性, 准静态) ----
# 最小约束: 底边 uy=0 (滚支, 允许自由 x 收缩) + 角点 ux=0 (防刚体平移)
# -> 板自由收缩, 拉力来自热梯度本身 (而非固定边), 裂纹才从冷顶边向下长
def bottom(x): return np.isclose(x[1], 0.0)
def corner(x): return np.logical_and(np.isclose(x[0], 0.0), np.isclose(x[1], 0.0))
Vy, _ = V_u.sub(1).collapse()
Vx, _ = V_u.sub(0).collapse()
dofs_by = fem.locate_dofs_geometrical((V_u.sub(1), Vy), bottom)
dofs_cx = fem.locate_dofs_geometrical((V_u.sub(0), Vx), corner)
zy = fem.Function(Vy); zx = fem.Function(Vx)
bc_u = [fem.dirichletbc(zy, dofs_by, V_u.sub(1)),
        fem.dirichletbc(zx, dofs_cx, V_u.sub(0))]
du_, vu = ufl.TrialFunction(V_u), ufl.TestFunction(V_u)
# 把热应变项移到右端: g(d)*C:eps(u) 内积 ... 用完整 sigma 含 T(已知)
a_u = ufl.inner(g(d)*(lam*ufl.tr(eps(du_))*ufl.Identity(2)+2*mu*eps(du_)), eps(vu))*ufl.dx
L_u = ufl.inner(g(d)*beta*T*ufl.Identity(2), eps(vu))*ufl.dx   # 热应力作为载荷
prob_u = LinearProblem(a_u, L_u, bcs=bc_u, u=u, petsc_options_prefix="u_",
                       petsc_options={"ksp_type": "preonly", "pc_type": "lu"})

# ---- 损伤子问题 ----
dd, qd = ufl.TrialFunction(V_d), ufl.TestFunction(V_d)
a_d = ((2.0*H + Gc_f/ell)*dd*qd + Gc_f*ell*ufl.dot(ufl.grad(dd), ufl.grad(qd)))*ufl.dx
L_d = 2.0*H*qd*ufl.dx
prob_d = LinearProblem(a_d, L_d, bcs=[], u=d, petsc_options_prefix="d_",
                       petsc_options={"ksp_type": "cg", "pc_type": "jacobi"})
psi_expr = fem.Expression(psi_pos(u), V_d.element.interpolation_points)
psi_func = fem.Function(V_d)

coords = V_d.tabulate_dof_coordinates(); xs, ys = coords[:, 0], coords[:, 1]
def save(step, t):
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 4.2))
    c1 = a1.tricontourf(xs, ys, T.x.array, levels=20, cmap="coolwarm")
    a1.set_aspect("equal"); a1.set_title(f"temperature  t={t:.3f}"); a1.set_ylabel("y")
    fig.colorbar(c1, ax=a1, label="dT")
    c2 = a2.tricontourf(xs, ys, d.x.array, levels=np.linspace(0,1,21), cmap="inferno")
    a2.set_aspect("equal"); a2.set_title("crack array (damage d)"); a2.set_xlabel("x"); a2.set_ylabel("y")
    fig.colorbar(c2, ax=a2, label="d")
    fig.tight_layout(); fig.savefig(f"{OUT_DIR}/thermal_{step:03d}.png", dpi=110); plt.close(fig)

# ---- 时间推进 ----
n_steps = 300
print(f"热震: dt={dt}, {n_steps}步", flush=True)
for step in range(1, n_steps+1):
    T_n.x.array[:] = T.x.array
    prob_T.solve()                       # 更新温度
    for it in range(4):                  # 交错: u <-> d
        d_old.x.array[:] = d.x.array
        prob_u.solve()
        psi_func.interpolate(psi_expr)
        H.x.array[:] = np.maximum(H.x.array, psi_func.x.array)
        prob_d.solve()
        d.x.array[:] = np.maximum(d.x.array, d_lb.x.array)
        if np.max(np.abs(d.x.array-d_old.x.array)) < 1e-3:
            break
    d_lb.x.array[:] = d.x.array
    if step % 30 == 0 or step == 1:
        ncrack = int(np.sum(d.x.array > 0.9))
        print(f"step {step:3d}  t={step*dt:.3f}  Tmin={T.x.array.min():.2f}  d_max={d.x.array.max():.3f}  ncrack_node={ncrack}", flush=True)
        save(step, step*dt)
print("完成. 结果在", OUT_DIR, flush=True)
