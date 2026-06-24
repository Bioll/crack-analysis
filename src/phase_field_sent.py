# -*- coding: utf-8 -*-
"""
阶段1 · 相场法生成裂纹 —— 单边缺口拉伸 (SENT) 基准算例
====================================================
模型: AT2 相场断裂模型 + 交错求解 (staggered scheme)
应变能分裂: Amor 体积-偏量分裂 (只让拉伸部分驱动开裂, 避免受压误裂)
求解器: dolfinx 0.11 (FEniCSx)

几何: 1mm x 1mm 方板, 左边中点起一道水平预制裂纹切到板心 (y=0.5, x<=0.5)
载荷: 底边固定, 顶边逐步向上拉伸位移
输出: 每隔若干加载步把损伤场 d 存成 PNG; 最后存载荷-位移曲线
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")            # 无桌面环境, 直接出图存文件
import matplotlib.pyplot as plt

from mpi4py import MPI
from petsc4py import PETSc
import ufl
import dolfinx
from dolfinx import fem, mesh
from dolfinx.fem.petsc import LinearProblem

# --------------------------------------------------------------------------
# 0. 输出目录 (放到 Windows 能直接打开的 results 文件夹)
# --------------------------------------------------------------------------
OUT_DIR = "/mnt/c/Users/Liu/desktop/裂纹分析预测/results/sent"
os.makedirs(OUT_DIR, exist_ok=True)

# --------------------------------------------------------------------------
# 1. 材料与相场参数 (Miehe 2010 经典 SENT 参数)
# --------------------------------------------------------------------------
E_lambda = 121.15e3   # 拉梅常数 lambda (MPa = N/mm^2)
E_mu     = 80.77e3    # 剪切模量 mu
Gc       = 2.7        # 断裂韧性 (N/mm)
ell      = 0.02       # 相场长度尺度 (mm), 需 >= 约2倍网格尺寸
k_res    = 1.0e-6     # 残余刚度, 防止完全断裂处刚度为0导致奇异

K_bulk = E_lambda + E_mu   # 2D 平面应变体积模量

# --------------------------------------------------------------------------
# 2. 网格与函数空间
# --------------------------------------------------------------------------
N = 150   # 每边单元数, h=0.0067, ell/h=3 (满足 h<=ell/2, 不过密)
domain = mesh.create_unit_square(MPI.COMM_WORLD, N, N, mesh.CellType.quadrilateral)

V_u = fem.functionspace(domain, ("Lagrange", 1, (2,)))   # 位移 (向量)
V_d = fem.functionspace(domain, ("Lagrange", 1))          # 损伤 (标量)

u   = fem.Function(V_u, name="displacement")
d   = fem.Function(V_d, name="damage")
d_old = fem.Function(V_d)     # 上一交错步的 d, 用于收敛判断
d_lb  = fem.Function(V_d)     # 不可逆下界 (裂纹不愈合)
H   = fem.Function(V_d)       # 历史场: 拉伸应变能的历史最大值

# --------------------------------------------------------------------------
# 3. 力学量: 应变 / Amor 分裂 / 退化应力
# --------------------------------------------------------------------------
def eps(v):
    return ufl.sym(ufl.grad(v))

def g(dmg):                              # 退化函数 (1-d)^2 + k
    return (1.0 - dmg) ** 2 + k_res

def macaulay_pos(x):
    return 0.5 * (x + abs(x))

def macaulay_neg(x):
    return 0.5 * (x - abs(x))

def sigma(v, dmg):
    # hybrid 模型: 动量方程用各向同性退化 (保持对 v 线性, 便于线性求解)
    e = eps(v)
    sig0 = E_lambda * ufl.tr(e) * ufl.Identity(2) + 2.0 * E_mu * e
    return g(dmg) * sig0

def psi_pos(v):                                      # 驱动开裂的拉伸应变能密度
    e = eps(v)
    tr = ufl.tr(e)
    e_dev = e - 0.5 * tr * ufl.Identity(2)
    return 0.5 * K_bulk * macaulay_pos(tr) ** 2 + E_mu * ufl.inner(e_dev, e_dev)

# --------------------------------------------------------------------------
# 4. 边界条件
# --------------------------------------------------------------------------
def bottom(x):  return np.isclose(x[1], 0.0)
def top(x):     return np.isclose(x[1], 1.0)

dofs_bot = fem.locate_dofs_geometrical(V_u, bottom)
dofs_top = fem.locate_dofs_geometrical(V_u, top)

u_bot = fem.Constant(domain, np.array([0.0, 0.0], dtype=PETSc.ScalarType))
u_top = fem.Constant(domain, np.array([0.0, 0.0], dtype=PETSc.ScalarType))  # uy 每步更新
bc_u = [fem.dirichletbc(u_bot, dofs_bot, V_u),
        fem.dirichletbc(u_top, dofs_top, V_u)]

# 预制裂纹【修正1】: 不再强设 d=1 宽带, 而是用历史场 H 在一条细线上播种,
# 让相场方程自己松弛出自然窄轮廓 -> 表面密度长度 ≈ 真实长度 0.5
def precrack_line(x):
    return np.logical_and(np.abs(x[1] - 0.5) < 0.75 / N, x[0] <= 0.5 + 1e-12)

seed_dofs = fem.locate_dofs_geometrical(V_d, precrack_line)
H_SEED = 1.0e4                       # 远大于 Gc/ell, 使该处 d->1
H.x.array[seed_dofs] = H_SEED
bc_d = []                            # 损伤子问题无需 Dirichlet (mass项使其正定)

# --------------------------------------------------------------------------
# 5. 变分问题 (两个线性子问题, 交错求解)
# --------------------------------------------------------------------------
# (a) 位移子问题: 给定 d, 解弹性平衡
du, vu = ufl.TrialFunction(V_u), ufl.TestFunction(V_u)
a_u = ufl.inner(sigma(du, d), eps(vu)) * ufl.dx
L_u = ufl.inner(fem.Constant(domain, np.array([0.0, 0.0], dtype=PETSc.ScalarType)), vu) * ufl.dx
prob_u = LinearProblem(a_u, L_u, bcs=bc_u, u=u,
                       petsc_options_prefix="u_",
                       petsc_options={"ksp_type": "preonly", "pc_type": "lu"})

# (b) 损伤子问题: 给定历史场 H, 解 AT2 相场方程 (线性)
#     ∫ [(2H + Gc/ell) d q + Gc*ell grad d·grad q] dx = ∫ 2H q dx
dd, qd = ufl.TrialFunction(V_d), ufl.TestFunction(V_d)
a_d = ((2.0 * H + Gc / ell) * dd * qd + Gc * ell * ufl.dot(ufl.grad(dd), ufl.grad(qd))) * ufl.dx
L_d = 2.0 * H * qd * ufl.dx
prob_d = LinearProblem(a_d, L_d, bcs=bc_d, u=d,
                       petsc_options_prefix="d_",
                       petsc_options={"ksp_type": "preonly", "pc_type": "lu"})

# 历史场更新用的投影
psi_expr = fem.Expression(psi_pos(u), V_d.element.interpolation_points)
psi_func = fem.Function(V_d)

# --------------------------------------------------------------------------
# 6. 反力计算 (顶边 y 方向反力 = ∫_top sigma_yy ds)
# --------------------------------------------------------------------------
import dolfinx.mesh as dmesh
fdim = domain.topology.dim - 1
top_facets = dmesh.locate_entities_boundary(domain, fdim, top)
mt = dmesh.meshtags(domain, fdim, np.sort(top_facets),
                    np.full(len(top_facets), 1, dtype=np.int32))
ds_top = ufl.Measure("ds", domain=domain, subdomain_data=mt)(1)
reaction_form = fem.form(sigma(u, d)[1, 1] * ds_top)

# --------------------------------------------------------------------------
# 6b. 裂纹嘴张开位移 CMOD【修正2】: 测裂纹两侧位移场 u_y 的跳变 (物理张开量),
#     而不是损伤带宽。取裂纹嘴(x=0)上下两侧 (y=0.5±off, off>ell 在损伤带外)。
# --------------------------------------------------------------------------
from dolfinx import geometry
_bbtree = geometry.bb_tree(domain, domain.topology.dim)
_off = 1.5 * ell
_p_top = np.array([[1e-6, 0.5 + _off, 0.0]])
_p_bot = np.array([[1e-6, 0.5 - _off, 0.0]])

def _eval_uy(pt):
    cand = geometry.compute_collisions_points(_bbtree, pt)
    cells = geometry.compute_colliding_cells(domain, cand, pt)
    return u.eval(pt, [cells.links(0)[0]])[1]   # u_y 分量

def cmod():
    return float(_eval_uy(_p_top) - _eval_uy(_p_bot))

# --------------------------------------------------------------------------
# 7. 绘图工具: 把损伤场存成 PNG
# --------------------------------------------------------------------------
coords = V_d.tabulate_dof_coordinates()
xs, ys = coords[:, 0], coords[:, 1]

def save_damage(step, disp):
    fig, ax = plt.subplots(figsize=(5, 4.5))
    tcf = ax.tricontourf(xs, ys, d.x.array, levels=np.linspace(0, 1, 21), cmap="inferno")
    fig.colorbar(tcf, ax=ax, label="damage  d")
    ax.set_aspect("equal"); ax.set_title(f"step {step:03d}   u_top = {disp*1e3:.3f} um")
    ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/damage_{step:03d}.png", dpi=120)
    plt.close(fig)

# --------------------------------------------------------------------------
# 8. 加载循环 (位移控制)
# --------------------------------------------------------------------------
n_steps   = 200          # du 减半, 步数增加以覆盖到断裂
du_step   = 0.5e-4        # 每步顶边位移增量 (mm), 0.05um, 裂纹走得更细腻
max_stag  = 5             # 每个加载步内最多交错迭代次数
stag_tol  = 1.0e-3        # 交错收敛容差 (||d-d_old||_inf)

disp_hist, force_hist, cmod_hist = [], [], []
d_snaps = []     # 每步的损伤场快照, 供阶段二分析

if MPI.COMM_WORLD.rank == 0:
    print(f"开始加载: {n_steps} 步, 每步 du={du_step} mm, 输出目录 {OUT_DIR}", flush=True)

for step in range(1, n_steps + 1):
    disp = step * du_step
    u_top.value[1] = disp

    # ---- 交错迭代: 交替解 u 和 d ----
    for it in range(max_stag):
        d_old.x.array[:] = d.x.array
        prob_u.solve()                         # 解位移
        psi_func.interpolate(psi_expr)         # 当前拉伸应变能
        H.x.array[:] = np.maximum(H.x.array, psi_func.x.array)   # 历史最大
        prob_d.solve()                         # 解损伤
        d.x.array[:] = np.maximum(d.x.array, d_lb.x.array)       # 不可逆
        diff = np.max(np.abs(d.x.array - d_old.x.array))
        if diff < stag_tol:
            break

    d_lb.x.array[:] = d.x.array                # 更新不可逆下界

    # ---- 反力 ----
    fy = fem.assemble_scalar(reaction_form)
    fy = MPI.COMM_WORLD.allreduce(fy, op=MPI.SUM)
    disp_hist.append(disp); force_hist.append(fy)
    cmod_hist.append(cmod())             # 裂纹嘴张开位移
    d_snaps.append(d.x.array.copy())     # 保存损伤场, 供阶段二分析

    if MPI.COMM_WORLD.rank == 0:
        print(f"step {step:3d}  u={disp*1e3:6.3f}um  F={fy:10.4f}N  "
              f"d_max={d.x.array.max():.3f}  stag_it={it+1}", flush=True)

    if step % 5 == 0 or step == 1:
        save_damage(step, disp)

# --------------------------------------------------------------------------
# 9. 载荷-位移曲线
# --------------------------------------------------------------------------
if MPI.COMM_WORLD.rank == 0:
    np.savetxt(f"{OUT_DIR}/load_displacement.csv",
               np.column_stack([disp_hist, force_hist]),
               delimiter=",", header="disp_mm,force_N", comments="")
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(np.array(disp_hist) * 1e3, force_hist, "-o", ms=3)
    ax.set_xlabel("top displacement (um)"); ax.set_ylabel("reaction force (N)")
    ax.set_title("SENT load-displacement"); ax.grid(True)
    fig.tight_layout(); fig.savefig(f"{OUT_DIR}/load_displacement.png", dpi=120)

    # 保存损伤场数据 (坐标 + 每步损伤 + 位移), 供阶段二分析脚本读取
    np.savez_compressed(f"{OUT_DIR}/fields.npz",
                        coords=coords,                       # 节点坐标 (Nnode,3)
                        d=np.array(d_snaps),                 # 损伤场 (nstep, Nnode)
                        disp=np.array(disp_hist),            # 顶边位移 (nstep,)
                        force=np.array(force_hist),          # 反力 (nstep,)
                        cmod=np.array(cmod_hist),            # 裂纹嘴张开位移 CMOD (nstep,)
                        ell=ell, N=N,
                        E_lambda=E_lambda, E_mu=E_mu, Gc=Gc)
    print("完成. 结果在", OUT_DIR, "  已保存 fields.npz 供阶段二分析", flush=True)
