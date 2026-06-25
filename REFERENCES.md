# 参考文献与开源代码对照 (References & Cross-Validation)

本项目的相场断裂/疲劳实现并非凭空设计，而是**复刻自该领域的奠基与基准工作**。
本文件逐项列出：① 奠基/基准论文 + 我们在哪段代码实现；② 可对照的开源代码 + 与我们的方法/参数对比 + 验证结论。

---

## 一、奠基 / 基准论文 (我们复刻的对象)

| 论文 | 贡献 | 我们的实现位置 |
|------|------|----------------|
| **Francfort & Marigo (1998)**, *Revisiting brittle fracture as an energy minimization problem*, JMPS 46(8) | 断裂=能量最小化的变分思想（理论源头） | 所有相场脚本的能量泛函基础 |
| **Bourdin, Francfort, Marigo (2000/2008)**, *Numerical experiments in revisited brittle fracture* / *The variational approach to fracture* | 变分断裂的**相场正则化**（Γ-收敛），引入损伤场 d | **`src/phase_field_bourdin1d.py`** (1D杆: 软化曲线+最优损伤剖面, 自有复刻) |
| **Amor, Marigo, Maurini (2009)**, *Regularized formulation of variational brittle fracture...*, JMPS 57(8) | **体积-偏量 (Amor) 能量分裂**，受压不开裂 | **`src/phase_field_amor.py`** (拉/压不对称对比, 自有复刻) |
| **Miehe, Hofacker, Welschinger (2010)**, *A phase field model for rate-independent crack propagation...*, CMAME 199 | **交错求解 (staggered)** + **SENT 拉伸/剪切基准** + 材料参数 | **`src/phase_field_sent.py`** (mode-I拉伸) + **`src/phase_field_shear.py`** (mode-II剪切) |
| **Miehe, Welschinger, Hofacker (2010)**, *Thermodynamically consistent phase-field models...*, IJNME 83 | 历史场 H 保证不可逆、热力学一致 | `H = max(H, psi_pos)` 历史场 |
| **Ambati, Gerasimov, De Lorenzis (2015)**, *A review on phase-field models... and a new fast hybrid formulation*, Comput. Mech. 55 | **hybrid 模型**：动量方程各向同性退化(线性)，拉伸能驱动损伤 | `sigma()` 各向同性退化 |
| **Carrara, Ambati, Alessi, De Lorenzis (2020)**, *A framework to model the fatigue behavior of brittle materials...*, CMAME 361 | **相场疲劳**：累积变量 ᾱ + 疲劳退化 f(ᾱ) 削弱 Gc | `src/phase_field_fatigue.py` |
| **Paris & Erdogan (1963)**, *A critical analysis of crack propagation laws* | 疲劳裂纹扩展律 da/dN=C(ΔK)^m | `src/paris_fatigue.py` |

---

## 二、可对照 / 复刻的开源代码

| 代码 | 框架/语言 | 与我们的方法对比 | 能否在本项目环境直接跑 |
|------|-----------|------------------|------------------------|
| **saugatsn/Phase-Field-Modeling** | 老版 FEniCS (legacy dolfin) | 几乎同款：staggered + AT1/AT2 + Amor/Miehe 分裂 + 历史场 + 反力 + 能量校验 | ❌ 老版 dolfin，与我们的 FEniCSx 0.11 不兼容，需移植 |
| **MdMasiurRahaman/PhaseFieldModelForMechanicalFracture** | Julia / Gridap | 同样的 SEN 拉伸算例，AT2 | ❌ Julia，另一语言 |
| **PhaseFieldX** (IMDEA, JOSS) | FEniCSx (`pip install phasefieldx`) | 完整相场框架(脆性/疲劳/各向异性) | ✅ 基于 FEniCSx，可装进我们的 `fenicsx` 环境对照 |
| **pfm-cracks** (Heister & Wick) | deal.II / C++ | 并行自适应网格相场扩展 | ❌ C++/deal.II，重型不同栈 |

### 参数对照 (本项目 vs saugatsn vs Miehe 2010 基准)

| 参数 | Miehe 2010 SENT | saugatsn | **本项目** |
|------|-----------------|----------|-----------|
| λ (Lamé) | 121.15e3 MPa | 121153.8 | **121.15e3** |
| μ (剪切模量) | 80.77e3 MPa | 80769.2 | **80.77e3** |
| Gc (断裂能) | 2.7 N/mm | 2.7 | **2.7** |
| ℓ (长度尺度) | 0.0075–0.015 mm | 0.015 | **0.02** (配 N=150 网格) |
| 模型 | AT2 | AT1/AT2 可切 | **AT2** |
| 能量分裂 | 谱分裂(Miehe) | Miehe/Amor 可切 | **Amor + hybrid 各向同性退化** |
| 求解 | 交错 staggered | 交错 staggered | **交错 staggered** |
| 退化函数 | (1-d)² | (1-d)² | **(1-d)²+k** |

> **材料参数与 Miehe 基准完全一致**；ℓ 略大(因网格更粗)、能量分裂用 Amor+hybrid(更稳、对 v 线性)。

---

## 三、验证结论

- 本项目 SENT 结果：弹性段线性上升 → **峰值反力 ~629 N** → 裂纹失稳贯穿 → 载荷骤降归零；
- 这是 Miehe 2010 SENT 基准的**典型脆性断裂响应**，saugatsn 等开源实现复现的也是同一基准；
- 方法同属一族 (变分相场 + AT2 + staggered + hybrid)，材料参数一致 → **本项目实现与文献基准定性吻合，方法学正确**。

> 定量逐点对比 (如反力曲线完全重合) 需统一 ℓ/网格/分裂方式后专门做；
> 若需活体交叉验证，**PhaseFieldX** 基于 FEniCSx，可装进本项目 `fenicsx` 环境运行其 SENT 例子对照。

## 四、复刻状态总览 (全部用自有代码在本地跑通)

| 论文 | 复刻算例 | 脚本 | 验证 |
|------|---------|------|------|
| Miehe 2010 (mode I) | SENT 拉伸 | `phase_field_sent.py` | 峰值 629N, 水平裂 ✅ |
| Miehe 2010 (mode II) | SENT 剪切 | `phase_field_shear.py` | 斜裂 ~66° (原文~70°) ✅ |
| Amor 2009 | 拉/压不对称 | `phase_field_amor.py` | 受压抑制开裂 ✅ |
| Bourdin/FM 2000-08 | 1D 杆软化+剖面 | `phase_field_bourdin1d.py` | 峰值应力误差 0.4% ✅ |
| Ambati 2015 | hybrid 退化 | 内置各脚本 | ✅ |
| Carrara 2020 | 相场疲劳 | `phase_field_fatigue.py` | a-N 潜伏+扩展 ✅ |
| Paris 1963 | da/dN | `paris_fatigue.py` | 斜率 m=3 ✅ |

## 五、真实疲劳数据来源 (用 `src/download_real_data.py` 获取)

| 数据 | 内容 | 来源 | 说明 |
|------|------|------|------|
| **Virkler** | 68 试件裂纹扩展 a-N (2024-T3 铝) | [WarrRich/Virkler-Data](https://github.com/WarrRich/Virkler-Data) | 从 Bogdanoff&Kozin 1985 图数字化的**近似版**, 底层实验为 Virkler 1977 |
| **NIMS** | 437 条钢材疲劳强度 (成分+热处理→强度) | [qq-shu/Fatigue-Dataset](https://github.com/qq-shu/Fatigue-Dataset) | 取自 [NIMS MatNavi](https://mits.nims.go.jp/), Agrawal et al. 2014 |

> 第三方数据不随仓库分发，请遵守各自许可。NIMS 官方数据表见 https://fds.nims.go.jp/ (需注册)。

## 六、链接

- saugatsn/Phase-Field-Modeling: https://github.com/saugatsn/Phase-Field-Modeling
- MdMasiurRahaman/PhaseFieldModelForMechanicalFracture: https://github.com/MdMasiurRahaman/PhaseFieldModelForMechanicalFracture
- **PhaseFieldX** (IMDEA Materials, FEniCSx): https://github.com/CastillonMiguel/phasefieldx  ( `pip install phasefieldx` )
- pfm-cracks (Heister & Wick, deal.II), 论文: https://www.sciencedirect.com/science/article/pii/S2665963820300361
- Hirshikesh et al. 2019 (saugatsn 的基础): https://doi.org/10.1016/j.compositesb.2019.04.003
