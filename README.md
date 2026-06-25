# 裂纹生成 · 分析 · 预测 (Crack Generation · Analysis · Prediction)

用**相场法 (phase-field)** 生成裂纹 → **提取特征并打标签** → **神经网络预测疲劳剩余寿命**的一条完整流程。

> ⚠️ **重要声明**：本项目所有数据均为**仿真/解析合成数据**，**非实验测量数据**。
> 神经网络在干净的合成数据上表现近乎完美 (R²≈0.9998)，这只说明流程跑通、模型能拟合确定性函数，
> **不代表能预测真实材料疲劳寿命**——真实疲劳数据离散性极大，难度高得多。
> 本仓库定位是**方法流程演示 (proof-of-concept)**，不是经实验验证的预测器。

---

## 流程总览

```
[阶段1 生成]            [阶段2 分析]              [阶段3 预测]
相场法仿真裂纹      →    提取几何/力学特征+标签   →   神经网络预测剩余寿命RUL
(SENT / 疲劳)           (长度/CMOD/K_c/...)         (PyTorch MLP)
```

## 目录结构

```
src/                       源代码
  phase_field_sent.py      阶段1: 相场法SENT拉伸(mode-I), 生成裂纹 + 损伤场/载荷曲线
  phase_field_shear.py     复刻 Miehe mode-II 剪切 (裂纹斜向扩展)
  phase_field_amor.py      复刻 Amor 2009 拉/压不对称 (受压抑制开裂)
  phase_field_bourdin1d.py 复刻 Bourdin/AT2 1D 杆 (软化曲线 + 最优损伤剖面)
  analyze_crack.py         阶段2: 提取裂纹长度/CMOD/曲折度/分叉/方向/K_c + 萌生/等级/失稳标签
  paris_fatigue.py         疲劳(快速): Paris 公式 da/dN, a-N 曲线 + 寿命
  phase_field_fatigue.py   疲劳(高保真): Carrara 2020 相场疲劳, 裂纹逐圈扩展
  replot_fatigue.py        相场疲劳曲线重绘工具
  fatigue_dataset_gen.py   阶段3 数据生成: 扫多工况造疲劳数据集
  train_nn.py              阶段3: 训练 MLP 预测剩余寿命 RUL
  download_real_data.py    下载真实疲劳数据 (Virkler + NIMS)
  virkler_real_data.py     真实 Virkler 数据探索 (展示疲劳散布)
  phase_field_holeplate.py 高难案例: 带孔板裂纹萌生 (gmsh挖孔)
  phase_field_dynamic.py   高难案例: 动态裂纹分叉 (惯性+显式时间积分, Borden 2012)
  phase_field_thermal.py   高难案例: 热震裂纹阵列 (热-力-相场耦合, Bourdin 2014)
  phase_field_lpanel.py    高难案例: L形板 (gmsh, 未干净复现)
  phase_field_lpanel_force.py  L形板力控版 (Neumann, 未干净复现)
results/                   结果图与小型 CSV
models/                    训练好的模型 (rul_mlp.pt)
data/                      数据集 (大文件 .gitignore, 由 fatigue_dataset_gen.py 生成)
environment_*.yml          conda 环境依赖
```

## 关键结果

| 内容 | 文件 |
|------|------|
| 相场裂纹萌生→扩展→断裂 | `results/sent/damage_*.png` |
| 载荷-位移曲线 | `results/sent/load_displacement.png` |
| 裂纹特征 + 标签 | `results/sent/analysis/` |
| Paris 疲劳 a-N / da-dN | `results/fatigue_paris/crack_growth.png` |
| 相场疲劳 a-N | `results/fatigue_phasefield/fatigue_aN.png` |
| RUL 预测精度 (R²=0.9998) | `results/nn/rul_prediction.png` |

## 环境与运行

需要 **WSL2 + conda**（FEniCSx 在 Windows 上无原生支持）。两个独立环境：

```bash
# 相场仿真环境 (FEniCSx/dolfinx + scikit-image)
conda env create -f environment_fenicsx.yml      # 或按 yml 手动装
# 神经网络环境 (PyTorch CPU)
conda env create -f environment_ml.yml

# 运行示例
conda activate fenicsx
python src/phase_field_sent.py      # 生成裂纹
python src/analyze_crack.py         # 分析裂纹
python src/paris_fatigue.py         # Paris 疲劳
python src/phase_field_fatigue.py   # 相场疲劳

conda activate ml
python src/fatigue_dataset_gen.py   # 造数据集
python src/train_nn.py              # 训练并评估
```

> 国内网络建议把 conda 频道换成清华镜像 (conda-forge)。

## 方法要点

- **相场断裂**: AT2 模型 + 交错求解 (staggered) + hybrid 各向同性退化；预制裂纹用历史场 H 播种 (自然窄轮廓)。
- **裂纹长度**: 相场表面密度积分 `∫[d²/2ℓ + ℓ/2·|∇d|²]dx`。
- **裂纹张开量**: 物理 CMOD (位移场跳变)，非损伤带宽。
- **断裂韧性**: 方法B `K_c=√(E'·Gc)`，相场扩展时强制 G=Gc。
- **疲劳**: Paris (经验/快) 与相场疲劳 (Carrara 2020, 能量退化/高保真) 两条路互验。
- **神经网络**: MLP, 输入(裂纹长, 应力, ΔK)→剩余寿命; 留出部分载荷测泛化。

## 参考文献与验证

本项目复刻自相场断裂/疲劳的奠基与基准工作（Miehe 2010、Ambati 2015、Carrara 2020、Paris 1963 等），
材料参数与 Miehe SENT 基准完全一致，方法与多个开源实现 (saugatsn、PhaseFieldX 等) 同属一族。
详见 **[REFERENCES.md](REFERENCES.md)**（含论文清单、代码对照表、参数验证）。

## 局限 (已知)

- 数据为合成数据，结果为流程演示，未经实验验证；
- 相场裂纹在对称 SENT 中会轻微偏折 (对称失稳，可接受)；
- 相场疲劳阈值 α_T 为经验标定，未对真实 da/dN 反标；
- 神经网络的高精度源于数据无噪声，真实场景难度远高于此。
