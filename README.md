# 张量积核非参数变量选择：归一化门控核 vs 原论文核（R 实现）

原论文：Yu, Wang, Zeng (2023), *Biometrics* 79(2): 951–963。推导见 [`docs/derivation.md`](docs/derivation.md)。

只依赖 base R（`stats`、`grDevices`），无需额外安装包。

## 文件

| 文件 | 内容 |
|---|---|
| `R/tpk.R` | 核心算法：带宽、高斯矩阵库、坐标子问题、α 步、交替坐标下降（原论文算法）、剖面 L-BFGS-B 工作集求解器（门控核）、交叉验证、refit |
| `R/sim.R` | 论文第 3 节的两个模拟设定（回归、分类） |
| `run_one.R` | 一次运行 = 一个 seed，跑完所选的全部 (design, n, p) 组合 |
| `summarize.R` | 汇总所有 seed 的结果 |
| `submit_array.sh` | SLURM array 脚本 |

## 运行

```bash
# 单个 seed（默认：design=reg,cls；n=100,200,400；p=200,400,1000；methods=gated,orig）
Rscript run_one.R seed=17

# 只跑部分设定或方法
Rscript run_one.R seed=17 design=reg n=100 p=200 methods=gated

# 服务器：两种方法分开提交（orig 慢得多），写到同一个 results/
sbatch --array=1-300 submit_array.sh methods=gated
sbatch --array=1-300 --time=48:00:00 submit_array.sh methods=orig

# 汇总
Rscript summarize.R in=results out=summary
```

- 每个 (design, n, p, seed, method) 存成 `results/<design>_n<n>_p<p>_seed<seed>_<method>.rds`；已存在的文件会跳过，作业被杀后直接重交即可。
- 数据只由 (seed, design, n, p) 决定，不同方法可以分开提交、写到同一个目录，用的是同一份数据。
- `seed` 缺省时取 `$SLURM_ARRAY_TASK_ID`。

## 方法（`methods=`）

| 名称 | 核 | 带宽 | 求解器 | 调参 |
|---|---|---|---|---|
| `orig` | $\prod(1+\lambda_m\kappa_m)$，$\lambda\in[0,10^5]$，$\ell_1$ | 全维平方距离中位数（同公开代码） | 公开代码的交替算法（一次 CD 扫描 / α 步，0 坐标永久删除，500 次上限，冷启动） | 3 折 CV 最小误差 |
| `orig_coord` | 同上 | 逐坐标距离中位数 | 同上 | 同上（默认不跑） |
| `gated` | $\prod\{(1-\rho_m)+\rho_m\kappa_m\}$，$\rho\in[0,1]$，$\ell_1$，截距 | 逐坐标距离中位数 | 剖面目标 + L-BFGS-B 工作集 + GEMM KKT 筛选，路径热启动 | 3 折 CV 最小误差 |

每个 `gated` 运行在结果里产生 4 行：

- `gated`：按门控核拟合本身的 CV 误差取最小（与论文相同的规则）；
- `gated_relax`：relaxed CV——每个网格点的支撑集上以 $\rho=1$ 加截距重新拟合，重拟合惩罚 $\gamma_1^{\rm r}$ 取遍同一网格，三者 $(\gamma_1,\gamma_2,\gamma_1^{\rm r})$ 按重拟合的 CV 误差联合选取（推导文档第 7 节）；
- `gated_refit`、`gated_relax_refit`：变量集合分别同上两行，预测函数换成在选出变量上重新调 $\gamma_1$ 与带宽倍数 $s\in\{1,2,4\}$ 的核岭回归（分类为加权指数损失）。

网格：$\gamma_1,\gamma_2\in\{2^{-15},2^{-13},\dots,2^{15}\}$。分类的 CV 准则默认为验证集上的类别平衡指数损失（`cls_cv=exp`）；`cls_cv=misclass` 改为误分类率。

## 输出（`summarize.R`）

`summary/` 下：

- `table_selection.csv`：TPR、TNR、平均选出个数、完全正确选择比例、收敛比例、平均耗时（对应论文表 1a/2a）；
- `table_error.csv`：回归为测试集 MSE（对含噪 $Y$；另给 RMSE 均值与对真 $f$ 的 MSE），分类为误分类率；均值、中位数、MAD；
- `true_var_freq.csv`：每个真变量被选中的频率；
- `weights.csv`：选中变量上 $\lambda$（orig）或 $\rho$（gated）的分位数，按真阳性/假阳性分开，另给 $\lambda\ge M/2$、取到上界的比例；
- `all_runs.csv`：逐次结果；`missing.csv`：缺失的 seed；`plots.pdf`：误差与权重箱线图。

## 实现要点（细节见推导文档）

1. **剖面目标。** 门控核把 $(\alpha,b)$ 解析/Newton 消去，得到盒约束上的光滑目标 $\phi(t)$；由包络定理，梯度为 $\langle H_q\oslash F_q,\Omega\rangle+\gamma_2$。L-BFGS-B 达到与交替坐标下降相同的目标值和解，函数求值次数少 2–4 倍。
2. **一次 GEMM 做 KKT 筛选。** 所有零坐标的梯度是 $\mathbb G^\top\mathrm{vec}(\Omega)$，$\mathbb G$ 是 $n^2\times p$ 的高斯矩阵库，每个 CV 折构造一次、整个网格共用。
3. **工作集 + 路径热启动 + `dfmax=30`。**
4. 原论文核保持公开代码的算法，只用 GEMM 找出第一次扫描中需要更新的坐标。

## 运行时间与资源（单线程，本仓库开发环境实测）

每个 (design, n, p, method) 一次运行（含 16×16 网格 3 折 CV 与最终拟合）：

| design | n | p | `gated` | `orig` |
|---|---|---|---|---|
| reg | 100 | 200 | 7–10 s | 176 s |
| reg | 200 | 400 | 25–28 s | 585 s |
| reg | 400 | 200 | — | > 45 min（未跑完） |
| reg | 400 | 1000 | 221 s | 1888 s |
| cls | 100 | 200 | 13–19 s | 15 s |
| cls | 400 | 1000 | 260 s | 373 s（空模型） |

内存峰值约 1.5 GB（CV 阶段）；$n=400,p=1000$ 的最终拟合需要 1.28 GB 的高斯矩阵库，`submit_array.sh` 申请 6 GB。原论文核的耗时主要来自交替算法每个网格点都跑满 500 次迭代（每次一个 $n\times n$ Cholesky），且随数据波动很大；单个 seed 跑完全部 18 个设定，`gated` 约 0.5 小时，`orig` 估计 5–10 小时。

## 测试中观察到的现象

（以下为开发时的小规模测试，不是 300 次模拟的结论。）

- **小规模多 seed（reg，$n=100,p=200$，8 个 seed）**：`gated` 的 TPR 0.80、平均选出 12.9 个、测试 MSE 51；`gated_relax` 的 TPR 0.55、平均 5.8 个、TNR 0.985，`gated_relax_refit` 的 MSE 50。论文表 1 的原方法：TPR 0.57、5.5 个、RMSE 7.57（MSE 约 57）。
- **$n=200,p=400$（reg）**：两个 seed 上 `gated` 选出全部 5 个真变量（另有 6–13 个噪声变量），`gated_relax` 选出 4 个和恰好 5 个，`gated_relax_refit` 的 MSE 为 17.3 和 5.6；同一数据上 `orig` 选出 3 个（TPR 0.6），MSE 42.9。
- **$n=400,p=1000$（reg）**：`gated` 恰好选出 X1–X5，`gated_refit` 的 MSE 为 10.2。
- **分类 CV 准则**：$n=100,p=200$ 的 8 个 seed 上，误分类率准则下 `gated` 的 TPR 0.46、平均 8 个、误分类 0.368；指数损失准则下 TPR 0.50、平均 4.8 个、误分类 0.323（Bayes 误差约 0.27）。默认用指数损失。分类中 X4 单独即可达到约 0.31 的误分类率（与 X2、X3 相关），X2+X3 的模型约 0.28，CV 常在两者之间摇摆。
- **原论文核 + 剖面求解器**：同一组 $(\gamma_1,\gamma_2)$ 下论文目标值从 48.6 降到 1.37，但测试 MSE 从 37 升到 103：$\lambda$ 增大把 $K$ 放大 $\prod(1+\lambda_m)$ 倍、岭惩罚失效而插值。原论文方法的结果依赖于交替算法在 500 次迭代处停止（多数网格点 `converged = FALSE`）。
- **全维带宽**：公开代码的 $\sigma^2$ 是全维平方距离中位数（约 $2p$），每个坐标因子 $\kappa_m\in[0.975,1]$（$p=200$）。有用变量的 $\lambda$ 必须很大，$\lambda_m(1-\kappa_m)$ 才有量级。
- **原论文核的分类**：没有公开代码，`orig` 的分类是按论文描述的重建（指数损失、类别平衡权、一步 Newton、单特征初值 $\lambda=1$）。它对 $(\gamma_1,\gamma_2)$ 极不稳定，多数网格点得到空模型；换成逐坐标带宽或收敛的 α 步都没有改善。比较分类结果时应注意这一点。
- **门控核的小 $\rho$ 退化方向**：$\rho$ 都很小时 $K_\rho\approx J+\sum_m\rho_m(G_m-J)$ 近似可加，$(\rho,\alpha,\gamma_1,\gamma_2)\to(c\rho,\alpha/c,\gamma_1/c,c\gamma_2)$ 几乎不改变拟合，按门控核拟合本身调参会选到小 $\gamma_1$、大 $\gamma_2$、许多 $\rho\approx0.01$ 的噪声变量（真阳性 $\rho$ 中位数约 0.18，假阳性约 0.015）。`gated_relax` 用 relaxed CV 处理这一点。
