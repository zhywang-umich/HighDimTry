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

# 服务器
sbatch --array=1-300 submit_array.sh
sbatch --array=1-300 submit_array.sh methods=gated          # 只跑门控核
sbatch --array=1-300 submit_array.sh methods=orig design=reg

# 汇总
Rscript summarize.R in=results out=summary
```

- 每个 (design, n, p, seed) 存成 `results/<design>_n<n>_p<p>_seed<seed>.rds`；已存在的文件会跳过，作业被杀后直接重交即可。
- 数据 seed 只由 (seed, design, n, p) 决定，所以 `methods=gated` 和 `methods=orig` 分开跑时用的是同一份数据。但同一个 (design, n, p, seed) 文件只写一次：分开跑不同方法时请用不同的 `out=` 目录，汇总时分别汇总或把文件合并后再汇总。
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

## 测试中观察到的现象

（以下为本仓库开发时的小规模测试，非 300 次模拟的结论。）

- **原论文核 + 剖面求解器**：同一组 $(\gamma_1,\gamma_2)$ 下论文目标值从 48.6 降到 1.37，但测试 MSE 从 37 升到 103：$\lambda$ 增大把 $K$ 放大 $\prod(1+\lambda_m)$ 倍、岭惩罚失效而插值。原论文方法的结果依赖于交替算法在 500 次迭代处提前停止（多数网格点 `converged = FALSE`）。
- **全维带宽**：公开代码的 $\sigma^2$ 是全维平方距离中位数（约 $2p$），每个坐标因子 $\kappa_m\in[0.975,1]$（$p=200$）。这正是有用变量的 $\lambda$ 被推到很大的原因之一：只有 $\lambda_m(1-\kappa_m)$ 足够大时该坐标才有作用。
- **原论文核的分类**：没有公开代码，`orig` 的分类是按论文描述的重建（指数损失、类别平衡权、一步 Newton、单特征初值 $\lambda=1$）。在测试中它对 $(\gamma_1,\gamma_2)$ 极不稳定，多数网格点得到空模型；换成逐坐标带宽或收敛的 α 步都没有改善。比较分类结果时应注意这一点。
- **门控核的小 $\rho$ 退化方向**：$\rho$ 都很小时 $K_\rho\approx J+\sum_m\rho_m(G_m-J)$ 近似可加，$(\rho,\alpha,\gamma_1,\gamma_2)\to(c\rho,\alpha/c,\gamma_1/c,c\gamma_2)$ 几乎不改变拟合，CV 最小误差可能选到小 $\gamma_1$、大 $\gamma_2$、许多 $\rho\approx0.01$ 的噪声变量。`gated_relax` 用 relaxed CV 处理这一点。
