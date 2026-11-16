# 推导：张量积核变量选择的坐标子问题与高效求解

本文给出 `R/tpk.R` 中实现所依据的公式。原论文核与归一化门控核写成同一形式，统一推导。

## 1. 记号

数据 $(X_i,Y_i)$，$i=1,\dots,n$，$X_i\in\mathbb R^p$。第 $m$ 个坐标的高斯矩阵

$$
(G_m)_{ij}=\exp\{-(X_{im}-X_{jm})^2/2\sigma_m^2\}.
$$

记 $J=\mathbf 1\mathbf 1^\top$，$\odot$、$\oslash$ 为逐元素乘、除。两种核都写成逐坐标因子的 Hadamard 积，每个因子对权重 $t_m$ 仿射：

$$
K_t=\bigodot_{m=1}^p F_m(t_m),\qquad F_m(t)=J+tH_m .
$$

| | 原论文核 | 门控核 |
|---|---|---|
| 因子 | $J+\lambda_m G_m$ | $(1-\rho_m)J+\rho_m G_m$ |
| $H_m$ | $G_m$ | $G_m-J$ |
| $t_m$ 的范围 $[0,U]$ | $[0,M]$ | $[0,1]$ |
| 截距 $b$ | 无（$Y$ 中心化） | 有，不惩罚 |
| 带宽 | $\sigma^2=$ 全维平方距离中位数（公开代码） | $\sigma_m=\mathrm{median}_{i<j}\lvert X_{im}-X_{jm}\rvert$ |

目标函数（$w_j$ 为权重：L2 取 1，分类取类别平衡权 $w_j=n/(2n_{Y_j})$）：

$$
\mathcal L(\alpha,b,t)=\frac1n\sum_{j=1}^n w_j\,l\{Y_j,\,b+(K_t\alpha)_j\}+\gamma_1\alpha^\top K_t\alpha+\gamma_2\sum_m t_m,\qquad 0\le t_m\le U .
$$

L2 损失 $l(y,f)=(y-f)^2$，指数损失 $l(y,f)=e^{-yf}$。原论文核的公开 MATLAB 代码用的是普通 $\ell_1$（$[0,M]$ 上截断到 $M$），本实现与之一致；论文正文中的截断惩罚 $\lambda_m I(\lambda_m<M/2)$ 在 `tpk_fit(..., truncated)` 中未使用。

## 2. 坐标子问题的显式表达式

固定 $q$。令 $W_q=\bigodot_{m\ne q}F_m(t_m)=K_t\oslash F_q(t_q)$，则

$$
K_t=W_q+t_q V_q,\qquad V_q=W_q\odot H_q .
$$

于是拟合值与 RKHS 范数都对 $t_q$ 仿射：

$$
b\mathbf 1+K_t\alpha=a_q+t_q b_q,\qquad \alpha^\top K_t\alpha=\alpha^\top W_q\alpha+t_q c_q,
$$

其中

$$
a_q=b\mathbf 1+W_q\alpha,\qquad
b_q=V_q\alpha=(W_q\odot G_q)\alpha-\delta\,W_q\alpha,\qquad
c_q=\alpha^\top b_q,\qquad
d_q=\gamma_1c_q+\gamma_2,
$$

原论文核 $\delta=0$，门控核 $\delta=1$。坐标子问题为

$$
\min_{0\le t\le U}\ h_q(t)=\frac1n\sum_j w_j\,l(Y_j,\,a_{jq}+t\,b_{jq})+d_q t .
$$

$h_q$ 是凸函数加线性项，故为区间上的一维凸问题。门控核的 $V_q$ 不一定半正定，$c_q$ 可以为负，但不影响凸性。

**L2 损失。** 记 $r_q=Y-a_q$，则

$$
t_q^{\rm new}=\Pi_{[0,U]}\left(\frac{r_q^\top b_q-n d_q/2}{\lVert b_q\rVert^2}\right).
$$

**指数损失。**

$$
h_q'(t)=-\frac1n\sum_j w_jY_jb_{jq}e^{-Y_j(a_{jq}+tb_{jq})}+d_q,\qquad
h_q''(t)=\frac1n\sum_j w_jb_{jq}^2e^{-Y_j(a_{jq}+tb_{jq})}\ge0 .
$$

若 $h_q'(0)\ge0$ 取 $0$，若 $h_q'(U)\le0$ 取 $U$，否则用带二分保护的 Newton 法求 $h_q'(t)=0$。

## 3. α 步

**L2 损失。** 平稳条件为 $K\{(K+n\gamma_1I)\alpha+b\mathbf 1-Y\}=0$ 和 $\mathbf 1^\top(Y-b\mathbf 1-K\alpha)=0$，取充分条件

$$
(K+n\gamma_1I)\alpha+b\mathbf 1=Y,\qquad \mathbf 1^\top\alpha=0 .
$$

令 $u=(K+n\gamma_1I)^{-1}Y$，$v=(K+n\gamma_1I)^{-1}\mathbf 1$（一次 Cholesky 两个右端），则

$$
b=\frac{\mathbf 1^\top u}{\mathbf 1^\top v},\qquad \alpha=u-bv .
$$

无截距时 $\alpha=(K+n\gamma_1I)^{-1}Y$，与论文 $(K^\top K+n\gamma_1K)^{-1}K^\top Y$ 等价。

**指数损失。** 记 $f=K\alpha+b\mathbf 1$，$u_j=w_je^{-Y_jf_j}$，$D=\mathrm{diag}(u)$，$s=-n^{-1}Y\odot u+2\gamma_1\alpha$。Newton 方程左乘 $K$ 的公因子消去后为 $(n+1)$ 维线性方程组

$$
\begin{pmatrix} n^{-1}DK+2\gamma_1I & n^{-1}D\mathbf 1\\ n^{-1}\mathbf 1^\top DK & n^{-1}\mathbf 1^\top D\mathbf 1\end{pmatrix}
\begin{pmatrix}\Delta\alpha\\ \Delta b\end{pmatrix}
=\begin{pmatrix}-s\\ n^{-1}\mathbf 1^\top(Y\odot u)\end{pmatrix},
$$

配合回溯线搜索。若当前点目标值大于 $(\alpha,b)=(0,0)$ 处的值，则从 $0$ 重新开始。

## 4. 零坐标的 KKT 检验：一次 GEMM

$t_q=0$ 时 $W_q=K$。$t_q$ 保持为 0 当且仅当 $h_q'(0)\ge0$，即

$$
z^\top b_q\le n(\gamma_1c_q+\gamma_2),\qquad
z=\begin{cases}2(Y-f), & \text{L2}\\ w\odot Y\odot e^{-Y\odot f}, & \text{指数损失}\end{cases}
$$

$z^\top b_q$ 与 $c_q$ 都是 $G_q$ 的线性泛函：

$$
z^\top(K\odot G_q)\alpha=\langle G_q,\,K\odot z\alpha^\top\rangle,\qquad
\alpha^\top(K\odot G_q)\alpha=\langle G_q,\,K\odot\alpha\alpha^\top\rangle .
$$

把所有坐标的高斯矩阵按列存成 $\mathbb G=[\mathrm{vec}\,G_1,\dots,\mathrm{vec}\,G_p]\in\mathbb R^{n^2\times p}$，则所有零坐标的检验统计量由一次矩阵乘法得到：

$$
\mathbb G^\top\big[\mathrm{vec}(K\odot z\alpha^\top),\ \mathrm{vec}(K\odot\alpha\alpha^\top)\big]\in\mathbb R^{p\times2},
$$

门控核再减去 $z^\top K\alpha$ 与 $\alpha^\top K\alpha$。每个训练集（每个 CV 折）只构造一次 $\mathbb G$，整个 $(\gamma_1,\gamma_2)$ 网格共用。

## 5. 剖面目标与 L-BFGS-B（门控核的求解器）

把 $(\alpha,b)$ 极小化掉：

$$
\phi(t)=\min_{\alpha,b}\mathcal L(\alpha,b,t),\qquad 0\le t\le U .
$$

L2 损失下，由第 3 节的平稳条件，残差 $Y-b\mathbf 1-K\alpha=n\gamma_1\alpha$，于是

$$
\phi(t)=\gamma_1\,\hat\alpha(t)^\top(K_t+n\gamma_1I)\hat\alpha(t)+\gamma_2\sum_mt_m=\gamma_1Y^\top\hat\alpha(t)+\gamma_2\sum_mt_m ,
$$

无截距时即 $\gamma_1Y^\top(K_t+n\gamma_1I)^{-1}Y+\gamma_2\sum_m t_m$。因 $t\ge0$，$\ell_1$ 项是线性的，$\phi$ 在盒子上光滑。

**梯度（包络定理）。** 在 $(\hat\alpha,\hat b)$ 处，

$$
\frac{\partial\phi}{\partial t_q}=\Big\langle\frac{\partial K_t}{\partial t_q},\ \gamma_1\hat\alpha\hat\alpha^\top-n^{-1}z\hat\alpha^\top\Big\rangle+\gamma_2
=\big\langle H_q\oslash F_q(t_q),\ \Omega\big\rangle+\gamma_2,
\qquad \Omega=K_t\odot(\gamma_1\hat\alpha\hat\alpha^\top-n^{-1}z\hat\alpha^\top),
$$

这里用了 $\partial K_t/\partial t_q=W_q\odot H_q=K_t\odot H_q\oslash F_q$。L2 损失下 $z=2n\gamma_1\hat\alpha$，$\Omega=-\gamma_1K_t\odot\hat\alpha\hat\alpha^\top$，梯度为 $-\gamma_1c_q+\gamma_2$，与第 2 节一致。

- 活跃坐标：每个 $O(n^2)$。
- 零坐标（$F_q=J$）：$\nabla_q\phi=\langle G_q,\Omega\rangle-\delta\langle J,\Omega\rangle+\gamma_2$，全部 $p$ 个由一次 $\mathbb G^\top\mathrm{vec}(\Omega)$ 得到。

**工作集算法。**

1. 在当前 $t$ 处求 $(\hat\alpha,\hat b)$ 与 $\Omega$，用一次 GEMM 计算所有零坐标的梯度。
2. 违反 KKT 的零坐标（$\nabla_q\phi<-\epsilon$）中取最多 `add_max`（默认 10）个加入工作集 $A$；若无违反者，停止。
3. 在 $A$ 上用 L-BFGS-B（`optim`，界 $[0,U]$）极小化 $\phi$；每次函数求值为一次 Cholesky（L2）或收敛的 Newton（指数损失）加 $|A|$ 个 $O(n^2)$ 运算。
4. 删去取值为 0 的坐标，回到 1。

终止点满足所有坐标的 KKT 条件。在交叉验证中，对每个 $\gamma_1$，$\gamma_2$ 从大到小走路径并用上一个解热启动；活跃坐标数超过 `dfmax`（默认 30）时该路径上更小的 $\gamma_2$ 记为 CV 误差 $+\infty$。

L2 损失下，$K\mapsto Y^\top(K+cI)^{-1}Y$ 是矩阵凸函数，$K_t$ 对每个 $t_q$ 仿射，故 $\phi$ 沿每个坐标是凸的。

## 6. 原论文核的求解器

原论文核使用公开代码的交替算法：$\alpha$ 步（L2 闭式，加 $10^{-4}I$；分类一步 Newton），然后一次坐标下降扫过候选坐标；第一次扫描后为 0 的坐标永久删除；$\lvert\Delta\mathcal L\rvert<10^{-3}$ 且 $\lVert\Delta\lambda\rVert_1<10^{-3}$ 或 500 次迭代停止；每个网格点冷启动。第一次扫描先用第 4 节的 GEMM 找出违反 KKT 的坐标，再对它们按边际相关性顺序做精确的逐坐标更新。

原论文核不用第 5 节的剖面求解器。剖面求解器在 L2 测试中把论文目标降到 1.37（交替算法停在 48.6），但其解的测试 MSE 为 103（交替算法 37）：$\lambda$ 增大使 $K_t$ 的量级按 $\prod(1+\lambda_m)$ 增长，岭惩罚随之失效而插值训练数据。

## 7. 小 $\rho$ 退化方向与 relaxed CV

$\rho$ 都很小时

$$
K_\rho=J+\sum_m\rho_m(G_m-J)+O(\lVert\rho\rVert^2),
$$

即近似可加核。在截距存在、$\mathbf 1^\top\alpha=0$ 时，$(\rho,\alpha,\gamma_1,\gamma_2)\mapsto(c\rho,\alpha/c,\gamma_1/c,c\gamma_2)$ 使拟合值与目标函数在一阶上不变，于是 $(\gamma_1,\gamma_2)$ 网格在该区域只通过 $\gamma_1\gamma_2$ 起作用；按门控核拟合本身的 CV 误差调参，常落在小 $\gamma_1$、大 $\gamma_2$ 处，许多噪声变量取 $\rho\approx 10^{-2}$。

relaxed CV：对每折、每个网格点 $(\gamma_1,\gamma_2)$ 得到支撑集 $\hat A$，在 $\hat A$ 上用 $\rho_m=1$（$x_{\hat A}$ 上的乘积高斯核）加截距重新拟合，其惩罚 $\gamma_1^{\rm r}$ 取遍同一网格，记录验证误差 $E(\gamma_1,\gamma_2,\gamma_1^{\rm r},k)$。以 $\frac13\sum_k E$ 的最小点联合选取 $(\gamma_1,\gamma_2,\gamma_1^{\rm r})$。重拟合只依赖 $\hat A$，按支撑集缓存，每个不同的 $\hat A$ 每折只算 16 次。

## 8. 二阶段 refit

在选出的 $\hat A$ 上拟合带截距的核岭回归（分类为加权指数损失），核为 $\prod_{m\in\hat A}\exp\{-(x_m-x'_m)^2/2(s\sigma_m)^2\}$；$\gamma_1$ 与公共带宽倍数 $s\in\{1,2,4\}$ 用同一组 3 折 CV 选取。选出的变量集合不变，只改变预测函数。

## 9. 计算量

- 高斯矩阵库 $\mathbb G$：$n^2p$ 个 double。$n=400,p=1000$ 时 1.28 GB；CV 折（$n\approx267$）0.57 GB，各折依次构造、用完释放。
- 门控核每次函数求值 $O(n^3/3+|A|n^2)$；每次 KKT 筛选 $O(n^2p)$（一次 BLAS 调用）。
