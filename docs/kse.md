# KSE：核逐步选择（REML + EBIC）

代码：`R/kse.R`；在 `run_one.R` 中为方法 `kse`（门控核，逐变量带宽）与 `kse_ard`（$\rho\equiv1$，乘积高斯/ARD 核）。

## 1. 出发点：权重 + $\ell_1$ 的变分结构

原论文与门控核都求解

$$
\min_{f,\,t\ge0}\ \frac1n\sum_i l\{Y_i,f(X_i)\}+\gamma_1\lVert f\rVert^2_{\mathcal H_t}+\gamma_2\sum_m t_m .
$$

门控核在 $\rho$ 都很小时，在去掉常数分量后

$$
K_\rho=J+\sum_m\rho_m\tilde G_m+O(\lVert\rho\rVert^2),\qquad \tilde G_m=G_m-J,
$$

对应 RKHS 是 $\bigoplus_m\tilde{\mathcal G}_m$，范数为 $\lVert f\rVert^2=\min_{f=\sum f_m}\sum_m\lVert f_m\rVert^2_{\tilde{\mathcal G}_m}/\rho_m$。对 $\rho_m$ 取极小：

$$
\min_{\rho_m>0}\Big\{\gamma_1\frac{\lVert f_m\rVert^2}{\rho_m}+\gamma_2\rho_m\Big\}=2\sqrt{\gamma_1\gamma_2}\,\lVert f_m\rVert,\qquad \hat\rho_m=\sqrt{\gamma_1/\gamma_2}\,\lVert f_m\rVert .
$$

所以在小权重区域，“核权重 + $\ell_1$”恰好是 COSSO / 稀疏可加模型的分量范数和惩罚，$(\gamma_1,\gamma_2)$ 只通过 $\gamma_1\gamma_2$ 起作用；原论文核在 $\lambda\to\infty$ 区域有同样的退化。由此得到三点：

1. 二维 CV 网格在退化区域只识别一维，3 折 CV 噪声大，且与 lasso 一样倾向多选（噪声变量取很小的权重）。
2. 带宽固定为中位数启发式，从不学习。
3. 若改为边际似然估计信号/噪声比 $\phi$ 而保留 $\gamma_2\sum\rho_m$，则只有 $\phi\rho_m$ 可识别，$\phi\to\infty,\rho\to0$ 使惩罚失效。

KSE 去掉 $\ell_1$ 与 $\gamma_1$：连续参数（门控、带宽、信号/噪声比）由 REML 估计，离散的支撑集由 EBIC 选择，用前向—后向逐步搜索。

## 2. 模型与 REML

$$
Y=b\mathbf 1+f+\varepsilon,\qquad \varepsilon\sim N(0,s^2I),\qquad f\sim\mathrm{GP}(0,\,s^2\phi K_S),
$$

$$
K_S(x,x')=\prod_{m\in S}\Big\{(1-\rho_m)+\rho_m\exp\Big(-e^{u_m}\frac{(x_m-x'_m)^2}{2\sigma_m^2}\Big)\Big\},
$$

$\sigma_m$ 为逐坐标距离中位数，$u_m$ 是对数逆带宽倍数，$\rho_m\in(0,1]$ 是门控（`kse_ard` 中 $\rho_m\equiv1$）。记 $A=I+\phi K_S$。把 $b$ 与 $s^2$ 剖面掉后的 REML 准则（略去常数）为

$$
-2\ell_R=(n-1)\log(r^\top A^{-1}r)+\log\det A+\log(\mathbf 1^\top A^{-1}\mathbf 1),
\qquad \hat b=\frac{\mathbf 1^\top A^{-1}Y}{\mathbf 1^\top A^{-1}\mathbf 1},\ r=Y-\hat b\mathbf 1 .
$$

REML 对协方差中加上 $cJ$ 不变，所以常数分量 $\prod(1-\rho_m)J$ 不影响准则，截距由 $\hat b$ 承担。

**梯度。** 记 $\Pi=A^{-1}-A^{-1}\mathbf 1\mathbf 1^\top A^{-1}/(\mathbf 1^\top A^{-1}\mathbf 1)$，$\alpha=A^{-1}r$，

$$
P=\Pi-\frac{n-1}{r^\top\alpha}\,\alpha\alpha^\top,\qquad
\frac{\partial(-2\ell_R)}{\partial\theta}=\Big\langle P,\frac{\partial A}{\partial\theta}\Big\rangle .
$$

对 $\log\phi$：$\langle\phi P,K_S\rangle$。记 $F_a$ 为第 $a$ 个因子，$E_a=\exp(-e^{u_a}\tilde D_a)$，$\tilde D_a=(x_{ia}-x_{ja})^2/2\sigma_a^2$，$L_a=\prod_{b\ne a}F_b$（前缀积与后缀积相乘，不做除法），则

$$
\frac{\partial(-2\ell_R)}{\partial\rho_a}=\big\langle\phi P\odot L_a,\ E_a-J\big\rangle,\qquad
\frac{\partial(-2\ell_R)}{\partial u_a}=-\rho_ae^{u_a}\big\langle\phi P\odot L_a,\ \tilde D_a\odot E_a\big\rangle .
$$

每次求值一次 Cholesky 与 $A^{-1}$（$O(n^3)$），梯度 $O(|S|n^2)$。

**得分检验（一次 GEMM）。** 对未选变量 $m$，在 $\rho_m=0,u_m=0$ 处

$$
\frac{\partial(-2\ell_R)}{\partial\rho_m}\Big|_{\rho_m=0}=\langle Q,\,G_m\rangle-\langle Q,J\rangle,\qquad Q=\phi K_S\odot P,
$$

全部 $p$ 个由 $\mathbb G^\top\mathrm{vec}(Q)$ 一次得到。$K_S$ 中已含选中变量，所以该得分衡量的是 $m$ 与已选变量的乘积结构（含交互）带来的改进。$S=\varnothing$ 时它退化为以高斯核度量的边际相关（类 HSIC）筛选。

## 3. 算法

1. **前向。** 从 $S=\varnothing$ 开始。每步用得分检验取最负的 `n_cand`（默认 5）个候选；对每个候选只优化 $(\log\phi,\rho_m,u_m)$（其他参数固定于当前值），取 $-2\ell_R$ 最小者加入 $S$，再对全部参数做 L-BFGS-B。记录路径上每个模型的 $-2\ell_R$。准则连续 2 步不改进或 $|S|=15$ 时停止。
2. **EBIC 选模型。** 

$$
\mathrm{EBIC}_\xi(S)=-2\ell_R(S)+|S|\,(k\log n+2\xi\log p),
$$

$k$ 为每个变量的参数个数（`kse`：$k=2$；`kse_ard`：$k=1$），在路径上取最小；结果分别报告 $\xi=0$（BIC）、$0.5$、$1$。
3. **后向。** 从选出的模型出发，逐个尝试删去一个变量并重新拟合，若 EBIC 下降则删去最优者，重复直至不再下降。
4. **预测。** 后验均值 $\hat b+\phi K_S(x,X)A^{-1}r$（门控与带宽为 REML 估计）。

分类用 $\pm1$ 标签上的高斯工作似然（最小二乘核分类），预测取符号。

没有 $(\gamma_1,\gamma_2)$ 网格，也没有交叉验证；整个过程只在全样本上拟合一次路径。
