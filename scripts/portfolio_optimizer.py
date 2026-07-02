import numpy as np
from scipy.stats import rankdata
from sklearn.decomposition import PCA


class LongOnlyPortfolioOptimizer:
    def __init__(
        self,
        N=5000,
        T=200,
        K_factors=10,
        lambda_risk=50.0,
        delta_ridge_ratio=0.01,
        gamma_turnover=0.001,
        target_vol=0.15,
        weight_limit=0.05,
        max_iter=300,
        alpha_clip=3.0,
    ):
        """
        纯多头量化选股优化器

        [数学模型]
        目标函数: max_w  α'w - (λ/2) w'Σw - (δ/2)||w||² - γ||w - w_prev||₁
        约束条件: 0 ≤ w_i ≤ L_max,  Σw_i ≤ 1
        其中:
          α (Alpha): 预期收益向量
          Σ (Sigma): 协方差矩阵 (通过因子模型隐式表达)
          w: 目标持仓权重向量
          w_prev: 上一期持仓权重向量
          λ (lambda_risk): 风险厌恶系数
          δ (delta_ridge): L2正则化系数, 提高数值稳定性
          γ (gamma_turnover): L1换手惩罚系数

        [核心算法]
        采用两步法:
        Step 1: FISTA加速近端梯度法求解带L1惩罚的Markowitz模型 → 得到组合"形状" (相对权重)
        Step 2: 将形状归一化到满仓(sum=1) → 按目标波动率缩放 → 最终绝对仓位
        这种解耦设计使得 λ 只控制集中度(形状), 不影响最终的绝对风险暴露。
        """
        self.N = N  # 股票池数量
        self.T = T  # 历史数据长度(如过去200个交易日)
        self.K = K_factors  # 协方差矩阵降维的因子数量(如前10个主成分)
        self.lambda_risk = lambda_risk  # λ: 风险厌恶系数, 越大组合越分散
        self.delta_ridge_ratio = delta_ridge_ratio  # δ: L2正则化占协方差对角线均值的比例
        self.gamma_turnover = gamma_turnover  # γ: 换手惩罚系数, 软阈值截断微小调仓
        self.target_vol = target_vol  # 目标年化波动率(如15%)
        self.weight_limit = weight_limit  # 单股权重上限 L_max (如5%)
        self.max_iter = max_iter  # FISTA 最大迭代次数
        self.alpha_clip = alpha_clip  # Alpha信号Z-score截断阈值(去极值)

    # ==================== Alpha 构建 ====================
    def _cross_sectional_zscore(self, S):
        """
        截面去极值与标准化
        使用中位数和MAD(中位数绝对偏差)代替均值和标准差, 抗异常值干扰。
        """
        median = np.median(S, axis=-1, keepdims=True)
        mad = np.median(np.abs(S - median), axis=-1, keepdims=True)
        mad[mad == 0] = 1.0  # 防止除以0
        return (S - median) / mad

    def _calculate_alpha(self, U_hist, S_hist, S_t):
        """
        构建 Alpha 向量 (预期收益)

        [公式] α = IC * σ * Z
        - IC: 历史平均Rank IC, 衡量信号预测能力, 决定Alpha整体正负方向
        - σ (sigma_t): 个股历史波动率, 将纯方向信号Z映射到收益率空间
        - Z (Z_t): 当期信号的截面Z-score, 决定个股间的相对强弱排序

        [关键修复]
        Alpha必须在收益率空间(数值极小, 如0.001), 与协方差矩阵(如0.0001)同尺度。
        若标准化到std=1, FISTA一步梯度值(~3.6)将远超权重上限0.05,
        导致所有正Alpha股票撞顶截断, 排序信息丢失, 退化为等权组合。
        """
        Z_hist = self._cross_sectional_zscore(S_hist)
        Z_t = self._cross_sectional_zscore(S_t.reshape(1, -1))[0]
        Z_t = np.clip(Z_t, -self.alpha_clip, self.alpha_clip)  # 截断极值

        # 计算历史Rank IC序列: 信号排序与未来收益排序的相关性
        ICs = []
        for i in range(self.T):
            rank_S = rankdata(Z_hist[i])
            rank_U = rankdata(U_hist[i])
            ic = np.corrcoef(rank_S, rank_U)[0, 1]
            ICs.append(ic)
        mean_IC = np.mean(ICs)

        # 个股波动率 σ
        sigma_t = np.std(U_hist, axis=0)
        sigma_t[sigma_t < 1e-8] = 1e-8

        # 收益率空间 Alpha = IC * σ * Z
        alpha = np.sign(mean_IC) * np.abs(mean_IC) * sigma_t * Z_t
        return alpha, mean_IC

    # ==================== 协方差估计 ====================
    def _estimate_covariance(self, U_hist):
        """
        PCA 因子模型估计协方差矩阵

        [数学原理]
        原始空间协方差: Σ = D_s (B Σ_f B' + D) D_s
        其中:
          D_s (std_dev): N×N 对角阵, 对角线为个股波动率 σ_i
          B: N×K 因子载荷阵
          Σ_f: K×K 因子协方差阵
          D: N×N 对角阵, 对角线为特异性方差(残差方差)

        [工程优化]
        为避免生成和运算 N×N 矩阵(N=5000时内存和算力爆炸),
        我们在标准化空间计算矩阵乘法, 最后通过 D_s 转换回原始空间。
        """
        mu = np.mean(U_hist, axis=0)
        U_centered = U_hist - mu
        std_dev = np.std(U_centered, axis=0)
        std_dev[std_dev < 1e-8] = 1e-8
        U_std = U_centered / std_dev  # 标准化收益: 均值0, 标准差1

        # PCA提取主成分因子
        pca = PCA(n_components=self.K)
        factors = pca.fit_transform(U_std)  # T×K 因子收益矩阵
        # 最小二乘法回归求载荷矩阵 B (N×K)
        B = np.linalg.lstsq(factors, U_std, rcond=None)[0].T
        # 因子协方差矩阵 Σ_f (K×K)
        Sigma_f = np.cov(factors, rowvar=False)

        # 特异性方差 D (N,)
        residuals = U_std - factors @ B.T
        D = np.var(residuals, axis=0)
        D[D < 1e-8] = 1e-8

        return B, Sigma_f, D, std_dev

    def _compute_actual_var(self, w, B, Sigma_f, D, std_dev):
        """
        在原始空间计算组合方差: Var_p = w' Σ w

        [推导]
        w' Σ w = w' D_s (B Σ_f B' + D) D_s w
               = (w ⊙ σ)' (B Σ_f B' + D) (w ⊙ σ)
        其中 ⊙ 表示逐元素相乘。这避免了显式构造 N×N 的 Σ 矩阵。
        """
        ws = w * std_dev  # w ⊙ σ, 转换到标准化空间
        # 矩阵乘法链: (B @ Σ_f @ B') 是系统性风险部分, D 是特异性风险部分
        var = ws @ (B @ (Sigma_f @ (B.T @ ws)) + D * ws)
        return var

    # ==================== FISTA 优化器 ====================
    def _power_iteration(self, Q_dot_w, num_iter=50):
        """
        幂迭代法估计矩阵 Q 的最大特征值 L

        [数学原理]
        FISTA算法需要步长 η = 1/L, 其中 L 是目标函数光滑部分梯度(Q矩阵)的Lipschitz常数。
        对于对称矩阵, Lipschitz常数等于其最大特征值的绝对值。
        幂迭代法: b_{k+1} = Q·b_k / ||Q·b_k||, 收敛后 ||Q·b|| 即为最大特征值。
        """
        b = np.random.randn(self.N)
        b = b / np.linalg.norm(b)
        for _ in range(num_iter):
            b_new = Q_dot_w(b)
            norm = np.linalg.norm(b_new)
            if norm < 1e-10:
                return 1.0
            b = b_new / norm
        return norm

    def _project_constraints(self, w):
        """
        将权重投影到约束空间: 0 ≤ w_i ≤ L_max, Σw_i ≤ 1

        [数学原理]
        这是一个带顶界的单纯形投影问题:
        min ||w - y||²  s.t. 0 ≤ w_i ≤ L_max, Σw_i ≤ 1
        通过拉格朗日乘子法, 解为 w_i = clip(y_i - ν, 0, L_max)
        其中 ν 通过二分法求解, 使得 Σw_i = 1 (当无约束解总和大于1时)。
        """
        # 1. 投影到箱约束 [0, weight_limit]
        w = np.clip(w, 0, self.weight_limit)

        # 2. 检查总和约束
        if np.sum(w) <= 1.0:
            return w  # 已在可行域内, 剩余资金作为现金

        # 3. 二分法寻找拉格朗日乘子 ν
        lower = np.min(w) - self.weight_limit
        upper = np.max(w)
        for _ in range(100):
            mid = (lower + upper) / 2.0
            s = np.sum(np.clip(w - mid, 0, self.weight_limit))
            if s > 1.0:
                lower = mid  # 需要增大 ν 来减小权重
            else:
                upper = mid
        nu = (lower + upper) / 2.0
        w = np.clip(w - nu, 0, self.weight_limit)
        return w

    def _fista_optimize(self, alpha, B, Sigma_f, D, std_dev, w_prev):
        """
        FISTA 加速近端梯度法求解 Markowitz + L1 换手惩罚

        [目标函数]
        max_w  α'w - (λ/2) w'Σw - (δ/2)||w||² - γ||w - w_prev||₁
        等价于:
        min_w  (1/2) w'Qw - α'w + γ||w - w_prev||₁
        其中 Q = λΣ + δI (原始空间)

        [FISTA迭代步骤]
        1. 梯度步: x = y - η * (Qy - α)   (对光滑的二次项求梯度)
        2. 近端步(软阈值): 处理非光滑的L1范数, 产生稀疏性, 抑制微小换手
           x = w_prev + S_{ηγ}(x - w_prev)
           其中 S 是软阈值算子: S_τ(z) = sign(z) * max(|z| - τ, 0)
        3. 投影步: 将 x 投影回约束空间
        4. 动量加速: t_{k+1} = (1 + sqrt(1+4t_k²))/2, y = x_k + ((t_k-1)/t_{k+1})(x_k - x_{k-1})
        """
        # 自适应 delta_ridge: 按协方差对角线均值的比例设定, 保证不同数据尺度下的数值稳定性
        diag_cov = (np.sum(B * (B @ Sigma_f), axis=1) + D) * std_dev**2
        delta_ridge = self.delta_ridge_ratio * np.mean(diag_cov)

        # 定义 Q·w 的隐式计算, 避免生成 N×N 矩阵
        def Q_dot_w(w):
            ws = w * std_dev
            temp = B @ (Sigma_f @ (B.T @ ws)) + D * ws
            return self.lambda_risk * std_dev * temp + delta_ridge * w

        # 估计步长 η
        L = self._power_iteration(Q_dot_w)
        eta = 1.0 / L

        x = w_prev.copy()  # 当前最优解
        y = w_prev.copy()  # 动量外推点
        t = 1.0  # 动量系数

        for iteration in range(self.max_iter):
            # 1. 计算在 y 点的梯度: ∇ = Qy - α
            grad = Q_dot_w(y) - alpha
            x_new = y - eta * grad

            # 2. 软阈值算子处理 L1 换手惩罚: γ||w - w_prev||₁
            threshold = eta * self.gamma_turnover
            diff = x_new - w_prev
            # S_{thr}(diff) = sign(diff) * max(|diff| - thr, 0)
            x_new = w_prev + np.sign(diff) * np.maximum(np.abs(diff) - threshold, 0)

            # 3. 投影到纯多头约束空间
            x_new = self._project_constraints(x_new)

            # 4. FISTA 动量更新
            t_new = (1.0 + np.sqrt(1.0 + 4.0 * t**2)) / 2.0
            y = x_new + ((t - 1.0) / t_new) * (x_new - x)

            # 收敛判断
            if np.linalg.norm(x_new - x) < 1e-8:
                break
            x = x_new
            t = t_new

        return x

    # ==================== 主函数 ====================
    def optimize(self, U_hist, S_hist, S_t, w_prev, verbose=True):
        """
        两步法优化主流程:
        Step 1: FISTA 求解 Markowitz → 组合"形状" (相对权重比例)
        Step 2: 归一化到满仓 → 波动率缩放 → 最终仓位
        """
        # 1. 构建收益率空间 Alpha
        alpha, mean_IC = self._calculate_alpha(U_hist, S_hist, S_t)
        if verbose:
            print(
                f'  [Alpha] IC={mean_IC:.4f}, alpha_std={np.std(alpha):.6f}, '
                f'range=[{np.min(alpha):.6f}, {np.max(alpha):.6f}]'
            )
        if mean_IC <= 0:
            if verbose:
                print('Warning: Mean IC <= 0. Keeping previous weights.')
            return w_prev

        # 2. 估计协方差矩阵 (因子模型隐式表达)
        B, Sigma_f, D, std_dev = self._estimate_covariance(U_hist)

        # 3. FISTA 优化 → 组合形状
        # 此时的 w_shape 只反映了相对强弱, 不受绝对仓位限制, 可能 sum < 1 或 sum > 1
        w_shape = self._fista_optimize(alpha, B, Sigma_f, D, std_dev, w_prev)
        if verbose:
            w_pos = w_shape[w_shape > 1e-8]
            cv = np.std(w_pos) / np.mean(w_pos) if len(w_pos) > 0 else 0
            print(
                f'  [FISTA] sum={np.sum(w_shape):.4f}, 持股={np.sum(w_shape > 1e-6)}, '
                f'max={np.max(w_shape):.4f}, CV={cv:.3f}'
            )

        # 4. 归一化到满仓 (sum=1)
        # 先归一化, 再投影, 确保满足权重上限约束
        total = np.sum(w_shape)
        if total > 1e-8:
            w_full = w_shape / total
        else:
            # Fallback: 若FISTA输出全0, 退化为正Alpha等权
            pos = alpha > 0
            w_full = np.zeros(self.N)
            w_full[pos] = 1.0 / max(np.sum(pos), 1)

        w_full = self._project_constraints(w_full)
        # 投影可能改变总和, 再次归一化
        total2 = np.sum(w_full)
        if total2 > 1e-8:
            w_full = w_full / total2
            w_full = self._project_constraints(w_full)
        if verbose:
            w_pos2 = w_full[w_full > 1e-8]
            cv2 = np.std(w_pos2) / np.mean(w_pos2) if len(w_pos2) > 0 else 0
            p90 = np.percentile(w_pos2, 90) if len(w_pos2) > 0 else 0
            p10 = np.percentile(w_pos2, 10) if len(w_pos2) > 0 else 1
            print(
                f'  [满仓] sum={np.sum(w_full):.4f}, 持股={np.sum(w_full > 1e-6)}, '
                f'max={np.max(w_full):.4f}, CV={cv2:.3f}, P90/P10={p90 / p10:.1f}x'
            )

        # 5. 目标波动率缩放
        # 计算满仓组合的实际年化波动率
        var_full = self._compute_actual_var(w_full, B, Sigma_f, D, std_dev)
        sigma_full = np.sqrt(max(var_full, 0))
        vol_annual = sigma_full * np.sqrt(252)  # 日波动率年化

        # 多头策略无法加杠杆, 因此 scale <= 1.0
        # 若满仓波动率 < 目标波动率, 则保持满仓(scale=1), 剩余资金为现金
        # 若满仓波动率 > 目标波动率, 则等比例降仓(scale < 1)
        scale = min(1.0, self.target_vol / vol_annual) if vol_annual > 1e-8 else 1.0
        if verbose:
            print(
                f'  [缩放] 满仓vol={vol_annual:.4f}, 目标={self.target_vol:.4f}, scale={scale:.4f}'
            )

        # 缩放并确保最终约束满足
        w_t = w_full * scale
        w_t = self._project_constraints(w_t)

        return w_t


# =====================================================================
#  数据生成 (含信号自相关性 ρ=0.9, 时变预测强度, IC_mean≈0.10, IC_std≈0.15)
# =====================================================================
def generate_data(N=5000, T=200, rho=0.9, sigma_signal=0.01, mu_beta=0.14, std_beta=0.18, seed=42):
    """
    生成具有真实统计特征的模拟数据

    [信号模型] (自相关 ρ=0.9)
        S_t = ρ * S_{t-1} + η_t
        信号高度自相关, 模拟现实中的动量/价值等缓慢变化因子, 从根本上降低换手率。

    [预测强度模型] (时变, 产生 IC 波动)
        β_t ~ N(μ_β, σ_β)
        - 某些日期 β_t > 0: 信号有效, IC > 0
        - 某些日期 β_t < 0: 信号反转, IC < 0
        - 某些日期 β_t ≈ 0: 信号失效, IC ≈ 0

    [收益率模型]
        U_t = 因子收益 + β_t * S_t + 特异性收益

    [参数标定] (seed=42)
        μ_β=0.14, σ_β=0.18 → IC_mean≈0.10, IC_std≈0.15, IC>0比例≈75%
    """
    np.random.seed(seed)

    # --- 1. 信号 AR(1) 过程 ---
    S_hist = np.zeros((T, N))
    S_hist[0] = np.random.randn(N) * sigma_signal
    for t in range(1, T):
        S_hist[t] = rho * S_hist[t - 1] + np.random.randn(N) * sigma_signal * np.sqrt(1 - rho**2)
    # 当期信号 (AR(1) 继续演化一步, 与上期相关 0.9)
    S_t = rho * S_hist[-1] + np.random.randn(N) * sigma_signal * np.sqrt(1 - rho**2)

    # --- 2. 时变预测强度 ---
    beta_t = np.random.randn(T) * std_beta + mu_beta

    # --- 3. 因子结构 (市场 + 风格) ---
    market_factor = np.random.randn(T, 1) * 0.01  # 市场因子: 日波动率 1%
    market_loadings = np.ones((N, 1)) + np.random.randn(N, 1) * 0.3  # Beta ~ N(1, 0.3)
    style_factors = np.random.randn(T, 4) * 0.005  # 风格因子: 日波动率 0.5%
    style_loadings = np.random.randn(N, 4) * 0.5
    factor_returns = market_factor @ market_loadings.T + style_factors @ style_loadings.T

    # --- 4. 收益率合成 ---
    U_hist = np.zeros((T, N))
    for t in range(T):
        U_hist[t] = factor_returns[t] + beta_t[t] * S_hist[t] + np.random.randn(N) * 0.01

    return U_hist, S_hist, S_t


def generate_prev_weights(optimizer, U_hist, S_hist, N):
    """
    用上一期信号 S_hist[-1] 通过优化器生成真实的 w_prev

    [时间线对齐]
      时刻 T-1 (上一期): 历史数据 U_hist[:T-1], S_hist[:T-1] | 当期信号 S_hist[-1] → w_prev
      时刻 T   (本期):   历史数据 U_hist[:T],   S_hist[:T]   | 当期信号 S_t        → w_t

    [为什么必须这样生成?]
    若用简单等权(持股~2500)作为w_prev, 与优化器输出(持股~400)结构完全不匹配,
    会导致换手率虚高至 1.3+。
    用同结构的优化器输出作为w_prev, S_t与S_hist[-1]相关0.9, 换手率自然降至 ~0.2。
    """
    S_prev = S_hist[-1]  # 上一期的当期信号
    U_hist_prev = U_hist[:-1]  # 上一期可用的历史数据 (T-1 期)
    S_hist_prev = S_hist[:-1]  # 上一期可用的历史信号 (T-1 期)

    # 初始持仓: 随机等权多头
    w_init = np.zeros(N)
    rng = np.random.RandomState(123)
    idx = rng.choice(N, 500, replace=False)
    w_init[idx] = 1.0 / 500

    w_prev = optimizer.optimize(U_hist_prev, S_hist_prev, S_prev, w_init, verbose=False)
    return w_prev


def verify_data(U_hist, S_hist, S_t, N, T):
    """验证数据的关键统计特征"""

    def zscore(S):
        m = np.median(S, axis=-1, keepdims=True)
        mad = np.median(np.abs(S - m), axis=-1, keepdims=True)
        mad[mad == 0] = 1.0
        return (S - m) / mad

    Z = zscore(S_hist)
    ICs = [np.corrcoef(rankdata(Z[i]), rankdata(U_hist[i]))[0, 1] for i in range(T)]
    autocorrs = [
        np.corrcoef(S_hist[1:, i], S_hist[:-1, i])[0, 1]
        for i in range(N)
        if np.std(S_hist[:, i]) > 1e-8
    ]

    print(f'  IC_mean:          {np.mean(ICs):.4f}  (目标 ≈ 0.10)')
    print(f'  IC_std:           {np.std(ICs):.4f}  (目标 ≈ 0.15)')
    print(f'  IC > 0 比例:      {np.mean(np.array(ICs) > 0):.1%}  (目标 ≈ 75%)')
    print(f'  信号自相关(均值):  {np.mean(autocorrs):.4f}  (目标 ≈ 0.88)')
    print(f'  S_t vs S_{{T-1}}:  {np.corrcoef(S_t, S_hist[-1])[0, 1]:.4f}')
    print(f'  个股年化波动率:    {np.mean(np.std(U_hist, axis=0)) * np.sqrt(252):.2f}')


# =====================================================================
#  主程序
# =====================================================================
if __name__ == '__main__':
    N, T = 5000, 200

    # ===== 1. 生成数据 =====
    print('=' * 70)
    print('数据生成 (信号 AR(1) ρ=0.9, 时变β, IC_mean≈0.10, IC_std≈0.15)')
    print('=' * 70)
    U_hist, S_hist, S_t = generate_data(N=N, T=T, seed=42)
    print(f'U_hist.shape={U_hist.shape}  S_hist.shape={S_hist.shape}  S_t.shape={S_t.shape}')
    verify_data(U_hist, S_hist, S_t, N, T)

    # ===== 2. 测试不同 lambda_risk =====
    print('\n' + '=' * 70)
    print('测试不同 lambda_risk (Alpha=IC*σ*Z, w_prev 用 S_hist[-1] 生成)')
    print('=' * 70)

    for lam in [10.0, 50.0, 100.0, 200.0, 500.0]:
        # 为每个 lambda 生成对应结构的 w_prev
        opt_prev = LongOnlyPortfolioOptimizer(
            N=N,
            T=T - 1,
            K_factors=10,
            lambda_risk=lam,
            delta_ridge_ratio=0.01,
            gamma_turnover=0.001,
            target_vol=0.15,
            weight_limit=0.05,
        )
        w_prev = generate_prev_weights(opt_prev, U_hist, S_hist, N)

        opt = LongOnlyPortfolioOptimizer(
            N=N,
            T=T,
            K_factors=10,
            lambda_risk=lam,
            delta_ridge_ratio=0.01,
            gamma_turnover=0.001,
            target_vol=0.15,
            weight_limit=0.05,
        )
        print(f'\n--- lambda_risk = {lam} ---')
        w_t = opt.optimize(U_hist, S_hist, S_t, w_prev, verbose=True)

        wp = w_t[w_t > 1e-6]
        cv = np.std(wp) / np.mean(wp) if len(wp) > 0 else 0
        p90 = np.percentile(wp, 90) if len(wp) > 0 else 0
        p10 = np.percentile(wp, 10) if len(wp) > 0 else 1
        turnover = np.sum(np.abs(w_t - w_prev))
        print(
            f'  [最终] 总仓位={np.sum(w_t):.4f}, 持股={len(wp)}, '
            f'max={np.max(w_t):.4f}, CV={cv:.3f}, P90/P10={p90 / p10:.1f}x, '
            f'换手={turnover:.4f}'
        )

    # ===== 3. 推荐配置 =====
    print('\n' + '=' * 70)
    print('推荐配置 (lambda_risk=50.0)')
    print('=' * 70)

    opt_prev = LongOnlyPortfolioOptimizer(
        N=N,
        T=T - 1,
        K_factors=10,
        lambda_risk=50.0,
        delta_ridge_ratio=0.01,
        gamma_turnover=0.001,
        target_vol=0.15,
        weight_limit=0.05,
    )
    w_prev = generate_prev_weights(opt_prev, U_hist, S_hist, N)

    opt = LongOnlyPortfolioOptimizer(
        N=N,
        T=T,
        K_factors=10,
        lambda_risk=50.0,
        delta_ridge_ratio=0.01,
        gamma_turnover=0.001,
        target_vol=0.15,
        weight_limit=0.05,
    )
    w_t = opt.optimize(U_hist, S_hist, S_t, w_prev, verbose=True)

    print('\n--- 最终验证 ---')
    wp = w_t[w_t > 1e-6]
    wp_prev = w_prev[w_prev > 1e-6]
    print(f'权重维度:       {w_t.shape}')
    print(f'总仓位:         {np.sum(w_t):.4f} (应 <= 1.0)')
    print(f'现金比例:       {1.0 - np.sum(w_t):.4f}')
    print(f'最小权重:       {np.min(w_t):.6f} (应 >= 0)')
    print(f'最大权重:       {np.max(w_t):.4f} (应 <= 0.05)')
    print(f'多头持仓股票数:  {len(wp)}')
    print(f'w_prev 持股数:   {len(wp_prev)}')
    print(f'权重CV:         {np.std(wp) / np.mean(wp):.3f} (应 > 0.3, 非等权)')
    print(f'P90/P10:        {np.percentile(wp, 90) / np.percentile(wp, 10):.1f}x (应 > 3x)')
    print(f'换手率(双边):   {np.sum(np.abs(w_t - w_prev)):.4f} (ρ=0.9 应 < 0.5)')

    B, Sigma_f, D, std_dev = opt._estimate_covariance(U_hist)
    var_opt = opt._compute_actual_var(w_t, B, Sigma_f, D, std_dev)
    vol_opt = np.sqrt(var_opt) * np.sqrt(252)
    print(f'年化波动率:     {vol_opt:.4f} (目标上限: 0.15)')
