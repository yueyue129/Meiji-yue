#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 16 15:11:39 2025

@author: yueyuechangxiangjian
"""


# 优化类
class PortfolioOptimizer:
    def __init__(self, cov_mat: pd.DataFrame, Mean_yield: pd.Series,
                 fund_info: Optional[pd.DataFrame] = None, Rf: Optional[float] = None):
        """
        初始化优化器
        :param cov_mat: 协方差矩阵，index和column均为资产名称
        :param Mean_yield: 资产的预期收益率
        :param fund_info: 投资比例限制信息
        :param Rf: 无风险利率
        """
        if not isinstance(cov_mat, pd.DataFrame):
            raise ValueError('cov_mat should be pandas DataFrame！')
        if not isinstance(Mean_yield, pd.Series):
            raise ValueError('Mean_yield should be pandas Series！')

        self.cov_mat = cov_mat.copy()
        self.Mean_yield = Mean_yield.copy()
        self.fund_info = fund_info
        self.Rf = Rf if Rf is not None else 0.0  # 默认无风险利率为0
        self.omega = cov_mat.values.copy()

    def portfolio_variance(self, x: np.ndarray) -> float:
        """计算组合方差"""
        return float(np.dot(x.T, np.dot(self.omega, x)))

    def portfolio_return(self, x: np.ndarray) -> float:
        """计算组合收益率"""
        return np.dot(x, self.Mean_yield.values)

    def portfolio_max_drawdown(self, x: np.ndarray, historical_returns: np.ndarray) -> float:
        """计算组合的最大回撤"""
        portfolio_returns = np.dot(historical_returns, x)
        cumulative_returns = np.cumprod(1 + portfolio_returns)
        peak = np.maximum.accumulate(cumulative_returns)
        drawdown = (peak - cumulative_returns) / peak
        return np.max(drawdown)

    def portfolio_cvar(self, x: np.ndarray, confidence_level: float = 0.95) -> float:
        """计算组合的条件风险价值（CVaR）"""
        port_return = self.portfolio_return(x)
        std_dev = np.sqrt(self.portfolio_variance(x))
        z_score = norm.ppf(1 - confidence_level)
        cvar = self.Rf - (port_return - z_score * std_dev)
        return cvar

    def fun1(self, x: np.ndarray) -> float:
        """最小方差目标函数"""
        return self.portfolio_variance(x)

    def fun2(self, x: np.ndarray) -> float:
        """风险平价目标函数"""
        port_var = self.portfolio_variance(x)
        marginal_risk = np.dot(self.omega, x) / np.sqrt(port_var)
        risk_contribution = x * marginal_risk
        target_risk = np.mean(risk_contribution)
        return np.sum((risk_contribution - target_risk) ** 2)

    def fun3(self, x: np.ndarray) -> float:
        """最大分散化目标函数"""
        den = np.dot(x, np.diag(self.omega))
        num = np.sqrt(self.portfolio_variance(x))
        return num / den

    def fun4(self, x: np.ndarray) -> float:
        """最大夏普比率目标函数"""
        port_return = self.portfolio_return(x)
        port_var = self.portfolio_variance(x)
        return - (port_return - self.Rf) / np.sqrt(port_var)

    def fun5(self, x: np.ndarray) -> float:
        """风险预算目标函数"""
        tmp = np.dot(self.omega, x)
        sigmatmp = np.sqrt(self.portfolio_variance(x))
        delta = (np.dot(x, tmp) / sigmatmp - sigmatmp * self.Mean_yield.values)
        return np.sum(delta ** 2)

    def fun6(self, x: np.ndarray) -> float:
        """改进的风险预算目标函数"""
        tmp = np.dot(self.omega, x)
        sigmatemp = np.sqrt(self.portfolio_variance(x))
        delta = (np.dot(x, tmp) / (sigmatemp * (self.Mean_yield.values - self.Rf)))
        return np.sum(delta ** 2)

    def optimize(self, method: str = 'risk budget', targ: Optional[float] = None,
                 historical_returns: Optional[np.ndarray] = None,
                 benchmark_returns: Optional[np.ndarray] = None,
                 confidence_level: float = 0.95) -> pd.Series:
        """
        优化权重配置
        :param method: 优化方法
        :param targ: 目标波动率或目标收益率
        :param historical_returns: 历史收益率数据
        :param benchmark_returns: 基准组合的收益率数据
        :param confidence_level: 用于CVaR计算的置信水平
        :return: 权重Series
        """
        method_to_fun = {
            'min variance': self.fun1,
            'risk parity': self.fun2,
            'max sharpe': self.fun4,
            'max diversification': self.fun3,
            'risk budget': self.fun5,
            'risk budget mod': self.fun6,
            'fix risky': self.fun4,
            'mean variance': self.fun1,
            'min max drawdown': self.portfolio_max_drawdown,
            'min cvar': self.portfolio_cvar,
            'min tracking error': self.portfolio_tracking_error
        }

        if method not in method_to_fun:
            raise ValueError('Method not supported')

        x0 = np.ones(self.omega.shape[0]) / self.omega.shape[0]
        bnds = tuple((0, None) for _ in x0)
        
        # 如果没有提供fund_info，则不添加投资比例限制
        cons = [{'type': 'eq', 'fun': lambda x: sum(x) - 1}]
        if self.fund_info is not None and '投资比例' in self.fund_info.columns:
            cons.append({'type': 'ineq', 'fun': lambda x: self.fund_info['投资比例'] - x})

        fun = method_to_fun[method]
        options = {'disp': False, 'maxiter': 1000, 'ftol': 1e-25}

        if method == 'fix risky':
            if targ is None:
                raise ValueError('targ must be provided for fix risky method')
            cons.append({'type': 'eq', 'fun': lambda x: np.sqrt(self.portfolio_variance(x)) - targ})
        elif method == 'mean variance':
            if targ is None:
                raise ValueError('targ must be provided for mean variance method')
            if (targ >= min(self.Mean_yield)) & (targ <= max(self.Mean_yield)):
                cons.append({'type': 'eq', 'fun': lambda x: np.dot(x, self.Mean_yield) - targ})
            else:
                raise ValueError('目标收益率不在可行域区间！！')
        elif method == 'min max drawdown' and historical_returns is not None:
            fun = lambda x: self.portfolio_max_drawdown(x, historical_returns)
        elif method == 'min cvar':
            fun = lambda x: self.portfolio_cvar(x, confidence_level)
        elif method == 'min tracking error' and benchmark_returns is not None:
            fun = lambda x: self.portfolio_tracking_error(x, benchmark_returns)

        res = minimize(fun, x0, bounds=bnds, constraints=cons, method='SLSQP', options=options)
        if not res.success:
            raise ValueError(f"Optimization failed: {res.message}")
            
        wts = pd.Series(index=self.cov_mat.index, data=res.x)
        return wts
