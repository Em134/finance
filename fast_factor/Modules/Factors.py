from typing import Any
from .BaseModules import BaseModule
import numpy as np
from numpy.lib.stride_tricks import as_strided as strided
import pandas as pd
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant
import statsmodels.api as sm
from typing import List, Dict
from tqdm import tqdm
import pandas as pd


class BaseFactor(BaseModule):
    def __init__(self) -> None:
        super().__init__()
        pass
    
    def calculate(self, *args: Any, **kwds: Any) -> Any:
        return args, kwds
    
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.calculate(*args, **kwds)


class RSRSFactor(BaseFactor):
    """
    RSRSFactor 是一个用于计算相对强弱回归斜率 (Relative Strength Regression Slope, RSRS) 指标的类。
    
    这个类实现了RSRS指标的计算，包括线性回归斜率（beta）和决定系数（R²）的计算，
    以及基于这些指标的Z-Score和MDF标准化得分的计算。这些指标可用于股票市场分析中，
    特别是在识别股票的趋势强度方面。

    参数:
        window_size (int): 计算线性回归斜率和R²值的窗口大小，默认为18。
        rolling_window_size (int): 计算滚动窗口内beta值的平均和标准差的窗口大小，默认为600。
    
    方法:
        calculate(df: pd.DataFrame) -> pd.DataFrame:
            计算给定DataFrame中的RSRS指标，包括beta、R²、Z-Score和MDF标准化得分。
            
    已经基于 numpy 加速计算。
    """
    def __init__(self, window_size=18, rolling_window_size=600) -> None:
        self.window_size = window_size
        self.rolling_window_size = rolling_window_size
        self.data = None

    @staticmethod
    def rolling_window(a: np.array, window: int):
        '生成滚动窗口，以三维数组的形式展示'
        shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
        strides = a.strides + (a.strides[-1],)
        return strided(a, shape=shape, strides=strides)

    def _calculate_beta_and_r2(self, x, y, window=18):
        if x.shape[0] < window:
            return np.nan, np.nan, np.nan

        # 使用NumPy进行线性回归
        x = x.reshape(-1, 1)
        ones_vector = np.ones((x.shape[0], 1))
        X = np.hstack([ones_vector, x])
        Y = y.reshape(-1, 1)

        # 计算beta值
        beta = np.linalg.lstsq(X, Y, rcond=None)[0].flatten()

        # 计算预测值
        y_pred = X @ beta

        # 计算总平方和（TSS）
        y_mean = np.mean(Y)
        TSS = np.sum((Y - y_mean) ** 2)

        # 计算残差平方和（RSS）
        RSS = np.sum((y_pred - y_mean) ** 2)

        # 计算R²值
        r2 = RSS / TSS

        return beta[1], r2

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        # 确保数据已经被设置
        if df is None:
            raise ValueError("Data DataFrame cannot be None.")

        # 获取low和high列
        lows = df['low'].values
        highs = df['high'].values

        # 使用优化的线性回归计算beta值
        betas, r2s = [], []

        # 创建滚动窗口
        lows_window = self.rolling_window(lows, self.window_size)
        highs_window = self.rolling_window(highs, self.window_size)

        # 计算所有窗口的beta值和R²值
        for low_win, high_win in zip(lows_window, highs_window):
            beta, r2 = self._calculate_beta_and_r2(low_win, high_win, window=self.window_size)
            betas.append(beta)
            r2s.append(r2)

        # 将beta值和R²值添加到原始DataFrame中
        # 注意这里我们需要确保beta和r2的长度与原始DataFrame相同
        beta_padded = np.concatenate((np.full(self.window_size - 1, np.nan), betas))
        r2_padded = np.concatenate((np.full(self.window_size - 1, np.nan), r2s))

        # 计算滚动窗口内的beta均值和标准差
        beta_rollwindow = self.rolling_window(beta_padded, self.rolling_window_size)
        beta_mean = np.mean(beta_rollwindow, axis=1)
        beta_std = np.std(beta_rollwindow, axis=1)

        # 计算z-score
        zscore = (beta_padded[self.rolling_window_size-1:] - beta_mean) / beta_std

        # 创建新的DataFrame
        result_df = df.copy()
        result_df['beta'] = beta_padded
        result_df['r2'] = r2_padded
        result_df['zscore'] = np.concatenate((np.full(self.rolling_window_size - 1, np.nan), zscore))

        result_df['mdf_std_score'] = result_df['r2'] * result_df['zscore']

        return result_df


class BarraBaseFactor(BaseFactor):
    def __init__(self) -> None:
        super().__init__()
        pass
    
    @staticmethod
    def pivot_df(df: pd.DataFrame, pivot_dict: Dict) -> pd.DataFrame:
        """
        将DataFrame转换为以 pivot_dict 指定的的唯一值为索引、列名、值的新DataFrame。

        参数:
        df : pd.DataFrame
            包含所用数据的df。

        返回:
        pd.DataFrame
            转换后的DataFrame。
        """
        # 确保数据已经被设置
        if df is None:
            raise ValueError("Data DataFrame cannot be None.")

        # 使用pivot方法进行转换
        pivoted_df = df.pivot(**pivot_dict)

        return pivoted_df


class BarraSizeFactor(BarraBaseFactor):
    """
    BarraSizeFactor 是一个用于计算Barra Size因子的类。
    要求输入的dataframe为一个投资组合，列为市值的时间序列，行为投资组合中的资产。

    定义：1.0 * LNCAP
    市值因子用LNCP来表示，表示公司股票总市值的自然对数。

    方法:
        calculate(df: pd.DataFrame, market_value_col: str) -> pd.DataFrame:
            计算给定DataFrame中的Barra Size因子。
    """

    def __init__(self, pivot_dict: Dict = None) -> None:
        super().__init__()
        if pivot_dict == None:
            self.pivot_dict = {'index': 'ts_code', 
                             'columns': 'trade_date', 
                             'values': 'total_mv'
                             }
        else:
            self.pivot_dict = pivot_dict
            

    def calculate(self, 
                  df: pd.DataFrame, 
                  ) -> pd.DataFrame:
        # 确保数据已经被设置
        if df is None:
            raise ValueError("Data DataFrame cannot be None.")
        df = self.pivot_df(df, self.pivot_dict)
        
        # 获取市值列并计算自然对数
        log_market_values = np.log(df.values)
        
        # 直接在原DataFrame上添加新列
        result_df = pd.DataFrame(log_market_values, index=df.index, columns=df.columns)

        return result_df
    
    
class BarraBetaFactor(BaseFactor):
    def __init__(self) -> None:
        super().__init__()
        pass
    
    @staticmethod
    def ewma_weights(length=252, half_life=63):
        """
        Calculate EWMA weights for a given length apnd half-life.

        Parameters:
        length (int): The number of data points.
        half_life (float): The half-life for the weights.

        Returns:
        np.ndarray: Array of EWMA weights.
        """
        # Calculate the smoothing factor alpha from the half-life
        alpha = 1 - np.exp(-np.log(2) / half_life)

        # Generate the weights
        weights = alpha * (1 - alpha) ** np.arange(length)[::-1]

        # Normalize the weights so they sum to 1
        weights /= weights.sum()

        return weights
    
    def get_beta_window(df, window=252, half_life=63):
        Y = df['']