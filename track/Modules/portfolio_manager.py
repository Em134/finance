import WindPy
import datetime
import numpy as np
import pandas as pd
from typing import Any, Union
from typing import Union, List, Tuple
from dataclasses import dataclass, field
from .date_time_module import TradingWeek
from .helper import PredictionFetcher, StockDataFetcher, DateHelper


@dataclass
class PortfolioState:
    """
    该类用于记录投资组合的状态, 包括净值、持仓状态、仓位比例以及最后一次交易的价格。

    Attributes:
        nav (float): 投资组合的净值, 表示投资组合的总价值 (每周计算到最后一次交易时为止)。
        nav_final (float): 投资组合的净值, 表示投资组合的总价值 (每周计算到最后一个交易日)。
        position (int): 持仓状态, 1 表示持有多头仓位, 0 表示持有空头仓位。
        position_size (float): 仓位比例, 表示投资组合中资产所占的投资比例。
        last_trade_price (float): 最后一次交易的价格。
        last_signal (int): 本周最后生成的交易信号。
    """
    
    nav: float = field(default=np.nan, metadata={'description': '投资组合的净值 (最后调仓时)'})
    nav_final: float = field(default=np.nan, metadata={'description': '投资组合的净值 (周末)'})
    position: int = field(default=np.nan, metadata={'description': '持仓状态, 1 表示多头, 0 表示空头'})
    position_size: float = field(default=np.nan, metadata={'description': '仓位比例, 周一调仓比例为1.0, 周三调仓比例为0.5'})
    last_trade_price: float = field(default=np.nan, metadata={'description': '最后一次交易的价格'})
    last_signal: int = field(default=np.nan, metadata={'description': '本周最后生成的交易信号'})

    def show(self) -> None:
        """
        打印投资组合状态的当前属性和值。
        """
        for attr, value in self.__dict__.items():
            print(f"{attr}: {value}")
            
    @property
    def is_initially_empty(self) -> bool:
        """
        检查除了 last_signal 以外的所有属性是否都为 np.nan。
        
        Returns:
            bool: 如果除了 last_signal 外其他属性都是 np.nan, 则返回 True; 否则返回 False。
        """
        for attr, value in self.__dict__.items():
            if attr != 'last_signal' and not np.isnan(value):
                return False
        return True


class WeeklyPortfolioManager:
    """
    每周投资组合管理器类，用于管理投资组合的每周交易并计算净值变化。

    Attributes:
        prediction_fetcher (PredictionFetcher): 用于获取预测数据的对象。
        state (PortfolioState): 当前投资组合的状态。
        nav_history (dict): 净值变化历史记录。
        code (str): 股票代码。

    初始化方法：
        def __init__(self, 
                     code: str = "000852.SH", 
                     prediction_fetcher: PredictionFetcher = PredictionFetcher('dataset/daily_data.xlsx')
                     ) -> None:
            - code (str): 要管理的股票代码，默认为"000852.SH"。
            - prediction_fetcher (PredictionFetcher): 用于获取预测数据的对象，默认为从指定的 Excel 文件中获取数据的 PredictionFetcher 对象。

    主要方法：
        def manage_portfolio_weekly(self, last_state: PortfolioState, this_week: TradingWeek):
            - last_state (PortfolioState): 上一周的投资组合状态。
            - this_week (TradingWeek): 本周的交易日及其他关键日期信息。

            管理每周的交易并计算净值变化。

            流程：
            - 如果上一周状态为空（初始状态），设置初始净值为 1，根据 last_signal 设置持仓状态和仓位比例，并获取开盘后半小时内的平均价格作为最后交易价格。
            - 如果不是初始状态，则进行周级别调仓，根据上一周的持仓状态和价格计算新的净值，并更新仓位比例和最后交易价格。
            - 如果本周允许周内调仓，根据周内调仓生成信号进行调仓操作。
            - 最后设置本周的最后生成的交易信号。

            返回：
            - 返回更新后的投资组合状态对象。

        def reposition(self, last_state: PortfolioState, this_week: TradingWeek, level: str='week'):
            - last_state (PortfolioState): 上一周的投资组合状态。
            - this_week (TradingWeek): 本周的交易日及其他关键日期信息。
            - level (str): 调仓级别，可以是'week'（周级别调仓）或'intra'（周内调仓），默认为'week'。

            根据不同的调仓级别进行调仓操作，并更新投资组合状态。

            流程：
            - 如果是周级别调仓，设置目标仓位比例为 1，获取本周第一个调仓日；如果是周内调仓，设置目标仓位比例为 0.5，获取周内调仓日。
            - 根据上一周的持仓状态和分钟价格数据，计算卖出价格和买入价格。
            - 根据上一周的持仓状态和价格计算新的净值，更新仓位比例、最后交易价格和持仓状态。
            - 将更新后的净值记录到净值历史记录中。

        def get_nav_history(self):
            获取净值变化历史记录。

            返回：
            - 返回净值变化历史记录字典。

        def get_final_state(self):
            获取本周结束时的组合状态。

            返回：
            - 返回一个包含本周净值、持仓状态和仓位比例的 PortfolioState 对象。
    """
    def __init__(self, 
                 code: str = "000852.SH", 
                 prediction_fetcher: PredictionFetcher = PredictionFetcher('dataset/daily_data.xlsx')
                 ) -> None:
        self.prediction_fetcher = prediction_fetcher
        self.state = PortfolioState()
        self.nav_history = {}
        self.code = code

    def manage_portfolio_weekly(self, last_state: PortfolioState, this_week: TradingWeek):
        """
        管理每周的交易并计算净值变化。

        参数:
        last_state (PortfolioState): 上一周的数据。
        this_week (TradingWeek): 这一周的交易日及其他关键日期信息。

        流程：
        1. 如果上一周状态为空（初始状态）：
            - 设置初始净值为 1。
            - 根据 last_signal 设置持仓状态和仓位比例。
            - 如果 last_signal 为 0，设置特定的时间范围，获取该时间范围内的股票数据并取平均价格作为最后交易价格；如果 last_signal 为 1，设置不同的时间范围并进行相同操作。
        2. 如果不是初始状态：
            - 调用 reposition 方法进行周级别调仓。
        3. 如果本周允许周内调仓：
            - 获取周内调仓生成的信号。
            - 调用 reposition 方法进行周内调仓。
        4. 设置本周的最后生成的交易信号。

        返回：
        - 返回更新后的投资组合状态对象。
        """
        self.state = PortfolioState()
        # 周级别调仓
        if last_state.is_initially_empty:
            # 如果当前为初始周
            self.state.nav = 1
            self.state.nav_final = 1
            self.state.position = last_state.last_signal
            self.state.position_size = 1
            if last_state.last_signal == 0:  # 如果上周最终信号为空
                start_delta = datetime.timedelta(hours=10) + datetime.timedelta(minutes=30)
                end_delta = datetime.timedelta(hours=11)
            else:
                start_delta = datetime.timedelta(hours=10)
                end_delta = datetime.timedelta(hours=10) + datetime.timedelta(minutes=30)
                
            self.state.last_trade_price = this_week.trading_days[0].minute_price(self.code, start_delta, end_delta).mean().values[0]
        else:
            self.reposition(last_state, this_week, 'week')
        
        # 如果进行周内调仓
        if this_week.intra_reposition:
            self.state.last_signal = self.prediction_fetcher.get_prediction_for_date(this_week.intra_signal_generating_day.dt)
            self.reposition(self.state, this_week, 'intra')
        
        self.state.last_signal = self.prediction_fetcher.get_prediction_for_date(this_week.final_signal_generating_day.dt)
        
        return self.state
            
    def reposition(self, last_state: PortfolioState, this_week: TradingWeek, level: str='week'):
        """
        根据不同的调仓级别进行调仓操作，并更新投资组合状态。

        参数:
        last_state (PortfolioState): 上一周的投资组合状态。
        this_week (TradingWeek): 本周的交易日及其他关键日期信息。
        level (str): 调仓级别，可以是'week'（周级别调仓）或'intra'（周内调仓），默认为'week'。

        流程：
        1. 如果是周级别调仓：
            - 设置目标仓位比例为 1，获取本周第一个调仓日。
        2. 如果是周内调仓：
            - 设置目标仓位比例为 0.5，获取周内调仓日。
        3. 根据上一周的持仓状态和分钟价格数据，计算卖出价格和买入价格。
        4. 如果上一周持仓状态为 1（多头）：
            - 根据公式计算新的净值。
        5. 如果上一周持仓状态为 0（空头）：
            - 根据公式计算新的净值。
        6. 更新仓位比例、最后交易价格和持仓状态。
        7. 将更新后的净值记录到净值历史记录中。
        """
        if level == 'week':
            target_position_size = 1
            target_day = this_week.first_repositioning_day
        elif level == 'intra':
            target_day = this_week.intra_repositioning_day
            target_position_size = 0.5
            
        min_delta = datetime.timedelta(hours=10)
        max_delta = datetime.timedelta(hours=11)
        all_price =target_day.minute_price(self.code, min_delta, max_delta)
        
        if last_state.position == 1:
            sell_price = all_price.iloc[len(all_price) // 2: ].mean().values
            buy_price = all_price.iloc[: len(all_price) // 2].mean().values
        else:
            sell_price = all_price.iloc[: len(all_price) // 2].mean().values
            buy_price = all_price.iloc[len(all_price) // 2: ].mean().values
        # sell_price = all_price.iloc[len(all_price) // 2: ].mean().values
        # buy_price = all_price.iloc[len(all_price) // 2: ].mean().values

        if last_state.position == 1:
            self.state.nav = last_state.nav * (1 + (sell_price - last_state.last_trade_price) / last_state.last_trade_price )* last_state.position_size + last_state.nav * (1 - last_state.position_size)
        else:
            self.state.nav = last_state.nav * (1 + (-sell_price + last_state.last_trade_price) / last_state.last_trade_price) * last_state.position_size + last_state.nav * (1 - last_state.position_size)
        
        self.state.position_size = target_position_size
        self.state.last_trade_price = buy_price
        self.state.position = last_state.last_signal 
        self.state.nav_final = self.state.nav
        self.nav_history[level] = self.state.nav
                

    def get_nav_history(self):
        """
        获取净值变化历史记录。

        返回：
        - 返回净值变化历史记录字典。
        """
        return self.nav_history

    def get_final_state(self):
        """
        获取本周结束时的组合状态。

        返回：
        - 返回一个包含本周净值、持仓状态和仓位比例的 PortfolioState 对象。
        """
        return PortfolioState(self.nav, self.position, self.position_size)



