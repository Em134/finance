import WindPy
import datetime
import numpy as np
import pandas as pd
from typing import Any, Union
from typing import Union, List, Tuple
from .helper import StockDataFetcher, DateHelper


class TradingDay:
    """
    一个封装了 datetime.datetime 功能的类，支持初始化时传入字符串或 datetime.datetime 对象。

    Attributes:
        dt (datetime.datetime): 内部的 datetime.datetime 实例。
    """

    def __init__(self,
                 input_date: Union[str, datetime.datetime],
                 date_helper: DateHelper = DateHelper(),
                 stock_helper: StockDataFetcher = StockDataFetcher(),
                 ) -> None:
        """
        初始化 TradingDay 类。

        参数:
        input_date (Union[str, datetime.datetime]): 输入的日期，可以是字符串或 datetime.datetime 对象。
        """
        self.date_helper = date_helper
        self.stock_helper = stock_helper
        if isinstance(input_date, str):
            self.dt = self.date_helper.convert_to_timestamp(input_date)
        elif isinstance(input_date, datetime.datetime):
            self.dt = input_date
        else:
            raise ValueError("input_date 必须是字符串或 datetime.datetime 类型。")

        if not self.date_helper.is_trading_day(self.dt):
            raise ValueError("给定日期不是交易日。")

    def close(self, code: str, close_type: str = "close", priceAdj: str = "U", cycle: str = "D"):
        return self.stock_helper.close(code=code, date=self.dt, close_type=close_type, priceAdj=priceAdj, cycle=cycle)

    def minute_price(self,
                     code: str,
                     start_delta: datetime.timedelta = datetime.timedelta(hours=9),
                     end_delta: datetime.timedelta = datetime.timedelta(hours=10),
                     frequency: str = '1'
                     ) -> pd.DataFrame:
        """
        获取股票的收盘价数据。

        参数:
        code (str): 股票代码。
        start_delta (datetime.timedelta): 开始时间, 默认为今天10点。
        end_delta (datetime.timedelta): 结束时间, 默认为今天10点半。
        frequency (str, optional): 数据获取频率, 默认为 '1' 分钟。
            - 可以是 '1', '3', '5', '10', '15', '30', '60' 等分钟级别, 或者 'D' 日频, 'W' 周频等。

        返回:
        pandas.DataFrame: 包含收盘价数据的 DataFrame。

        示例:
        >>> DataFetcher = DataFetcher()
        >>> data_df = DataFetcher.get_stock_close_price('000852.SH')
        >>> print(data_df)
        """
        start_time = self.dt + start_delta
        end_time = self.dt + end_delta
        return self.stock_helper.get_stock_price(code=code, start_date=start_time, end_date=end_time,
                                                 frequency=frequency)

    def __getattr__(self, name):
        """
        代理所有未定义的属性和方法到内部的 datetime.datetime 实例。
        """
        return getattr(self.dt, name)

    def __repr__(self):
        """
        返回对象的字符串表示形式。
        """
        return repr(self.dt)

    def __str__(self):
        """
        返回对象的字符串表示形式。
        """
        return str(self.dt)

    def format(self, fmt: str) -> str:
        """
        格式化日期时间字符串。

        参数:
        fmt (str): 格式字符串，如 '%Y-%m-%d %H:%M:%S'。

        返回:
        str: 格式化的日期时间字符串。
        """
        return self.dt.strftime(fmt)

    def __add__(self, other: datetime.timedelta) -> 'TradingDay':
        """
        支持加法操作。

        参数:
        other (timedelta): 要加上的时间间隔。

        返回:
        TradingDay: 新的 TradingDay 实例。
        """
        new_dt = self.dt + other
        return TradingDay(new_dt)

    def __sub__(self, other: Union['TradingDay', datetime.timedelta]) -> Union['TradingDay', datetime.timedelta]:
        """
        支持减法操作。

        参数:
        other (Union[TradingDay, timedelta]): 要减去的时间间隔或 TradingDay 实例。

        返回:
        Union[TradingDay, timedelta]: 新的 TradingDay 实例或时间间隔。
        """
        if isinstance(other, datetime.timedelta):
            new_dt = self.dt - other
            return TradingDay(new_dt)
        elif isinstance(other, TradingDay):
            return self.dt - other.dt
        else:
            raise TypeError("Unsupported operand type for -: 'TradingDay' and '{}'".format(type(other).__name__))

    def __lt__(self, other):
        if isinstance(other, TradingDay):
            return self.dt < other.dt
        elif isinstance(other, datetime.datetime):
            return self.dt < other
        else:
            raise TypeError("Unsupported operand type for <: 'TradingDay' and '{}'".format(type(other).__name__))

    def __le__(self, other):
        if isinstance(other, TradingDay):
            return self.dt <= other.dt
        elif isinstance(other, datetime.datetime):
            return self.dt <= other
        else:
            raise TypeError("Unsupported operand type for <=: 'TradingDay' and '{}'".format(type(other).__name__))

    def __gt__(self, other):
        if isinstance(other, TradingDay):
            return self.dt > other.dt
        elif isinstance(other, datetime.datetime):
            return self.dt > other
        else:
            raise TypeError("Unsupported operand type for >: 'TradingDay' and '{}'".format(type(other).__name__))

    def __ge__(self, other):
        if isinstance(other, TradingDay):
            return self.dt >= other.dt
        elif isinstance(other, datetime.datetime):
            return self.dt >= other
        else:
            raise TypeError("Unsupported operand type for >=: 'TradingDay' and '{}'".format(type(other).__name__))
    
    
class TradingWeek:
    def __init__(self, 
                 mark_day: Union[str, datetime.datetime, TradingDay], 
                 stock_helper: StockDataFetcher = StockDataFetcher(), 
                 date_helper: DateHelper = DateHelper(), 
                 intra_reposition: bool = True
                 ) -> None:
        """
        初始化 Week 类，用于计算指定周的交易日信息。

        参数:
        date_helper (DateHelper): 提供辅助方法的对象，例如获取目标周。
        mark_day (Union[str, datetime.datetime, TradingDay]): 用于计算目标周的标记日期。
        intra_reposition (bool): 是否启用周内重新定位, 默认为True。
        """
        self.intra_reposition = intra_reposition
        if isinstance(mark_day, TradingDay):
            mark_day = mark_day.dt
        
        self.date_helper = date_helper
        self.stock_helper = stock_helper
        
        self.start, self.end = date_helper.get_target_week(mark_day)
        self.trading_days = self._get_trading_days()
        self.early_week_days, self.late_week_days = self._split_week_days()

        if not self.trading_days:
            self.has_trading_days = False
        else:
            self.has_trading_days = True
            if isinstance(self.trading_days[0], datetime.datetime):
                self.first_repositioning_day = TradingDay(self.trading_days[0])
            else:
                self.first_repositioning_day = self.trading_days[0]
            if isinstance(self.trading_days[-1], datetime.datetime):
                self.final_signal_generating_day = TradingDay(self.trading_days[-1])
            else:
                self.final_signal_generating_day = self.trading_days[-1]

            if not self.early_week_days:
                self.intra_reposition = False

            if self.intra_reposition:
                if isinstance(self.early_week_days[-1], datetime.datetime):
                    self.intra_signal_generating_day = TradingDay(self.early_week_days[-1])
                else:
                    self.intra_signal_generating_day = self.early_week_days[-1]
                if self.late_week_days and isinstance(self.late_week_days[0], datetime.datetime):
                    self.intra_repositioning_day = TradingDay(self.late_week_days[0])
                else:
                    self.intra_repositioning_day = self.late_week_days[0] if self.late_week_days else self.early_week_days[-1]

    def _get_trading_days(self) -> List[TradingDay]:
        """
        使用 WindPy 库获取从开始日期到结束日期之间的所有交易日。
        """
        trading_days_data = self.stock_helper.w.tdays(self.start, self.end, '').Data
        if len(trading_days_data) == 0:
            return []
        trading_days_datetime = self.stock_helper.w.tdays(self.start, self.end, '').Data[0]
        trading_days = [TradingDay(day) if isinstance(day, datetime.datetime) else day for day in trading_days_datetime]
        return trading_days

    def _split_week_days(self) -> Tuple[List[TradingDay], List[TradingDay]]:
        """
        将交易日分为两个列表：早于等于周二的交易日和晚于周二的交易日。
        如果找不到周二的交易日, 则所有交易日都视为早期交易日。
        """
        tuesday = next((day for day in self.trading_days if day.weekday() == 1), None)
        if tuesday:
            early_week = [day for day in self.trading_days if day <= tuesday]
            late_week = [day for day in self.trading_days if day > tuesday]
        else:
            early_week = list(self.trading_days)
            late_week = []
        return early_week, late_week

    def no_trading_days_error(self) -> None:
        """
        当没有交易日时触发的错误处理方法。
        抛出 ValueError 异常。
        """
        raise ValueError("在开始和结束日期之间未找到交易日。")






