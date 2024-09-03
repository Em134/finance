import tushare as ts
import datetime
from typing import List, Dict, Union, Tuple, Any
import os
import yaml
from WindPy import w
from dateutil.relativedelta import relativedelta
import pandas as pd
from datetime import timedelta
from chinese_calendar import is_workday


class DateHelper:
    """
    DateHelper 是一个用于处理日期和时间的辅助类, 提供了一系列静态方法来格式化、转换和操作日期。

    该类包含的方法能够处理多种日期时间格式, 并支持常见的日期时间操作, 如：
    - 将日期转换为特定格式。
    - 转换日期时间字符串为 pd.Timestamp 或 datetime 对象。
    - 根据给定的时间单位调整日期。
    - 获取指定周的起始和结束日期。
    - 根据时间精度获取当前时间的字符串表示。

    所有方法均为静态方法, 可以直接通过类名调用, 无需实例化。
    """
    @staticmethod
    def format_date_for_wind(date: Union[str, datetime.datetime]) -> str:
        """
        将日期转换为 Wind API 需要的格式。

        参数:
        date (str or datetime.datetime): 日期。

        返回:
        str: 转换后的日期, 格式为 "YYYYMMDD HH:MM:SS"。
        """
        if isinstance(date, datetime.datetime):
            return date.strftime("%Y%m%d %H:%M:%S")
        elif isinstance(date, str):
            try:
                # 支持 "YYYYMMDD" 格式
                date_obj = datetime.datetime.strptime(date, "%Y%m%d")
                return date_obj.strftime("%Y%m%d 00:00:00")
            except ValueError:
                try:
                    # 支持 "YYYY-MM-DD" 格式
                    date_obj = datetime.datetime.strptime(date, "%Y-%m-%d")
                    return date_obj.strftime("%Y%m%d 00:00:00")
                except ValueError:
                    try:
                        # 支持 "YYYYMMDD HH:MM:SS" 格式
                        date_obj = datetime.datetime.strptime(date, "%Y%m%d %H:%M:%S")
                        return date_obj.strftime("%Y%m%d %H:%M:%S")
                    except ValueError:
                        try:
                            # 支持 "YYYY-MM-DD HH:MM:SS" 格式
                            date_obj = datetime.datetime.strptime(date, "%Y-%m-%d %H:%M:%S")
                            return date_obj.strftime("%Y%m%d %H:%M:%S")
                        except ValueError:
                            raise ValueError("Invalid date format. Expected 'YYYYMMDD', 'YYYY-MM-DD', 'YYYYMMDD HH:MM:SS', or 'YYYY-MM-DD HH:MM:SS'.")
        else:
            raise TypeError("Date must be a string or datetime.datetime object.")

    @staticmethod
    def convert_to_datetime(date_input: Union[str, pd.Timestamp, datetime.datetime]) -> pd.Timestamp:
        """
        将多种格式的日期字符串转换为 datetime 对象。

        参数:
        date_input (str): 输入的日期字符串, 支持多种格式。

        返回:
        datetime.datetime: 转换后的 datetime 对象。
        """
        if isinstance(date_input, datetime.datetime):
            return date_input
        elif isinstance(date_input, pd.Timestamp):
            return date_input.to_pydatetime()
        
        formats = [
            "%Y%m%d",           # YYYYMMDD
            "%Y-%m-%d",         # YYYY-MM-DD
            "%Y%m%d %H:%M:%S",  # YYYYMMDD HH:MM:SS
            "%Y-%m-%d %H:%M:%S",# YYYY-MM-DD HH:MM:SS
            "%Y/%m/%d",         # YYYY/MM/DD
            "%Y.%m.%d",         # YYYY.MM.DD
            "%Y/%m/%d %H:%M:%S",# YYYY/MM/DD HH:MM:SS
            "%Y.%m.%d %H:%M:%S",# YYYY.MM.DD HH:MM:SS
            "%Y年%m月%d日",     # YYYY年MM月DD日
            "%Y年%m月%d日 %H时%M分%S秒",# YYYY年MM月DD日 HH时MM分SS秒
            "%Y年%m月%d日 %H:%M:%S",# YYYY年MM月DD日 HH:MM:SS
            "%Y年%m月%d日 %H:%M",   # YYYY年MM月DD日 HH:MM
            "%Y年%m月%d日 %H时%M分", # YYYY年MM月DD日 HH时MM分
            "%Y年%m月%d",       # YYYY年MM月DD
            "%Y年%m月",         # YYYY年MM月
            "%Y年",             # YYYY年
            "%Y%m",             # YYYYMM
            "%Y%m%d %H:%M",     # YYYYMMDD HH:MM
            "%Y-%m-%d %H:%M",   # YYYY-MM-DD HH:MM
            "%Y/%m/%d %H:%M",   # YYYY/MM/DD HH:MM
            "%Y.%m.%d %H:%M",   # YYYY.MM.DD HH:MM
            "%Y%m%d %H时%M分%S秒", # YYYYMMDD HH时MM分SS秒
            "%Y%m%d %H时%M分",  # YYYYMMDD HH时MM分
            "%Y%m%d %H:%M",     # YYYYMMDD HH:MM
            "%Y%m%d %H时",      # YYYYMMDD HH时
            "%Y%m%d %H",        # YYYYMMDD HH
            "%Y-%m-%d %H时%M分%S秒", # YYYY-MM-DD HH时MM分SS秒
            "%Y-%m-%d %H时%M分", # YYYY-MM-DD HH时MM分
            "%Y-%m-%d %H时",    # YYYY-MM-DD HH时
            "%Y-%m-%d %H",      # YYYY-MM-DD HH
            "%Y/%m/%d %H时%M分%S秒", # YYYY/MM/DD HH时MM分SS秒
            "%Y/%m/%d %H时%M分", # YYYY/MM/DD HH时MM分
            "%Y/%m/%d %H时",    # YYYY/MM/DD HH时
            "%Y/%m/%d %H",      # YYYY/MM/DD HH
            "%Y.%m.%d %H时%M分%S秒", # YYYY.MM.DD HH时MM分SS秒
            "%Y.%m.%d %H时%M分", # YYYY.MM.DD HH时MM分
            "%Y.%m.%d %H时",    # YYYY.MM.DD HH时
            "%Y.%m.%d %H",      # YYYY.MM.DD HH
            "%Y年%m月%d日%H:%M:%S", # YYYY年MM月DD日HH:MM:SS
            "%Y年%m月%d日%H:%M",   # YYYY年MM月DD日HH:MM
            "%Y年%m月%d日%H时%M分%S秒", # YYYY年MM月DD日HH时MM分SS秒
            "%Y年%m月%d日%H时%M分", # YYYY年MM月DD日HH时MM分
            "%Y年%m月%d日%H时",     # YYYY年MM月DD日HH时
            "%Y年%m月%d日%H",       # YYYY年MM月DD日HH
            "%Y%m%d%H%M%S",          # YYYYMMDDHHMMSS
            "%Y%m%d%H%M",            # YYYYMMDDHHMM
            "%Y%m%d%H",              # YYYYMMDDHH
            "%Y%m%d%H时%M分%S秒",    # YYYYMMDDHH时MM分SS秒
            "%Y%m%d%H时%M分",        # YYYYMMDDHH时MM分
            "%Y%m%d%H时",            # YYYYMMDDHH时
            "%Y%m%d%H",              # YYYYMMDDHH
            "%Y-%m-%d%H:%M:%S",      # YYYY-MM-DDHH:MM:SS
            "%Y-%m-%d%H:%M",         # YYYY-MM-DDHH:MM
            "%Y-%m-%d%H时%M分%S秒",  # YYYY-MM-DDHH时MM分SS秒
            "%Y-%m-%d%H时%M分",      # YYYY-MM-DDHH时MM分
            "%Y-%m-%d%H时",          # YYYY-MM-DDHH时
            "%Y-%m-%d%H",            # YYYY-MM-DDHH
            "%Y/%m/%d%H:%M:%S",      # YYYY/MM/DDHH:MM:SS
            "%Y/%m/%d%H:%M",         # YYYY/MM/DDHH:MM
            "%Y/%m/%d%H时%M分%S秒",  # YYYY/MM/DDHH时MM分SS秒
            "%Y/%m/%d%H时%M分",      # YYYY/MM/DDHH时MM分
            "%Y/%m/%d%H时",          # YYYY/MM/DDHH时
            "%Y/%m/%d%H",            # YYYY/MM/DDHH
            "%Y.%m.%d%H:%M:%S",      # YYYY.MM.DDHH:MM:SS
            "%Y.%m.%d%H:%M",         # YYYY.MM.DDHH:MM
            "%Y.%m.%d%H时%M分%S秒",  # YYYY.MM.DDHH时MM分SS秒
            "%Y.%m.%d%H时%M分",      # YYYY.MM.DDHH时MM分
            "%Y.%m.%d%H时",          # YYYY.MM.DDHH时
            "%Y.%m.%d%H",            # YYYY.MM.DDHH
            "%Y年%m月%d日%H:%M:%S",  # YYYY年MM月DD日HH:MM:SS
            "%Y年%m月%d日%H:%M",     # YYYY年MM月DD日HH:MM
            "%Y年%m月%d日%H时%M分%S秒", # YYYY年MM月DD日HH时MM分SS秒
            "%Y年%m月%d日%H时%M分",  # YYYY年MM月DD日HH时MM分
            "%Y年%m月%d日%H时",      # YYYY年MM月DD日HH时
            "%Y年%m月%d日%H",        # YYYY年MM月DD日HH
        ]

        for fmt in formats:
            try:
                return datetime.datetime.strptime(date_input, fmt)
            except ValueError:
                continue
        raise ValueError("Invalid date format.")
    
    @staticmethod
    def get_target_week(target_date: Union[str, pd.Timestamp]) -> Tuple[pd.Timestamp, pd.Timestamp]:
        """
        设置当前周的数据和日期范围。

        Args:
            target_date (Union[str, pd.Timestamp]): 目标日期, 可以是字符串或 pd.Timestamp。

        Returns:
            Tuple[pd.Timestamp, pd.Timestamp]: 包含当前周的开始和结束日期的元组。
        """
        # 将输入的日期转换为 pd.Timestamp
        timestamp_date = DateHelper.convert_to_timestamp(target_date)

        # 确定当前周的开始和结束日期
        start_of_week = timestamp_date - pd.Timedelta(days=timestamp_date.weekday())
        end_of_week = start_of_week + pd.Timedelta(days=6)

        # 返回当前周的开始和结束日期
        return start_of_week, end_of_week
    
    @staticmethod
    def get_date_str_days_ago(days_ago: int = 1) -> str:
        """
        获取几天前的日期并转换为 'YYYYMMDD' 格式。
        
        参数:
        days_ago : int
            几天前的天数, 例如 1 表示昨天, 2 表示前天等。
        
        返回:
        str: 几天前的日期, 格式为 'YYYYMMDD'。
        """
        # 获取今天的日期
        today = datetime.datetime.now()
        
        # 计算几天前的日期
        date_days_ago = today - datetime.timedelta(days=days_ago)
        
        # 将日期格式化为 'YYYYMMDD'
        date_str = date_days_ago.strftime('%Y%m%d')
        
        return date_str

    @staticmethod
    def convert_date_format_1(date_str: str) -> str:
        """
        将形如 "20221110" 或 "20221110 12:34:56" 的日期字符串转换为 "2022-11-10" 或 "2022-11-10 12:34:56" 的格式。
        
        参数:
        date_str (str): 输入的日期字符串, 格式为 "YYYYMMDD" 或 "YYYYMMDD HH:MM:SS"。
        
        返回:
        str: 转换后的日期字符串, 格式为 "YYYY-MM-DD" 或 "YYYY-MM-DD HH:MM:SS"。
        """
        # 尝试使用 "YYYYMMDD" 或 "YYYYMMDD HH:MM:SS" 格式解析日期
        try:
            date_obj = datetime.datetime.strptime(date_str, "%Y%m%d %H:%M:%S")
            date_format = "%Y-%m-%d %H:%M:%S"
        except ValueError:
            try:
                date_obj = datetime.datetime.strptime(date_str, "%Y%m%d")
                date_format = "%Y-%m-%d"
            except ValueError:
                raise ValueError("Invalid date format. Expected 'YYYYMMDD' or 'YYYYMMDD HH:MM:SS'.")
        
        # 将 datetime 对象转换为相应格式的字符串
        formatted_date = date_obj.strftime(date_format)
        
        return formatted_date
    
    @staticmethod
    def convert_date_format_2(date_str: str) -> str:
        """
        将形如 "2022-11-10" 或 "2022-11-10 12:34:56" 的日期字符串转换为 "20221110" 或 "20221110 12:34:56" 的格式。
        
        参数:
        date_str (str): 输入的日期字符串, 格式为 "YYYY-MM-DD" 或 "YYYY-MM-DD HH:MM:SS"。
        
        返回:
        str: 转换后的日期字符串, 格式为 "YYYYMMDD" 或 "YYYYMMDD HH:MM:SS"。
        """
        # 尝试使用 "YYYY-MM-DD" 或 "YYYY-MM-DD HH:MM:SS" 格式解析日期
        try:
            date_obj = datetime.datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
            date_format = "%Y%m%d %H:%M:%S"
        except ValueError:
            try:
                date_obj = datetime.datetime.strptime(date_str, "%Y-%m-%d")
                date_format = "%Y%m%d"
            except ValueError:
                raise ValueError("Invalid date format. Expected 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM:SS'.")
        
        # 将 datetime 对象转换为相应格式的字符串
        formatted_date = date_obj.strftime(date_format)
        
        return formatted_date
    
    @staticmethod
    def convert_to_timestamp(date_input: Union[str, datetime.datetime]) -> datetime:
        """
        将输入的日期转换为pd.Timestamp格式。
        
        Args:
            date_input (Union[str, pd.Timestamp, datetime.datetime]): 输入的日期, 可以是字符串或pd.Timestamp。
            
        Returns:
            pd.Timestamp: 转换后的日期。
        """
        if isinstance(date_input, pd.Timestamp):
            return date_input
        elif isinstance(date_input, str):
            return pd.to_datetime(date_input)
        elif isinstance(date_input, datetime.datetime):
            return pd.to_datetime(date_input)
        else:
            raise ValueError("Invalid date format. Please provide a string, datetime.datetime or pd.Timestamp.")
        
    @staticmethod
    def datetime_to_string(dt: datetime.datetime, format_type: str = 'default') -> str:
        """
        将 datetime 对象转换为指定格式的日期字符串。
        
        参数:
        dt (datetime.datetime): 输入的 datetime 对象。
        format_type (str): 输出日期字符串的格式类型。默认为 'default', 可选值为 'default', 'hyphenated', 'with_time', 或 'with_time_hyphenated'。
        
        返回:
        str: 转换后的日期字符串。
        """
        # 根据 format_type 选择相应的日期格式
        if format_type == 'default':
            date_str = dt.strftime("%Y%m%d")
        elif format_type == 'hyphenated':
            date_str = dt.strftime("%Y-%m-%d")
        elif format_type == 'with_time':
            date_str = dt.strftime("%Y%m%d%H%M%S")
        elif format_type == 'with_time_hyphenated':
            date_str = dt.strftime("%Y-%m-%d %H:%M:%S")
        else:
            raise ValueError("Invalid format_type. Expected 'default', 'hyphenated', 'with_time', or 'with_time_hyphenated'.")
        
        return date_str
    
    @staticmethod
    def adjust_date(input_date, amount: int = 1, operation: str = "add", unit: str = "day", time_precision: str = "day"):
        """
        根据给定的操作和单位调整日期, 并根据输入日期的类型返回相同类型的结果。
        如果时间精度设置为秒级, 则输出结果为秒级精度, 且在进行加减操作时也基于秒级精度。
        如果时间精度设置为其他级别, 则输出结果为该级别的精度。

        参数:
        input_date (str or datetime.datetime or float): 输入的日期, 可以是 "YYYYMMDD" 或 "YYYY-MM-DD" 或 "YYYYMMDD HH:MM:SS" 或 "YYYY-MM-DD HH:MM:SS" 格式的字符串, 也可以是 datetime.datetime 对象或 timestamp 浮点数。
        operation (str): 操作类型, 可以是 "add" 或 "subtract"。
        unit (str): 单位类型, 可以是 "year", "month", "day", "hour", "minute", 或 "second"。
        amount (int): 要加减的单位数量。
        time_precision (str): 时间精确度, 可以是 "year", "month", "day", "hour", "minute", 或 "second"。默认为 "day"。

        返回:
        str or datetime.datetime or float: 调整后的日期, 类型与输入日期相同, 并根据 time_precision 参数确定精度。
        """
        # 将输入日期转换为 datetime 对象

        date_obj = DateHelper.convert_to_datetime(input_date)
        # 调整日期
        if unit == "year":
            year = date_obj.year + (amount if operation == "add" else -amount)
            try:
                adjusted_date = date_obj.replace(year=year)
            except ValueError:
                # Handle February 29 in non-leap years
                adjusted_date = datetime.datetime(year, date_obj.month + 1 if date_obj.month == 2 else date_obj.month, 1)
        elif unit == "month":
            month = date_obj.month + (amount if operation == "add" else -amount)
            year, month = divmod(month - 1, 12)
            year += date_obj.year
            month += 1
            try:
                adjusted_date = date_obj.replace(year=year, month=month)
            except ValueError:
                # Handle days that do not exist in the new month
                adjusted_date = datetime.datetime(year, month, 1)
        elif unit == "day":
            delta = timedelta(days=amount if operation == "add" else -amount)
            adjusted_date = date_obj + delta
        elif unit == "hour":
            delta = timedelta(hours=amount if operation == "add" else -amount)
            adjusted_date = date_obj + delta
        elif unit == "minute":
            delta = timedelta(minutes=amount if operation == "add" else -amount)
            adjusted_date = date_obj + delta
        elif unit == "second":
            delta = timedelta(seconds=amount if operation == "add" else -amount)
            adjusted_date = date_obj + delta
        else:
            raise ValueError("Invalid unit type. Supported units are 'year', 'month', 'day', 'hour', 'minute', and 'second'.")

        # 根据时间精度返回适当格式的结果
        if time_precision == "year":
            result = datetime.datetime(adjusted_date.year, 1, 1)
        elif time_precision == "month":
            result = datetime.datetime(adjusted_date.year, adjusted_date.month, 1)
        elif time_precision == "day":
            result = datetime.datetime(adjusted_date.year, adjusted_date.month, adjusted_date.day)
        elif time_precision == "hour":
            result = datetime.datetime(adjusted_date.year, adjusted_date.month, adjusted_date.day, adjusted_date.hour)
        elif time_precision == "minute":
            result = datetime.datetime(adjusted_date.year, adjusted_date.month, adjusted_date.day, adjusted_date.hour, adjusted_date.minute)
        elif time_precision == "second":
            result = adjusted_date
        else:
            raise ValueError("Invalid time precision. Supported precisions are 'year', 'month', 'day', 'hour', 'minute', and 'second'.")

        # 根据输入类型返回相应格式的结果
        if isinstance(input_date, str):
            if time_precision == "year":
                return result.strftime("%Y")
            elif time_precision == "month":
                return result.strftime("%Y-%m")
            elif time_precision == "day":
                return result.strftime("%Y-%m-%d")
            elif time_precision == "hour":
                return result.strftime("%Y-%m-%d %H")
            elif time_precision == "minute":
                return result.strftime("%Y-%m-%d %H:%M")
            elif time_precision == "second":
                return result.strftime("%Y-%m-%d %H:%M:%S")

        elif isinstance(input_date, datetime.datetime):
            return result
        elif isinstance(input_date, (float, int)):
            return result.timestamp()
    
    @staticmethod
    def get_current_time(time_precision: str = "second") -> str:
        """
        获取当前时间, 根据指定的时间精确度返回格式化的当前时间字符串。

        参数:
        time_precision (str): 时间精确度, 可以是 "year", "month", "day", "hour", "minute", 或 "second"。默认为 "second"。

        返回:
        str: 当前时间字符串, 格式化为指定的时间精确度。
        """
        now = datetime.datetime.now()

        # 根据 time_precision 选择相应的日期格式
        if time_precision == "year":
            date_str = now.strftime("%Y")
        elif time_precision == "month":
            date_str = now.strftime("%Y-%m")
        elif time_precision == "day":
            date_str = now.strftime("%Y-%m-%d")
        elif time_precision == "hour":
            date_str = now.strftime("%Y-%m-%d %H")
        elif time_precision == "minute":
            date_str = now.strftime("%Y-%m-%d %H:%M")
        elif time_precision == "second":
            date_str = now.strftime("%Y-%m-%d %H:%M:%S")
        else:
            raise ValueError("Invalid time_precision. Expected 'year', 'month', 'day', 'hour', 'minute', or 'second'.")
        
        return date_str                            

    @staticmethod
    def generate_date_range(start_date: Union[str, datetime.datetime, pd.Timestamp], 
                            end_date: Union[str, datetime.datetime, pd.Timestamp], 
                            step: int = 1) -> List[datetime.datetime]:
        """
        生成指定日期范围内的所有日期列表。

        参数:
        start_date (Union[str, datetime.datetime, pandas.Timestamp]): 起始日期。
        end_date (Union[str, datetime.datetime, pandas.Timestamp]): 结束日期。
        step (int): 步长, 即每隔多少天生成一个日期, 默认为1天。

        返回:
        List[datetime.datetime]: 日期范围内的所有日期列表。
        """
        start_date = DateHelper.convert_to_datetime(start_date)
        end_date = DateHelper.convert_to_datetime(end_date)
        
        date_list = []
        current_date = start_date
        
        while current_date < end_date:
            if not isinstance(current_date, datetime.datetime):
                date_list.append(current_date.to_pydatetime())
            else:
                date_list.append(current_date)
            current_date += timedelta(days=step)
        
        return date_list

    @staticmethod
    def to_trading_days(source_days: List[pd.Timestamp]) -> List[pd.Timestamp]:
        """
        将给定的日期列表转换为最近的交易日列表。

        参数:
        source_days (List[pd.Timestamp]): 需要转换的日期列表。

        返回:
        List[pd.Timestamp]: 最近的交易日列表。
        """

        def find_max_below_threshold(lst, threshold):
            """
            在列表中查找最大值, 该值小于给定的阈值。

            参数:
            lst (List[pd.Timestamp]): 列表。
            threshold (pd.Timestamp): 阈值。

            返回:
            pd.Timestamp: 小于阈值的最大值, 如果不存在则返回 None。
            """
            # 使用列表推导式筛选出小于threshold的元素
            below_threshold = [x for x in lst if x <= threshold]
            
            # 如果筛选后的列表为空, 则返回None, 否则返回最大值
            return max(below_threshold) if below_threshold else None

        # 获取交易日列表
        trading_days = w.tdays(source_days[0] - relativedelta(days=7), source_days[-1] + relativedelta(days=7), "").Data[0]

        # 转换源日期列表
        res_list = [
            day if day in trading_days else pd.to_datetime(find_max_below_threshold(trading_days, day))
            for day in source_days
        ]

        return res_list
    
    @staticmethod
    def is_trading_day(date: Union[str, datetime.datetime]) -> bool:
        """
        判断给定日期是否为交易日。
        
        参数:
        date (datetime.datetime): 要检查的日期。
        
        返回:
        bool: 如果是交易日返回 True, 否则返回 False。
        """
        return is_workday(DateHelper.convert_to_datetime(date))


class StockDataFetcher:
    def __init__(self, ts_token: str = None, date_helper: DateHelper = DateHelper()) -> None:
        """
        初始化 DataFetcher 类, 尝试从配置文件中读取 tushare_token, 并尝试启动 Wind 终端。
        
        参数:
        ts_token (str): 可选的 tushare token, 如果未提供, 则尝试从配置文件中读取。
        """
        
        self.date_helper = date_helper
        
        # 尝试从配置文件中读取 tushare_token
        try:
            with open('configs/config.yaml', 'r') as file:
                config = yaml.safe_load(file)
                ts_token = config.get('tushare_token', 0)
        except FileNotFoundError:
            ts_token = 0  # 如果配置文件不存在, 则使用默认值 0
        except Exception as e:
            print(f"Error reading configuration: {e}")
            ts_token = 0  # 如果出现其他错误, 则使用默认值 0

        # 设置 tushare token
        if ts_token:
            ts.set_token(ts_token)
            self.pro = ts.pro_api()
        else:
            # 如果 ts_token 为 0, 则不设置 tushare token
            self.pro = None  # 或者设置为其他默认值
        
        # 尝试启动 Wind 终端
        self._try_start_wind_terminal()

    def _try_start_wind_terminal(self):
        """
        尝试启动 Wind 终端。
        """
        try:
            # 尝试启动 Wind 终端
            self.w = w
            self.w.start()
            print("Wind terminal started successfully.")
        except NameError:
            print("Wind terminal not available. Make sure the Wind library is imported and accessible.")
        except Exception as e:
            print(f"Error starting Wind terminal: {e}")

    def get_risk_free_rate(self) -> float:
        """
        获取无风险利率。
        
        返回:
        float: 无风险利率值。
        """
        if self.pro is None:
            raise ValueError("Tushare API client not initialized. Please provide a valid ts_token.")
        
        # 获取昨天的日期
        trade_date = self.date_helper.get_date_str_days_ago(1)
        
        # 查询无风险利率
        _df = self.pro.yc_cb(ts_code='1001.CB', curve_type='0', trade_date=trade_date)
        
        # 获取5年期的无风险利率
        risk_free_rate = _df[_df['curve_term'] == 5]['yield'].values[0] * 0.01
        
        return risk_free_rate
    
    def get_stock_price(self, code: str = '000852.SH', 
                              start_date: Union[str, datetime.datetime] = None, 
                              end_date: Union[str, datetime.datetime] = None, 
                              frequency: str = '1'
                              ) -> pd.DataFrame:
        """
        获取股票的收盘价数据。

        参数:
        code (str): 股票代码。
        start_date (str or datetime.datetime, optional): 开始日期, 默认为昨天。
            - 如果是 str, 支持 "YYYYMMDD", "YYYY-MM-DD", "YYYYMMDD HH:MM:SS", "YYYY-MM-DD HH:MM:SS" 格式。
            - 如果是 datetime.datetime, 将直接使用该对象。
            默认值: None, 表示使用昨天的日期。
        end_date (str or datetime.datetime, optional): 结束日期, 默认为当前时间。
            - 如果是 str, 支持 "YYYYMMDD", "YYYY-MM-DD", "YYYYMMDD HH:MM:SS", "YYYY-MM-DD HH:MM:SS" 格式。
            - 如果是 datetime.datetime, 将直接使用该对象。
            默认值: None, 表示使用当前时间。
        frequency (str, optional): 数据获取频率, 默认为 '1' 分钟。
            - 可以是 '1', '3', '5', '10', '15', '30', '60' 等分钟级别, 或者 'D' 日频, 'W' 周频等。

        返回:
        pandas.DataFrame: 包含收盘价数据的 DataFrame。

        示例:
        >>> DataFetcher = DataFetcher()
        >>> data_df = DataFetcher.get_stock_close_price('000852.SH')
        >>> print(data_df)
        """
        # 如果没有提供开始日期, 则默认为昨天
        if start_date is None:
            start_date = self.date_helper.adjust_date(self.date_helper.get_current_time(), 1, 'subtract', 'day', 'second')

        # 如果没有提供结束日期, 则默认为当前时间
        if end_date is None:
            end_date = self.date_helper.get_current_time()

        # 将日期转换为 Wind API 需要的格式
        start_date_str = self.date_helper.format_date_for_wind(start_date)
        end_date_str = self.date_helper.format_date_for_wind(end_date)

        # 使用 Wind API 获取数据
        data_df = self.w.wsi(code, 'close', start_date_str, end_date_str, f"BarSize={frequency}", usedf=True)[1]

        return data_df

    def close(self, code: str, date: Union[str, datetime.datetime] = datetime.datetime.now(), close_type: str="close", priceAdj: str="U", cycle: str="D"):
        return w.wss(code, close_type, f"tradeDate={self.date_helper.datetime_to_string(date)};priceAdj={priceAdj};cycle={cycle}")
    

class PredictionFetcher:
    def __init__(self, source_path: str, date_column: str = '交易日期', prediction_column: str = 'predictions') -> None:
        """
        初始化PredictionFetcher类。

        参数:
        source_path (str): 数据源文件路径。
        date_column (str): 日期列的名称, 默认为'交易日期'。
        prediction_column (str): 预测值列的名称, 默认为'predictions'。
        """
        # 根据文件类型读取数据
        if source_path.endswith('.csv'):
            self.data = pd.read_csv(source_path)
        elif source_path.endswith(('.xls', '.xlsx')):
            self.data = pd.read_excel(source_path)
        else:
            raise ValueError("Unsupported file format. Please provide a CSV or Excel file.")
        
        # 将日期列转换为datetime格式
        self.date_column = date_column
        self.data[date_column] = pd.to_datetime(self.data[date_column])
        
        self.prediction_column = prediction_column
        
        # 按照prediction_column列的值进行升序排序
        self.data.sort_values(by=self.date_column, ascending=True, inplace=True)
        # 重置索引
        self.data.reset_index(drop=True, inplace=True)
        
        # 初始化DateHelper实例
        self.date_helper = DateHelper()

    def get_prediction_for_date(self, target_date: Union[str, datetime.datetime]) -> float:
        """
        获取给定日期的预测值。

        参数:
        target_date (Union[str, datetime.datetime]): 目标日期。

        返回:
        float: 给定日期的预测值。
        """
        # 转换日期为datetime格式
        target_date = self.date_helper.convert_to_datetime(target_date)
        # 查找对应日期的预测值
        prediction = self.data[self.data[self.date_column] == target_date][self.prediction_column].values
        # 返回预测值, 如果没有找到则返回None
        return prediction[0] if prediction.size > 0 else None
    
    def get_format_signal(self, target_date: Union[str, datetime.datetime]) -> Dict:
        """
        获取目标日期所在周的所有预测值。

        参数:
        target_date (Union[str, datetime.datetime]): 目标日期。

        返回:
        list: 一周内每一天的预测值列表。
        """
        # 获取目标周的开始和结束日期
        start_day, end_day = self.date_helper.get_target_week(target_date)
        tuesday_signal = self.get_prediction_for_date(start_day + relativedelta(days=1))
        friday_signal = self.get_prediction_for_date(start_day + relativedelta(days=4))
        
        return{'tuesday_signal': tuesday_signal, 'friday_signal': friday_signal}
    
    def get_week_predictions(self, target_date: Union[str, datetime.datetime]) -> list:
        """
        获取目标日期所在周的所有预测值。

        参数:
        target_date (Union[str, datetime.datetime]): 目标日期。

        返回:
        list: 一周内每一天的预测值列表。
        """
        # 获取目标周的开始和结束日期
        start_day, end_day = self.date_helper.get_target_week(target_date)
        # 生成一周内的日期列表
        week_list = [start_day + relativedelta(days=i) for i in range(7)]
        # 获取这一周内每一天的数据
        week_data = self.data[self.data[self.date_column].isin(week_list)]
        
        # 返回这一周内每一天的预测值
        return week_data[self.prediction_column].tolist()



