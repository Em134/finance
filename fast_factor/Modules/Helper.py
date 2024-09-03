import tushare as ts
import datetime
from typing import List, Dict
import os 
import yaml


ts.set_token('519550beb1e16be8a2bbd60fd5148d3d851074e6f5ec866832560432')
pro = ts.pro_api()


class Helper:
    def __init__(self, ts_token=None) -> None:
        if ts_token is not None:
            ts.set_token(ts_token)
            self.pro = ts.pro_api()
            self.get_risk_free_rate()
        else:
            self.pro = None
    
    @staticmethod
    def get_date_str_days_ago(days_ago: int = 1) -> str:
        """
        获取几天前的日期并转换为 'YYYYMMDD' 格式。

        参数:
        days_ago : int
            几天前的天数，例如 1 表示昨天，2 表示前天等。

        返回:
        str: 几天前的日期，格式为 'YYYYMMDD'。
        """
        # 获取今天的日期
        today = datetime.datetime.now()
        
        # 计算几天前的日期
        date_days_ago = today - datetime.timedelta(days=days_ago)
        
        # 将日期格式化为 'YYYYMMDD'
        date_str = date_days_ago.strftime('%Y%m%d')
        
        return date_str

    def get_risk_free_rate(self) -> float:
        """
        获取无风险利率。
        
        返回:
        float: 无风险利率值。
        """
        if self.pro is None:
            raise ValueError("Tushare API client not initialized. Please provide a valid ts_token.")
        
        # 获取昨天的日期
        trade_date = self.get_date_str_days_ago(1)
        
        # 查询无风险利率
        _df = self.pro.yc_cb(ts_code='1001.CB', curve_type='0', trade_date=trade_date)
        
        # 获取5年期的无风险利率
        risk_free_rate = _df[_df['curve_term'] == 5]['yield'].values[0] * 0.01
        
        return risk_free_rate
    
    @staticmethod
    def prior_index(input_list: List) -> str:
        """
        返回最优先的指数
        Args:
            input_list (List): _description_

        Returns:
            str: 优先的指数
        """
        if '沪深300指数' in input_list:
            return '沪深300指数'
        if '中证2000' in input_list:
            return '中证2000'
        if '中证1000' in input_list:
            return '中证1000'
        if '中证500' in input_list:
            return '中证500'
        
        return input_list[0]
    
    @staticmethod
    def get_config():
        with open(os.path.join("../configs/config.yaml"), encoding="utf-8") as f:
            config = yaml.load(f.read(), Loader=yaml.FullLoader)
        return config
    
    @staticmethod
    def convert_date_format_1(date_str: str) -> str:
        """
        将形如 "20221110" 的日期字符串转换为 "2022-11-10" 的格式。

        参数:
        date_str (str): 输入的日期字符串，格式为 "YYYYMMDD"。

        返回:
        str: 转换后的日期字符串，格式为 "YYYY-MM-DD"。
        """
        # 将输入的日期字符串解析为 datetime 对象
        date_obj = datetime.datetime.strptime(date_str, "%Y%m%d")

        # 将 datetime 对象转换为 "YYYY-MM-DD" 格式的字符串
        formatted_date = date_obj.strftime("%Y-%m-%d")

        return formatted_date
    
    @staticmethod
    def convert_date_format_2(date_str: str) -> str:
        """
        将形如 "2022-11-10" 的日期字符串转换为 "20221110" 的格式。

        参数:
        date_str (str): 输入的日期字符串，格式为 "YYYY-MM-DD"。

        返回:
        str: 转换后的日期字符串，格式为 "YYYYMMDD"。
        """
        # 将输入的日期字符串解析为 datetime 对象
        date_obj = datetime.datetime.strptime(date_str, "%Y-%m-%d")

        # 将 datetime 对象转换为 "YYYYMMDD" 格式的字符串
        formatted_date = date_obj.strftime("%Y%m%d")

        return formatted_date
    
    @staticmethod
    def convert_to_datetime(date_str: str) -> datetime.datetime :
        """
        将形如 "20221110" 或 "2022-11-10" 的日期字符串转换为 datetime 对象。

        参数:
        date_str (str): 输入的日期字符串，格式为 "YYYYMMDD" 或 "YYYY-MM-DD"。

        返回:
        datetime.datetime: 转换后的 datetime 对象。
        """
        # 尝试使用 "YYYYMMDD" 格式解析日期
        try:
            date_obj = datetime.datetime.strptime(date_str, "%Y%m%d")
        except ValueError:
            # 如果解析失败，则尝试使用 "YYYY-MM-DD" 格式解析日期
            try:
                date_obj = datetime.datetime.strptime(date_str, "%Y-%m-%d")
            except ValueError:
                raise ValueError("Invalid date format. Expected 'YYYYMMDD' or 'YYYY-MM-DD'.")
        
        return date_obj
    
    @staticmethod
    def datetime_to_string(dt: datetime.datetime, format_type: str = 'default') -> str:
        """
        将 datetime 对象转换为指定格式的日期字符串。

        参数:
        dt (datetime.datetime): 输入的 datetime 对象。
        format_type (str): 输出日期字符串的格式类型。默认为 'default'，可选值为 'default' 或 'hyphenated'。

        返回:
        str: 转换后的日期字符串。
        """
        # 根据 format_type 选择相应的日期格式
        if format_type == 'default':
            date_str = dt.strftime("%Y%m%d")
        elif format_type == 'hyphenated':
            date_str = dt.strftime("%Y-%m-%d")
        else:
            raise ValueError("Invalid format_type. Expected 'default' or 'hyphenated'.")

        return date_str
    
    @staticmethod
    def adjust_date(input_date, amount: int = 1, operation: str = "add", unit: str = "day"):
        """
        根据给定的操作和单位调整日期，并根据输入日期的类型返回相同类型的结果。

        参数:
        input_date (str or datetime.datetime): 输入的日期，可以是 "YYYYMMDD" 或 "YYYY-MM-DD" 格式的字符串，也可以是 datetime.datetime 对象。
        operation (str): 操作类型，可以是 "add" 或 "subtract"。
        unit (str): 单位类型，可以是 "year", "month", 或 "day"。
        amount (int): 要加减的单位数量。

        返回:
        str or datetime.datetime: 调整后的日期，类型与输入日期相同。
        """
        # 将输入日期转换为 datetime 对象
        if isinstance(input_date, str):
            try:
                date_obj = datetime.datetime.strptime(input_date, "%Y%m%d")
                format_type = "default"
            except ValueError:
                try:
                    date_obj = datetime.datetime.strptime(input_date, "%Y-%m-%d")
                    format_type = "hyphenated"
                except ValueError:
                    raise ValueError("Invalid date format. Expected 'YYYYMMDD' or 'YYYY-MM-DD'.")
        elif isinstance(input_date, datetime.datetime):
            date_obj = input_date
            format_type = None
        else:
            raise TypeError("Input date must be a string or datetime.datetime object.")
        
        # 根据操作类型和单位调整日期
        if operation == "add":
            if unit == "year":
                adjusted_date = date_obj + datetime.timedelta(days=365 * amount)
            elif unit == "month":
                # 由于 timedelta 不支持月份，这里采用一个简单的近似
                adjusted_date = date_obj + datetime.timedelta(days=30 * amount)
            elif unit == "day":
                adjusted_date = date_obj + datetime.timedelta(days=amount)
            else:
                raise ValueError("Invalid unit. Expected 'year', 'month', or 'day'.")
        elif operation == "subtract":
            if unit == "year":
                adjusted_date = date_obj - datetime.timedelta(days=365 * amount)
            elif unit == "month":
                # 由于 timedelta 不支持月份，这里采用一个简单的近似
                adjusted_date = date_obj - datetime.timedelta(days=30 * amount)
            elif unit == "day":
                adjusted_date = date_obj - datetime.timedelta(days=amount)
            else:
                raise ValueError("Invalid unit. Expected 'year', 'month', or 'day'.")
        else:
            raise ValueError("Invalid operation. Expected 'add' or 'subtract'.")
        
        # 根据输入日期的类型返回结果
        if format_type == "default":
            return adjusted_date.strftime("%Y%m%d")
        elif format_type == "hyphenated":
            return adjusted_date.strftime("%Y-%m-%d")
        else:
            return adjusted_date


    
    
    