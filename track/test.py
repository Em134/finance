import unittest
import datetime
from Modules import DateHelper

# 假设 DateHelper 中的函数都在同一模块中，直接引入函数进行测试
format_date_for_wind = lambda date: DateHelper.format_date_for_wind(date)
convert_to_timestamp = lambda date_input: DateHelper.convert_to_timestamp(date_input)
set_current_week = lambda target_date: DateHelper.get_target_week(target_date)
get_date_str_days_ago = lambda days_ago=1: DateHelper.get_date_str_days_ago(days_ago)
convert_date_format_1 = lambda date_str: DateHelper.convert_date_format_1(date_str)
convert_date_format_2 = lambda date_str: DateHelper.convert_date_format_2(date_str)
convert_to_datetime = lambda date_str: DateHelper.convert_to_datetime(date_str)
datetime_to_string = lambda dt, format_type='default': DateHelper.datetime_to_string(dt, format_type)
adjust_date = lambda input_date, amount=1, operation="add", unit="day", time_precision="day": DateHelper.adjust_date(input_date, amount, operation, unit, time_precision)
get_time_precision = lambda date_str: DateHelper.get_time_precision(date_str)
get_current_time = lambda time_precision="second": DateHelper.get_current_time(time_precision)

class TestDateHelperFunctions(unittest.TestCase):

    def test_format_date_for_wind(self):
        date_str = "20240830"
        date_obj = datetime.datetime(2024, 8, 30, 12, 34, 56)
        self.assertEqual(format_date_for_wind(date_str), "20240830 00:00:00")
        self.assertEqual(format_date_for_wind(date_obj), "20240830 12:34:56")

    def test_convert_to_timestamp(self):
        date_str = "2024-08-30"
        timestamp = datetime.datetime(2024, 8, 30)
        self.assertEqual(convert_to_timestamp(date_str), timestamp)

    def test_set_current_week(self):
        target_date = "2024-08-30"
        start, end = set_current_week(target_date)
        self.assertTrue(start <= datetime.datetime(2024, 8, 30) <= end)

    def test_get_date_str_days_ago(self):
        days_ago = 2
        date_str = get_date_str_days_ago(days_ago)
        expected_date = datetime.datetime.now() - datetime.timedelta(days=days_ago)
        self.assertEqual(date_str, expected_date.strftime('%Y%m%d'))

    def test_convert_date_format_1(self):
        date_str = "20240830"
        self.assertEqual(convert_date_format_1(date_str), "2024-08-30")
        date_str = "20240830 12:34:56"
        self.assertEqual(convert_date_format_1(date_str), "2024-08-30 12:34:56")


    def test_convert_to_datetime(self):
        date_strs = ["20240830", "2024-08-30", "20240830 12:34:56", "2024-08-30 12:34:56"]
        for date_str in date_strs:
            date_obj = convert_to_datetime(date_str)
            self.assertIsInstance(date_obj, datetime.datetime)

    def test_datetime_to_string(self):
        date_obj = datetime.datetime(2024, 8, 30, 12, 34, 56)
        self.assertEqual(datetime_to_string(date_obj, 'default'), "20240830")
        self.assertEqual(datetime_to_string(date_obj, 'hyphenated'), "2024-08-30")
        self.assertEqual(datetime_to_string(date_obj, 'with_time'), "20240830123456")
        self.assertEqual(datetime_to_string(date_obj, 'with_time_hyphenated'), "2024-08-30 12:34:56")

    def test_adjust_date(self):
        input_date = "2024-08-30"
        adjusted_date = adjust_date(input_date, amount=2, operation="add", unit="day")
        expected_date = datetime.datetime(2024, 9, 1)
        self.assertEqual(convert_to_datetime(adjusted_date), expected_date)

    def test_get_current_time(self):
        precisions = ["year", "month", "day", "hour", "minute", "second"]
        for precision in precisions:
            current_time = get_current_time(precision)
            date_obj = datetime.datetime.now()
            if precision == "year":
                self.assertEqual(current_time, date_obj.strftime("%Y"))
            elif precision == "month":
                self.assertEqual(current_time, date_obj.strftime("%Y-%m"))
            elif precision == "day":
                self.assertEqual(current_time, date_obj.strftime("%Y-%m-%d"))
            elif precision == "hour":
                self.assertEqual(current_time, date_obj.strftime("%Y-%m-%d %H"))
            elif precision == "minute":
                self.assertEqual(current_time, date_obj.strftime("%Y-%m-%d %H:%M"))
            elif precision == "second":
                self.assertEqual(current_time, date_obj.strftime("%Y-%m-%d %H:%M:%S"))

if __name__ == '__main__':
    unittest.main()