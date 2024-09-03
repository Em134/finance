# finance
## track
- 创建 TradingDay 包裹了datetime.datetime，使每个日期拥有更多方法包括获取当日指定股票收盘价、指定分钟序列等。
- 实现 TradingWeek ，基于 TradingDay 实现每周的交易日、调仓日等功能。
- 实现 Manager 基于宏观择时模型进行调仓。

## industry_rank
- 基于三份研报实现的行业轮动复现。
- 采用包括barra等因子、基于XGB、Logistic等机器学习预测轮动行业。

## fast_factor
- 对RSRS、Barra因子的计算进行numpy卷积加速。相较于基于pandas的运算能提速约100倍。

