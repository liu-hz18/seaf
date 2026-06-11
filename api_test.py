import baostock as bs
import pandas as pd

# 重要迭代：接下来我们要接入 baostock 的真实历史日k线数据，不再使用 seafquant\data_generator.py 给出的模拟数据
# NOTE: 每日API请求不能超过5万次，超过后进入黑名单控制
# T日调用，只能获得截至T-1日的数据
# 因为调用次数的限制，我们在获取数据时，每次调用需要获取多日的数据（如单只股票一年的数据）
# 约定: 数据的下载日期从 2007-01-01 到今日（今日没有数据，最新到 T-1 日），每个股票需要编排获取的日期。
# 开发时，需要先按小数据区间，如一年，开展测试，包括数据下载、入库、生成器迭代过程、接入 pipeline.py 等
# 调用api时偶尔有卡住或请求失败的现象，需要注意研究并实现防api卡死的解决方案。

# 调用API的代码如下：

# TODO: bs.login 和 bs.logout 可以用 python with 语法来修饰，更加简洁一些
#### 登陆系统 ####
lg = bs.login()
# 显示登陆返回信息
print('login respond error_code:' + lg.error_code)
print('login respond  error_msg:' + lg.error_msg)

code = 'sh.600000'
start_date = '2025-06-00'
end_date = '2025-06-12'

#### 获取交易日信息 ####
rs = bs.query_trade_dates(start_date=start_date, end_date=end_date)
print('query_trade_dates respond error_code:'+rs.error_code)
print('query_trade_dates respond  error_msg:'+rs.error_msg)

#### 打印结果集 ####
data_list = []
while (rs.error_code == '0') & rs.next():
    # 获取一条记录，将记录合并在一起
    data_list.append(rs.get_row_data())
result = pd.DataFrame(data_list, columns=rs.fields)
result.to_csv("D:\\trade_datas.csv", encoding="utf-8", index=False)
print(result)
# 输出 result：
#    calendar_date is_trading_day
# 0     2026-06-01              1
# 1     2026-06-02              1
# 2     2026-06-03              1
# 3     2026-06-04              1
# 4     2026-06-05              1
# 5     2026-06-06              0
# 6     2026-06-07              0
# 7     2026-06-08              1
# 8     2026-06-09              1
# 9     2026-06-10              1
# 10    2026-06-11              1
# 11    2026-06-12              1


#### 获取某日所有证券信息 ####
# 注意：每日都应该获取新的信息，因为有新上市/退市的股票。但是同时有调用次数限制，注意不要把调用次数消耗完
# code prefix 含义如下：
# sh.600/601/603/605 = 主板大盘股
# sh.688 = 科创板
# sz.000/001/002/003/004 开头 = 深市A股主板（含原中小板）
# sz.300/301/302 开头 = 创业板
# 其余的prefix是指数等，我们不需要
rs = bs.query_all_stock(day=start_date)  #当参数“day”为空时，默认取当天日期。闭市后日K线数据更新，该接口才会返回当天数据，否则返回空。
print('query_all_stock respond error_code:'+rs.error_code)
print('query_all_stock respond  error_msg:'+rs.error_msg)

keep_prefixes = (
    # 沪市主板
    'sh.600', 'sh.601', 'sh.603', 'sh.605',
    # 科创板
    'sh.688',
    # 深市主板
    'sz.000', 'sz.001', 'sz.002', 'sz.003', 'sz.004',
    # 创业板
    'sz.300', 'sz.301', 'sz.302',
)

#### 打印结果集 ####
data_list = []
while (rs.error_code == '0') & rs.next():
    # 获取一条记录，将记录合并在一起
    data_list.append(rs.get_row_data())
result = pd.DataFrame(data_list, columns=rs.fields)
mask = result['code'].str.startswith(keep_prefixes)
result = result[mask].reset_index(drop=True)

#### 结果集输出到csv文件 ####
result.to_csv("D:\\all_stock.csv", encoding="utf-8", index=False)
print(result)
# 输出 result:
#            code tradeStatus code_name
# 0     sh.600000           1      浦发银行
# 1     sh.600004           1      白云机场
# 2     sh.600006           1      东风股份
# 3     sh.600007           1      中国国贸
# 4     sh.600008           1      首创环保
# ...         ...         ...       ...
# 5202  sz.301682           1      宏明电子
# 5203  sz.301683           1      慧谷新材
# 5204  sz.301687           1       新广益
# 5205  sz.301696           1      三瑞智能
# 5206  sz.302132           1      中航成飞


#### 获取沪深A股历史日K线数据 ####
# 详细指标参数，参见“历史行情指标参数”章节；“分钟线”参数与“日线”参数不同。“分钟线”不包含指数。
# 日线指标：date,code,open,high,low,close,volume,amount,adjustflag,turn,pctChg,peTTM,pbMRQ,psTTM,pcfNcfTTM
rs = bs.query_history_k_data_plus(
    code=code,
    fields='date,code,open,high,low,close,preclose,volume,amount,adjustflag,turn,tradestatus,pctChg,isST,peTTM,pbMRQ,psTTM,pcfNcfTTM',
    start_date=start_date,
    end_date=end_date,
    frequency='d',
    adjustflag='3',
)
print('query_history_k_data_plus respond error_code:' + rs.error_code)
print('query_history_k_data_plus respond  error_msg:' + rs.error_msg)

#### 打印结果集 ####
data_list = []
while (rs.error_code == '0') & rs.next():
    # 获取一条记录，将记录合并在一起
    data_list.append(rs.get_row_data())
result = pd.DataFrame(data_list, columns=rs.fields)

#### 结果集输出到csv文件 ####
result.to_csv('./history_A_stock_k_data_daily.csv', encoding="utf-8", index=False)
print(result)
# 输出 result：
#          date       code    open    high  ...     peTTM     pbMRQ     psTTM pcfNcfTTM
# 0  2026-06-01  sh.600000  9.3200  9.3500  ...  6.173636  0.411766  1.777685  2.980474
# 1  2026-06-02  sh.600000  9.3000  9.4300  ...  6.167012  0.411324  1.775777  2.977276
# 2  2026-06-03  sh.600000  9.2800  9.3100  ...  6.127267  0.408673  1.764333  2.958089
# 3  2026-06-04  sh.600000  9.2700  9.3000  ...  6.087523  0.406022  1.752889  2.938901
# 4  2026-06-05  sh.600000  9.1800  9.3500  ...  6.186884  0.412649  1.781499  2.986870
# 5  2026-06-08  sh.600000  9.3000  9.4300  ...  6.206756  0.413975  1.787222  2.996464
# 6  2026-06-09  sh.600000  9.3300  9.4400  ...  6.206756  0.413975  1.787222  2.996464
# 7  2026-06-10  sh.600000  9.3700  9.5900  ...  6.352486  0.423695  1.829184  3.066818

# [8 rows x 18 columns]

#### 登出系统 ####
bs.logout()
