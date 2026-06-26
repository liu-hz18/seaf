import time

import baostock as bs
import pandas as pd

# 1. 生成大约50个股票代码 (上证 600000 - 600049)
codes = [f'sh.60{i:04d}' for i in range(500)]

# 日期建议使用过去的交易日，原代码的2026年没有数据
start_date = '2023-06-20'
end_date = '2023-06-26'

# 2. 整体计时开始
total_start_time = time.time()

# 登录计时
login_start = time.time()
lg = bs.login()
login_time = time.time() - login_start
print(f'登录耗时: {login_time:.2f} 秒')
print(f'登录状态: {lg.error_code} {lg.error_msg}\n')

all_data_list = []

if lg.error_code == '0':
    # 3. 查询计时开始
    query_start_time = time.time()

    # 不重复登录，耗时 75s / 500 stocks
    for code in codes:
        print(f"正在查询: {code}") # 取消注释可看详细进度
        lg = bs.login()
        rs = bs.query_history_k_data_plus(
            code=code,
            fields='date,code,open,high,low,close,preclose,volume,amount,adjustflag,turn,tradestatus,pctChg,isST,peTTM,pbMRQ,psTTM,pcfNcfTTM',
            start_date=start_date,
            end_date=end_date,
            frequency='d',
            adjustflag='3',
        )

        data_list = []
        while (rs.error_code == '0') and rs.next():
            data = rs.get_row_data()
            data_list.append(data)

        if rs.error_code != '0':
            print(f'{code} 查询出错: {rs.error_code} {rs.error_msg}')
        lg = bs.logout()
        if data_list:
            df = pd.DataFrame(data_list, columns=rs.fields)
            print(f"{code}: {df}")
            all_data_list.append(df)

    query_total_time = time.time() - query_start_time
    print('=' * 40)
    print(f'50个股票查询总耗时: {query_total_time:.2f} 秒')
    print(f'平均每个股票耗时: {query_total_time / len(codes):.4f} 秒')
    print('=' * 40)

    # 4. 合并数据并展示
    if all_data_list:
        result = pd.concat(all_data_list, ignore_index=True)
        print(f'\n合并后的数据总行数: {len(result)}')
        print(result)
    else:
        print('未获取到任何数据')

    # bs.logout() # 测试时可注释掉，避免反复登录开销
else:
    print(f'登录失败: {lg.error_code} {lg.error_msg}')

total_time = time.time() - total_start_time
print(f'\n脚本总运行时间: {total_time:.2f} 秒')
