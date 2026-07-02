import sys

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.dates import DateFormatter, MonthLocator


def calculate_yang_zhang(df: pd.DataFrame, window: int):
    """
    计算 Yang-Zhang 波动率估计量
    结合了隔夜跳空风险 和日内波动风险
    """
    # 1. 计算对数收益率
    # 隔夜收益率：今开 / 昨收
    log_oc = np.log(df['open'] / df['close'].shift(1))
    # 日内收益率关系
    log_ho = np.log(df['high'] / df['open'])
    log_lo = np.log(df['low'] / df['open'])
    log_co = np.log(df['close'] / df['open'])

    # 2. 计算 Rogers-Satchell 估计量 (日内波动部分)
    # RS = ln(H/O)*[ln(H/O)-ln(C/O)] + ln(L/O)*[ln(L/O)-ln(C/O)]
    rs = log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)

    # 3. 计算权重系数 k
    # k = 0.34 / (1.34 + (N+1)/(N-1))
    k = 0.34 / (1.34 + (window + 1) / (window - 1))

    # 4. 组合方差
    # Yang-Zhang 方差 = k * 隔夜方差均值 + 日内方差均值
    var_overnight = (log_oc ** 2).ewm(span=window, min_periods=window).mean()
    var_intraday = rs.ewm(span=window, min_periods=window).mean()

    yz_variance = k * var_overnight + var_intraday

    # 5. 取根号得到波动率
    return np.sqrt(yz_variance)


# 配置显示和编码
sys.stdout.reconfigure(encoding='utf-8')
pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', 50)
pd.set_option('display.max_colwidth', 20)
pd.set_option('display.width', 1024)

# 设置中文字体，以防绘图时中文乱码（根据你的系统环境可能需要调整）
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

DB_PATH = 'quant_stock.duckdb'
code = 'sh.603083'
windows = [60]

con = duckdb.connect(DB_PATH, read_only=True, config={'threads': 16})
df = con.execute(
    "SELECT code, name, date, open, close, high, low, close_uq, tradestatus FROM hot_daily_stock "
    "WHERE code = ? ORDER BY code, date",
    [code],
).df()

# --- plot close and close_uq ---
# 创建图表
fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# 第一个子图：绘制 close 和 close_uq
axes[0].plot(df['date'], df['close'] / df.loc[0, 'close'], label='Close (后复权收盘价)', alpha=0.6)
axes[0].plot(df['date'], df['close_uq'] / df.loc[0, 'close_uq'], label='Close_UQ (原始收盘价)', alpha=0.8)
axes[0].set_title(f"{df['name'].iloc[0]} ({code}) - Price Comparison")
axes[0].set_ylabel("Price")
axes[0].legend()
axes[0].grid(True, linestyle='--', alpha=0.5)

# --- rolling std of close ---
for window in windows:
    df[f'vol_{window}'] = df['close'].apply(np.log).diff(periods=1).ewm(span=window, min_periods=window).std()
    df[f'yzvol_{window}'] = calculate_yang_zhang(df, window=window)

# --- plot vol of stock ---
# 第二个子图：绘制 vol
for window in windows:
    axes[1].plot(df['date'], df[f'yzvol_{window}'], label=f'YZ Volatility (Diff Std, Window={window})', alpha=0.6)
for window in windows:
    axes[1].plot(df['date'], df[f'vol_{window}'], label=f'Volatility (Diff Std, Window={window})', alpha=0.6, linestyle='--')
axes[1].set_title("Stock Volatility (Rolling Std of Price Difference)")
axes[1].set_xlabel("Date")
axes[1].set_ylabel("Volatility")
axes[1].legend()
axes[1].grid(True, linestyle='--', alpha=0.5)

# 获取底部的 x 轴对象
for ax in axes:
    # 1. 设置主刻度定位器：每个月一个刻度
    ax.xaxis.set_major_locator(MonthLocator(interval=2))
    # 2. 设置刻度格式化器：显示为 "年-月" (例如 2023-01)
    ax.xaxis.set_major_formatter(DateFormatter('%Y-%m'))
    # 3. 自动旋转标签，防止日期重叠
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

plt.tight_layout()
plt.show()

# 打印前几行数据查看计算结果
print(df[['date', 'close', f'vol_{windows[0]}']].tail())
