import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.dates import DateFormatter, MonthLocator

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

DB_PATH = 'quant_stock.duckdb'

con = duckdb.connect(DB_PATH, read_only=True, config={'threads': 16})
# 1. 读取全部股票（不按 code 过滤），只保留正常交易日
df = con.execute(
    "SELECT code, date, close, tradestatus "
    "FROM hot_daily_stock "
    # "WHERE tradestatus = 1"
    "ORDER BY code, date"
).df()

# 2. 计算每只股票的简单收益率（用后复权收盘价 close），简单收益率的平均值就是当日的等权收益率
df = df.sort_values(["code", "date"]).reset_index(drop=True)
df["ret"] = df.groupby("code", group_keys=False)["close"].apply(
    lambda s: s / s.shift(1)
)
df = df.dropna(subset=["ret"])          # 丢弃每只股票首日（NaN）

# 3. 截面等权均值（按日期 groupby 取均值）
cs_mean = df.groupby("date")["ret"].mean().sort_index().apply(np.log)  # 取 log 来减少累乘的舍入误差

# 4. 累加得到累计对数收益，再 exp 还原成净值曲线（起点=1）
nav = pd.DataFrame({
    "date":    cs_mean.index,
    "log_nav": cs_mean.cumsum().values,
    "nav":     np.exp(cs_mean.cumsum()).values,
    "daily_log_ret": cs_mean,
})

fig, ax = plt.subplots(figsize=(12, 5))

# 画图时给 label，legend 才有内容
ax.plot(nav["date"], nav["nav"], label="等权净值")
ax.plot(nav["date"], nav["log_nav"], label="等权净值(对数)")

ax.legend()
ax.grid(True, linestyle='--', alpha=0.5)

# 1. 主刻度：每 2 个月一个
ax.xaxis.set_major_locator(MonthLocator(interval=2))
# 2. 格式化：年-月
ax.xaxis.set_major_formatter(DateFormatter('%Y-%m'))
# 3. 旋转标签
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

plt.tight_layout()
plt.show()
