import duckdb

con = duckdb.connect('quant_stock.duckdb', read_only=True)

print("=== hot_daily_stock 概览 ===")
r = con.execute("SELECT MIN(date), MAX(date), COUNT(DISTINCT date), COUNT(DISTINCT code) FROM hot_daily_stock").fetchone()
print(f"日期范围: {r[0]} ~ {r[1]}")
print(f"交易日数: {r[2]}")
print(f"股票数量: {r[3]}")

print("\n=== daily_stocks 日期范围 ===")
r = con.execute("SELECT MIN(date), MAX(date) FROM daily_stocks").fetchone()
print(f"{r[0]} ~ {r[1]}")

print("\n=== stock_list 日期范围 ===")
r = con.execute("SELECT MIN(first_seen), MAX(last_seen) FROM stock_list").fetchone()
print(f"{r[0]} ~ {r[1]}")

print("\n=== trading_calendar ===")
r = con.execute("SELECT MIN(date), MAX(date), COUNT(*) FILTER (WHERE is_trading=1) FROM trading_calendar").fetchone()
print(f"日期范围: {r[0]} ~ {r[1]}, 其中交易日: {r[2]} 天")

con.close()
