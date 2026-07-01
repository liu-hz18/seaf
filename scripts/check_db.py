import duckdb

# TODO: 解决访问为 None 的问题

DB_PATH = 'quant_stock.duckdb'
code = 'sh.600753'
chunk_start = '2008-06-11'
chunk_end = '2008-07-22'

con = duckdb.connect(DB_PATH, read_only=True, config={'threads': 16})
# con.execute("PRAGMA memory_limit='2GB'")

# existing = con.execute(
#     "SELECT COUNT(*) FROM hot_daily_stock "
#     "WHERE code = ? AND \"date\" >= ? AND \"date\" <= ?",
#     [code, chunk_start, chunk_end],
# ).fetchone()[0]

# print(existing)

rows = con.execute(
    "SELECT code, name, date, close, close_uq, turn, volume, amount, tradestatus, isST FROM hot_daily_stock "
    "WHERE \"date\" >= ? AND \"date\" <= ? AND code = ? ORDER BY code, date",
    [chunk_start, chunk_end, code],
).df()
print(rows)

rows['vwap'] = (rows['amount'] / rows['volume']) * (rows['close'] / rows['close_uq'])
rows['market_cap'] = (rows['volume'] / (rows['turn'] / 100.0)) * rows['close_uq']

print(rows[['code', 'close', 'close_uq', 'vwap', 'market_cap']])

# columns = [desc[0] for desc in rows.description]
# print(columns)


# day_str = '2010-11-16'
# df = con.execute(
#     'SELECT code, name FROM daily_stocks WHERE date = ? ORDER BY code', [day_str]
# ).df()
# print(df)

con.close()

# 删除表的全部行
# con = duckdb.connect(DB_PATH, read_only=False, config={'threads': 16})
# con.execute("TRUNCATE daily_stocks;")
# # 或者 con.execute("DELETE FROM daily_stocks;")
# con.close()
