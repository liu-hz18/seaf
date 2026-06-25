import duckdb

# TODO: 解决访问为 None 的问题

DB_PATH = 'quant_stock.duckdb'
code = 'sh.601003'
chunk_start = '2016-01-01'
chunk_end = '2016-12-31'

con = duckdb.connect(DB_PATH, read_only=True, config={'threads': 16})
# con.execute("PRAGMA memory_limit='2GB'")

# existing = con.execute(
#     "SELECT COUNT(*) FROM hot_daily_stock "
#     "WHERE code = ? AND \"date\" >= ? AND \"date\" <= ?",
#     [code, chunk_start, chunk_end],
# ).fetchone()[0]

# print(existing)

rows = con.execute(
    "SELECT code, date, close FROM hot_daily_stock "
    "WHERE \"date\" >= ? AND \"date\" <= ? ORDER BY code, date",
    [chunk_start, chunk_end],
)
print(rows)

for row in rows.fetchall():
    print(row)

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
