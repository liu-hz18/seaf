"""
BaoStock 数据模块 — 数据库模式、字段常量、类型映射与股票前缀。

提取自 baostock_data.py，降低耦合度，便于独立测试和维护。
"""

from __future__ import annotations

# ═══════════════════════════════════════════════════════════════════════════
# baostock API 返回字段（后复权 adjustflag='1'）
# ═══════════════════════════════════════════════════════════════════════════

BAOSTOCK_FIELDS = (
    'date,code,open,high,low,close,preclose,volume,amount,'
    'adjustflag,turn,tradestatus,pctChg,isST,'
    'peTTM,pbMRQ,psTTM,pcfNcfTTM'
)

# 不复权 close 专用字段（轻量 API 调用）
CLOSE_UQ_FIELDS = 'date,code,close'

# ═══════════════════════════════════════════════════════════════════════════
# DataFrame 列类型映射（baostock 返回全为字符串，需转换）
# ═══════════════════════════════════════════════════════════════════════════

BAOSTOCK_DTYPES: dict[str, str] = {
    'date': 'str',
    'code': 'str',
    'open': 'float64',
    'high': 'float64',
    'low': 'float64',
    'close': 'float64',
    'preclose': 'float64',
    'volume': 'int64',
    'amount': 'float64',
    'adjustflag': 'int8',
    'turn': 'float64',
    'tradestatus': 'int8',
    'pctChg': 'float64',
    'isST': 'int8',
    'peTTM': 'float64',
    'pbMRQ': 'float64',
    'psTTM': 'float64',
    'pcfNcfTTM': 'float64',
}

# ═══════════════════════════════════════════════════════════════════════════
# 保留的股票前缀（沪深A股）
# ═══════════════════════════════════════════════════════════════════════════

STOCK_PREFIXES: tuple[str, ...] = (
    'sh.600',
    'sh.601',
    'sh.603',
    'sh.605',  # 沪市主板
    'sh.688',  # 科创板
    'sz.000',
    'sz.001',
    'sz.002',
    'sz.003',
    'sz.004',  # 深市主板
    'sz.300',
    'sz.301',
    'sz.302',  # 创业板
)

# ═══════════════════════════════════════════════════════════════════════════
# DuckDB DDL
# ═══════════════════════════════════════════════════════════════════════════

DDL_HOT_TABLE = """
CREATE TABLE IF NOT EXISTS hot_daily_stock (
    "date"        DATE,
    "code"        VARCHAR,
    "name"        VARCHAR,
    "open"        DOUBLE,
    "high"        DOUBLE,
    "low"         DOUBLE,
    "close"       DOUBLE,
    "close_uq"    DOUBLE,
    "preclose"    DOUBLE,
    "volume"      BIGINT,
    "amount"      DOUBLE,
    "adjustflag"  SMALLINT,
    "turn"        DOUBLE,
    "tradestatus" SMALLINT,
    "pctChg"      DOUBLE,
    "peTTM"       DOUBLE,
    "pbMRQ"       DOUBLE,
    "psTTM"       DOUBLE,
    "pcfNcfTTM"   DOUBLE,
    "isST"        SMALLINT,
    PRIMARY KEY ("code", "date")
)
"""

DDL_STOCK_LIST = """
CREATE TABLE IF NOT EXISTS stock_list (
    "code" VARCHAR,
    "name" VARCHAR,
    "first_seen" DATE,
    "last_seen"  DATE,
    PRIMARY KEY ("code")
)
"""

DDL_TRADING_CALENDAR = """
CREATE TABLE IF NOT EXISTS trading_calendar (
    "date"        DATE PRIMARY KEY,
    "is_trading"  SMALLINT
)
"""

DDL_DAILY_STOCKS = """
CREATE TABLE IF NOT EXISTS daily_stocks (
    "date"        DATE,
    "code"        VARCHAR,
    "name"        VARCHAR,
    PRIMARY KEY ("date", "code")
)
"""

# `__all__` 包装当前常量列表便于使用 form-seafquant.baostock_schema import *
__all__ = [
    'BAOSTOCK_DTYPES',
    'BAOSTOCK_FIELDS',
    'CLOSE_UQ_FIELDS',
    'DDL_DAILY_STOCKS',
    'DDL_HOT_TABLE',
    'DDL_STOCK_LIST',
    'DDL_TRADING_CALENDAR',
    'STOCK_PREFIXES',
]
