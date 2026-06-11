# 量化回测本地数据库选型与开发方案 (含基本面与状态字段)
## 1. 需求定义与约束
- **数据规模**：10年+历史，约5000只股票，日频（行情、估值、状态等16+字段），总行数约1250万+行，压缩前约2-3GB。
- **写入模式**：每日增量更新（T+1拉取前日全市场日线数据），需支持偶尔的历史数据修正（Upsert）。
- **读取模式**：按时间顺序流式读取回测，频繁进行时间范围扫描和截面聚合。**关键特征：宽表模式，回测时通常只需读取特定列（如 close, peTTM, tradestatus），需严格过滤停牌和ST股**。
- **核心目标**：最大化回测读取效率（列裁剪、分区裁剪），最小化本地资源占用，低运维成本。
## 2. 核心架构：读写分离的冷热分层
由于 **Parquet 格式不支持行级追加**，而回测极度依赖 Parquet 的列存扫描性能，本方案采用 **“DuckDB热数据表(写入缓冲) + Parquet冷数据文件(只读归档)”** 的双层架构。
```mermaid
flowchart LR
    subgraph 写入路径
        A[每日行情/估值数据源] -->|批量Upsert| B(DuckDB内部表: hot_daily_stock)
    end
    subgraph 归档路径
        B -->|定时任务 T-1数据| C[导出为Parquet文件]
        C --> D[(本地磁盘: data/daily_stock/date=.../)]
        B -->|清理已归档数据| E[(清理热表)]
    end
    subgraph 读取路径
        F[回测引擎] -->|流式fetchmany| G(DuckDB查询引擎)
        D -->|read_parquet 列裁剪+谓词下推| G
    end
3. 技术选型与依据
4. 数据模型定义
逻辑表结构严格对齐需求字段，针对数据特征优化数据类型以节省存储和提升计算效率：
CREATE TABLE IF NOT EXISTS daily_stock (
    "date"        DATE,        -- 交易所行情日期 (分区键)
    "code"        VARCHAR,     -- 证券代码 (主键之一)
    "name"        VARCHAR,     -- 证券名称
    "open"        DOUBLE,      -- 开盘价(后复权)
    "high"        DOUBLE,      -- 最高价(后复权)
    "low"         DOUBLE,      -- 最低价(后复权)
    "close"       DOUBLE,      -- 收盘价(后复权)
    "close_uq"    DOUBLE,      -- 收盘价(不复权)
    "preclose"    DOUBLE,      -- 前收盘价(后复权)
    "volume"      BIGINT,      -- 成交量（股）
    "amount"      DOUBLE,      -- 成交额（人民币元）
    "turn"        DOUBLE,      -- 换手率(%)
    "tradestatus" SMALLINT,    -- 交易状态(1：正常交易 0：停牌)
    "pctChg"      DOUBLE,      -- 涨跌幅(%)
    "peTTM"       DOUBLE,      -- 滚动市盈率
    "pbMRQ"       DOUBLE,      -- 市净率
    "psTTM"       DOUBLE,      -- 滚动市销率
    "pcfNcfTTM"   DOUBLE,      -- 滚动市现率
    "isST"        SMALLINT,    -- 是否ST股(1是，0否)
    PRIMARY KEY ("code", "date")
);
Parquet物理分区设计：按 "date" 进行Hive风格分区，每交易日一个目录。
data/daily_stock/date=2020-01-02/part-0.parquet
data/daily_stock/date=2020-01-03/part-0.parquet
5. 标准操作流程 (SOP)
5.1 初始化环境
import duckdb
import os
con = duckdb.connect('quant_stock.duckdb') # 持久化DuckDB文件
con.execute("SET memory_limit='2GB'")       # 限制内存占用
# 创建热数据写入缓冲表
con.execute("""
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
""")
os.makedirs('data/daily_stock', exist_ok=True)
5.2 每日增量写入
将每日拉取的新数据批量 Upsert 到 DuckDB 热表。
def upsert_daily_data(new_rows_df):
    # 利用 DuckDB 的 DataFrame 直接注册和批量写入
    con.register('new_data_temp', new_rows_df)
    con.execute("""
        INSERT INTO hot_daily_stock
        SELECT * FROM new_data_temp
        ON CONFLICT ("code", "date") DO UPDATE
        SET 
            "name"=excluded."name", "open"=excluded."open", "high"=excluded."high", "low"=excluded."low", 
            "close"=excluded."close", "close_uq"=excluded."close_uq","preclose"=excluded."preclose",
            "volume"=excluded."volume", "amount"=excluded."amount", 
            "adjustflag"=excluded."adjustflag", "turn"=excluded."turn",
            "tradestatus"=excluded."tradestatus", "pctChg"=excluded."pctChg",
            "peTTM"=excluded."peTTM", "pbMRQ"=excluded."pbMRQ", 
            "psTTM"=excluded."psTTM", "pcfNcfTTM"=excluded."pcfNcfTTM",
            "isST"=excluded."isST"
    """)
    con.unregister('new_data_temp')
5.3 定期数据归档
每日收盘后，将 T-1 日数据导出至 Parquet，并清理热表。
def archive_data(archive_date_str):
    partition_dir = f"data/daily_stock/date={archive_date_str}"
    os.makedirs(partition_dir, exist_ok=True)
    # 导出至 Parquet，使用 ZSTD 压缩以获得最佳体积/性能平衡
    con.execute(f"""
        COPY (
            SELECT * EXCLUDE ("date") -- 分区目录已包含日期，文件内无需冗余存储
            FROM hot_daily_stock
            WHERE "date" = '{archive_date_str}'
        ) TO '{partition_dir}/part-0.parquet' (FORMAT PARQUET, COMPRESSION 'zstd')
    """)
    # 清理热表缓冲
    con.execute(f"DELETE FROM hot_daily_stock WHERE \"date\" = '{archive_date_str}'")
5.4 回测流式读取
利用 DuckDB 直接映射分区 Parquet 目录，结合列裁剪和截面过滤（剔除停牌和ST），流式吐出数据。
def stream_backtest_data(start_date, end_date, batch_size=5000):
    # 典型的量化回测场景：只要正常交易且非ST的股票的特定列
    sql = f"""
        SELECT "code", "date", "close", "preclose", "volume", "peTTM", "pctChg"
        FROM read_parquet('data/daily_stock/*/*/part-0.parquet')
        WHERE "date" BETWEEN '{start_date}' AND '{end_date}'
          AND "tradestatus" = 1   -- 过滤停牌
          AND "isST" = 0          -- 过滤ST股
        ORDER BY "date", "code"
    """
    result = con.execute(sql)
    while True:
        rows = result.fetchmany(batch_size)
        if not rows:
            break
        yield rows # 流式传递给回测引擎
6. 架构边界与扩容指南
当前架构针对 单机、百GB级以下、宽表列存、写入频次低（日频）、读取计算重 的场景是最优解。
何时需要重构架构？
数据频率提升：若需处理分钟级/Tick级数据（数据量达TB级），单机Parquet无法支撑。
高并发写入：若多进程/实时秒级写入，DuckDB的嵌入式锁会成为瓶颈。
扩容方向：写入侧迁移至 TimescaleDB；计算侧迁移至分布式 OLAP (如 ClickHouse 集群)。
