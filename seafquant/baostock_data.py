"""
BaoStock 真实数据源 — 编排层 (BaoStockDataCallable)。

══════════════════════════════════════════════════════════════
架构总览
══════════════════════════════════════════════════════════════

  pipeline.py
    └── Flow.add_source('src_data', BaoStockDataCallable(), ...)
          └── SourceNode 子进程调用 __call__()
                ├── [1] _init_db()          → 初始化 DuckDB 四表
                ├── [2] 交易日历管理         → trading_calendar 表 (优先 DB, 缺失 API)
                ├── [3] _download_all()     → 多进程并行下载缺失 chunk
                │     ├── 每 STOCK_LIST_INTERVAL 天: _fetch_stock_list() → API 取股票列表
                │     ├── _year_chunks()     → 按年度边界切分 (trade_day → day_now)
                │     ├── _chunk_complete()  → 检查 chunk 内所有交易日是否已入库
                │     ├── _run_worker() → mp.Process → download_stock_worker()
                │     │     ├── _fetch_main_data()  → 后复权 OHLCV (adjustflag='1')
                │     │     └── _fetch_close_uq()   → 不复权 close (adjustflag='3')
                │     └── 主进程入库 (INSERT OR REPLACE INTO hot_daily_stock)
                ├── [4] 逐日产出             → _read_day() + _frame_to_f3d() → yield Frame3D
                └── [5] 归档                 → _archive_date() 导出 Parquet

══════════════════════════════════════════════════════════════
模块拆分
══════════════════════════════════════════════════════════════

  baostock_schema.py   → DDL / 字段常量 / 类型映射 / 股票前缀
  baostock_api.py      → bao_session() 上下文 / query_with_retry() 重试
  baostock_worker.py   → download_stock_worker() 多进程入口 + 子函数

══════════════════════════════════════════════════════════════
数据库表 (DuckDB + Parquet)
══════════════════════════════════════════════════════════════

  hot_daily_stock     → 全部日K数据 (DuckDB)
  trading_calendar    → 交易日历缓存 (DuckDB)
  daily_stocks        → 每日股票列表缓存 (DuckDB)
  stock_list          → 股票追踪 (code, name, first_seen, last_seen)

══════════════════════════════════════════════════════════════
数据流
══════════════════════════════════════════════════════════════

  baostock API
    ├── adjustflag='1' (后复权) → open/high/low/close/volume/...
    └── adjustflag='3' (不复权) → close_uq
        │
        ▼
  download_stock_worker() ──返回 records──→ 主进程
        │
        ▼
  INSERT OR REPLACE INTO hot_daily_stock  (DuckDB)
        │
        ▼
  _read_day() ──从 DuckDB 读取──→ DataFrame
        │
        ▼
  _frame_to_f3d() ──转换为──→ Frame3D → yield 给下游节点
"""

from __future__ import annotations

import logging
from contextlib import suppress
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from qpipe.frame3d import Frame3D
from seafquant.baostock_api import bao_session, query_with_retry
from seafquant.baostock_schema import (
    DDL_DAILY_STOCKS,
    DDL_HOT_TABLE,
    DDL_STOCK_LIST,
    DDL_TRADING_CALENDAR,
    STOCK_PREFIXES,
)
from seafquant.baostock_worker import download_stock_worker

if TYPE_CHECKING:
    from collections.abc import Iterator


# ── 多进程入口 ────────────────────────────────────────────────
# 必须在模块级别定义，Windows spawn 模式要求 pickle 可达。
# 接收单个 task dict，调用 Worker 函数并将结果放入 mp.Queue。


def _run_worker(task: dict, result_queue) -> None:
    """多进程入口：调用 download_stock_worker 并将结果放入队列。

    task 必须包含键: code, name, start, end, _log_files.
    """
    try:
        data = download_stock_worker(task)
        result_queue.put({'status': 'ok', 'data': data})
    except Exception as e:
        result_queue.put({
            'status': 'error',
            'error': str(e),
            'code': task.get('code', '?'),
        })


# ══════════════════════════════════════════════════════════════
# BaoStockDataCallable
# ══════════════════════════════════════════════════════════════


class BaoStockDataCallable:
    """BaoStock 数据源可调用对象 (pickle 安全)。

    实现迭代器协议，逐日产出 Frame3D，供 qpipe.Flow 的 SourceNode 使用。
    用法: flow.add_source('src_data', BaoStockDataCallable(start_date=...), [...])
    """

    # ── 构造 ────────────────────────────────────────────────

    def __init__(
        self,
        start_date: str = '2010-01-01',
        end_date: str | None = None,
        db_path: str = 'quant_stock.duckdb',
        precision: int = 2,
        max_stocks: int | None = None,
        mlflow_run_id: str = '',
    ) -> None:
        """参数:
        start_date:   下载起始日期 (YYYY-MM-DD).
        end_date:     下载终止日期，默认今天.
        db_path:      DuckDB 数据库文件路径.
        precision:    OHLC 价格精度 (小数位数).
        max_stocks:   限制下载股票数量 (None = 全部).
        mlflow_run_id: MLflow run ID，用于指标记录.
        """
        self.start_date = start_date
        self.end_date = end_date or pd.Timestamp.now().strftime('%Y-%m-%d')
        self.db_path = db_path
        self.mlflow_run_id = mlflow_run_id
        self.precision = precision
        self.max_stocks = max_stocks

    # ── MLflow 辅助 ──────────────────────────────────────────

    def _mlflow_log(self, metrics: dict[str, float], step: int = 0) -> None:
        """安全写入 MLflow 指标。

        mlflow_run_id 为空时静默跳过；写入失败也不抛异常，
        避免因监控问题中断数据下载流程。
        """
        if not self.mlflow_run_id:
            return
        try:
            from qpipe.utils import mlflow_log_metrics

            mlflow_log_metrics(
                self.mlflow_run_id,
                'baostock_data',
                {k: float(v) for k, v in metrics.items()},
                step=step,
            )
        except Exception:
            pass

    # ══════════════════════════════════════════════════════════
    # 数据库初始化
    # ══════════════════════════════════════════════════════════

    def _init_db(self):
        """打开 DuckDB 连接，确保四张表均已创建。

        每次调用返回新连接，调用者负责 close()。
        连接配置: memory_limit=2GB, httpfs 扩展 (用于 Parquet 读取).
        """
        import duckdb

        con = duckdb.connect(self.db_path)
        con.execute("SET memory_limit='2GB'")
        with suppress(Exception):
            con.execute('INSTALL httpfs')
        with suppress(Exception):
            con.execute('LOAD httpfs')
        con.execute(DDL_HOT_TABLE)         # 热数据表
        con.execute(DDL_STOCK_LIST)        # 股票追踪表
        con.execute(DDL_TRADING_CALENDAR)  # 交易日历表
        con.execute(DDL_DAILY_STOCKS)      # 每日股票列表表
        return con

    # ══════════════════════════════════════════════════════════
    # 股票列表管理
    # ══════════════════════════════════════════════════════════

    def _fetch_stock_list(self, bs, day_str: str) -> pd.DataFrame:
        """获取指定交易日全部 A 股列表。

        优先从 daily_stocks 表读取缓存；若缓存缺失或 max_stocks 变大，
        则调 baostock API (bs.query_all_stock) 获取并存入 daily_stocks。
        每次 API 调用耗时约 60 秒。

        返回: DataFrame(columns=['code','name'])，已过滤 non-A-share.
        """
        con = self._init_db()
        cached_count = con.execute(
            'SELECT COUNT(*) FROM daily_stocks WHERE date = ?', [day_str]
        ).fetchone()[0]
        if cached_count > 0:
            if self.max_stocks and cached_count < self.max_stocks:
                # max_stocks 变大 → 缓存不全，重新拉取
                logging.info(
                    f'[baostock] Stock list for {day_str}: cached={cached_count} '
                    f'< max_stocks={self.max_stocks}, re-fetching'
                )
                con.execute('DELETE FROM daily_stocks WHERE date = ?', [day_str])
            else:
                logging.debug(
                    f'[baostock] Stock list for {day_str} already cached ({cached_count} stocks)'
                )
                df = con.execute(
                    'SELECT code, name FROM daily_stocks WHERE date = ?', [day_str]
                ).df()
                con.close()
                return df

        # API 获取 → 过滤沪深 A 股 → 入库
        df = query_with_retry(
            bs,
            lambda: bs.query_all_stock(day=day_str),
            f'query_all_stock({day_str})',
        )
        if df.empty:
            logging.warning(f'[baostock] No stocks returned for {day_str}')
            con.close()
            return df

        mask = df['code'].str.startswith(STOCK_PREFIXES)
        df = df[mask].reset_index(drop=True)
        if self.max_stocks and len(df) > self.max_stocks:
            df = df.head(self.max_stocks)

        con2 = self._init_db()
        for _, row in df.iterrows():
            con2.execute(
                'INSERT OR REPLACE INTO daily_stocks VALUES (?, ?, ?)',
                [day_str, row['code'], row.get('code_name', '')],
            )
        con2.close()

        logging.info(f'[baostock] {day_str}: {len(df)} A-share stocks (API → DB)')
        return df

    # ══════════════════════════════════════════════════════════
    # 年份切分 & 完整性检查
    # ══════════════════════════════════════════════════════════

    @staticmethod
    def _year_chunks(start: str, end: str) -> list[tuple[str, str]]:
        """按年度边界切分 [start, end] → [(cs, ce), ...].

        start=2007-05-01, end=2026-06-17 →
        [('2007-05-01','2007-12-31'), ('2008-01-01','2008-12-31'), ...,
         ('2026-01-01','2026-06-17')]

        切分目的:
        - 首/尾年可能不满全年，中间年份为完整自然年
        - 作为下载任务的最小单元，网断只影响单个 chunk
        """
        s = pd.Timestamp(start)
        e = pd.Timestamp(end)
        if s > e:
            return []
        chunks: list[tuple[str, str]] = []
        year_start = s
        while year_start <= e:
            year_end = min(pd.Timestamp(year=str(year_start.year) + '-12-31'), e)
            chunks.append((
                year_start.strftime('%Y-%m-%d'),
                year_end.strftime('%Y-%m-%d'),
            ))
            year_start = pd.Timestamp(year=str(year_start.year + 1) + '-01-01')
        return chunks

    def _chunk_complete(
        self, code: str, chunk_start: str, chunk_end: str, trading_days: list[str],
    ) -> bool:
        """检查 chunk 内所有交易日是否已完整入库。

        比较 hot_daily_stock 中的实际行数与 chunk 包含的交易日数。
        任意缺失 → 返回 False → chunk 加入下载队列。
        网络失败导致无数据 → 下次运行自动检测到缺失并重试。
        """
        con = self._init_db()
        try:
            existing = con.execute(
                "SELECT COUNT(*) FROM hot_daily_stock "
                "WHERE code = ? AND \"date\" >= ? AND \"date\" <= ?",
                [code, chunk_start, chunk_end],
            ).fetchone()[0]
            td_in_chunk = sum(
                1 for d in trading_days
                if chunk_start <= str(d)[:10] <= chunk_end
            )
            return existing >= td_in_chunk
        finally:
            con.close()

    # ══════════════════════════════════════════════════════════
    # 数据下载编排 (多进程)
    # ══════════════════════════════════════════════════════════

    def _download_all(self, trading_days: list[str]) -> None:
        """主循环: 遍历交易日 → 获取股票列表 → 年边界切分 → 完整性检测 → 多进程下载。

        流程:
          每天 (每隔 STOCK_LIST_INTERVAL 天调 API 取股票列表)
            ├── 对每只股票: _year_chunks(trade_day, day_now)
            ├── 对每个 chunk: _chunk_complete() 检测完整性
            ├── 不完整的 chunk → 提交 mp.Process 下载
            └── Worker 返回数据 → INSERT OR REPLACE 入库

        并发控制:
          - MAX_WORKERS 个进程/批，边跑边收防止 mp.Queue 满死锁
          - 每天 API 调用上限 MAX_DAILY_CALLS (45000)
          - 30 分钟全局超时兜底
          - SIGINT 信号 → _stop_flag → 本轮结束后优雅退出

        容错:
          - Worker 失败 → 错误日志，数据不入库 → 下次运行自动重试
          - 进程卡死 → p.join(timeout=...) + terminate()
        """
        import multiprocessing as _mp
        import queue as _queue_lib
        import signal as _signal
        import threading
        import time as _time

        # ── 常量 ──────────────────────────────────────────────
        MAX_WORKERS = 8           # 最大并行进程数
        MAX_DAILY_CALLS = 45000   # 每日 API 调用上限
        STOCK_LIST_INTERVAL = 20  # 每 N 天调一次股票列表 API

        # ── 状态变量 ──────────────────────────────────────────
        _api_calls = 0
        _total_rows = 0
        _stop_flag = threading.Event()

        # 日志文件路径 → 传给 Worker 子进程
        _log_files: list[str] = []
        for h in logging.getLogger().handlers:
            if isinstance(h, logging.FileHandler):
                _log_files.append(h.baseFilename)

        # Ctrl+C 信号处理
        def _on_interrupt(signum, frame):
            logging.warning('[baostock] SIGINT received, stopping...')
            _stop_flag.set()
        _signal.signal(_signal.SIGINT, _on_interrupt)

        # ── 主循环 ────────────────────────────────────────────
        _last_fetched_df: pd.DataFrame | None = None
        for day_idx, day in enumerate(trading_days):
            if _stop_flag.is_set() or _api_calls >= MAX_DAILY_CALLS:
                logging.error(f'[baostock] Stopped at day {day}.')
                break

            # ① 股票列表 (每隔 STOCK_LIST_INTERVAL 天 API 获取，其余复用缓存)
            if day_idx % STOCK_LIST_INTERVAL == 0 or day_idx == len(trading_days) - 1:
                logging.info(
                    f'[baostock] Stock list day {day_idx}/{len(trading_days)}: {day}'
                )
                with bao_session() as bs:
                    stocks_df = self._fetch_stock_list(bs, day)
                if not stocks_df.empty:
                    _last_fetched_df = stocks_df
            else:
                stocks_df = (
                    _last_fetched_df if _last_fetched_df is not None
                    else pd.DataFrame()
                )
                # 回填 daily_stocks 表，避免跳过的日期缓存不全
                if not stocks_df.empty:
                    con_bf = self._init_db()
                    for _, row in stocks_df.iterrows():
                        con_bf.execute(
                            'INSERT OR REPLACE INTO daily_stocks VALUES (?, ?, ?)',
                            [str(day)[:10], row['code'],
                             row.get('name', row.get('code_name', ''))],
                        )
                    con_bf.close()

            if stocks_df is None or stocks_df.empty:
                continue

            # ② 年边界切分 + 完整性预检 → 生成下载任务
            #    day_now = 最新有数据可用的交易日 (18:00 前用 T-1)
            trade_day_str = str(day)[:10]
            _now = pd.Timestamp.now()
            _day_now = (
                str(trading_days[-1])[:10]
                if _now.hour >= 18
                else str(trading_days[-2])[:10]
            )

            tasks: list[dict] = []
            for _, row in stocks_df.iterrows():
                code = row['code']
                name = row.get('name', row.get('code_name', ''))
                chunks = self._year_chunks(trade_day_str, _day_now)
                for cs, ce in chunks:
                    if not self._chunk_complete(code, cs, ce, trading_days):
                        tasks.append({
                            'code': code, 'name': name,
                            'start': cs, 'end': ce,
                            '_log_files': _log_files,
                        })

            if not tasks:
                continue  # 本日无缺失数据

            logging.info(
                f'[baostock] Day {trade_day_str}: '
                f'{len(tasks)} chunks / {len(stocks_df)} stocks need download'
            )

            # ③ 多进程并行下载
            result_queue: _mp.Queue = _mp.Queue()
            all_procs: list[_mp.Process] = []
            total_tasks = len(tasks)
            completed = 0
            deadline = _time.time() + 1800   # 30 分钟

            for batch_start in range(0, total_tasks, MAX_WORKERS):
                batch = tasks[batch_start:batch_start + MAX_WORKERS]
                procs = []
                for t in batch:
                    p = _mp.Process(target=_run_worker, args=(t, result_queue))
                    p.start()
                    procs.append(p)
                    all_procs.append(p)

                # 边收结果边等本批完成 (防止 Queue 满死锁)
                batch_completed = 0
                while batch_completed < len(batch):
                    try:
                        msg = result_queue.get(timeout=10)
                    except _queue_lib.Empty:
                        if all(not p.is_alive() for p in procs):
                            break
                        if _time.time() > deadline:
                            logging.error('[baostock] Global timeout, aborting')
                            break
                        continue
                    batch_completed += 1
                    completed += 1

                    if msg['status'] != 'ok':
                        logging.warning(
                            f'[baostock] Worker failed: {msg.get("error")}'
                        )
                        continue

                    result = msg['data']
                    _api_calls += result['calls']
                    _total_rows += result['rows']
                    records = result.get('data', [])
                    if not records:
                        continue  # 空结果 → 不插入，下次运行自动重试

                    # INSERT OR REPLACE 入库
                    kdf = pd.DataFrame(records)
                    cols = [
                        'date', 'code', 'name', 'open', 'high', 'low',
                        'close', 'close_uq', 'preclose', 'volume', 'amount',
                        'adjustflag', 'turn', 'tradestatus', 'pctChg',
                        'peTTM', 'pbMRQ', 'psTTM', 'pcfNcfTTM', 'isST',
                    ]
                    kdf = kdf[[c for c in cols if c in kdf.columns]]
                    con_w = self._init_db()
                    con_w.register('_tmp', kdf)
                    con_w.execute(
                        'INSERT OR REPLACE INTO hot_daily_stock '
                        'SELECT * FROM _tmp'
                    )
                    con_w.unregister('_tmp')
                    con_w.execute(
                        'INSERT OR REPLACE INTO stock_list VALUES (?, ?, ?, ?)',
                        [result['code'], result['name'],
                         self.start_date, result['end']],
                    )
                    con_w.close()

                    if completed % 5 == 0 or completed == total_tasks:
                        con_check = self._init_db()
                        db_rows = con_check.execute(
                            'SELECT COUNT(*) FROM hot_daily_stock'
                        ).fetchone()[0]
                        con_check.close()
                        logging.info(
                            f'[baostock] Day {str(day)[:10]}: '
                            f'{completed}/{total_tasks} chunks, '
                            f'API={_api_calls}, rows={_total_rows},'
                            f' hot_rows={db_rows}'
                        )

                # 清理残留进程
                for p in procs:
                    if p.is_alive():
                        p.terminate()
                        p.join(timeout=5)

                if _time.time() > deadline:
                    break

            if _stop_flag.is_set():
                break

            # ④ 定期汇总日志 + MLflow
            if day_idx % 10 == 0:
                con_sum = self._init_db()
                db_rows = con_sum.execute(
                    'SELECT COUNT(*) FROM hot_daily_stock'
                ).fetchone()[0]
                con_sum.close()
                self._mlflow_log({
                    'download_day': float(day_idx),
                    'api_calls': float(_api_calls),
                    'cum_upserted': float(_total_rows),
                    'hot_rows': float(db_rows),
                }, step=day_idx)

        logging.info(
            f'[baostock] Download complete. API={_api_calls}, rows={_total_rows}'
        )

    # ══════════════════════════════════════════════════════════
    # 数据归档 (DuckDB → Parquet)
    # ══════════════════════════════════════════════════════════

    # ══════════════════════════════════════════════════════════
    # 日截面数据读取
    # ══════════════════════════════════════════════════════════

    def _read_day(self, date_str: str) -> pd.DataFrame | None:
        """从 DuckDB 读取指定交易日截面数据。"""
        con = self._init_db()
        try:
            df = con.execute(
                "SELECT * FROM hot_daily_stock WHERE \"date\" = ?", [date_str]
            ).df()
            return None if df.empty else df
        finally:
            con.close()

    # ══════════════════════════════════════════════════════════
    # DataFrame → Frame3D 转换
    # ══════════════════════════════════════════════════════════

    def _frame_to_f3d(self, df: pd.DataFrame) -> Frame3D:
        """将数据库 DataFrame 转换为 Frame3D。

        产出列 (15): stock_name, open, high, low, close, close_uq,
                      turnover, volume, market_cap,
                      peTTM, pbMRQ, psTTM, pcfNcfTTM,
                      tradestatus, isST

        缺失列默认 np.nan (避免虚假 0.0 污染模型).
        Index: MultiIndex (key=date, code=stock_code).
        """
        # 缺失列填充 np.nan
        needed = {
            'open': np.nan, 'high': np.nan, 'low': np.nan, 'close': np.nan,
            'turnover': np.nan, 'volume': np.nan, 'market_cap': np.nan,
            'peTTM': np.nan, 'pbMRQ': np.nan, 'psTTM': np.nan,
            'pcfNcfTTM': np.nan,
        }
        for col, default in needed.items():
            if col not in df.columns:
                df[col] = default

        # 价格精度对齐
        for col in ['open', 'high', 'low', 'close']:
            df[col] = np.round(df[col].astype(float), self.precision)

        # 构建 MultiIndex
        arrays = [df['date'].values, df['code'].values]
        mi = pd.MultiIndex.from_arrays(arrays, names=['key', 'code'])
        stock_names = df.get('name', pd.Series('', index=df.index)).tolist()

        out_df = pd.DataFrame(
            {
                'stock_name': stock_names,
                'open': df['open'].values,
                'high': df['high'].values,
                'low': df['low'].values,
                'close': df['close'].values,
                'close_uq': df['close_uq'].values,
                # baostock turn 是百分比 → 除以 100 转为小数
                'turnover': df.get('turn', pd.Series(np.nan, index=df.index)).values / 100.0,
                'volume': df['volume'].values,
                'market_cap': df['market_cap'].values,
                'peTTM': df['peTTM'].values,
                'pbMRQ': df['pbMRQ'].values,
                'psTTM': df['psTTM'].values,
                'pcfNcfTTM': df['pcfNcfTTM'].values,
                'tradestatus': df.get('tradestatus', pd.Series(1, index=df.index)).values,
                'isST': df.get('isST', pd.Series(0, index=df.index)).values,
            },
            index=mi,
        )
        return Frame3D(out_df)

    # ══════════════════════════════════════════════════════════
    # 主入口 — 生成器
    # ══════════════════════════════════════════════════════════

    def __call__(self) -> Iterator[Frame3D]:
        """生成器入口: 补全缺失数据 → 按交易日历逐日 yield Frame3D。

        调用链:
          [1] 预检: 统计现有数据量 (hot_rows + parquet_partitions)
          [2] 交易日历: 优先 trading_calendar 表，缺失调 API 补齐
          [3] 下载: _download_all(trading_days) 多进程并行补全
          [4] 后检: 统计下载后数据量
          [5] 迭代: 逐日 _read_day() + 停牌前向填充 + _frame_to_f3d() → yield
          [6] 归档: 每 30 天将热表数据导出 Parquet
        """
        logging.info(
            f'[baostock] Start date={self.start_date}, end_date={self.end_date}, '
            f'db={self.db_path}'
        )

        # ── 预检 ───────────────────────────────────────────────
        con0 = self._init_db()
        pre_rows = con0.execute(
            "SELECT COUNT(*) FROM hot_daily_stock"
        ).fetchone()[0]
        logging.info(f'[baostock] Pre-download: hot_rows={pre_rows}')
        con0.close()

        # ── 交易日历 ───────────────────────────────────────────
        con_td = self._init_db()
        cached_td = con_td.execute(
            "SELECT COUNT(*) FROM trading_calendar"
        ).fetchone()[0]
        con_td.close()

        if cached_td > 0:
            con_td2 = self._init_db()
            db_range = con_td2.execute(
                'SELECT MIN(date), MAX(date) '
                'FROM trading_calendar WHERE is_trading = 1'
            ).fetchone()
            db_start, db_end = db_range[0], db_range[1]
            # 若缓存不完整 → API 补全
            if (db_start is None or str(db_start) > self.start_date
                    or str(db_end) < self.end_date):
                logging.info(
                    f'[baostock] DB cache ({db_start}~{db_end}) incomplete, '
                    f'fetching missing from API...'
                )
                with bao_session() as bs:
                    td_df = query_with_retry(
                        bs,
                        lambda: bs.query_trade_dates(
                            start_date=self.start_date, end_date=self.end_date
                        ),
                        'trade_dates',
                    )
                con_td3 = self._init_db()
                for _, row in td_df.iterrows():
                    con_td3.execute(
                        'INSERT OR REPLACE INTO trading_calendar VALUES (?, ?)',
                        [row['calendar_date'], int(row['is_trading_day'])],
                    )
                con_td3.close()
            td_df = con_td2.execute(
                'SELECT date FROM trading_calendar '
                'WHERE is_trading = 1 AND date >= ? AND date <= ? '
                'ORDER BY date',
                [self.start_date, self.end_date],
            ).df()
            con_td2.close()
            trading_days = td_df['date'].tolist()
            logging.info(f'[baostock] Trading days from DB: {len(trading_days)}')
        else:
            with bao_session() as bs:
                td_df = query_with_retry(
                    bs,
                    lambda: bs.query_trade_dates(
                        start_date=self.start_date, end_date=self.end_date
                    ),
                    'trade_dates',
                )
            con_td3 = self._init_db()
            for _, row in td_df.iterrows():
                con_td3.execute(
                    'INSERT OR REPLACE INTO trading_calendar VALUES (?, ?)',
                    [row['calendar_date'], int(row['is_trading_day'])],
                )
            con_td3.close()
            trading_days = sorted(
                td_df[td_df['is_trading_day'] == '1']['calendar_date'].tolist()
            )
            logging.info(
                f'[baostock] Trading days from API → DB: {len(trading_days)}'
            )

        self._mlflow_log({
            'total_trading_days': float(len(trading_days)),
        }, step=0)

        # ── 下载 ───────────────────────────────────────────────
        self._download_all(trading_days)

        # ── 后检 ───────────────────────────────────────────────
        con1 = self._init_db()
        post_rows = con1.execute(
            "SELECT COUNT(*) FROM hot_daily_stock"
        ).fetchone()[0]
        logging.info(f'[baostock] Post-download hot_rows={post_rows}')
        self._mlflow_log({
            'post_download_hot_rows': float(post_rows),
        }, step=len(trading_days))
        con1.close()

        # ── 逐日迭代 ───────────────────────────────────────────
        day_count = 0
        prev_f3d: Frame3D | None = None
        for day in trading_days:
            day_count += 1
            df = self._read_day(day)
            if df is None or df.empty:
                logging.debug(f'[baostock] No data for {day}')
                continue

            # 停牌股票前向填充: baostock 停牌日价格=0 → 取前日数据
            if prev_f3d is not None and 'tradestatus' in df.columns:
                suspended = df['tradestatus'] == 0
                if suspended.any():
                    prev_df = prev_f3d.df.droplevel('key')
                    price_cols = [
                        'open', 'high', 'low', 'close', 'preclose',
                        'volume', 'amount', 'turn',
                    ]
                    for col in price_cols:
                        if col in df.columns and col in prev_df.columns:
                            prev_map = prev_df[col].to_dict()
                            df.loc[suspended, col] = (
                                df.loc[suspended, 'code']
                                .map(prev_map)
                                .fillna(df.loc[suspended, col])
                            )

            f3d = self._frame_to_f3d(df)

            n_stocks = f3d.df.index.get_level_values('code').nunique()
            n_cols = len(f3d.df.columns)
            if day_count % 100 == 0 or day_count == 1:
                logging.info(
                    f'[baostock] Yielding day={day} '
                    f'({day_count}/{len(trading_days)}): '
                    f'stocks={n_stocks}, cols={n_cols}'
                )
                self._mlflow_log({
                    'daily_n_stocks': float(n_stocks),
                    'daily_n_cols': float(n_cols),
                }, step=day_count)

            yield (day_count, f3d)
            prev_f3d = f3d

        logging.info(
            f'[baostock] Generator exhausted. Total days yielded: {day_count}'
        )
