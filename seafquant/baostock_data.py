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
from tqdm import tqdm

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


TIMEING_INTEVAL = 20


class BaoStockDataCallable:
    """BaoStock 数据源可调用对象 (pickle 安全)。

    实现迭代器协议，逐日产出 Frame3D，供 qpipe.Flow 的 SourceNode 使用。
    用法: flow.add_source('src_data', BaoStockDataCallable(start_date=...), [...])
    """

    # ── 构造 ────────────────────────────────────────────────

    def __init__(
        self,
        start_date: str = '2007-01-01',
        end_date: str | None = None,
        update_start_date: str = '2007-01-01',
        db_path: str = 'quant_stock.duckdb',
        precision: int = 2,
        max_stocks: int | None = None,
        update_db: bool = False,
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
        self.update_start_date = update_start_date
        self.db_path = db_path
        self.mlflow_run_id = mlflow_run_id
        self.precision = precision
        self.max_stocks = max_stocks
        self.update_db = update_db

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

    def _init_db(self, read_only: bool = False):
        """打开 DuckDB 连接，确保四张表均已创建。

        每次调用返回新连接，调用者负责 close()。
        连接配置: memory_limit=2GB, httpfs 扩展 (用于 Parquet 读取).
        """
        import duckdb

        con = duckdb.connect(self.db_path, read_only=read_only, config={'threads': 16})
        con.execute("PRAGMA memory_limit='2GB'")
        if not read_only:
            with suppress(Exception):
                con.execute('INSTALL httpfs')
            with suppress(Exception):
                con.execute('LOAD httpfs')
            con.execute(DDL_HOT_TABLE)  # 热数据表
            con.execute(DDL_STOCK_LIST)  # 股票追踪表
            con.execute(DDL_TRADING_CALENDAR)  # 交易日历表
            con.execute(DDL_DAILY_STOCKS)  # 每日股票列表表
        return con

    # ══════════════════════════════════════════════════════════
    # 股票列表管理
    # ══════════════════════════════════════════════════════════

    def _fetch_stock_list(self, day_str: str) -> pd.DataFrame:
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
        logging.info(f'[{day_str}] db cached stock list num: {cached_count}')
        if cached_count > 0:
            if self.max_stocks and cached_count < self.max_stocks:
                # max_stocks 变大 → 缓存不全，重新拉取
                logging.info(
                    f'Stock list for {day_str}: cached={cached_count} '
                    f'< max_stocks={self.max_stocks}, re-fetching'
                )
                con.execute('DELETE FROM daily_stocks WHERE date = ?', [day_str])
            else:
                logging.debug(f'Stock list for {day_str} already cached ({cached_count} stocks)')
                df = con.execute(
                    'SELECT code, name FROM daily_stocks WHERE date = ? ORDER BY code', [day_str]
                ).df()
                con.close()
                return df

        # API 获取 → 过滤沪深 A 股 → 入库
        with bao_session() as bs:
            df = query_with_retry(
                lambda: bs.query_all_stock(day=day_str),
                f'query_all_stock({day_str})',
            )
        if df.empty:
            logging.warning(f'No stocks returned for {day_str}')
            con.close()
            return df

        mask = df['code'].str.startswith(STOCK_PREFIXES)
        df = df[mask].reset_index(drop=True)
        if self.max_stocks and len(df) > self.max_stocks:
            df = df.head(self.max_stocks)

        con2 = self._init_db()

        # 1. 构造待插入的 DataFrame (注意列名和顺序要对应)
        insert_df = pd.DataFrame(
            {
                'date': day_str,
                'code': df['code'],
                'name': df['code_name'] if 'code_name' in df.columns else '',
            }
        )

        # 2. 将 DataFrame 注册为虚拟表，然后一条 SQL 批量 INSERT OR REPLACE
        con2.register('_tmp_insert_df', insert_df)
        con2.execute('INSERT OR REPLACE INTO daily_stocks SELECT * FROM _tmp_insert_df')
        con2.unregister('_tmp_insert_df')

        con2.close()

        logging.info(f'{day_str}: {len(df)} A-share stocks (API → DB)')
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
            year_end = min(pd.Timestamp(f'{year_start.year}-12-31'), e)
            chunks.append(
                (
                    year_start.strftime('%Y-%m-%d'),
                    year_end.strftime('%Y-%m-%d'),
                )
            )
            year_start = pd.Timestamp(f'{year_start.year + 1}-01-01')
        return chunks

    def _chunk_complete(
        self,
        code: str,
        chunk_start: str,
        chunk_end: str,
        trading_days: list[str],
    ) -> bool:
        """检查 chunk 内所有交易日是否已完整入库。

        比较 hot_daily_stock 中的实际行数与 chunk 包含的交易日数。
        任意缺失 → 返回 False → chunk 加入下载队列。
        网络失败导致无数据 → 下次运行自动检测到缺失并重试。
        """
        task_con = self._init_db(read_only=True)
        td_in_chunk = sum(1 for d in trading_days if chunk_start <= str(d)[:10] <= chunk_end)
        try:
            existing = task_con.execute(
                'SELECT COUNT(*) FROM hot_daily_stock '
                'WHERE code = ? AND "date" >= ? AND "date" <= ?',
                [code, chunk_start, chunk_end],
            ).fetchone()
        except Exception as e:
            logging.error(f'Exception: {e}')
        finally:
            task_con.close()
        if not existing:
            logging.debug(
                f'[{code}][{chunk_start}-{chunk_end}] db_rows=NONE, request_rows={td_in_chunk}'
            )
            return False
        if existing[0] < td_in_chunk:
            logging.debug(
                f'[{code}][{chunk_start}-{chunk_end}] db_rows={existing[0]}, request_rows={td_in_chunk}'
            )
        return existing[0] >= td_in_chunk

    # ══════════════════════════════════════════════════════════
    # 数据下载编排 (多进程)
    # ══════════════════════════════════════════════════════════

    def _download_all(self, trading_days: list[str]) -> None:
        """主循环: 遍历交易日 → 获取股票列表 → 年边界切分 → 完整性检测 → 多进程下载。

        流程:
          每天 (每隔 STOCK_LIST_INTERVAL 天调 API 取股票列表)
            ├── 对每只股票: _year_chunks(trade_day, day_now)
            ├── 对每个 chunk: _chunk_complete() 检测完整性
            ├── 去重 (code, start, end) 防止跨批次重复提交
            ├── 不完整的 chunk → ProcessPoolExecutor 并行下载
            └── Worker 返回数据 → INSERT OR REPLACE 入库

        并发控制:
          - ProcessPoolExecutor(max_workers=MAX_WORKERS) 进程池
          - as_completed 流式收集，先完成先入库
          - 每天 API 调用上限 MAX_DAILY_CALLS (45000)
          - 24 小时全局超时兜底
          - SIGINT 信号 → _stop_flag → 取消剩余 future 后优雅退出

        容错:
          - Worker 失败 → 错误日志，数据不入库 → 下次运行自动重试
          - with 退出时自动 shutdown(wait=True)，无需手动 terminate/join
          - dump_to_db 异常 → 错误日志，不中断其他任务入库
        """
        import signal as _signal
        import threading
        import time as _time

        # ── 常量 ──────────────────────────────────────────────
        MAX_WORKERS = 1  # 最大并行进程数
        MAX_DAILY_CALLS = 10_0000  # 每日 API 调用上限
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
            logging.warning('SIGINT received, stopping...')
            _stop_flag.set()

        _signal.signal(_signal.SIGINT, _on_interrupt)

        # ── 主循环 ────────────────────────────────────────────
        # _last_fetched_df: pd.DataFrame | None = None
        for day_idx, day in enumerate(trading_days):
            if _stop_flag.is_set() or _api_calls >= MAX_DAILY_CALLS:
                logging.error(f'[{day_idx}][{day}] Stopped. {_stop_flag.is_set()=} {_api_calls=}')
                break

            # ① 股票列表 (每隔 STOCK_LIST_INTERVAL 天 API 获取)
            if day_idx % STOCK_LIST_INTERVAL == 0 or day_idx == len(trading_days) - 1:
                logging.info(f'[{day_idx}][{day}] Fetching Stock list...')
                stocks_df = self._fetch_stock_list(day)
                logging.info(f'[{day_idx}][{day}] {stocks_df=}')

            if stocks_df is None or stocks_df.empty:
                continue

            # 每隔 STOCK_LIST_INTERVAL 更新股票日频数据
            if day_idx % STOCK_LIST_INTERVAL != 0 and day_idx != len(trading_days) - 1:
                continue

            # ② 年边界切分 + 完整性预检 → 生成下载任务
            #    day_now = 最新有数据可用的交易日 (18:00 前用 T-1)
            trade_day_str = str(day)[:10]
            _now = pd.Timestamp.now()
            # NOTE: 这个 day_now 设置为当年年底就可以，
            # 因为我们的 stock_df 是按照 STOCK_LIST_INTERVAL 更新的，
            # 只要保证 数据获取长度 > STOCK_LIST_INTERVAL 即可，不然也是浪费 api 调用次数
            # 如果是最后一年，再利用 trading_days 列表进行计算
            _day_now = str(trading_days[-1])[:10] if _now.hour >= 18 else str(trading_days[-2])[:10]
            next_trade_day = trading_days[day_idx+STOCK_LIST_INTERVAL] if day_idx+STOCK_LIST_INTERVAL < len(trading_days) else _day_now
            next_trade_day_str = str(next_trade_day)[:10]
            year_end_str = next_trade_day_str[:4] + '-12-31'
            # 2. 取 year_end_str 和 _day_now 中的较小值，确保不超过 _day_now
            fetch_start_day = trade_day_str
            fetch_end_day = min(year_end_str, _day_now)
            logging.info(
                f'[{day_idx}][{day}] fetch stocks day range: [{fetch_start_day}, {fetch_end_day}]'
            )

            # 多线程并发检查 chunk 完整性（DuckDB 支持并发读）
            from concurrent.futures import ThreadPoolExecutor, as_completed

            def _check_stock(row) -> list[dict]:
                # NOTE: 多线程读取 duckdb, 每个线程应该内部建立db连接，不可公用，否则会出现数据获取不全的问题
                code = row['code']
                name = row.get('name', row.get('code_name', ''))
                chunks = self._year_chunks(fetch_start_day, fetch_end_day)  # noqa: B023
                stock_tasks = []
                for cs, ce in chunks:
                    if not self._chunk_complete(code, cs, ce, trading_days):
                        task = {
                            'code': code,
                            'name': name,
                            'start': cs,
                            'end': ce,
                            '_log_files': _log_files,
                        }
                        stock_tasks.append(task)
                return stock_tasks

            tasks: list[dict] = []
            # NOTE: 这里多线程几乎没有提升
            try:
                with ThreadPoolExecutor(max_workers=4) as check_executor:
                    futures = {
                        check_executor.submit(_check_stock, row): row['code']
                        for _, row in stocks_df.iterrows()
                    }
                    for future in tqdm(
                        as_completed(futures), total=len(stocks_df)
                    ):  # this will waiting for as soon as each task compelete
                        task = future.result()
                        if len(task) > 0:
                            logging.debug(
                                f'[{day_idx}][{day}][{task[0]["code"]}][{task[0]["name"]}] {len(task)} chunks'
                            )
                            tasks.extend(task)
            except Exception as e:
                logging.error(f'[{day_idx}][{day}] Exception: {e}')

            if not tasks:
                continue  # 本日无缺失数据

            # 去重：以 (code, start, end) 为 key。
            # 进程池复用 + DuckDB read_only 快照可能导致
            # _chunk_complete 在跨批次时错误返回 False，
            # 造成同一 chunk 被重复提交。此处兜底去重。
            seen: dict[tuple, bool] = {}
            deduped: list[dict] = []
            for t in tasks:
                key = (t['code'], t['start'], t['end'])
                if key not in seen:
                    seen[key] = True
                    deduped.append(t)
            if len(deduped) < len(tasks):
                logging.warning(
                    f'[{day_idx}][{day}] Dedup: '
                    f'{len(tasks)} → {len(deduped)} tasks '
                    f'({len(tasks) - len(deduped)} duplicates removed)'
                )
            tasks = deduped

            for num, task in enumerate(tasks):
                task['taskid'] = num

            logging.info(
                f'[{day_idx}][{day}] {len(tasks)} chunks / {len(stocks_df)} stocks need download'
            )

            def dump_to_db(data: dict) -> None:
                records = data.get('data', [])
                if not records:
                    return  # 空结果 → 不插入，下次运行自动重试
                kdf = pd.DataFrame(records)
                cols = [
                    'date',
                    'code',
                    'name',
                    'open',
                    'high',
                    'low',
                    'close',
                    'close_uq',
                    'preclose',
                    'volume',
                    'amount',
                    'adjustflag',
                    'turn',
                    'tradestatus',
                    'pctChg',
                    'peTTM',
                    'pbMRQ',
                    'psTTM',
                    'pcfNcfTTM',
                    'isST',
                ]
                kdf = kdf[[c for c in cols if c in kdf.columns]]
                con_w = None
                try:
                    con_w = self._init_db()
                    con_w.register('_tmp', kdf)
                    con_w.execute('INSERT OR REPLACE INTO hot_daily_stock SELECT * FROM _tmp')
                    con_w.unregister('_tmp')
                    con_w.execute(
                        'INSERT OR REPLACE INTO stock_list VALUES (?, ?, ?, ?)',
                        [data['code'], data['name'], self.update_start_date, data['end']],
                    )
                    logging.info(
                        f'[{day_idx}][{day}] dump SUCCESS for '  # noqa: B023
                        f'{data.get("code", "?")}/'
                        f'{data.get("start", "?")}~{data.get("end", "?")}, shape={kdf.shape}'
                    )
                except Exception as exc:
                    logging.error(
                        f'[{day_idx}][{day}] dump FAILED for '  # noqa: B023
                        f'{data.get("code", "?")}/'
                        f'{data.get("start", "?")}~{data.get("end", "?")}, shape={kdf.shape}: {exc}'
                    )
                finally:
                    if con_w is not None:
                        with suppress(Exception):
                            con_w.close()

            # ③ 多进程并行下载 (进程池)。注意：夜间下载接口显著更稳定
            from concurrent.futures import ProcessPoolExecutor, as_completed

            total_tasks = len(tasks)
            completed = 0
            deadline = _time.time() + 3600 * 24  # 24h 全局超时

            if total_tasks > 0:
                with ProcessPoolExecutor(max_workers=min(MAX_WORKERS, total_tasks)) as pool:
                    # 一次提交所有任务；submit 不阻塞
                    future_to_task: dict = {}
                    for t in tasks:
                        future = pool.submit(download_stock_worker, t)
                        future_to_task[future] = t
                        logging.debug(f'[{day_idx}][{day}] submit task: {t}')

                    # as_completed 流式收集 —— 先完成先处理，无需手动管 Queue
                    for future in as_completed(future_to_task):
                        task = future_to_task[future]

                        # SIGINT / 超时 → 取消剩余任务
                        if _stop_flag.is_set():
                            for f in future_to_task:
                                f.cancel()
                            logging.warning(f'[{day_idx}][{day}] Stopped by SIGINT')
                            break
                        if _time.time() > deadline:
                            for f in future_to_task:
                                f.cancel()
                            logging.error(f'[{day_idx}][{day}] Global timeout, aborting')
                            break

                        completed += 1

                        # 获取结果 (download_stock_worker 返回 dict)
                        try:
                            data = future.result(timeout=10)
                        except Exception as exc:
                            logging.warning(
                                f'[{day_idx}][{day}] Worker exception: '
                                f'{task.get("code", "?")} {exc}'
                            )
                            continue

                        # Worker 返回空 → rows=0, data=[] → 不插入，下次自动重试
                        if data.get('rows', 0) == 0 and not data.get('data'):
                            continue

                        _api_calls += data['calls']
                        _total_rows += data['rows']

                        # INSERT OR REPLACE 入库
                        dump_to_db(data)

                        if completed % 5 == 0 or completed == total_tasks:
                            con_check = self._init_db(read_only=True)
                            db_rows = con_check.execute(
                                'SELECT COUNT(*) FROM hot_daily_stock'
                            ).fetchone()[0]
                            con_check.close()
                            logging.info(
                                f'[{day_idx}][{day}] '
                                f'{completed}/{total_tasks} chunks, '
                                f'API={_api_calls}, rows={_total_rows},'
                                f' db_rows={db_rows}'
                            )
                    # with 退出时自动 shutdown(wait=True)，无需手动 terminate/join

                logging.info(f'[{day_idx}][{day}] Multi process fetch done')
                if _stop_flag.is_set():
                    break

            # 这里的记录是跳变的（阶梯式的），因为我们会在年初获取全年的数据，后续年内获取的就少了
            con_sum = self._init_db(read_only=True)
            db_rows = con_sum.execute('SELECT COUNT(*) FROM hot_daily_stock').fetchone()[0]
            con_sum.close()
            self._mlflow_log(
                {
                    'download_day': float(day_idx),
                    'api_calls': float(_api_calls),
                    'cum_upserted': float(_total_rows),
                    'db_rows': float(db_rows),
                },
                step=day_idx,
            )
            logging.info(
                f'[{day_idx}][{day}] api_calls={_api_calls}, total_rows={_total_rows}, db_rows={db_rows}'
            )

        logging.info(f'[{day_idx}][{day}] Download complete. API={_api_calls}, rows={_total_rows}')

    # ══════════════════════════════════════════════════════════
    # 日截面数据读取
    # ══════════════════════════════════════════════════════════
    def _read_day(self, date_str: str) -> pd.DataFrame | None:
        """从 DuckDB 读取指定交易日截面数据。"""
        con = self._init_db()
        try:
            df = con.execute(
                'SELECT * FROM hot_daily_stock WHERE "date" = ? ORDER BY code', [date_str]
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
            'open': np.nan,
            'high': np.nan,
            'low': np.nan,
            'close': np.nan,
            'turnover': np.nan,
            'volume': np.nan,
            'market_cap': np.nan,
            'peTTM': np.nan,
            'pbMRQ': np.nan,
            'psTTM': np.nan,
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
          [1] 预检: 统计现有数据量 (db_rows)
          [2] 交易日历: 优先 trading_calendar 表，缺失调 API 补齐
          [3] 下载: _download_all(trading_days) 多进程并行补全
          [4] 后检: 统计下载后数据量
          [5] 迭代: 逐日 _read_day() + 停牌前向填充 + _frame_to_f3d() → yield
          [6] 归档: 每 30 天将热表数据导出 Parquet
        """
        logging.info(f'backtest dates=[{self.start_date}, {self.end_date}], db={self.db_path}, update-dates=[{self.update_start_date}, {self.end_date}]')

        # ── 预检 ───────────────────────────────────────────────
        con0 = self._init_db(read_only=True)
        pre_rows = con0.execute('SELECT COUNT(*) FROM hot_daily_stock').fetchone()[0]
        logging.info(f'Pre-download: db_rows={pre_rows}')

        # ── 交易日历 ───────────────────────────────────────────
        db_range = con0.execute('SELECT MIN(date), MAX(date) FROM trading_calendar').fetchone()
        db_start, db_end = db_range[0], db_range[1]
        con0.close()

        need_fetch = (
            db_start is None or str(db_start) > min(self.update_start_date, self.start_date) or str(db_end) < self.end_date
        )
        logging.info(
            f'db_range=[{db_start}, {db_end}], request=[{min(self.update_start_date, self.start_date)}, {self.end_date}], '
            f'need_fetch={need_fetch}'
        )

        if need_fetch and self.update_db:
            logging.info('Cache incomplete (or empty), fetching from API...')
            with bao_session() as bs:
                td_df = query_with_retry(
                    lambda: bs.query_trade_dates(
                        start_date=min(self.update_start_date, self.start_date), end_date=self.end_date
                    ),
                    'trade_dates',
                )
            con = self._init_db()
            con.executemany(
                'INSERT OR REPLACE INTO trading_calendar VALUES (?, ?)',
                [(r['calendar_date'], int(r['is_trading_day'])) for _, r in td_df.iterrows()],
            )
            con.commit()
            con.close()

        # ── 下载 ───────────────────────────────────────────────
        if self.update_db:
            con = self._init_db(read_only=True)
            # DB trading days
            td_df = con.execute(
                'SELECT date FROM trading_calendar '
                'WHERE is_trading = 1 AND date >= ? AND date <= ? '
                'ORDER BY date',
                [self.update_start_date, self.end_date],
            ).df()
            con.close()

            db_trading_days = td_df['date'].tolist()
            logging.info(f'Update DB Trading days length: {len(db_trading_days)}')
            if len(db_trading_days) > 0:
                logging.info(f'Update DB Trading days: [{db_trading_days[0]}-{db_trading_days[-1]}]')
            self._mlflow_log(
                {
                    'db_trading_days': float(len(db_trading_days)),
                },
                step=0,
            )
            self._download_all(db_trading_days)

            # ── 后检 ───────────────────────────────────────────────
            con = self._init_db(read_only=True)
            post_rows = con.execute('SELECT COUNT(*) FROM hot_daily_stock').fetchone()[0]
            logging.info(f'Post-download db_rows={post_rows}')
            self._mlflow_log(
                {
                    'post_download_db_rows': float(post_rows),
                },
                step=len(db_trading_days),
            )
            con.close()

        # 获得回测期间交易日历
        con = self._init_db(read_only=True)
        td_df = con.execute(
            'SELECT date FROM trading_calendar '
            'WHERE is_trading = 1 AND date >= ? AND date <= ? '
            'ORDER BY date',
            [self.start_date, self.end_date],
        ).df()
        con.close()

        trading_days = td_df['date'].tolist()
        logging.info(f'Update DB Trading days length: {len(trading_days)}')
        if len(trading_days) > 0:
            logging.info(f'Update DB Trading days: [{trading_days[0]}-{trading_days[-1]}]')
        self._mlflow_log(
            {
                'backtest_trading_days': float(len(trading_days)),
            },
            step=0,
        )

        # ── 逐日迭代 ───────────────────────────────────────────
        con_db = self._init_db(read_only=True)
        _PRICE_COLS = ['open', 'high', 'low', 'close', 'preclose', 'volume', 'amount', 'turn']
        prev_f3d: Frame3D | None = None
        for day_count, day in enumerate(trading_days):
            df = con_db.execute(
                'SELECT * FROM hot_daily_stock WHERE "date" = ? ORDER BY code',
                [day],
            ).df()
            if df.empty:
                logging.warning(f'[{day_count}/{len(trading_days)}][{day}] No data current day.')
                continue

            # ── 停牌前向填充 (向量化) ──
            if prev_f3d is not None and 'tradestatus' in df.columns:
                suspended = df['tradestatus'] == 0
                if suspended.any():
                    prev_df = prev_f3d.df.droplevel('key')
                    cols = [c for c in _PRICE_COLS if c in df.columns and c in prev_df.columns]
                    if cols:
                        # 一次性 reindex 对齐，替代逐列 to_dict + map
                        prev_aligned = prev_df.reindex(df.loc[suspended, 'code'].values)[cols]
                        prev_aligned.index = df.index[suspended]

                        df.loc[suspended, cols] = prev_aligned.where(
                            prev_aligned.notna(),
                            df.loc[suspended, cols],
                        )

            f3d = self._frame_to_f3d(df)

            n_stocks = f3d.df.index.get_level_values('code').nunique()
            n_cols = len(f3d.df.columns)
            self._mlflow_log(
                {'daily_n_stocks': float(n_stocks), 'daily_n_cols': float(n_cols)},
                step=day_count,
            )
            if day_count % TIMEING_INTEVAL == 0:
                logging.info(
                    f'[{day_count}/{len(trading_days)}][{day}] stocks={n_stocks}, cols={n_cols}'
                )

            logging.debug(f'[{day_count}/{len(trading_days)}][{day}] {f3d=}')
            yield (day_count, f3d)
            prev_f3d = f3d

        con_db.close()
        logging.info(f'Generator exhausted. Total days yielded: {len(trading_days)}')
