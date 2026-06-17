"""
BaoStock 多进程 Worker — 下载单只股票单个 chunk 的数据并返回。

不写数据库，由主进程集中入库。
"""

from __future__ import annotations

import logging
import os
import sys
import time
from contextlib import suppress
from typing import Any

import numpy as np
import pandas as pd

from seafquant.baostock_schema import BAOSTOCK_DTYPES, BAOSTOCK_FIELDS, CLOSE_UQ_FIELDS

_MAX_RETRIES: int = 3
_RETRY_BASE_DELAY: float = 2.0


def _configure_worker_logging(log_files: list[str]) -> None:
    """配置 Worker 日志：继承主进程 FileHandler + stderr。"""
    handlers: list[Any] = [logging.StreamHandler(sys.stderr)]
    for lf in log_files:
        with suppress(Exception):
            os.makedirs(os.path.dirname(lf), exist_ok=True)
            handlers.append(logging.FileHandler(lf, encoding='utf-8'))
    logging.basicConfig(
        level=logging.DEBUG,
        format='[%(levelname)s][%(asctime)s][%(filename)s:%(lineno)d] %(message)s',
        handlers=handlers,
    )


def _login_with_retry(bs, code: str) -> bool:
    """登录 baostock，最多重试 3 次。"""
    for attempt in range(3):
        lg = bs.login()
        if lg.error_code == '0':
            return True
        if attempt < 2:
            time.sleep(1.0 * (attempt + 1))
    logging.warning(f'[baostock-worker] {code}: login failed after 3 retries')
    return False


def _should_relogin(err_msg: str) -> bool:
    """判断是否应重新登录（session 过期 / Socket 故障）。"""
    return '未登录' in err_msg or '10001001' in err_msg or '10002007' in err_msg


def _convert_kdf_types(kdf: pd.DataFrame) -> pd.DataFrame:
    """将 baostock 字符串 DataFrame 转换为正确的 Python 类型。"""
    for col, dtype in BAOSTOCK_DTYPES.items():
        if col in kdf.columns:
            with suppress(Exception):
                if dtype in {'int64', 'int8'}:
                    kdf[col] = pd.to_numeric(kdf[col], errors='coerce').fillna(0).astype(dtype)
                elif 'float' in dtype:
                    kdf[col] = pd.to_numeric(kdf[col], errors='coerce')
    return kdf


def _fetch_main_data(
    bs, code: str, s: str, e: str, relogin_fn=None,
) -> pd.DataFrame:
    """API：后复权 OHLCV（adjustflag='1'）。"""
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            rs = bs.query_history_k_data_plus(
                code=code, fields=BAOSTOCK_FIELDS,
                start_date=s, end_date=e,
                frequency='d', adjustflag='1',
            )
            if rs.error_code != '0':
                raise ConnectionError(f'API error: {rs.error_msg} (code={rs.error_code})')
            data_list = []
            while rs.next():
                data_list.append(rs.get_row_data())
            if data_list:
                return pd.DataFrame(data_list, columns=rs.fields)
            return pd.DataFrame()
        except Exception as exc:
            _err_msg = str(exc)
            if _should_relogin(_err_msg) and relogin_fn and attempt == 1:
                logging.warning(f'[baostock-worker] {code}: socket error, re-login')
                relogin_fn()
            if attempt == _MAX_RETRIES:
                raise
            delay = _RETRY_BASE_DELAY * (2 ** (attempt - 1))
            logging.warning(
                f'[baostock-worker] {code}: attempt {attempt}/{_MAX_RETRIES} '
                f'failed for {s}~{e}: {exc}. Retrying in {delay:.1f}s...'
            )
            time.sleep(delay)
    return pd.DataFrame()


def _fetch_close_uq(
    bs, code: str, s: str, e: str, relogin_fn=None,
) -> pd.DataFrame:
    """API：不复权收盘价（adjustflag='3'），仅 close 字段。"""
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            rs = bs.query_history_k_data_plus(
                code=code, fields=CLOSE_UQ_FIELDS,
                start_date=s, end_date=e,
                frequency='d', adjustflag='3',
            )
            if rs.error_code != '0':
                raise ConnectionError(f'close_uq API error: {rs.error_msg} (code={rs.error_code})')
            uq_list = []
            while rs.next():
                uq_list.append(rs.get_row_data())
            if uq_list:
                uq_df = pd.DataFrame(uq_list, columns=['date', 'code', 'close_uq'])
                uq_df['close_uq'] = pd.to_numeric(uq_df['close_uq'], errors='coerce')
                return uq_df
            return pd.DataFrame()
        except Exception as exc:
            _err_msg = str(exc)
            if _should_relogin(_err_msg) and relogin_fn and attempt == 1:
                logging.warning(f'[baostock-worker] {code}: socket error, re-login')
                relogin_fn()
            if attempt == _MAX_RETRIES:
                return pd.DataFrame()
            time.sleep(1.0 * attempt)
    return pd.DataFrame()


def download_stock_worker(args: dict) -> dict:
    """多进程 Worker：下载单只股票单个 chunk 的数据并返回。

    args: code, name, start, end, _log_files
    返回: {'code': str, 'name': str, 'start': str, 'end': str,
            'calls': int, 'rows': int, 'data': list[dict]}
    失败时 rows=0, data=[] —— 主进程不插入任何行，下次运行自动重试。
    """
    code: str = args['code']
    name: str = args['name']
    s: str = args['start']
    e: str = args['end']
    log_files: list[str] = args.get('_log_files', [])

    _configure_worker_logging(log_files)

    import baostock as bs

    if not _login_with_retry(bs, code):
        return {
            'code': code, 'name': name, 'start': s, 'end': e,
            'calls': 0, 'rows': 0, 'data': [],
        }

    def _relogin():
        bs.logout()
        bs.login()

    try:
        logging.debug(f'[baostock-worker] {code}: downloading {s}~{e}')
        t0 = time.time()

        # 主数据（后复权）
        try:
            kdf = _fetch_main_data(bs, code, s, e, _relogin)
        except Exception as exc:
            logging.error(
                f'[baostock-worker] {code}: ALL RETRIES EXHAUSTED for {s}~{e}: {exc}'
            )
            return {
                'code': code, 'name': name, 'start': s, 'end': e,
                'calls': _MAX_RETRIES, 'rows': 0, 'data': [],
            }

        # 不复权 close
        if not kdf.empty:
            uq_df = _fetch_close_uq(bs, code, s, e, _relogin)
            if not uq_df.empty:
                kdf = kdf.merge(uq_df, on=['date', 'code'], how='left')
            else:
                kdf['close_uq'] = np.nan

        elapsed = time.time() - t0
        if not kdf.empty:
            kdf = _convert_kdf_types(kdf)
            if 'tradestatus' in kdf.columns:
                kdf = kdf[kdf['tradestatus'] == 1]
            if not kdf.empty:
                kdf['name'] = name
                records = kdf.to_dict('records')
                logging.debug(
                    f'[baostock-worker] {code}: {s}~{e} → {len(records)} rows in {elapsed:.1f}s'
                )
                return {
                    'code': code, 'name': name, 'start': s, 'end': e,
                    'calls': 2, 'rows': len(records), 'data': records,
                }

        logging.debug(
            f'[baostock-worker] {code}: {s}~{e} → empty ({elapsed:.1f}s)'
        )
        return {
            'code': code, 'name': name, 'start': s, 'end': e,
            'calls': 2, 'rows': 0, 'data': [],
        }
    finally:
        bs.logout()
