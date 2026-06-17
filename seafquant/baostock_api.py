"""
BaoStock API 封装 — 登录管理、带重试查询、中断信号检测。

提取自 baostock_data.py，独立于数据库和业务逻辑。
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

# API 重试配置
_MAX_RETRIES: int = 3
_RETRY_BASE_DELAY: float = 2.0  # 秒


@contextmanager
def bao_session():
    """baostock 登录/登出上下文管理器（登录失败自动重试 3 次）。"""
    import baostock as bs

    lg = None
    for attempt in range(3):
        lg = bs.login()
        if lg.error_code == '0':
            break
        if attempt < 2:
            logging.warning(
                f'[baostock] Login attempt {attempt + 1}/3 failed: '
                f'{lg.error_msg}(code={lg.error_code}). Retrying...'
            )
            time.sleep(1.0 * (attempt + 1))
    if lg is None or lg.error_code != '0':
        _msg = f'{lg.error_msg if lg else "N/A"}'
        logging.error(f'[baostock] Login failed after 3 attempts: {_msg}')
        raise ConnectionError(f'BaoStock login failed: {_msg}')
    try:
        yield bs
    finally:
        bs.logout()


def query_with_retry(bs, query_fn, desc: str = '') -> pd.DataFrame:
    """带重试的 API 查询，返回 DataFrame。

    支持外部中断：设置 query_with_retry._stop_event 可提前终止重试。
    """
    import threading as _thr

    import pandas as pd

    last_err: Exception | None = None
    _stop_evt = getattr(query_with_retry, '_stop_event', None)
    if _stop_evt is None:
        _stop_evt = query_with_retry._stop_event = _thr.Event()  # type: ignore[attr-defined]
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            rs = query_fn()
            if rs.error_code != '0':
                raise ConnectionError(f'[{desc}] API error: {rs.error_msg} (code={rs.error_code})')
            data_list = []
            while rs.next():
                data_list.append(rs.get_row_data())
            if not data_list:
                return pd.DataFrame()
            return pd.DataFrame(data_list, columns=rs.fields)
        except Exception as e:
            last_err = e
            if attempt < _MAX_RETRIES:
                if _stop_evt.is_set():
                    raise KeyboardInterrupt('Download stopped by user') from e
                delay = _RETRY_BASE_DELAY * (2 ** (attempt - 1))
                logging.warning(
                    f'[{desc}] Attempt {attempt}/{_MAX_RETRIES} failed: {e}. '
                    f'Retrying in {delay:.1f}s...'
                )
                time.sleep(delay)
    raise last_err  # type: ignore[misc]


__all__ = ['bao_session', 'query_with_retry']
