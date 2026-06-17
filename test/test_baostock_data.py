"""
BaoStockDataCallable 单元测试 — 覆盖 API 封装、数据库操作、Frame3D 转换。
使用内存 DuckDB + 模拟 baostock API，不依赖真实网络。
"""

from __future__ import annotations

import os
import tempfile
from contextlib import suppress
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from qpipe.frame3d import Frame3D


# ═══════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════


@pytest.fixture
def temp_db():
    """临时 DuckDB 目录，测试后整体删除。"""
    d = tempfile.mkdtemp(prefix='duckdb_test_')
    path = os.path.join(d, 'test.duckdb')
    yield path
    with suppress(Exception):
        import shutil
        shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def temp_parquet_dir():
    """临时 Parquet 归档目录。"""
    d = tempfile.mkdtemp(prefix='baostock_test_')
    yield d
    with suppress(Exception):
        import shutil
        shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def sample_kline_df():
    """构造模拟 baostock 返回的日K线 DataFrame（字符串类型）。"""
    return pd.DataFrame({
        'date': ['2020-01-02', '2020-01-03', '2020-01-06'],
        'code': ['sh.600000', 'sh.600000', 'sh.600000'],
        'open': ['10.0', '10.2', '10.1'],
        'high': ['10.5', '10.4', '10.3'],
        'low': ['9.9', '10.0', '10.0'],
        'close': ['10.3', '10.1', '10.2'],
        'preclose': ['9.8', '10.3', '10.1'],
        'volume': ['1000000', '1200000', '1100000'],
        'amount': ['10300000', '12120000', '11220000'],
        'adjustflag': ['3', '3', '3'],
        'turn': ['0.5', '0.6', '0.55'],
        'tradestatus': ['1', '1', '1'],
        'pctChg': ['5.1', '-1.94', '0.99'],
        'isST': ['0', '0', '0'],
        'peTTM': ['6.5', '6.4', '6.45'],
        'pbMRQ': ['0.85', '0.84', '0.85'],
        'psTTM': ['0.5', '0.49', '0.5'],
        'pcfNcfTTM': ['1.2', '1.19', '1.2'],
    })


@pytest.fixture
def sample_stock_list_df():
    """模拟 baostock query_all_stock 返回。"""
    return pd.DataFrame({
        'code': ['sh.600000', 'sh.600004', 'sh.601000'],
        'tradeStatus': ['1', '1', '1'],
        'code_name': ['浦发银行', '白云机场', '测试股份'],
    })


@pytest.fixture
def sample_trade_dates_df():
    """模拟 baostock query_trade_dates 返回。"""
    return pd.DataFrame({
        'calendar_date': ['2020-01-02', '2020-01-03', '2020-01-06'],
        'is_trading_day': ['1', '1', '1'],
    })


# ═══════════════════════════════════════════════════════════════════════════
# 1. 模块导入 + 基础构造函数
# ═══════════════════════════════════════════════════════════════════════════


class TestBaoStockImport:
    """测试模块可导入，BaoStockDataCallable 可实例化。"""

    def test_import(self):
        from seafquant.baostock_data import BaoStockDataCallable
        assert BaoStockDataCallable is not None

    def test_construct_defaults(self):
        from seafquant.baostock_data import BaoStockDataCallable
        b = BaoStockDataCallable()
        assert b.start_date == '2010-01-01'
        assert b.precision == 2
        assert b.max_stocks is None

    def test_construct_custom(self):
        from seafquant.baostock_data import BaoStockDataCallable
        b = BaoStockDataCallable(
            start_date='2020-01-01',
            end_date='2020-12-31',
            db_path='test.duckdb',
            precision=4,
            max_stocks=100,
        )
        assert b.start_date == '2020-01-01'
        assert b.end_date == '2020-12-31'
        assert b.db_path == 'test.duckdb'
        assert b.precision == 4
        assert b.max_stocks == 100


# ═══════════════════════════════════════════════════════════════════════════
# 2. 数据库初始化
# ═══════════════════════════════════════════════════════════════════════════


class TestDatabaseInit:
    """测试 DuckDB 初始化、表创建、归档。"""

    def test_init_db_creates_tables(self, temp_db):
        from seafquant.baostock_data import BaoStockDataCallable
        b = BaoStockDataCallable(db_path=temp_db)
        con = b._init_db()
        tables = con.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        table_names = {t[0] for t in tables}
        assert 'hot_daily_stock' in table_names
        assert 'stock_list' in table_names
        con.close()

    def test_archive_date(self, temp_db, temp_parquet_dir, sample_kline_df):
        import duckdb
        from seafquant.baostock_data import BaoStockDataCallable

        # 先写入一条数据到 hot 表
        con = duckdb.connect(temp_db)
        con.execute("SET memory_limit='2GB'")
        from seafquant.baostock_schema import DDL_HOT_TABLE as _DDL_HOT_TABLE
        con.execute(_DDL_HOT_TABLE)
        con.execute("""
            INSERT INTO hot_daily_stock VALUES
            ('2020-01-02', 'sh.600000', '浦发银行', 10.0, 10.5, 9.9, 10.3, 10.3, 9.8,
             1000000, 10300000.0, 3, 0.5, 1, 5.1, 6.5, 0.85, 0.5, 1.2, 0)
        """)
        con.close()

        b = BaoStockDataCallable(db_path=temp_db, parquet_dir=temp_parquet_dir)
        b._archive_date('2020-01-02')

        # 验证 Parquet 文件存在
        parquet_path = os.path.join(temp_parquet_dir, 'date=2020-01-02', 'part-0.parquet')
        assert os.path.exists(parquet_path)

        # 验证 hot 表已清理
        con2 = duckdb.connect(temp_db)
        remaining = con2.execute(
            "SELECT COUNT(*) FROM hot_daily_stock WHERE date = '2020-01-02'"
        ).fetchone()[0]
        assert remaining == 0
        con2.close()


# ═══════════════════════════════════════════════════════════════════════════
# 3. 数据读取（_read_day）
# ═══════════════════════════════════════════════════════════════════════════


class TestReadDay:
    """测试从数据库读取指定交易日数据。"""

    def test_read_day_from_hot_table(self, temp_db):
        import duckdb
        from seafquant.baostock_data import BaoStockDataCallable; from seafquant.baostock_schema import DDL_HOT_TABLE as _DDL_HOT_TABLE

        con = duckdb.connect(temp_db)
        con.execute(_DDL_HOT_TABLE)
        con.execute("""
            INSERT INTO hot_daily_stock VALUES
            ('2020-01-02', 'sh.600000', '浦发银行', 10.0, 10.5, 9.9, 10.3, 10.3, 9.8,
             1000000, 10300000.0, 3, 0.5, 1, 5.1, 6.5, 0.85, 0.5, 1.2, 0),
            ('2020-01-02', 'sh.600004', '白云机场', 7.0, 7.2, 6.9, 7.1, 7.1, 6.95,
             500000, 3550000.0, 3, 0.3, 1, 2.16, 12.0, 1.2, 0.8, 2.0, 0)
        """)
        con.close()

        b = BaoStockDataCallable(db_path=temp_db)
        df = b._read_day('2020-01-02')
        assert df is not None
        assert len(df) == 2
        assert sorted(df['code'].tolist()) == ['sh.600000', 'sh.600004']

    def test_read_day_empty(self, temp_db):
        from seafquant.baostock_data import BaoStockDataCallable

        b = BaoStockDataCallable(db_path=temp_db)
        df = b._read_day('2099-01-01')
        assert df is None

    def test_read_day_keeps_all_stocks(self, temp_db):
        """_read_day 保留 ST 和停牌股票，过滤由下游 strategy 节点负责。"""
        import duckdb
        from seafquant.baostock_data import BaoStockDataCallable; from seafquant.baostock_schema import DDL_HOT_TABLE as _DDL_HOT_TABLE

        con = duckdb.connect(temp_db)
        con.execute(_DDL_HOT_TABLE)
        con.execute("""
            INSERT INTO hot_daily_stock VALUES
            ('2020-01-02', 'sh.600000', '正常', 10.0, 10.5, 9.9, 10.3, 10.3, 9.8,
             1000000, 10300000.0, 3, 0.5, 1, 5.1, 6.5, 0.85, 0.5, 1.2, 0),
            ('2020-01-02', 'sh.600001', 'ST股', 5.0, 5.2, 4.9, 5.1, 5.1, 4.95,
             200000, 1020000.0, 3, 0.1, 1, -2.0, 99.0, 0.5, 0.2, 0.5, 1)
        """)
        con.close()

        b = BaoStockDataCallable(db_path=temp_db)
        df = b._read_day('2020-01-02')
        assert df is not None
        assert len(df) == 2  # ST 和正常股票均保留


# ═══════════════════════════════════════════════════════════════════════════
# 4. Frame3D 转换
# ═══════════════════════════════════════════════════════════════════════════


class TestFrameConversion:
    """测试数据库 DataFrame → Frame3D 转换。"""

    def test_frame_to_f3d_basic(self, temp_db):
        from seafquant.baostock_data import BaoStockDataCallable

        b = BaoStockDataCallable(db_path=temp_db)
        df = pd.DataFrame({
            'date': ['2020-01-02', '2020-01-02'],
            'code': ['sh.600000', 'sh.600004'],
            'name': ['浦发银行', '白云机场'],
            'open': [10.0, 7.0],
            'high': [10.5, 7.2],
            'low': [9.9, 6.9],
            'close': [10.3, 7.1],
            'close_uq': [10.3, 7.1],
            'turn': [0.5, 0.3],
            'volume': [1000000, 500000],
        })
        f3d = b._frame_to_f3d(df)
        assert isinstance(f3d, Frame3D)
        assert f3d.df.index.names == ['key', 'code']
        assert 'turnover' in f3d.df.columns
        assert 'stock_name' in f3d.df.columns
        # turnover 应为百分比/100
        assert abs(f3d.df.loc[('2020-01-02', 'sh.600000'), 'turnover'] - 0.005) < 1e-6

    def test_frame_to_f3d_missing_cols(self, temp_db):
        from seafquant.baostock_data import BaoStockDataCallable

        b = BaoStockDataCallable(db_path=temp_db)
        df = pd.DataFrame({
            'date': ['2020-01-02'],
            'code': ['sh.600000'],
            'close': [10.0],
            'close_uq': [10.0],
        })
        f3d = b._frame_to_f3d(df)
        assert 'open' in f3d.df.columns
        assert 'market_cap' in f3d.df.columns
        # 缺失列应有默认值 (nan)
        assert np.isnan(f3d.df['open'].iloc[0])


# ═══════════════════════════════════════════════════════════════════════════
# 5. 股票前缀过滤
# ═══════════════════════════════════════════════════════════════════════════


class TestStockPrefixFilter:
    """测试沪深 A 股前缀过滤逻辑。"""

    def test_prefix_filter(self):
        from seafquant.baostock_schema import STOCK_PREFIXES as _STOCK_PREFIXES
        codes = pd.Series([
            'sh.600000',   # 沪主板 — 保留
            'sh.688001',   # 科创板 — 保留
            'sz.000001',   # 深主板 — 保留
            'sz.300750',   # 创业板 — 保留
            'sh.000001',   # 上证指数 — 过滤
            'sz.399001',   # 深证成指 — 过滤
            'bj.430047',   # 北交所 — 过滤
        ])
        mask = codes.str.startswith(_STOCK_PREFIXES)
        filtered = codes[mask].tolist()
        assert 'sh.600000' in filtered
        assert 'sh.688001' in filtered
        assert 'sz.000001' in filtered
        assert 'sz.300750' in filtered
        assert 'sh.000001' not in filtered
        assert 'bj.430047' not in filtered


# ═══════════════════════════════════════════════════════════════════════════
# 6. 数据类型转换
# ═══════════════════════════════════════════════════════════════════════════


class TestDtypeConversion:
    """测试 baostock 字符串 → 正确 dtype 转换。"""

    def test_dtype_conversion(self):
        from seafquant.baostock_schema import BAOSTOCK_DTYPES as _BAOSTOCK_DTYPES
        # 确认关键字段存在且映射正确
        assert _BAOSTOCK_DTYPES['open'] == 'float64'
        assert _BAOSTOCK_DTYPES['close'] == 'float64'
        assert _BAOSTOCK_DTYPES['volume'] == 'int64'
        assert _BAOSTOCK_DTYPES['isST'] == 'int8'
        assert _BAOSTOCK_DTYPES['tradestatus'] == 'int8'
        assert _BAOSTOCK_DTYPES['peTTM'] == 'float64'

    def test_stock_prefix_coverage(self):
        from seafquant.baostock_schema import STOCK_PREFIXES as _STOCK_PREFIXES
        # 沪市主板
        assert 'sh.600' in _STOCK_PREFIXES
        assert 'sh.601' in _STOCK_PREFIXES
        # 科创板
        assert 'sh.688' in _STOCK_PREFIXES
        # 创业板
        assert 'sz.300' in _STOCK_PREFIXES


# ═══════════════════════════════════════════════════════════════════════════
# 7. 集成：下载 → 入库 → 读取 → 转换（mock API）
# ═══════════════════════════════════════════════════════════════════════════


class TestIntegrationMocked:
    """端到端集成测试（mock baostock API，真实 DuckDB）。"""

    def test_download_and_read_flow(
        self, temp_db, temp_parquet_dir,
        sample_kline_df, sample_stock_list_df, sample_trade_dates_df,
    ):
        """模拟完整流程：交易日历 → 股票列表 → K线下载 → 入库 → 读取 → 转换。"""
        import duckdb
        from unittest.mock import MagicMock, PropertyMock

        from seafquant.baostock_data import BaoStockDataCallable
        from seafquant.baostock_schema import (
            DDL_HOT_TABLE as _DDL_HOT_TABLE,
            DDL_STOCK_LIST as _DDL_STOCK_LIST,
        )

        # 初始化数据库
        con = duckdb.connect(temp_db)
        con.execute(_DDL_HOT_TABLE)
        con.execute(_DDL_STOCK_LIST)
        con.close()

        b = BaoStockDataCallable(
            start_date='2020-01-02',
            end_date='2020-01-06',
            db_path=temp_db,
            parquet_dir=temp_parquet_dir,
        )

        # 模拟 baostock API
        mock_bs = MagicMock()

        # Mock login
        mock_lg = MagicMock()
        mock_lg.error_code = '0'
        mock_lg.error_msg = 'success'
        mock_bs.login.return_value = mock_lg

        # Mock trade_dates 查询
        td_rs = MagicMock()
        td_rs.error_code = '0'
        td_rs.fields = ['calendar_date', 'is_trading_day']
        td_rs.next.side_effect = [
            True, True, True, False
        ]
        td_rs.get_row_data.side_effect = [
            ['2020-01-02', '1'],
            ['2020-01-03', '1'],
            ['2020-01-06', '1'],
        ]
        mock_bs.query_trade_dates.return_value = td_rs

        # Mock all_stock 查询
        as_rs = MagicMock()
        as_rs.error_code = '0'
        as_rs.fields = ['code', 'tradeStatus', 'code_name']
        as_rs.next.side_effect = [True, True, False]
        as_rs.get_row_data.side_effect = [
            ['sh.600000', '1', '浦发银行'],
            ['sh.600004', '1', '白云机场'],
        ]
        mock_bs.query_all_stock.return_value = as_rs

        # Mock k_data 查询
        kd_rs = MagicMock()
        kd_rs.error_code = '0'
        kd_rs.fields = [
            'date', 'code', 'open', 'high', 'low', 'close',
            'preclose', 'volume', 'amount', 'adjustflag',
            'turn', 'tradestatus', 'pctChg', 'isST',
            'peTTM', 'pbMRQ', 'psTTM', 'pcfNcfTTM',
        ]
        kd_rs.next.side_effect = [True, True, True, False]
        kd_rs.get_row_data.side_effect = [
            ['2020-01-02', 'sh.600000', '10.0', '10.5', '9.9', '10.3',
             '9.8', '1000000', '10300000', '3', '0.5', '1',
             '5.1', '0', '6.5', '0.85', '0.5', '1.2'],
            ['2020-01-03', 'sh.600000', '10.2', '10.4', '10.0', '10.1',
             '10.3', '1200000', '12120000', '3', '0.6', '1',
             '-1.94', '0', '6.4', '0.84', '0.49', '1.19'],
            ['2020-01-06', 'sh.600000', '10.1', '10.3', '10.0', '10.2',
             '10.1', '1100000', '11220000', '3', '0.55', '1',
             '0.99', '0', '6.45', '0.85', '0.5', '1.2'],
        ]
        mock_bs.query_history_k_data_plus.return_value = kd_rs

        # 直接调用内部方法测试流程
        with patch('seafquant.baostock_data.bao_session',
                   return_value=MagicMock(__enter__=lambda s: mock_bs,
                                          __exit__=lambda *a: None)):
            # 测试股票列表获取
            stocks = b._fetch_stock_list(mock_bs, '2020-01-02')
            assert len(stocks) == 2

            # 手动插入数据后测试读取
            con2 = duckdb.connect(temp_db)
            con2.execute("""
                INSERT INTO hot_daily_stock VALUES
                ('2020-01-02', 'sh.600000', '浦发银行', 10.0, 10.5, 9.9, 10.3, 10.3, 9.8,
                 1000000, 10300000.0, 3, 0.5, 1, 5.1, 6.5, 0.85, 0.5, 1.2, 0)
            """)
            con2.close()

            df = b._read_day('2020-01-02')
            assert df is not None
            assert len(df) == 1

            f3d = b._frame_to_f3d(df)
            assert isinstance(f3d, Frame3D)
            assert 'sh.600000' in str(f3d.df.index.get_level_values('code').tolist())


# ═══════════════════════════════════════════════════════════════════════════
# 8. _query_with_retry — 重试 + 指数退避
# ═══════════════════════════════════════════════════════════════════════════


class TestQueryWithRetry:
    """测试 _query_with_retry 的重试逻辑。"""

    def test_success_first_attempt(self):
        from seafquant.baostock_api import query_with_retry as _query_with_retry
        mock_bs = MagicMock()
        rs = MagicMock()
        rs.error_code = '0'
        rs.fields = ['a', 'b']
        rs.next.side_effect = [True, False]
        rs.get_row_data.return_value = ['1', '2']
        mock_bs.query.return_value = rs
        df = _query_with_retry(mock_bs, lambda: mock_bs.query(), 'test')
        assert len(df) == 1
        assert mock_bs.query.call_count == 1

    def test_retry_on_error_code(self):
        from seafquant.baostock_api import query_with_retry as _query_with_retry
        mock_bs = MagicMock()
        rs_bad = MagicMock(error_code='-1')
        rs_good = MagicMock(error_code='0', fields=['a'])
        rs_good.next.side_effect = [True, False]
        rs_good.get_row_data.return_value = ['ok']
        mock_bs.query.side_effect = [rs_bad, rs_good]
        df = _query_with_retry(mock_bs, lambda: mock_bs.query(), 'test')
        assert len(df) == 1
        assert mock_bs.query.call_count == 2

    def test_retry_on_exception(self):
        from seafquant.baostock_api import query_with_retry as _query_with_retry
        mock_bs = MagicMock()
        rs_good = MagicMock(error_code='0', fields=['a'])
        rs_good.next.side_effect = [True, False]
        rs_good.get_row_data.return_value = ['ok']
        mock_bs.query.side_effect = [ConnectionError('fail'), rs_good]
        df = _query_with_retry(mock_bs, lambda: mock_bs.query(), 'test')
        assert len(df) == 1
        assert mock_bs.query.call_count == 2

    def test_empty_result(self):
        from seafquant.baostock_api import query_with_retry as _query_with_retry
        mock_bs = MagicMock()
        rs = MagicMock(error_code='0', fields=['a'])
        rs.next.return_value = False
        mock_bs.query.return_value = rs
        df = _query_with_retry(mock_bs, lambda: mock_bs.query(), 'test')
        assert df.empty

    def test_all_retries_exhausted(self):
        from seafquant.baostock_api import query_with_retry as _query_with_retry
        mock_bs = MagicMock()
        mock_bs.query.side_effect = ConnectionError('persistent fail')
        with pytest.raises(ConnectionError):
            _query_with_retry(mock_bs, lambda: mock_bs.query(), 'test')


# ═══════════════════════════════════════════════════════════════════════════
# 9. _download_stock_worker — 多进程 worker 函数
# ═══════════════════════════════════════════════════════════════════════════


class TestDownloadStockWorker:
    """测试 _download_stock_worker 模块级函数。"""

    def test_worker_returns_correct_structure(self):
        from seafquant.baostock_worker import download_stock_worker as _download_stock_worker
        args = {
            'code': 'sh.600000', 'name': 'test',
            'start': '2020-01-01', 'end': '2020-01-01',
        }
        with patch.dict('sys.modules', {'baostock': MagicMock()}) as _mocks:
            mock_bs = _mocks['baostock']
            mock_lg = MagicMock(error_code='0')
            mock_bs.login.return_value = mock_lg
            rs = MagicMock(error_code='0', fields=['a'])
            rs.next.return_value = False
            mock_bs.query_history_k_data_plus.return_value = rs
            result = _download_stock_worker(args)
            assert result['code'] == 'sh.600000'
            assert result['name'] == 'test'
            assert 'calls' in result
            assert 'rows' in result
            assert 'data' in result

    def test_worker_login_failure(self):
        from seafquant.baostock_worker import download_stock_worker as _download_stock_worker
        args = {
            'code': 'sh.600000', 'name': 'test',
            'start': '2020-01-01', 'end': '2020-01-01',
        }
        with patch.dict('sys.modules', {'baostock': MagicMock()}) as _mocks:
            mock_bs = _mocks['baostock']
            mock_lg = MagicMock(error_code='-1', error_msg='fail')
            mock_bs.login.return_value = mock_lg
            result = _download_stock_worker(args)
            assert result['calls'] == 0
            assert result['rows'] == 0

    def test_worker_start_gt_end(self):
        from seafquant.baostock_worker import download_stock_worker as _download_stock_worker
        args = {
            'code': 'sh.600000', 'name': 'test',
            'start': '2025-01-01', 'end': '2020-01-01',
        }
        with patch.dict('sys.modules', {'baostock': MagicMock()}) as _mocks:
            mock_bs = _mocks['baostock']
            mock_lg = MagicMock(error_code='0')
            mock_bs.login.return_value = mock_lg
            result = _download_stock_worker(args)
            assert result['rows'] == 0  # 数据空，不插入 DB

    def test_worker_fetches_data(self):
        from seafquant.baostock_worker import download_stock_worker as _download_stock_worker
        args = {
            'code': 'sh.600000', 'name': '浦发',
            'start': '2020-01-01', 'end': '2020-01-02',
        }
        with patch.dict('sys.modules', {'baostock': MagicMock()}) as _mocks:
            mock_bs = _mocks['baostock']
            mock_lg = MagicMock(error_code='0')
            mock_bs.login.return_value = mock_lg
            rs = MagicMock(error_code='0', fields=['date', 'code'])
            rs.next.side_effect = [True, False]
            rs.get_row_data.return_value = ['2020-01-02', 'sh.600000']
            mock_bs.query_history_k_data_plus.return_value = rs
            result = _download_stock_worker(args)
            assert result['calls'] >= 1
            assert result['rows'] == 1
            assert len(result['data']) == 1


# ═══════════════════════════════════════════════════════════════════════════
# 10. _fetch_stock_list — 缓存与 max_stocks 联动
# ═══════════════════════════════════════════════════════════════════════════


class TestFetchStockListCache:
    """测试 _fetch_stock_list 的 DB 缓存与 max_stocks 联动。"""

    def test_cache_hit(self, temp_db):
        import duckdb
        from seafquant.baostock_data import BaoStockDataCallable; from seafquant.baostock_schema import DDL_DAILY_STOCKS as _DDL_DAILY_STOCKS
        con = duckdb.connect(temp_db)
        con.execute(_DDL_DAILY_STOCKS)
        con.execute(
            "INSERT INTO daily_stocks VALUES ('2020-01-02', 'sh.600000', '浦发')"
        )
        con.close()
        b = BaoStockDataCallable(db_path=temp_db, max_stocks=None)
        mock_bs = MagicMock()
        mock_bs.query_all_stock = MagicMock()
        df = b._fetch_stock_list(mock_bs, '2020-01-02')
        assert len(df) == 1
        assert df['code'].iloc[0] == 'sh.600000'
        mock_bs.query_all_stock.assert_not_called()

    def test_cache_rejected_if_max_stocks_increased(self, temp_db):
        import duckdb
        from seafquant.baostock_data import BaoStockDataCallable; from seafquant.baostock_schema import DDL_DAILY_STOCKS as _DDL_DAILY_STOCKS
        con = duckdb.connect(temp_db)
        con.execute(_DDL_DAILY_STOCKS)
        con.execute(
            "INSERT INTO daily_stocks VALUES ('2020-01-02', 'sh.600000', '浦发')"
        )
        con.close()
        b = BaoStockDataCallable(db_path=temp_db, max_stocks=10)
        mock_bs = MagicMock()
        rs = MagicMock(error_code='0', fields=['code', 'tradeStatus', 'code_name'])
        rs.next.side_effect = [True, True, False]
        rs.get_row_data.side_effect = [
            ['sh.600000', '1', '浦发'], ['sh.600004', '1', '白云'],
        ]
        mock_bs.query_all_stock.return_value = rs
        df = b._fetch_stock_list(mock_bs, '2020-01-02')
        assert len(df) == 2
        mock_bs.query_all_stock.assert_called_once()


# ═══════════════════════════════════════════════════════════════════════════
# 11. _frame_to_f3d — 完整列映射 + 精度
# ═══════════════════════════════════════════════════════════════════════════


class TestFrameToF3dComplete:
    """测试 _frame_to_f3d 的完整 baostock 列映射与精度。"""

    def test_all_15_cols_present(self, temp_db):
        from seafquant.baostock_data import BaoStockDataCallable
        b = BaoStockDataCallable(db_path=temp_db, precision=2)
        df = pd.DataFrame({
            'date': ['2020-01-02'], 'code': ['sh.600000'], 'name': ['浦发'],
            'open': [10.0], 'high': [10.5], 'low': [9.9], 'close': [10.3],
            'close_uq': [10.3], 'turn': [0.5], 'volume': [1000000],
            'peTTM': [6.5], 'pbMRQ': [0.85], 'psTTM': [0.5], 'pcfNcfTTM': [1.2],
            'tradestatus': [1], 'isST': [0],
        })
        f3d = b._frame_to_f3d(df)
        expected = [
            'stock_name', 'open', 'high', 'low', 'close', 'close_uq',
            'turnover', 'volume', 'market_cap',
            'peTTM', 'pbMRQ', 'psTTM', 'pcfNcfTTM',
            'tradestatus', 'isST',
        ]
        for col in expected:
            assert col in f3d.df.columns, f'Missing column: {col}'
        assert len(f3d.df.columns) == len(expected)

    def test_precision_rounding(self, temp_db):
        from seafquant.baostock_data import BaoStockDataCallable
        b = BaoStockDataCallable(db_path=temp_db, precision=2)
        df = pd.DataFrame({
            'date': ['2020-01-02'], 'code': ['sh.600000'],
            'open': [10.1234], 'close': [10.3456], 'close_uq': [10.3456],
        })
        f3d = b._frame_to_f3d(df)
        assert f3d.df['open'].iloc[0] == 10.12
        assert f3d.df['close'].iloc[0] == 10.35


# ═══════════════════════════════════════════════════════════════════════════
# 12. _init_db — 四张表全部创建
# ═══════════════════════════════════════════════════════════════════════════


class TestInitDbAllTables:
    """测试 _init_db 创建全部 4 张表。"""

    def test_all_four_tables_created(self, temp_db):
        from seafquant.baostock_data import BaoStockDataCallable
        b = BaoStockDataCallable(db_path=temp_db)
        con = b._init_db()
        tables = con.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
        table_names = [t[0] for t in tables]
        assert 'daily_stocks' in table_names
        assert 'hot_daily_stock' in table_names
        assert 'stock_list' in table_names
        assert 'trading_calendar' in table_names
        con.close()


# ═══════════════════════════════════════════════════════════════════════════
# 13. 参数化测试 — chunk_months / precision
# ═══════════════════════════════════════════════════════════════════════════


class TestParameters:
    """测试 chunk_months / precision 参数传递。"""

    # chunk_months 已移除——改为年边界切分，无需此参数

    @pytest.mark.parametrize('precision', [0, 2, 4])
    def test_precision_values(self, precision):
        from seafquant.baostock_data import BaoStockDataCallable
        b = BaoStockDataCallable(precision=precision)
        assert b.precision == precision