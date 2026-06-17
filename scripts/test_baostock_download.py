"""
BaoStock 下载模块独立测试脚本 — 单线程，小范围验证。

用法：
  python scripts/test_baostock_download.py
  python scripts/test_baostock_download.py --start 2026-01-01 --end 2026-06-15 --max-stocks 5
"""

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from seafquant.baostock_data import BaoStockDataCallable


def main():
    parser = argparse.ArgumentParser(description='BaoStock 下载模块测试')
    parser.add_argument('--start', default='2026-01-01', help='起始日期')
    parser.add_argument('--end', default=None, help='结束日期（默认今天）')
    parser.add_argument('--max-stocks', type=int, default=10, help='最大股票数')
    parser.add_argument('--db', default='test_download.duckdb', help='数据库路径')
    parser.add_argument('--parquet-dir', default='data/test_download', help='Parquet 目录')
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG,
        format='[%(levelname)s][%(asctime)s][%(filename)s:%(lineno)d] %(message)s',
    )

    logging.info(f'Args: {args}')

    callable_obj = BaoStockDataCallable(
        start_date=args.start,
        end_date=args.end,
        db_path=args.db,
        parquet_dir=args.parquet_dir,
        max_stocks=args.max_stocks,
    )

    logging.info('Starting download + iteration...')
    day_count = 0
    row_count = 0
    for f3d in callable_obj():
        day_count += 1
        n = len(f3d.df)
        row_count += n
        if day_count % 10 == 0 or day_count == 1:
            cols = list(f3d.df.columns)
            logging.info(
                f'Day {day_count}: {n} rows, cols={cols}'
            )

    logging.info(f'Done. Total: {day_count} days, {row_count} rows')


if __name__ == '__main__':
    main()
