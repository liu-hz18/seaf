import pandas as pd
import logging
from typing import Iterator

from .frame3d import Frame3D
from .flow import Flow


##### --- 简单测试用的数据流与节点定义 ---
def gen_src1_frames():
    return gen_source_frames('src1', price_base=10)

def gen_src2_frames():
    return gen_source_frames('src2', price_base=100)

def stat_node_fn(name: str, f3d: Frame3D) -> Frame3D:
    df = f3d.df.copy()
    # print(f"[{name}] {df}")
    # 假设所有输入列都做 mean/std
    for col in df.columns:
        # 这里只是示例，实际只需计算你感兴趣的列
        df[f'{col}_mean'] = df[col].mean()
        df[f'{col}_std'] = df[col].std()
    return Frame3D(df)

def printer_node(name: str, f3d: Frame3D) -> Frame3D:
    print(f"\n[Printer][{name}] Frame3D:\n", f3d.df, "\n")
    return f3d

# NOTE: 只支持一次 put 一个 day 的数据。不要 put 多 day 的数据
def gen_source_frames(name, price_base) -> Iterator[Frame3D]:
    stocks_per_iter = 2
    days = 7
    for i in range(days):
        arrays = [
            pd.to_datetime([f'2025-01-{i+1:02d}' for j in range(stocks_per_iter)]),
            ['A', 'B']
        ]
        mi = pd.MultiIndex.from_arrays(arrays, names=['key', 'name'])
        df = pd.DataFrame({
            f'{name}_price': [price_base + i, price_base + 2 + i],
            f'{name}_vol': [price_base + 100 + i, price_base + 105 + i]
        }, index=mi)
        yield Frame3D(df)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    flow = Flow()

    flow.add_source(
        name='src1',
        gen_func=gen_src1_frames,
        output_to=[
            'sumprod:0',
            'printer:0'
        ]
    )
    flow.add_source(
        name='src2',
        gen_func=gen_src2_frames,
        output_to=[
            'sumprod:1',
            'printer:1'
        ]
    )

    flow.add_node(
        name='sumprod',  # node name
        func=stat_node_fn,
        input_from=[
            'sumprod:0',
            'sumprod:1'
        ],  # queue name
        output_to=['printer'],
        window=3,
        min_periods=2,
        input_columns=['src1_price', 'src2_price'],
        output_columns=['src1_price_mean', 'src2_price_mean'],
    )

    flow.add_node(
        name='printer-src',
        func=printer_node,
        input_from=['printer:0', 'printer:1'],
        output_to=[]
    )

    flow.add_node(
        name='printer-final',
        func=printer_node,
        input_from='printer',
        output_to=[]
    )

    flow.start()
    flow.join()
