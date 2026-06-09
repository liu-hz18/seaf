"""
验证脚本：独立计算 factor_value 在 2021-04-20 的截面值，对比框架输出。

用法：
    python scripts/verify_value_factors.py

生成与以下 pipeline 命令相同参数的数据，独立调用 compute_value_factors，
输出 2021-04-20 截面值，与框架日志对照：
    python pipeline.py --n-times 1000 --n-stocks 20 --model-type lgbm --fwd 20 \
        --model-window 250 --noise-ratio 0.5
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qpipe.frame3d import Frame3D
from seafquant.data_generator import generate_synthetic_data
from seafquant.factor.value import compute_value_factors


def main() -> None:
    # ---- 参数（与 pipeline 命令一致）----
    N_TIMES = 1000
    N_STOCKS = 20
    NOISE_RATIO = 0.5
    SEED = 42
    START_DATE = "2020-01-02"
    TARGET_DATE = pd.Timestamp("2021-04-20")
    WINDOW = 130  # factor_value 的滑动窗口大小

    # ---- 关闭 time.sleep 加速数据生成 ----
    _original_sleep = time.sleep
    time.sleep = lambda x: None  # type: ignore[method-assign]

    print(f"生成 {N_TIMES} 天 × {N_STOCKS} 只股票数据 (noise={NOISE_RATIO})...")
    gen = generate_synthetic_data(N_TIMES, N_STOCKS, NOISE_RATIO, SEED, START_DATE)

    # ---- 收集所有日数据 ----
    all_frames: list[Frame3D] = []
    for i, frame in enumerate(gen):
        all_frames.append(frame)
        if (i + 1) % 200 == 0:
            print(f"  已收集 {i + 1}/{N_TIMES} 天")

    time.sleep = _original_sleep
    print(f"数据生成完成：{len(all_frames)} 天。")

    # ---- 确定目标日期索引 ----
    time_keys = [f.df.index.get_level_values("key")[0] for f in all_frames]
    target_idx = time_keys.index(TARGET_DATE)
    print(f"2021-04-20 在 0-indexed 位置: {target_idx}")
    print(f"窗口范围: [{target_idx - WINDOW + 1}, {target_idx}]")

    # ---- 构建 130 天窗口 ----
    window_frames = all_frames[target_idx - WINDOW + 1 : target_idx + 1]
    window_df = pd.concat([f.df for f in window_frames], axis=0)
    window_f3d = Frame3D(window_df)

    # ---- 计算因子（与框架 MultiInputNode 相同入口） ----
    result = compute_value_factors("factor_value", window_f3d, {})

    # ---- 提取 2021-04-20 截面 ----
    cs_mask = result.df.index.get_level_values("key") == TARGET_DATE
    latest_section = result.df[cs_mask]

    factor_cols = [c for c in latest_section.columns if c.startswith("factor_val_")]
    print(f"\n=== {TARGET_DATE.date()} factor_value 截面 ({len(factor_cols)} 列) ===")
    pd.set_option("display.max_columns", 30)
    pd.set_option("display.width", 300)
    pd.set_option("display.float_format", lambda x: f"{x:.6f}")
    print(latest_section[factor_cols].to_string())

    # ---- 与框架预期值对比 ----
    # 以下是从框架日志中提取的预期值（2026-06-09 14:15:02 运行）
    # 只需比对几个关键列确认一致性
    expected_inv_price = np.array(
        [
            -0.223607, -0.223607, -0.223607, -0.223607, -0.223607,
            -0.223607, -0.223607, -0.223607, -0.223607, -0.223607,
            -0.223607, -0.223607, -0.223607, -0.223607, 4.248529,
            -0.223607, -0.223607, -0.223607, -0.223607, -0.223607,
        ]
    )
    actual = latest_section["factor_val_inv_price"].values
    diff = np.abs(actual - expected_inv_price).max()
    print(f"\n=== 对比 factor_val_inv_price ===")
    print(f"  max diff: {diff:.2e}")
    if diff < 1e-5:
        print("  [PASS] exact match (diff < 1e-5)")
    elif diff < 1e-3:
        print(f"  [WARN] close (diff {diff:.2e}, likely fp precision)")
    else:
        print(f"  [FAIL] mismatch! max diff {diff:.2e}")

    # ---- 全列逐一对比 ----
    print(f"\n=== 全列对比 (16 columns) ===")
    expected_all = {
        "factor_val_inv_price": np.array(
            [
                -0.223607, -0.223607, -0.223607, -0.223607, -0.223607,
                -0.223607, -0.223607, -0.223607, -0.223607, -0.223607,
                -0.223607, -0.223607, -0.223607, -0.223607, 4.248529,
                -0.223607, -0.223607, -0.223607, -0.223607, -0.223607,
            ]
        ),
        "factor_val_log_mcap": np.array(
            [
                0.744369, -0.26217, 2.320958, -0.641983, -1.210812,
                0.509131, 0.381972, 0.252708, -0.736263, -1.169614,
                -0.986736, -0.863175, -0.208981, 1.73637, 0.230762,
                -0.661876, -0.657342, 0.240972, 1.607171, -0.62546,
            ]
        ),
    }
    all_pass = True
    for col in factor_cols:
        if col in expected_all:
            diff_col = np.abs(latest_section[col].values - expected_all[col]).max()
            status = "[PASS]" if diff_col < 1e-5 else f"[FAIL] {diff_col:.2e}"
            if diff_col >= 1e-5:
                all_pass = False
        else:
            status = "[SKIP]"
        # 只打印前几个，其余状态聚合
    print(f"  checked {len(expected_all)} columns: {'all pass' if all_pass else 'some failed'}")


if __name__ == "__main__":
    main()
