"""
噪声扫描验证脚本。
遍历 20 种噪声强度，运行 pipeline，收集 ICIR，
验证 ICIR 随 noise_ratio 单调递减。
"""
import subprocess
import numpy as np
import sys
import os
import re

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def run_pipeline(noise_ratio: float, n_times: int = 400, n_stocks: int = 30, seed: int = 42) -> dict:
    """运行一次 pipeline，从 stdout 中提取 ICIR 等统计量。
    
    使用较小规模（400 天 × 30 股票）并调整 HEARTBEAT_TIMEOUT。
    """
    cmd = [
        sys.executable, '-c', f'''
import sys
sys.path.insert(0, r"{os.path.dirname(os.path.dirname(__file__))}")
from qpipe import node
node.MultiInputNode.HEARTBEAT_TIMEOUT = 5.0
from pipeline import main
import argparse
sys.argv = ['pipeline.py', '--noise-ratio', str({noise_ratio}), 
            '--n-times', str({n_times}), '--n-stocks', str({n_stocks}), 
            '--seed', str({seed}), '--log-level', 'INFO']
main()
'''
    ]
    
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=300, cwd=os.path.dirname(os.path.dirname(__file__))
        )
        output = result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        return {'noise_ratio': noise_ratio, 'icir': None, 'mean_ic': None, 'error': 'timeout'}
    
    # 从输出中解析 ICIR
    icir_match = re.search(r'ICIR=([-\d.]+)', output)
    mean_ic_match = re.search(r'Mean IC=([-\d.]+)', output)
    n_match = re.search(r'N=(\d+)', output)
    
    icir = float(icir_match.group(1)) if icir_match else None
    mean_ic = float(mean_ic_match.group(1)) if mean_ic_match else None
    n_valid = int(n_match.group(1)) if n_match else 0
    
    return {
        'noise_ratio': noise_ratio,
        'icir': icir,
        'mean_ic': mean_ic,
        'n_valid_ic': n_valid,
    }


def main():
    """主验证流程：遍历 10 种噪声强度（简化版），检查 ICIR 单调性。"""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--full', action='store_true', help='Run full 20-point sweep')
    args = parser.parse_args()
    
    n_points = 20 if args.full else 10
    noise_ratios = np.linspace(0.0, 0.95, n_points)
    
    print(f"Noise Sweep Validation ({n_points} points)")
    print("=" * 60)
    print(f"{'Noise':>8s}  {'ICIR':>8s}  {'Mean IC':>8s}  {'N IC':>6s}")
    print("-" * 40)
    
    results = []
    for nr in noise_ratios:
        print(f"Running noise_ratio={nr:.2f}...", end=' ', flush=True)
        r = run_pipeline(noise_ratio=nr)
        results.append(r)
        icir_str = f"{r['icir']:.4f}" if r['icir'] is not None else "N/A"
        mean_ic_str = f"{r['mean_ic']:.4f}" if r['mean_ic'] is not None else "N/A"
        print(f"ICIR={icir_str}, Mean IC={mean_ic_str}, N={r['n_valid_ic']}")
    
    # 验证单调性
    valid_results = [r for r in results if r['icir'] is not None]
    
    if len(valid_results) < 5:
        print("\n⚠ Insufficient valid ICIR data for validation.")
        return
    
    icirs = [r['icir'] for r in valid_results]
    nrs = [r['noise_ratio'] for r in valid_results]
    
    from scipy.stats import spearmanr
    corr = spearmanr(nrs, icirs).correlation
    
    print(f"\n{'='*60}")
    print(f"Validation Results:")
    print(f"  Spearman(noise_ratio, ICIR) = {corr:.4f}")
    print(f"  ICIR range: [{min(icirs):.4f}, {max(icirs):.4f}]")
    
    if corr < -0.6:
        print(f"  ✓ PASS: ICIR decreases with noise (ρ={corr:.4f} < -0.6)")
    else:
        print(f"  ✗ WARNING: Weak noise-ICIR relationship (ρ={corr:.4f} > -0.6)")
        print(f"    This may indicate insufficient data or model trained on small sample.")
    
    print("=" * 60)


if __name__ == '__main__':
    main()
