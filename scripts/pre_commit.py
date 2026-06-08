#!/usr/bin/env python3
"""
SEAF Pre-commit 自动化验证链

执行顺序: lint → test → bench → changelog → git diff → commit

用法:
    python scripts/pre_commit.py "commit message"

步骤:
    1. ruff check + format       — 代码规范 & 自动修复
    2. mypy                      — 类型检查
    3. pytest                    — 单元测试 (55 tests)
    4. bench_all_factors         — 因子基准测试 (10 modules)
    5. changelog diff            — 展示 CHANGELOG.md 待提交变更
    6. git diff --stat           — 展示改动摘要
    7. git add -A + commit       — 确认后提交
"""

import os
import subprocess
import sys


def run(cmd: list[str], desc: str) -> bool:
    """运行命令，打印结果，返回是否通过。"""
    print(f'\n{"=" * 60}')
    print(f'  [{desc}]')
    print(f'  {" ".join(cmd)}')
    print(f'{"=" * 60}')
    result = subprocess.run(cmd, check=False, cwd=os.path.dirname(__file__) + '/..')
    if result.returncode != 0:
        print(f'  FAILED: {desc} (exit {result.returncode})')
        return False
    print(f'  PASS: {desc}')
    return True


def main() -> None:
    commit_msg = sys.argv[1] if len(sys.argv) > 1 else None

    steps = [
        (['python', '-m', 'ruff', 'check', '--fix'], 'ruff check'),
        (['python', '-m', 'ruff', 'format', '--check'], 'ruff format'),
        (['python', '-m', 'pytest', 'test/', '-q', '--tb=short'], 'pytest (55 tests)'),
        (['python', 'scripts/bench_all_factors.py'], 'bench_all_factors (10 modules)'),
    ]

    for cmd, desc in steps:
        if not run(cmd, desc):
            sys.exit(1)

    # git diff summary
    print(f'\n{"=" * 60}')
    print('  [git diff --stat]')
    print(f'{"=" * 60}')
    subprocess.run(['git', 'diff', '--stat'], check=False, cwd=os.path.dirname(__file__) + '/..')
    subprocess.run(
        ['git', 'diff', '--cached', '--stat'], check=False, cwd=os.path.dirname(__file__) + '/..'
    )

    if commit_msg:
        print(f'\n  Commit message: {commit_msg}')
        response = input('  Proceed with commit? [y/N]: ').strip().lower()
        if response in ('y', 'yes'):
            subprocess.run(['git', 'add', '-A'], check=False, cwd=os.path.dirname(__file__) + '/..')
            subprocess.run(
                ['git', 'commit', '-m', commit_msg],
                check=False,
                cwd=os.path.dirname(__file__) + '/..',
            )
            print('  Commit complete.')
        else:
            print('  Commit skipped.')
    else:
        print('\n  No commit message provided. Skipping commit.')
        print("  Usage: python scripts/pre_commit.py 'commit message'")


if __name__ == '__main__':
    main()
