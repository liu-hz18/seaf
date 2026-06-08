"""检查最新一次 mlflow 实验的指标。"""
import mlflow

mlflow.set_tracking_uri('sqlite:///mlruns.db')
exps = mlflow.search_experiments()
non_default = sorted([e for e in exps if e.name != 'Default'], key=lambda e: e.creation_time)
if non_default:
    exp = non_default[-1]
    runs = mlflow.search_runs([exp.experiment_id])
    print(f'Experiment="{exp.name}"  runs={len(runs)}')
    if len(runs) > 0:
        metric_cols = sorted([c for c in runs.columns if c.startswith('metrics.')])
        print(f'Total metric columns: {len(metric_cols)}')
        for c in metric_cols:
            vals = runs[c].dropna()
            print(f'  {c}: n={len(vals):4d}  range=[{vals.min():.4f}, {vals.max():.4f}]')
        param_cols = [c for c in runs.columns if c.startswith('params.')]
        for c in sorted(param_cols):
            v = runs[c].dropna()
            if len(v) > 0:
                print(f'  {c} = {v.iloc[0]}')
