"""验证最新 mlflow 实验的 IC 指标 step。"""
import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri('sqlite:///mlruns.db')
exps = sorted(
    [e for e in mlflow.search_experiments() if e.name != 'Default'],
    key=lambda e: e.creation_time,
)
print(f'Experiments: {[e.name for e in exps]}')
exp = exps[-1]
runs = mlflow.search_runs([exp.experiment_id])
rid = runs['run_id'].iloc[0]
client = MlflowClient()

for metric_name in ['ic_analysis.pearson_ic', 'ic_analysis.rank_ic', 'model.pred_signal_max']:
    try:
        history = client.get_metric_history(rid, metric_name)
        steps = [h.step for h in history]
        print(f'{metric_name}: {len(history)} pts, unique_steps={len(set(steps))}')
        if steps:
            print(f'  steps: {steps[:3]}..{steps[-3:]}')
    except Exception as exc:
        print(f'{metric_name}: ERROR - {exc}')
