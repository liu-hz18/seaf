"""
模型训练与预测节点 — LightGBM 滚动训练，预测每日截面信号。

训练逻辑：
- 每 20 天用最近 200 天因子数据重新训练 LightGBM。
- Label：20 日后截面超额收益率（cs_zscore）。
- 时间穿越防护：只用 time 0~179 训练（label 需要 t+20，窗口共 200 天因子 + 20 天前瞻 close）。
- 预测：对窗口内最后一天（time 199）的因子做 predict，输出 pred_signal。

输入：来自因子节点的 128 列因子数据 + 来自 source 的 close 列。
输出：pred_signal（截面标准化后的预测值）。
"""
import numpy as np
import pandas as pd
import logging
from typing import Tuple, Any
from lightgbm import LGBMRegressor
from qpipe.frame3d import Frame3D
from sklearn.model_selection import TimeSeriesSplit


def model_train_predict(name: str, f3d: Frame3D, context: Any) -> Tuple[Frame3D, Any]:
    """模型训练与预测主函数。
    
    f3d 包含 window 天的数据（因子列 + close 列）。
    context 维护训练状态：
        - trained_model: LightGBM model
        - is_trained: bool
        - retrain_every: int (默认 20)
        - days_since_train: int
        - feature_cols: list
    """
    if context is None:
        context = {
            'trained_model': None,
            'is_trained': False,
            'retrain_every': 20,
            'days_since_train': 0,
            'feature_cols': None,
        }
    
    df = f3d.df.copy()
    
    # 识别因子列（不以原始数据列名开头）
    raw_cols = {'open', 'high', 'low', 'close', 'turnover', 'volume', 'market_cap'}
    if context['feature_cols'] is None:
        context['feature_cols'] = [c for c in df.columns if c not in raw_cols and not c.startswith('_')]
    
    feature_cols = context['feature_cols']
    
    # 提取 close 列
    if 'close' not in df.columns:
        raise ValueError(f"[{name}] Model node requires 'close' column in input data")
    
    # ===== 提取时间维度数据 =====
    times = sorted(df.index.get_level_values('key').unique())
    n_times = len(times)
    n_stocks = len(df.index.get_level_values('name').unique())
    
    if n_times < 20:
        # 窗口不足，不做预测
        empty_result = Frame3D(pd.DataFrame({'pred_signal': [0.0] * len(df.loc[times[-1]])}, 
                                            index=df.loc[times[-1]].index))
        return empty_result, context
    
    # 检查是否需要训练
    context['days_since_train'] += 1
    should_train = (not context['is_trained']) or (context['days_since_train'] >= context['retrain_every'])
    
    if should_train:
        # ===== 训练流程 =====
        logging.info(f"[{name}] Triggering retrain. days_since_train={context['days_since_train']}")
        
        # 需要用 200 天因子 + 20 天前瞻 close
        train_times = times[:200]       # 因子数据时间
        close_times = times[:220]       # close 需要 220 天（200 + 20 前瞻）
        
        if len(times) < 220:
            # 历史数据不够，等下一轮
            context['days_since_train'] = 0
            empty_result = Frame3D(pd.DataFrame({'pred_signal': [0.0] * len(df.loc[times[-1]])}, 
                                                index=df.loc[times[-1]].index))
            return empty_result, context
        
        # 构建特征矩阵 X
        X_list = []
        y_list = []
        
        for t_idx in range(min(180, len(train_times))):  # 0..179 用于训练（180 个时间点）
            t = train_times[t_idx]
            # 特征：该时间截面的因子值
            cs_mask = df.index.get_level_values('key') == t
            X_cs = df.loc[cs_mask, feature_cols].values.astype(float)
            
            # Label：20 日后截面超额收益
            t_label = train_times[t_idx + 20]  # t+20 对应的 key
            cs_mask_label = df.index.get_level_values('key') == t_label
            close_t20 = df.loc[cs_mask_label, 'close'].values
            
            # 当前 close
            close_t = df.loc[cs_mask, 'close'].values
            
            # 20 日前瞻收益
            fwd_ret = close_t20 / close_t - 1
            
            X_list.append(X_cs)
            y_list.append(fwd_ret)
        
        X = np.vstack(X_list)
        y = np.hstack(y_list)
        
        # 移除 NaN 样本
        valid = ~np.isnan(y) & ~np.any(np.isnan(X), axis=1)
        X = X[valid]
        y = y[valid]
        
        if len(y) < 50:
            logging.warning(f"[{name}] Insufficient training samples: {len(y)}")
            context['days_since_train'] = 0
            empty_result = Frame3D(pd.DataFrame({'pred_signal': [0.0] * len(df.loc[times[-1]])}, 
                                                index=df.loc[times[-1]].index))
            return empty_result, context
        
        # 截面标准化 label
        y_mean = np.mean(y)
        y_std = np.std(y)
        y_xd = (y - y_mean) / (y_std + 1e-10)
        
        # LightGBM 训练
        model = LGBMRegressor(
            n_estimators=100,
            max_depth=6,
            num_leaves=31,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=42,
            verbose=-1,
        )
        
        # TimeSeriesSplit 交叉验证评估
        tscv = TimeSeriesSplit(n_splits=3)
        cv_scores = []
        for train_idx, val_idx in tscv.split(X):
            if len(val_idx) < 10:
                continue
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y_xd[train_idx], y_xd[val_idx]
            model.fit(X_tr, y_tr)
            pred = model.predict(X_val)
            # IC on validation
            from scipy.stats import spearmanr
            try:
                ic = spearmanr(pred, y_val).correlation
                cv_scores.append(ic)
            except Exception:
                pass
        
        # Full training
        model.fit(X, y_xd)
        
        context['trained_model'] = model
        context['is_trained'] = True
        context['days_since_train'] = 0
        
        cv_mean = np.mean(cv_scores) if cv_scores else 0.0
        logging.info(
            f"[{name}] Training done: samples={len(y)}, features={len(feature_cols)}, "
            f"cv_ic_mean={cv_mean:.4f}"
        )
    
    # ===== 预测流程 =====
    model = context['trained_model']
    latest_t = times[-1]
    cs_mask_latest = df.index.get_level_values('key') == latest_t
    X_latest = df.loc[cs_mask_latest, feature_cols].values.astype(float)
    
    # 预测
    pred_raw = model.predict(X_latest)
    
    # 截面标准化预测信号
    pred_mean = np.nanmean(pred_raw)
    pred_std = np.nanstd(pred_raw)
    if pred_std > 0:
        pred_signal = (pred_raw - pred_mean) / pred_std
    else:
        pred_signal = np.zeros_like(pred_raw)
    
    logging.info(
        f"[{name}] Predict day t={latest_t}, "
        f"signal_mean={pred_signal.mean():.4f}, signal_std={pred_signal.std():.4f}"
    )
    
    # 构造输出 Frame3D
    result_df = pd.DataFrame(
        {'pred_signal': pred_signal},
        index=df.loc[cs_mask_latest].index
    )
    
    return Frame3D(result_df), context
