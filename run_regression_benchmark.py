"""Fast runner for C-MAPSS regression benchmark (same logic as cmapss-rul-regression.ipynb)."""
import os, glob, warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor
import xgboost as xgb

warnings.filterwarnings('ignore')

RUL_CAP = 125
ROLLING_WINDOW = 5
DATASET_IDS = ['FD001', 'FD002', 'FD003', 'FD004']
from paths import DATA_DIR, RESULTS

OUTPUT_DIR = str(RESULTS)
os.makedirs(OUTPUT_DIR, exist_ok=True)
DATA_DIR = str(DATA_DIR)

COL_NAMES = (
    ['unit', 'cycle', 'op_setting_1', 'op_setting_2', 'op_setting_3']
    + [f'sensor_{i+1}' for i in range(21)]
)


def discover_files(data_dir):
    mapping = {}
    for path in glob.glob(os.path.join(data_dir, '*')):
        name = os.path.basename(path).upper()
        for fd in DATASET_IDS:
            if fd in name:
                mapping.setdefault(fd, {})
                if 'TRAIN' in name:
                    mapping[fd]['train'] = path
                if 'TEST' in name:
                    mapping[fd]['test'] = path
                if 'RUL' in name:
                    mapping[fd]['rul'] = path
    return mapping


def load_dataset(train_path, test_path, rul_path):
    train = pd.read_csv(train_path, sep=r'\s+', header=None, names=COL_NAMES)
    train['RUL'] = train.groupby('unit')['cycle'].transform('max') - train['cycle']
    test = pd.read_csv(test_path, sep=r'\s+', header=None, names=COL_NAMES)
    rul = pd.read_csv(rul_path, sep=r'\s+', header=None, names=['RUL_final'])
    max_cycle = test.groupby('unit')['cycle'].max().reset_index().sort_values('unit').reset_index(drop=True)
    max_cycle['RUL_final'] = rul['RUL_final'].values
    test = test.merge(max_cycle[['unit', 'RUL_final']], on='unit', how='left')
    test['RUL'] = test.groupby('unit')['cycle'].transform('max') - test['cycle'] + test['RUL_final']
    return train.drop(columns=[]), test.drop(columns=['RUL_final'])


def select_features(train_df, candidate_cols, min_std=1e-6):
    stds = train_df[candidate_cols].std()
    return stds[stds > min_std].index.tolist()


def build_feature_matrix(df, base_cols, window=ROLLING_WINDOW):
    parts = []
    grouped = df.groupby('unit', sort=False)
    for col in base_cols:
        parts.append(df[[col]])
        parts.append(grouped[col].transform(lambda s: s - s.iloc[0]).to_frame(f'{col}_rel'))
        parts.append(grouped[col].transform(
            lambda s: s.rolling(window, min_periods=1).mean()
        ).to_frame(f'{col}_rmean'))
        parts.append(grouped[col].transform(
            lambda s: s.rolling(window, min_periods=1).std().fillna(0)
        ).to_frame(f'{col}_rstd'))
    return pd.concat(parts, axis=1)


def prepare_xy(train_df, test_df, feature_cols):
    train_feat = build_feature_matrix(train_df, feature_cols)
    test_feat = build_feature_matrix(test_df, feature_cols)
    y_train = train_df['RUL'].clip(upper=RUL_CAP).values
    y_test = test_df['RUL'].values
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_feat)
    X_test = scaler.transform(test_feat)
    return X_train, y_train, X_test, y_test, list(train_feat.columns)


def regression_metrics(y_true, y_pred):
    y_pred = np.clip(y_pred, 0, None)
    return {
        'RMSE': float(np.sqrt(mean_squared_error(y_true, y_pred))),
        'MAE': float(mean_absolute_error(y_true, y_pred)),
        'R2': float(r2_score(y_true, y_pred)),
    }


def get_models(n_train):
    large = n_train > 40000
    n_est = 120 if large else 250
    return {
        'Ridge': Ridge(alpha=5.0),
        'ElasticNet': ElasticNet(alpha=0.005, l1_ratio=0.2, max_iter=8000),
        'RandomForest': RandomForestRegressor(
            n_estimators=n_est, max_depth=24, min_samples_leaf=2, n_jobs=-1, random_state=42),
        'ExtraTrees': ExtraTreesRegressor(
            n_estimators=n_est, max_depth=24, min_samples_leaf=2, n_jobs=-1, random_state=42),
        'HistGradientBoosting': HistGradientBoostingRegressor(
            max_iter=250 if large else 400, learning_rate=0.06, max_depth=12,
            min_samples_leaf=15, random_state=42),
        'XGBoost': xgb.XGBRegressor(
            n_estimators=250 if large else 500, max_depth=8, learning_rate=0.05,
            subsample=0.85, colsample_bytree=0.85, reg_lambda=1.0,
            objective='reg:squarederror', random_state=42, n_jobs=-1),
    }


def evaluate_model(model, X_train, y_train, X_test, y_test, test_df):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics_all = regression_metrics(y_test, y_pred)
    last_idx = test_df.groupby('unit')['cycle'].idxmax().values
    metrics_last = regression_metrics(y_test[last_idx], y_pred[last_idx])
    return metrics_all, metrics_last, y_pred


def main():
    file_map = discover_files(DATA_DIR)
    all_results = []
    best_per_dataset = {}

    for fd in DATASET_IDS:
        print('\n' + '=' * 60)
        print(f'Dataset: {fd}')
        train_df, test_df = load_dataset(
            file_map[fd]['train'], file_map[fd]['test'], file_map[fd]['rul'])
        candidate = [c for c in train_df.columns if c.startswith('sensor_') or c.startswith('op_setting_')]
        feature_cols = select_features(train_df, candidate)
        X_train, y_train, X_test, y_test, expanded_cols = prepare_xy(train_df, test_df, feature_cols)
        print(f'  train={len(X_train):,}  test={len(X_test):,}  features={len(expanded_cols)}')

        fd_best_rmse = np.inf
        fd_best = None
        for name, model in get_models(len(X_train)).items():
            print(f'  {name}...', end=' ', flush=True)
            metrics_all, metrics_last, y_pred = evaluate_model(
                model, X_train, y_train, X_test, y_test, test_df)
            print(f"RMSE={metrics_last['RMSE']:.2f} MAE={metrics_last['MAE']:.2f} R2={metrics_last['R2']:.4f}")
            row = {
                'Dataset': fd, 'Model': name,
                'RMSE_all': metrics_all['RMSE'], 'MAE_all': metrics_all['MAE'], 'R2_all': metrics_all['R2'],
                'RMSE_last': metrics_last['RMSE'], 'MAE_last': metrics_last['MAE'], 'R2_last': metrics_last['R2'],
                'n_features': len(expanded_cols), 'n_train': len(X_train), 'n_test': len(X_test),
            }
            all_results.append(row)
            if metrics_last['RMSE'] < fd_best_rmse:
                fd_best_rmse = metrics_last['RMSE']
                fd_best = row.copy()
        best_per_dataset[fd] = fd_best
        print(f"  >> Best: {fd_best['Model']}  RMSE={fd_best['RMSE_last']:.2f}  R2={fd_best['R2_last']:.4f}")

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(os.path.join(OUTPUT_DIR, 'all_model_results.csv'), index=False)

    best_rows = [{
        'Dataset': fd,
        'Best Model': best_per_dataset[fd]['Model'],
        'RMSE': best_per_dataset[fd]['RMSE_last'],
        'MAE': best_per_dataset[fd]['MAE_last'],
        'R2': best_per_dataset[fd]['R2_last'],
        'RMSE (all rows)': best_per_dataset[fd]['RMSE_all'],
        'MAE (all rows)': best_per_dataset[fd]['MAE_all'],
        'R2 (all rows)': best_per_dataset[fd]['R2_all'],
    } for fd in DATASET_IDS]
    best_df = pd.DataFrame(best_rows).set_index('Dataset')
    best_df.to_csv(os.path.join(OUTPUT_DIR, 'best_per_dataset.csv'))
    print('\n=== BEST PER DATASET (last-cycle metrics) ===')
    print(best_df.round(4).to_string())


if __name__ == '__main__':
    main()
