"""Build self-contained Jupyter notebooks (all code inline, no scripts folder)."""
import json
import textwrap
from pathlib import Path

from paths import ROOT

NB_DIR = ROOT / "notebooks"
NB_DIR.mkdir(exist_ok=True)

ROOT_SETUP = textwrap.dedent("""
from pathlib import Path
ROOT = Path.cwd()
if ROOT.name == 'notebooks':
    ROOT = ROOT.parent
RESULTS = ROOT / 'results'
DOCS = ROOT / 'docs'
MODELS = ROOT / 'models'
for p in (RESULTS, DOCS, MODELS):
    p.mkdir(exist_ok=True)

DATA_DIR = ROOT / 'data'
for candidate in [ROOT / 'data', ROOT, ROOT.parent,
                  Path(r'C:\\kaggle\\input\\cmapss-jet-engine-simulated-data'),
                  Path('/kaggle/input/cmapss-jet-engine-simulated-data')]:
    if candidate.is_dir() and list(candidate.glob('train_FD*.txt')):
        DATA_DIR = candidate
        break
print('ROOT:', ROOT)
print('DATA_DIR:', DATA_DIR)
""").strip()


def nb(cells):
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11.0"},
        },
        "cells": cells,
    }


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": [s]}


def code(s):
    return {"cell_type": "code", "metadata": {}, "outputs": [], "execution_count": None, "source": [s]}


def write_nb(name, cells):
    path = NB_DIR / name
    path.write_text(json.dumps(nb(cells), indent=1), encoding="utf-8")
    print(f"Wrote {path}")


def main():
    # ------------------------------------------------------------------ 01 ML
    write_nb("01_ml_regression_benchmark.ipynb", [
        md("# C-MAPSS RUL — ML Regression Benchmark\n\n"
           "Trains 6 sklearn models on FD001–FD004.\n"
           "Features: raw + unit-relative + rolling mean/std (window=5).\n"
           "Metric: **last-cycle RMSE / MAE / R²** (RUL capped at 125 for training)."),
        code(ROOT_SETUP),
        md("## Imports & configuration"),
        code(textwrap.dedent("""
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
            COL_NAMES = (
                ['unit', 'cycle', 'op_setting_1', 'op_setting_2', 'op_setting_3']
                + [f'sensor_{i+1}' for i in range(21)]
            )
        """).strip()),
        md("## Data loading"),
        code(textwrap.dedent("""
            def discover_files(data_dir):
                mapping = {}
                for path in glob.glob(os.path.join(str(data_dir), '*')):
                    name = os.path.basename(path).upper()
                    for fd in DATASET_IDS:
                        if fd in name:
                            mapping.setdefault(fd, {})
                            if 'TRAIN' in name: mapping[fd]['train'] = path
                            if 'TEST' in name: mapping[fd]['test'] = path
                            if 'RUL' in name: mapping[fd]['rul'] = path
                return mapping

            def load_dataset(train_path, test_path, rul_path):
                train = pd.read_csv(train_path, sep=r'\\s+', header=None, names=COL_NAMES)
                train['RUL'] = train.groupby('unit')['cycle'].transform('max') - train['cycle']
                test = pd.read_csv(test_path, sep=r'\\s+', header=None, names=COL_NAMES)
                rul = pd.read_csv(rul_path, sep=r'\\s+', header=None, names=['RUL_final'])
                max_cycle = test.groupby('unit')['cycle'].max().reset_index().sort_values('unit')
                max_cycle['RUL_final'] = rul['RUL_final'].values
                test = test.merge(max_cycle[['unit', 'RUL_final']], on='unit', how='left')
                test['RUL'] = test.groupby('unit')['cycle'].transform('max') - test['cycle'] + test['RUL_final']
                return train, test.drop(columns=['RUL_final'])

            def select_features(train_df, candidate_cols, min_std=1e-6):
                stds = train_df[candidate_cols].std()
                return stds[stds > min_std].index.tolist()

            file_map = discover_files(DATA_DIR)
            list(file_map.keys())
        """).strip()),
        md("## Feature engineering"),
        code(textwrap.dedent("""
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
        """).strip()),
        md("## Models & evaluation"),
        code(textwrap.dedent("""
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
                return metrics_all, metrics_last
        """).strip()),
        md("## Run benchmark on all datasets"),
        code(textwrap.dedent("""
            all_results = []
            best_per_dataset = {}

            for fd in DATASET_IDS:
                print('\\n' + '=' * 60)
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
                    metrics_all, metrics_last = evaluate_model(
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
                print(f"  >> Best: {fd_best['Model']}  RMSE={fd_best['RMSE_last']:.2f}")

            results_df = pd.DataFrame(all_results)
            results_df.to_csv(RESULTS / 'all_model_results.csv', index=False)
            best_df = pd.DataFrame([{
                'Dataset': fd, 'Best Model': best_per_dataset[fd]['Model'],
                'RMSE': best_per_dataset[fd]['RMSE_last'], 'MAE': best_per_dataset[fd]['MAE_last'],
                'R2': best_per_dataset[fd]['R2_last'],
            } for fd in DATASET_IDS])
            best_df.to_csv(RESULTS / 'best_per_dataset.csv', index=False)
            print('\\nSaved:', RESULTS / 'all_model_results.csv')
        """).strip()),
        md("## Results"),
        code("display(results_df.round(3))\ndisplay(best_df.round(3))"),
    ])

    # ------------------------------------------------------------------ 02 DL
    dl_script = (ROOT / "run_dl_benchmark.py").read_text(encoding="utf-8")
    # Strip argparse main block; notebooks run cells directly
    dl_body = dl_script.split('def main():')[0]
    dl_body = (
        dl_body
        .replace("from paths import DATA_DIR, RESULTS\n\n", "")
        .replace("OUTPUT_DIR = str(RESULTS)", "OUTPUT_DIR = str(RESULTS)")
        .replace('RESULTS_CSV = os.path.join(OUTPUT_DIR, "dl_model_results.csv")',
                 'RESULTS_CSV = str(RESULTS / "dl_model_results.csv")')
        .replace("DATA_DIR = str(DATA_DIR)", "DATA_DIR = str(DATA_DIR)")
        .replace("os.makedirs(OUTPUT_DIR, exist_ok=True)\n", "")
    )

    write_nb("02_dl_benchmark.ipynb", [
        md("# C-MAPSS RUL — Deep Learning Benchmark (PyTorch)\n\n"
           "Models: LSTM, GRU, CNN1D, TCN, Transformer on 30-step sequences."),
        code(ROOT_SETUP),
        md("## Imports, data utilities, models & training"),
        code(dl_body.strip()),
        md("## GPU check"),
        code("import torch\nprint('Device:', DEVICE)\nif torch.cuda.is_available():\n    print(torch.cuda.get_device_name(0))"),
        md("## Run one model (example: Transformer)"),
        code(textwrap.dedent("""
            file_map = discover_files(DATA_DIR)
            existing = pd.read_csv(RESULTS_CSV) if os.path.isfile(RESULTS_CSV) else None
            results_dl = run_one_model('Transformer', file_map, existing)
            results_dl.round(3)
        """).strip()),
        md("## Run all DL models"),
        code(textwrap.dedent("""
            # Uncomment to run all 5 models (~30-60 min on GPU):
            # file_map = discover_files(DATA_DIR)
            # existing = pd.read_csv(RESULTS_CSV) if os.path.isfile(RESULTS_CSV) else None
            # for model_name in DL_MODELS:
            #     existing = run_one_model(model_name, file_map, existing)
            # print(existing.pivot(index='Dataset', columns='Model', values='RMSE_last').round(2))
        """).strip()),
        md("## View saved results"),
        code("pd.read_csv(RESULTS / 'dl_model_results.csv').round(3)"),
    ])

    # ------------------------------------------------------------------ 03 Graph
    graph_script = (ROOT / "run_graph_benchmark.py").read_text(encoding="utf-8")
    graph_body = graph_script.split('def main():')[0]
    graph_body = (
        graph_body
        .replace("from paths import DATA_DIR, RESULTS\n\n", "")
        .replace('RESULTS_CSV = os.path.join(OUTPUT_DIR, "graph_model_results.csv")',
                 'RESULTS_CSV = str(RESULTS / "graph_model_results.csv")')
        .replace("os.makedirs(OUTPUT_DIR, exist_ok=True)\n", "")
    )

    write_nb("03_graph_benchmark.ipynb", [
        md("# C-MAPSS RUL — Graph Neural Network Benchmark (PyG)\n\n"
           "Models: TemporalGCN, TemporalGAT."),
        code(ROOT_SETUP),
        md("## Imports, graph construction, models & training"),
        code(graph_body.strip()),
        md("## Run TemporalGAT (completed in benchmark)"),
        code(textwrap.dedent("""
            file_map = discover_files(DATA_DIR)
            existing = pd.read_csv(RESULTS_CSV) if os.path.isfile(RESULTS_CSV) else None
            graph_results = run_one_model('TemporalGAT', file_map, existing)
            graph_results.round(3)
        """).strip()),
        md("## Run all graph models"),
        code(textwrap.dedent("""
            # Uncomment to run all 4 graph models:
            # for name in GRAPH_MODELS:
            #     existing = run_one_model(name, file_map, existing)
        """).strip()),
        md("## View results"),
        code("pd.read_csv(RESULTS / 'graph_model_results.csv').round(3)"),
    ])

    # ------------------------------------------------------------------ 04 Documentation
    write_nb("04_documentation_and_analysis.ipynb", [
        md("# C-MAPSS RUL — Documentation & Analysis\n\n"
           "Sensor importance, comparison plots, epoch metrics, and verified report generation."),
        code(ROOT_SETUP + "\nimport matplotlib.pyplot as plt\nimport seaborn as sns\nsns.set_theme(style='whitegrid')"),
        md("## Load all benchmark results"),
        code(textwrap.dedent("""
            ml = pd.read_csv(RESULTS / 'all_model_results.csv'); ml['Family'] = 'ML'
            dl = pd.read_csv(RESULTS / 'dl_model_results.csv'); dl['Family'] = 'DL'
            g = pd.read_csv(RESULTS / 'graph_model_results.csv'); g['Family'] = 'Graph'
            all_df = pd.concat([
                ml.rename(columns={'RMSE_last':'RMSE','MAE_last':'MAE','R2_last':'R2'}),
                dl.rename(columns={'RMSE_last':'RMSE','MAE_last':'MAE','R2_last':'R2'}),
                g.rename(columns={'RMSE_last':'RMSE','MAE_last':'MAE','R2_last':'R2'}),
            ], ignore_index=True)[['Model','Dataset','Family','RMSE','MAE','R2']]
            all_df = all_df.drop_duplicates(subset=['Model','Dataset'], keep='last')
            all_df.to_csv(RESULTS / 'all_results_master.csv', index=False)
            best = all_df.loc[all_df.groupby('Dataset')['RMSE'].idxmin()]
            best.to_csv(RESULTS / 'best_per_dataset_all.csv', index=False)
            display(best.round(3))
        """).strip()),
        md("## ML epoch metrics (stages 10 / 50 / 100 / 200)"),
        code(textwrap.dedent("""
            from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
            import xgboost as xgb

            CHECKPOINTS = [10, 50, 100, 200]
            epoch_rows = []
            for fd in DATASET_IDS:
                train_df, test_df = load_dataset(file_map[fd]['train'], file_map[fd]['test'], file_map[fd]['rul'])
                candidate = [c for c in train_df.columns if c.startswith('sensor_') or c.startswith('op_setting_')]
                feature_cols = select_features(train_df, candidate)
                X_train, y_train, X_test, y_test, _ = prepare_xy(train_df, test_df, feature_cols)
                last_idx = test_df.groupby('unit')['cycle'].idxmax().values
                for stage in CHECKPOINTS:
                    for name, model in [
                        ('XGBoost', xgb.XGBRegressor(n_estimators=stage, max_depth=8, learning_rate=0.05, random_state=42, n_jobs=-1)),
                        ('ExtraTrees', ExtraTreesRegressor(n_estimators=stage, max_depth=16, n_jobs=-1, random_state=42)),
                        ('HistGradientBoosting', HistGradientBoostingRegressor(max_iter=stage, learning_rate=0.06, max_depth=12, random_state=42)),
                    ]:
                        model.fit(X_train, y_train)
                        pred = model.predict(X_test)
                        rmse = float(np.sqrt(mean_squared_error(y_test[last_idx], pred[last_idx])))
                        epoch_rows.append({'Model': name, 'Dataset': fd, 'Epoch': stage, 'RMSE': rmse})
            epoch_ml = pd.DataFrame(epoch_rows)
            epoch_ml.to_csv(DOCS / 'epoch_metrics_ml.csv', index=False)
            display(epoch_ml.pivot_table(index='Model', columns='Epoch', values='RMSE').round(2))
        """).strip()),
        md("## Sensor importance (ExtraTrees)"),
        code(textwrap.dedent("""
            FIGURES = DOCS / 'figures'
            FIGURES.mkdir(exist_ok=True)
            imp_rows = []
            for fd in ['FD001', 'FD002', 'FD003', 'FD004']:
                train_df, test_df = load_dataset(file_map[fd]['train'], file_map[fd]['test'], file_map[fd]['rul'])
                candidate = [c for c in train_df.columns if c.startswith('sensor_') or c.startswith('op_setting_')]
                feature_cols = select_features(train_df, candidate)
                X_train, y_train, _, _, expanded = prepare_xy(train_df, test_df, feature_cols)
                m = ExtraTreesRegressor(n_estimators=200, max_depth=24, n_jobs=-1, random_state=42)
                m.fit(X_train, y_train)
                for feat, val in zip(expanded, m.feature_importances_):
                    key = feat.split('_r')[0] if '_r' in feat else feat
                    for sfx in ['_rel','_rmean','_rstd']:
                        if key.endswith(sfx): key = key[:-len(sfx)]
                    imp_rows.append({'Dataset': fd, 'Feature': key, 'Importance': float(val)})
            sensor_imp = pd.DataFrame(imp_rows).groupby(['Dataset','Feature'])['Importance'].sum().reset_index()
            sensor_imp.to_csv(DOCS / 'sensor_importance.csv', index=False)
            for fd in ['FD001','FD004']:
                sub = sensor_imp[sensor_imp.Dataset==fd].nlargest(10, 'Importance')
                sub.plot.barh(x='Feature', y='Importance', figsize=(8,5), title=f'Top features {fd}', legend=False)
                plt.tight_layout()
                plt.savefig(FIGURES / f'sensor_importance_{fd}.png', dpi=150)
                plt.show()
        """).strip()),
        md("## Comparison plots"),
        code(textwrap.dedent("""
            pivot = all_df.pivot_table(index='Model', columns='Dataset', values='RMSE')
            plt.figure(figsize=(8, max(5, len(pivot)*0.35)))
            sns.heatmap(pivot, annot=True, fmt='.1f', cmap='RdYlGn_r')
            plt.title('RMSE heatmap (last-cycle)')
            plt.tight_layout()
            plt.savefig(FIGURES / 'heatmap_rmse_all.png', dpi=150)
            plt.show()
            for fd in DATASET_IDS:
                sub = all_df[all_df.Dataset==fd].sort_values('RMSE')
                plt.figure(figsize=(9, max(4, len(sub)*0.35)))
                sns.barplot(data=sub, y='Model', x='RMSE', hue='Family', dodge=False)
                plt.title(f'{fd} — RMSE last-cycle')
                plt.tight_layout()
                plt.savefig(FIGURES / f'compare_{fd}_RMSE_last.png', dpi=150)
                plt.show()
        """).strip()),
        md("## View pre-computed DL epoch metrics & report"),
        code(textwrap.dedent("""
            dl_epochs = DOCS / 'epoch_metrics_dl.csv'
            if dl_epochs.exists():
                display(pd.read_csv(dl_epochs).pivot_table(index='Model', columns='Epoch', values='RMSE').round(2))
            report = DOCS / 'CMAPSS_RUL_Benchmark_Documentation.md'
            if report.exists():
                from IPython.display import Markdown
                display(Markdown(report.read_text(encoding='utf-8')[:4000] + '\\n\\n...'))
            else:
                print('Run verify cell below to generate report from CSVs.')
        """).strip()),
        md("## Regenerate verified markdown report"),
        code(textwrap.dedent("""
            def df_to_md(df):
                cols = list(df.columns)
                lines = ['| ' + ' | '.join(str(c) for c in cols) + ' |', '| ' + ' | '.join('---' for _ in cols) + ' |']
                for _, row in df.iterrows():
                    lines.append('| ' + ' | '.join(str(row[c]) for c in cols) + ' |')
                return '\\n'.join(lines)

            b = {r.Dataset: r for _, r in best.iterrows()}
            lines = ['# NASA C-MAPSS RUL — Benchmark Documentation\\n',
                     '*Metrics verified from results/*.csv*\\n',
                     '## Best Model Per Dataset\\n']
            bt = best[['Dataset','Model','Family','RMSE','MAE','R2']].copy().round(4)
            lines.append(df_to_md(bt))
            lines.append('\\n## All Models RMSE\\n')
            lines.append(df_to_md(all_df.pivot_table(index='Model', columns='Dataset', values='RMSE').round(2).reset_index()))
            (DOCS / 'CMAPSS_RUL_Benchmark_Documentation.md').write_text('\\n'.join(lines), encoding='utf-8')
            print('Report saved:', DOCS / 'CMAPSS_RUL_Benchmark_Documentation.md')
        """).strip()),
    ])

    # Need DATASET_IDS and helpers in notebook 04 - add dependency cell
    # Insert after ROOT_SETUP in 04 - need imports from notebook 01
    # I'll add a cell that re-defines minimal helpers or imports from running 01 first

    # Fix notebook 04 - add imports and helper definitions at start
    nb04_path = NB_DIR / "04_documentation_and_analysis.ipynb"
    nb04 = json.loads(nb04_path.read_text(encoding="utf-8"))
    helper_cell = code(textwrap.dedent("""
        # Re-use ML helpers (run notebook 01 first, or define here)
        import os, glob, warnings
        import numpy as np
        import pandas as pd
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        warnings.filterwarnings('ignore')
        RUL_CAP, ROLLING_WINDOW = 125, 5
        DATASET_IDS = ['FD001','FD002','FD003','FD004']
        COL_NAMES = ['unit','cycle','op_setting_1','op_setting_2','op_setting_3'] + [f'sensor_{i+1}' for i in range(21)]

        def discover_files(data_dir):
            mapping = {}
            for path in glob.glob(os.path.join(str(data_dir), '*')):
                name = os.path.basename(path).upper()
                for fd in DATASET_IDS:
                    if fd in name:
                        mapping.setdefault(fd, {})
                        if 'TRAIN' in name: mapping[fd]['train'] = path
                        if 'TEST' in name: mapping[fd]['test'] = path
                        if 'RUL' in name: mapping[fd]['rul'] = path
            return mapping

        def load_dataset(train_path, test_path, rul_path):
            train = pd.read_csv(train_path, sep=r'\\s+', header=None, names=COL_NAMES)
            train['RUL'] = train.groupby('unit')['cycle'].transform('max') - train['cycle']
            test = pd.read_csv(test_path, sep=r'\\s+', header=None, names=COL_NAMES)
            rul = pd.read_csv(rul_path, sep=r'\\s+', header=None, names=['RUL_final'])
            max_cycle = test.groupby('unit')['cycle'].max().reset_index().sort_values('unit')
            max_cycle['RUL_final'] = rul['RUL_final'].values
            test = test.merge(max_cycle[['unit','RUL_final']], on='unit', how='left')
            test['RUL'] = test.groupby('unit')['cycle'].transform('max') - test['cycle'] + test['RUL_final']
            return train, test.drop(columns=['RUL_final'])

        def select_features(train_df, cols, min_std=1e-6):
            stds = train_df[cols].std()
            return stds[stds > min_std].index.tolist()

        def build_feature_matrix(df, base_cols, window=5):
            parts, grouped = [], df.groupby('unit', sort=False)
            for col in base_cols:
                parts += [df[[col]],
                    grouped[col].transform(lambda s: s-s.iloc[0]).to_frame(f'{col}_rel'),
                    grouped[col].transform(lambda s: s.rolling(window,min_periods=1).mean()).to_frame(f'{col}_rmean'),
                    grouped[col].transform(lambda s: s.rolling(window,min_periods=1).std().fillna(0)).to_frame(f'{col}_rstd')]
            return pd.concat(parts, axis=1)

        def prepare_xy(train_df, test_df, feature_cols):
            train_feat = build_feature_matrix(train_df, feature_cols)
            test_feat = build_feature_matrix(test_df, feature_cols)
            scaler = StandardScaler()
            X_train = scaler.fit_transform(train_feat)
            X_test = scaler.transform(test_feat)
            return X_train, train_df['RUL'].clip(upper=RUL_CAP).values, X_test, test_df['RUL'].values, list(train_feat.columns)

        file_map = discover_files(DATA_DIR)
    """).strip())
    nb04["cells"].insert(2, helper_cell)
    nb04_path.write_text(json.dumps(nb04, indent=1), encoding="utf-8")
    print(f"Updated {nb04_path}")

    # ------------------------------------------------------------------ 05 Save models
    write_nb("05_save_best_models.ipynb", [
        md("# Save Best Models (.pkl / .pth)\n\n"
           "Saves best model per dataset. Run notebooks **01** and **02** first, or run the setup cell below."),
        code(ROOT_SETUP),
        md("## Configuration — best models from benchmark"),
        code(textwrap.dedent("""
            import json, pickle, joblib
            import torch

            BEST = {
                'FD001': {'family': 'ML', 'model': 'ExtraTrees'},
                'FD002': {'family': 'DL', 'model': 'Transformer'},
                'FD003': {'family': 'ML', 'model': 'HistGradientBoosting'},
                'FD004': {'family': 'ML', 'model': 'HistGradientBoosting'},
            }
        """).strip()),
        md("## Save ML models"),
        code(textwrap.dedent("""
            # Requires helpers from notebook 01 (discover_files, load_dataset, select_features, prepare_xy, get_models, build_feature_matrix)
            from sklearn.preprocessing import StandardScaler

            manifest = []
            file_map = discover_files(DATA_DIR)

            for fd, spec in BEST.items():
                if spec['family'] != 'ML':
                    continue
                train_df, test_df = load_dataset(file_map[fd]['train'], file_map[fd]['test'], file_map[fd]['rul'])
                candidate = [c for c in train_df.columns if c.startswith('sensor_') or c.startswith('op_setting_')]
                feature_cols = select_features(train_df, candidate)
                X_train, y_train, X_test, y_test, expanded_cols = prepare_xy(train_df, test_df, feature_cols)
                train_feat = build_feature_matrix(train_df, feature_cols)
                scaler = StandardScaler()
                scaler.fit(train_feat)
                model = get_models(len(X_train))[spec['model']]
                model.fit(X_train, y_train)
                bundle = {'model': model, 'scaler': scaler, 'feature_cols': feature_cols,
                          'expanded_cols': expanded_cols, 'dataset': fd, 'model_name': spec['model'],
                          'family': 'ML', 'rul_cap': 125}
                out = MODELS / f"{fd}_{spec['model']}.pkl"
                joblib.dump(bundle, out)
                manifest.append({'dataset': fd, 'model': spec['model'], 'family': 'ML', 'path': str(out)})
                print('Saved', out)
        """).strip()),
        md("## Save DL model (Transformer FD002)\n\n"
           "> Run **notebook 02** first so `build_model`, `prepare_sequence_data`, `train_model`, `SEQ_LEN`, and `DEVICE` are defined."),
        code(textwrap.dedent("""
            # Requires notebook 02 definitions (build_model, prepare_sequence_data, train_model, DEVICE, SEQ_LEN)
            fd, model_name = 'FD002', 'Transformer'
            train_df, test_df = load_dataset(file_map[fd]['train'], file_map[fd]['test'], file_map[fd]['rul'])
            candidate = [c for c in train_df.columns if c.startswith('sensor_') or c.startswith('op_setting_')]
            feature_cols = select_features(train_df, candidate)
            X_train, y_train, train_meta, X_test, y_test, test_meta = prepare_sequence_data(
                train_df, test_df, feature_cols, SEQ_LEN)
            batch = 16
            model = build_model(model_name, X_train.shape[2])
            model = train_model(model, X_train, y_train, train_meta, batch_size=batch)
            meta = {'dataset': fd, 'model_name': model_name, 'family': 'DL',
                    'feature_cols': feature_cols, 'seq_len': SEQ_LEN, 'n_features': X_train.shape[2]}
            pth = MODELS / f'{fd}_{model_name}.pth'
            pkl = MODELS / f'{fd}_{model_name}_meta.pkl'
            torch.save(model.state_dict(), pth)
            with open(pkl, 'wb') as f:
                pickle.dump(meta, f)
            manifest.append({'dataset': fd, 'model': model_name, 'family': 'DL',
                             'weights': str(pth), 'metadata': str(pkl)})
            print('Saved', pth, pkl)
        """).strip()),
        md("## Write manifest"),
        code(textwrap.dedent("""
            (MODELS / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
            print(json.dumps(manifest, indent=2))
        """).strip()),
    ])

    # Add ML helpers to notebook 05
    nb05 = json.loads((NB_DIR / "05_save_best_models.ipynb").read_text(encoding="utf-8"))
    nb05["cells"].insert(2, nb04["cells"][2])  # same helper cell as 04
    # Add get_models from 01
    nb05["cells"].insert(3, code(textwrap.dedent("""
        from sklearn.linear_model import Ridge, ElasticNet
        from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor
        import xgboost as xgb

        def get_models(n_train):
            large = n_train > 40000
            n_est = 120 if large else 250
            return {
                'ExtraTrees': ExtraTreesRegressor(n_estimators=n_est, max_depth=24, n_jobs=-1, random_state=42),
                'HistGradientBoosting': HistGradientBoostingRegressor(
                    max_iter=250 if large else 400, learning_rate=0.06, max_depth=12, random_state=42),
            }
    """).strip()))
    (NB_DIR / "05_save_best_models.ipynb").write_text(json.dumps(nb05, indent=1), encoding="utf-8")

    # Add note to 05 about running 02 for DL - include minimal note in markdown
    print("Done — all notebooks are self-contained.")


if __name__ == "__main__":
    main()
