"""Collect DL epoch metrics and loss curves (FD001 primary, saves incrementally)."""
import json
import sys
from pathlib import Path

import pandas as pd

from paths import ROOT

DOCS = ROOT / "docs"
FIGS = DOCS / "figures"
sys.path.insert(0, str(ROOT))

from generate_documentation import (  # noqa: E402
    CHECKPOINT_EPOCHS,
    MAX_EPOCHS,
    collect_dl_epoch_data,
    detect_overfit_epoch,
    discover_files,
    plot_loss_curves,
    train_dl_with_history,
    write_markdown_report,
    load_all_results,
    plot_model_comparisons,
    compute_sensor_importance,
    collect_ml_epoch_data,
    classify_model_behavior,
    df_to_md,
)
from run_dl_benchmark import build_model, prepare_sequence_data  # noqa: E402
from run_regression_benchmark import load_dataset, select_features  # noqa: E402

MODELS = ["LSTM", "GRU", "CNN1D", "TCN", "Transformer"]
DATASETS = ["FD001"]  # representative; full benchmark scores in results/*.csv


def run_incremental():
    file_map = discover_files(str(ROOT))
    epoch_path = DOCS / "epoch_metrics_dl.csv"
    existing = pd.read_csv(epoch_path) if epoch_path.exists() else pd.DataFrame()
    done = set(zip(existing.Model, existing.Dataset)) if len(existing) else set()

    rows = existing.to_dict("records") if len(existing) else []
    histories = {}
    hist_path = DOCS / "training_histories.json"
    if hist_path.exists():
        with open(hist_path) as f:
            histories = json.load(f)

    for model_name in MODELS:
        for fd in DATASETS:
            if (model_name, fd) in done:
                print(f"Skip {model_name}/{fd}")
                continue
            print(f"Training {model_name}/{fd} ({MAX_EPOCHS} epochs)...", flush=True)
            train_df, test_df = load_dataset(
                file_map[fd]["train"], file_map[fd]["test"], file_map[fd]["rul"]
            )
            candidate = [c for c in train_df.columns if c.startswith("sensor_") or c.startswith("op_setting_")]
            feature_cols = select_features(train_df, candidate)
            X_train, y_train, train_meta, X_test, y_test, test_meta = prepare_sequence_data(
                train_df, test_df, feature_cols, 30
            )
            batch = 16 if fd in ("FD002", "FD004") else 32
            model = build_model(model_name, X_train.shape[2])
            hist, ckpts = train_dl_with_history(
                model, X_train, y_train, train_meta, X_test, y_test, test_meta, batch_size=batch
            )
            del model
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            key = f"{model_name}_{fd}"
            histories[key] = hist
            of = detect_overfit_epoch(hist)
            for c in ckpts:
                rows.append({
                    "Model": model_name, "Dataset": fd, "Family": "DL",
                    "Epoch": c["Epoch"], "RMSE": c["RMSE"], "MAE": c["MAE"], "R2": c["R2"],
                    "train_loss": c["train_loss"], "val_loss": c["val_loss"],
                    "overfit_epoch": of,
                })
            plot_loss_curves(hist, f"{model_name} — {fd} Train vs Val Loss",
                             FIGS / f"loss_{model_name}_{fd}.png")
            pd.DataFrame(rows).to_csv(epoch_path, index=False)
            with open(hist_path, "w") as f:
                json.dump(histories, f)
            print(f"  Done {model_name}/{fd}, overfit ~epoch {of}", flush=True)

    print("DL history collection complete.")


if __name__ == "__main__":
    run_incremental()
