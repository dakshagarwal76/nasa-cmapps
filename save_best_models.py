"""Train and save best model per dataset as .pkl (ML) or .pth + metadata .pkl (DL)."""
from __future__ import annotations

import json
import os
import pickle
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from paths import ROOT, MODELS as MODELS_DIR

MODELS_DIR.mkdir(exist_ok=True)

import sys
sys.path.insert(0, str(ROOT))

from run_regression_benchmark import (  # noqa: E402
    DATASET_IDS,
    discover_files,
    get_models,
    load_dataset,
    prepare_xy,
    select_features,
)
from run_dl_benchmark import (  # noqa: E402
    DEVICE,
    SEQ_LEN,
    build_model,
    prepare_sequence_data,
    train_model,
)

# Best model per dataset from verified benchmark (last-cycle RMSE)
BEST = {
    "FD001": {"family": "ML", "model": "ExtraTrees"},
    "FD002": {"family": "DL", "model": "Transformer"},
    "FD003": {"family": "ML", "model": "HistGradientBoosting"},
    "FD004": {"family": "ML", "model": "HistGradientBoosting"},
}


def save_ml_model(fd: str, model_name: str, file_map: dict) -> Path:
    train_df, test_df = load_dataset(
        file_map[fd]["train"], file_map[fd]["test"], file_map[fd]["rul"]
    )
    candidate = [c for c in train_df.columns if c.startswith("sensor_") or c.startswith("op_setting_")]
    feature_cols = select_features(train_df, candidate)
    X_train, y_train, X_test, y_test, expanded_cols = prepare_xy(train_df, test_df, feature_cols)

    from sklearn.preprocessing import StandardScaler
    train_feat = __import__("run_regression_benchmark", fromlist=["build_feature_matrix"]).build_feature_matrix(
        train_df, feature_cols
    )
    scaler = StandardScaler()
    scaler.fit(train_feat)

    model = get_models(len(X_train))[model_name]
    model.fit(X_train, y_train)

    bundle = {
        "model": model,
        "scaler": scaler,
        "feature_cols": feature_cols,
        "expanded_cols": expanded_cols,
        "dataset": fd,
        "model_name": model_name,
        "family": "ML",
        "rul_cap": 125,
    }
    out = MODELS_DIR / f"{fd}_{model_name}.pkl"
    joblib.dump(bundle, out)
    print(f"Saved {out}")
    return out


def save_dl_model(fd: str, model_name: str, file_map: dict) -> tuple[Path, Path]:
    train_df, test_df = load_dataset(
        file_map[fd]["train"], file_map[fd]["test"], file_map[fd]["rul"]
    )
    candidate = [c for c in train_df.columns if c.startswith("sensor_") or c.startswith("op_setting_")]
    feature_cols = select_features(train_df, candidate)
    X_train, y_train, train_meta, X_test, y_test, test_meta = prepare_sequence_data(
        train_df, test_df, feature_cols, SEQ_LEN
    )
    batch = 16 if fd in ("FD002", "FD004") else 32
    model = build_model(model_name, X_train.shape[2])
    model = train_model(model, X_train, y_train, train_meta, batch_size=batch)

    meta = {
        "dataset": fd,
        "model_name": model_name,
        "family": "DL",
        "feature_cols": feature_cols,
        "seq_len": SEQ_LEN,
        "n_features": X_train.shape[2],
    }
    pth = MODELS_DIR / f"{fd}_{model_name}.pth"
    pkl = MODELS_DIR / f"{fd}_{model_name}_meta.pkl"
    torch.save(model.state_dict(), pth)
    with open(pkl, "wb") as f:
        pickle.dump(meta, f)
    print(f"Saved {pth} and {pkl}")
    return pth, pkl


def main():
    file_map = discover_files(str(ROOT))
    manifest = []
    for fd, spec in BEST.items():
        if spec["family"] == "ML":
            p = save_ml_model(fd, spec["model"], file_map)
            manifest.append({"dataset": fd, "model": spec["model"], "family": "ML", "path": str(p)})
        else:
            pth, pkl = save_dl_model(fd, spec["model"], file_map)
            manifest.append({
                "dataset": fd, "model": spec["model"], "family": "DL",
                "weights": str(pth), "metadata": str(pkl),
            })
    (MODELS_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Manifest: {MODELS_DIR / 'manifest.json'}")


if __name__ == "__main__":
    main()
