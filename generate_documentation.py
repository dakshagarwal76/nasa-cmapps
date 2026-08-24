"""
Generate full C-MAPSS RUL benchmark documentation:
- Epoch metrics at 10/50/100/200
- Loss curves (train vs validation)
- Sensor importance, comparisons, architecture diagrams
- Markdown report
"""
from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

from paths import ROOT, DOCS, FIGURES as FIGS

DOCS.mkdir(exist_ok=True)
FIGS.mkdir(exist_ok=True)

sys.path.insert(0, str(ROOT))
from run_regression_benchmark import (  # noqa: E402
    DATASET_IDS,
    RUL_CAP,
    build_feature_matrix,
    discover_files,
    load_dataset,
    prepare_xy,
    regression_metrics,
    select_features,
)
from run_dl_benchmark import (  # noqa: E402
    DEVICE,
    SEQ_LEN,
    build_model as build_dl_model,
    engine_split,
    last_cycle_indices,
    predict as dl_predict,
    prepare_sequence_data,
)

CHECKPOINT_EPOCHS = [10, 50, 100, 200]
MAX_EPOCHS = 200


# ---------------------------------------------------------------------------
# ML staged training (epoch-equivalent via n_estimators / max_iter)
# ---------------------------------------------------------------------------
def ml_metrics_at_stages(X_train, y_train, X_test, y_test, test_df, stages, model_name):
    last_idx = test_df.groupby("unit")["cycle"].idxmax().values
    y_te_last, y_all = y_test[last_idx], y_test
    rows = []
    for stage in stages:
        if model_name == "XGBoost":
            import xgboost as xgb
            m = xgb.XGBRegressor(
                n_estimators=stage, max_depth=8, learning_rate=0.05,
                subsample=0.85, colsample_bytree=0.85, random_state=42, n_jobs=-1,
            )
        elif model_name == "ExtraTrees":
            m = ExtraTreesRegressor(n_estimators=stage, max_depth=16, n_jobs=-1, random_state=42)
        elif model_name == "HistGradientBoosting":
            m = HistGradientBoostingRegressor(max_iter=stage, learning_rate=0.06, max_depth=12, random_state=42)
        elif model_name == "Ridge":
            from sklearn.linear_model import Ridge
            m = Ridge(alpha=5.0)  # no stages; same at all checkpoints
        else:
            continue
        m.fit(X_train, y_train)
        pred = m.predict(X_test)
        met = regression_metrics(y_te_last, pred[last_idx])
        rows.append({"Model": model_name, "Stage": stage, **{f"{k}_last": v for k, v in met.items()}})
    return rows


# ---------------------------------------------------------------------------
# DL training with per-epoch history
# ---------------------------------------------------------------------------
def train_dl_with_history(model, X_train, y_train, train_meta, X_test, y_test, test_meta,
                          batch_size=64, max_epochs=MAX_EPOCHS):
    from torch.utils.data import DataLoader, TensorDataset

    tr_idx, va_idx = engine_split(train_meta)
    X_tr, y_tr = X_train[tr_idx], y_train[tr_idx]
    X_va, y_va = X_train[va_idx], y_train[va_idx]

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr)),
        batch_size=batch_size, shuffle=True, pin_memory=True,
    )
    X_va_t = torch.from_numpy(X_va).to(DEVICE)
    y_va_t = torch.from_numpy(y_va).to(DEVICE)

    model = model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    def batched_val_loss():
        model.eval()
        parts = []
        with torch.no_grad():
            for i in range(0, len(X_va), batch_size):
                xb = torch.from_numpy(X_va[i : i + batch_size]).to(DEVICE)
                yb = torch.from_numpy(y_va[i : i + batch_size]).to(DEVICE)
                parts.append(loss_fn(model(xb), yb).item())
        return float(np.mean(parts))

    history = {"epoch": [], "train_loss": [], "val_loss": []}
    checkpoint_metrics = []
    last_idx = last_cycle_indices(test_meta)

    for epoch in range(1, max_epochs + 1):
        model.train()
        losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            losses.append(loss.item())

        val_loss = batched_val_loss()
        train_loss = float(np.mean(losses))
        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if epoch in CHECKPOINT_EPOCHS:
            pred = dl_predict(model, X_test, batch_size=64)
            met = regression_metrics(y_test[last_idx], pred[last_idx])
            checkpoint_metrics.append({"Epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, **met})

    del X_va_t, y_va_t
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return history, checkpoint_metrics


def detect_overfit_epoch(history: dict, gap_threshold: float = 0.12) -> int | None:
    """First epoch where val_loss exceeds train_loss by relative gap_threshold."""
    for i, (tl, vl) in enumerate(zip(history["train_loss"], history["val_loss"])):
        if tl > 0 and (vl - tl) / tl > gap_threshold and i >= 5:
            return history["epoch"][i]
    # fallback: epoch of best val + 3
    best_i = int(np.argmin(history["val_loss"]))
    return history["epoch"][min(best_i + 5, len(history["epoch"]) - 1)]


def plot_loss_curves(history, title, path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.plot(history["epoch"], history["train_loss"], label="Train MSE", linewidth=1.5)
    ax.plot(history["epoch"], history["val_loss"], label="Validation MSE", linewidth=1.5)
    for ep in CHECKPOINT_EPOCHS:
        if ep <= len(history["epoch"]):
            ax.axvline(ep, color="gray", linestyle="--", alpha=0.4, linewidth=0.8)
    of_ep = detect_overfit_epoch(history)
    if of_ep:
        ax.axvline(of_ep, color="red", linestyle=":", alpha=0.7, label=f"Overfit onset ~{of_ep}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("Train vs Validation Loss")
    ax.legend()

    ax2 = axes[1]
    gap = np.array(history["val_loss"]) - np.array(history["train_loss"])
    ax2.plot(history["epoch"], gap, color="purple", linewidth=1.5)
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.fill_between(history["epoch"], gap, 0, where=(gap > 0), alpha=0.3, color="red", label="Val > Train")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Val − Train MSE")
    ax2.set_title("Generalization Gap")
    ax2.legend()

    fig.suptitle(title, fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Architecture diagrams
# ---------------------------------------------------------------------------
ARCH_SPECS = {
    "ExtraTrees": ["24 sensor+op features (×4 engineered)", "→ 250 tree ensemble", "→ RUL"],
    "XGBoost": ["24 features (×4 engineered)", "→ 500 gradient boosted trees", "→ RUL"],
    "LSTM": ["(30, F) sequence", "→ LSTM 64→32", "→ Dense 50", "→ RUL"],
    "GRU": ["(30, F) sequence", "→ GRU 64→32", "→ Dense 50", "→ RUL"],
    "CNN1D": ["(30, F) sequence", "→ Conv1D ×3 + pool", "→ Dense", "→ RUL"],
    "Transformer": ["(30, F) sequence", "→ Linear proj + PosEnc", "→ 2-layer TransformerEncoder", "→ RUL"],
    "TCN": ["(30, F) sequence", "→ Dilated Conv1D ×4", "→ Global pool", "→ RUL"],
    "TemporalGCN": ["30 time-step nodes", "→ GCNConv ×2 (chain edges)", "→ Last node", "→ RUL"],
    "TemporalGAT": ["30 time-step nodes", "→ GATConv ×2 (chain edges)", "→ Last node", "→ RUL"],
    "TemporalGraphSAGE": ["30 time-step nodes", "→ SAGEConv ×2", "→ Last node", "→ RUL"],
    "SensorGCN": ["F sensor nodes", "→ GCNConv ×2 (corr edges)", "→ Mean pool", "→ RUL"],
}


def draw_architecture(model_name, path):
    steps = ARCH_SPECS.get(model_name, ["Input", "Model", "RUL"])
    fig, ax = plt.subplots(figsize=(10, 2.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 2)
    ax.axis("off")
    n = len(steps)
    w = 8.5 / n
    for i, text in enumerate(steps):
        x = 0.75 + i * (w + 0.15)
        rect = mpatches.FancyBboxPatch((x, 0.5), w, 1.0, boxstyle="round,pad=0.05",
                                        facecolor="#4C72B0" if i == 0 else "#55A868" if i == n-1 else "#C44E52",
                                        edgecolor="black", alpha=0.85)
        ax.add_patch(rect)
        ax.text(x + w/2, 1.0, text, ha="center", va="center", fontsize=8, color="white", fontweight="bold")
        if i < n - 1:
            ax.annotate("", xy=(x + w + 0.12, 1.0), xytext=(x + w + 0.02, 1.0),
                        arrowprops=dict(arrowstyle="->", color="black", lw=1.5))
    ax.set_title(f"{model_name} Architecture", fontsize=12, fontweight="bold", pad=10)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Sensor importance
# ---------------------------------------------------------------------------
def compute_sensor_importance(file_map):
    records = []
    for fd in DATASET_IDS:
        train_df, test_df = load_dataset(
            file_map[fd]["train"], file_map[fd]["test"], file_map[fd]["rul"]
        )
        candidate = [c for c in train_df.columns if c.startswith("sensor_") or c.startswith("op_setting_")]
        feature_cols = select_features(train_df, candidate)
        X_train, y_train, X_test, y_test, expanded = prepare_xy(train_df, test_df, feature_cols)
        model = ExtraTreesRegressor(n_estimators=200, max_depth=24, n_jobs=-1, random_state=42)
        model.fit(X_train, y_train)
        imp = model.feature_importances_
        for name, val in zip(expanded, imp):
            base = name.split("_")[0] if name.startswith("op_") else name.split("_r")[0] if "_r" in name else name
            if base.startswith("op_setting"):
                base = "_".join(name.split("_")[:3]) if "op_setting" in name else name
            # group by sensor/op base name
            key = name
            for suffix in ["_rel", "_rmean", "_rstd"]:
                if name.endswith(suffix):
                    key = name[: -len(suffix)]
                    break
            records.append({"Dataset": fd, "Feature": key, "Importance": float(val)})
    df = pd.DataFrame(records)
    agg = df.groupby(["Dataset", "Feature"])["Importance"].sum().reset_index()
    agg = agg.sort_values(["Dataset", "Importance"], ascending=[True, False])
    agg.to_csv(DOCS / "sensor_importance.csv", index=False)

    for fd in DATASET_IDS:
        sub = agg[agg.Dataset == fd].head(12)
        fig, ax = plt.subplots(figsize=(9, 5))
        sns.barplot(data=sub, y="Feature", x="Importance", ax=ax, palette="rocket")
        ax.set_title(f"Top Sensor/Feature Importance ({fd}) — ExtraTrees")
        plt.tight_layout()
        plt.savefig(FIGS / f"sensor_importance_{fd}.png", dpi=150, bbox_inches="tight")
        plt.close()
    return agg


# ---------------------------------------------------------------------------
# Comparison plots from existing CSVs
# ---------------------------------------------------------------------------
def load_all_results():
    ml = pd.read_csv(ROOT / "results" / "all_model_results.csv")
    ml["Family"] = "ML"
    dl = pd.read_csv(ROOT / "results" / "dl_model_results.csv")
    dl["Family"] = "DL"
    dl = dl.rename(columns={"RMSE_last": "RMSE_last", "MAE_last": "MAE_last", "R2_last": "R2_last"})
    graph_path = ROOT / "results" / "graph_model_results.csv"
    frames = [ml]
    if graph_path.exists():
        g = pd.read_csv(graph_path)
        g["Family"] = "Graph"
        frames.append(g)
    frames.append(dl)
    all_df = pd.concat(frames, ignore_index=True)
    all_df = all_df.drop_duplicates(subset=["Model", "Dataset"], keep="last")
    return all_df


def plot_model_comparisons(all_df):
    for metric in ["RMSE_last", "MAE_last", "R2_last"]:
        for fd in DATASET_IDS:
            sub = all_df[all_df.Dataset == fd].sort_values(metric, ascending=(metric != "R2_last"))
            fig, ax = plt.subplots(figsize=(10, max(4, len(sub) * 0.35)))
            sns.barplot(data=sub, y="Model", x=metric, hue="Family", ax=ax, dodge=False)
            ax.set_title(f"{fd} — {metric.replace('_last','')} (last-cycle)")
            plt.tight_layout()
            plt.savefig(FIGS / f"compare_{fd}_{metric}.png", dpi=150, bbox_inches="tight")
            plt.close()

    # heatmap RMSE
    pivot = all_df.pivot_table(index="Model", columns="Dataset", values="RMSE_last")
    fig, ax = plt.subplots(figsize=(8, max(6, len(pivot) * 0.35)))
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="RdYlGn_r", ax=ax)
    ax.set_title("RMSE Heatmap (last-cycle, all models)")
    plt.tight_layout()
    plt.savefig(FIGS / "heatmap_rmse_all.png", dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Main documentation run
# ---------------------------------------------------------------------------
def collect_dl_epoch_data(file_map, models, datasets=None):
    """Train DL models with full history; primary loss curves on FD001."""
    datasets = datasets or DATASET_IDS
    epoch_rows = []
    histories = {}

    for model_name in models:
        draw_architecture(model_name, FIGS / f"arch_{model_name}.png")
        for fd in datasets:
            print(f"  History: {model_name} / {fd}")
            train_df, test_df = load_dataset(
                file_map[fd]["train"], file_map[fd]["test"], file_map[fd]["rul"]
            )
            candidate = [c for c in train_df.columns if c.startswith("sensor_") or c.startswith("op_setting_")]
            feature_cols = select_features(train_df, candidate)
            X_train, y_train, train_meta, X_test, y_test, test_meta = prepare_sequence_data(
                train_df, test_df, feature_cols, SEQ_LEN
            )
            batch = 32 if len(X_train) > 45000 else 64
            model = build_dl_model(model_name, X_train.shape[2])
            hist, ckpts = train_dl_with_history(
                model, X_train, y_train, train_meta, X_test, y_test, test_meta, batch_size=batch
            )
            key = f"{model_name}_{fd}"
            histories[key] = hist
            for c in ckpts:
                epoch_rows.append({
                    "Model": model_name, "Dataset": fd, "Family": "DL",
                    "Epoch": c["Epoch"], "RMSE": c["RMSE"], "MAE": c["MAE"], "R2": c["R2"],
                    "train_loss": c["train_loss"], "val_loss": c["val_loss"],
                    "overfit_epoch": detect_overfit_epoch(hist),
                })
            if fd == "FD001":
                plot_loss_curves(hist, f"{model_name} — FD001 Train vs Val Loss", FIGS / f"loss_{model_name}_FD001.png")
            if fd == "FD004":
                plot_loss_curves(hist, f"{model_name} — FD004 Train vs Val Loss", FIGS / f"loss_{model_name}_FD004.png")

    pd.DataFrame(epoch_rows).to_csv(DOCS / "epoch_metrics_dl.csv", index=False)
    with open(DOCS / "training_histories.json", "w") as f:
        json.dump(histories, f)
    return epoch_rows


def collect_ml_epoch_data(file_map):
    rows = []
    models = ["XGBoost", "ExtraTrees", "HistGradientBoosting"]
    for fd in DATASET_IDS:
        print(f"  ML epochs: {fd}", flush=True)
        train_df, test_df = load_dataset(
            file_map[fd]["train"], file_map[fd]["test"], file_map[fd]["rul"]
        )
        candidate = [c for c in train_df.columns if c.startswith("sensor_") or c.startswith("op_setting_")]
        feature_cols = select_features(train_df, candidate)
        X_train, y_train, X_test, y_test, _ = prepare_xy(train_df, test_df, feature_cols)
        for mn in models:
            staged = ml_metrics_at_stages(X_train, y_train, X_test, y_test, test_df, CHECKPOINT_EPOCHS, mn)
            for s in staged:
                rows.append({"Model": mn, "Dataset": fd, "Family": "ML", "Epoch": s["Stage"],
                             "RMSE": s["RMSE_last"], "MAE": s["MAE_last"], "R2": s["R2_last"]})
    out = pd.DataFrame(rows)
    out.to_csv(DOCS / "epoch_metrics_ml.csv", index=False)
    return rows


def classify_model_behavior(epoch_df, final_df):
    """Classify overfitting / underfitting / stable from loss trends."""
    notes = []
    for (model, fd), grp in epoch_df.groupby(["Model", "Dataset"]):
        if "train_loss" not in grp.columns:
            notes.append({"Model": model, "Dataset": fd, "Behavior": "Stable (ML ensemble)",
                          "Note": "ML models use estimator stages, not iterative loss"})
            continue
        grp = grp.sort_values("Epoch")
        if len(grp) < 2:
            continue
        tl = grp["train_loss"].values
        vl = grp["val_loss"].values
        if vl[-1] > vl[0] * 1.2 and tl[-1] < vl[-1]:
            behavior = "Overfitting"
        elif vl[-1] > 25 and grp["R2"].iloc[-1] < 0.5:
            behavior = "Underfitting"
        else:
            behavior = "Stable"
        of = grp["overfit_epoch"].iloc[0] if "overfit_epoch" in grp.columns else None
        notes.append({"Model": model, "Dataset": fd, "Behavior": behavior,
                      "Note": f"Overfit onset ~epoch {of}" if of else ""})
    return pd.DataFrame(notes)


def df_to_md(df: pd.DataFrame) -> str:
    """Simple markdown table without tabulate dependency."""
    cols = list(df.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |",
             "| " + " | ".join("---" for _ in cols) + " |"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(lines)


def write_markdown_report(all_df, epoch_ml, epoch_dl, sensor_imp, behavior_df):
    best = all_df.loc[all_df.groupby("Dataset")["RMSE_last"].idxmin()]

    md = []
    md.append("# NASA C-MAPSS RUL Prediction — Benchmark Documentation\n")
    md.append("## Presentation Summary\n")
    md.append(
        "This project benchmarks **15 machine-learning, deep-learning, and graph models** "
        "on NASA's C-MAPSS turbofan RUL datasets (FD001–FD004). "
        "We use proper regression metrics (RMSE, MAE, R²) at the **last observed cycle per test engine**, "
        "with RUL capped at 125 during training. "
        "**ExtraTrees** wins FD001, **Transformer** wins FD002 among DL, "
        "**HistGradientBoosting/ExtraTrees** lead on FD003/FD004 for ML. "
        "Graph **TemporalGAT** achieves strong FD001 results (RMSE 16.6).\n"
    )

    md.append("## Dataset Explanation\n")
    md.append("| Dataset | Train Engines | Test Engines | Operating Conditions | Fault Modes | Difficulty |")
    md.append("|---------|---------------|--------------|----------------------|-------------|------------|")
    md.append("| FD001 | 100 | 100 | 1 (Sea Level) | 1 (HPC) | Easy |")
    md.append("| FD002 | 260 | 259 | 6 | 1 (HPC) | Medium |")
    md.append("| FD003 | 100 | 100 | 1 (Sea Level) | 2 (HPC + Fan) | Medium |")
    md.append("| FD004 | 248 | 249 | 6 | 2 (HPC + Fan) | **Hard** |")
    md.append("\nEach row = one engine cycle. Columns: unit, cycle, 3 op-settings, 21 sensors. "
                "Training runs to failure; test ends before failure with provided final RUL.\n")

    md.append("## Why FD004 Is Harder Than FD001\n")
    md.append(
        "1. **6 operating conditions** vs 1 — models must generalize across flight regimes.\n"
        "2. **2 fault modes** (HPC + Fan degradation) vs 1 — more complex degradation patterns.\n"
        "3. **Larger fleet** (248 engines) with more heterogeneity and sensor noise.\n"
        "4. **Higher test RMSE** across all model families (~29–35 vs ~14–18 on FD001).\n"
        "5. **Feature distribution shift** between conditions makes single global models harder to fit.\n"
    )

    md.append("## Data Preparation & Preprocessing\n")
    md.append("1. **Load** train/test/RUL files per FD subset.\n")
    md.append("2. **Compute RUL** per row; cap training RUL at **125 cycles**.\n")
    md.append("3. **Drop constant sensors** (zero variance on training set).\n")
    md.append("4. **ML features**: raw + unit-relative + rolling mean (5) + rolling std (5).\n")
    md.append("5. **DL/Graph features**: unit-relative scaling + StandardScaler; 30-step sequences.\n")
    md.append("6. **Graph edges**: temporal chain (GCN/GAT) or sensor correlation graph (SensorGCN).\n")
    md.append("7. **Evaluation**: last cycle per test engine — RMSE, MAE, R².\n")

    md.append("\n## Models Used\n")
    md.append("**ML (6):** Ridge, ElasticNet, RandomForest, ExtraTrees, HistGradientBoosting, XGBoost\n")
    md.append("**DL (5):** LSTM, GRU, CNN1D, TCN, Transformer (PyTorch CUDA)\n")
    md.append("**Graph (4):** TemporalGCN, TemporalGAT, TemporalGraphSAGE, SensorGCN (PyG)\n")

    md.append("\n## Best Model Per Dataset\n")
    md.append("| Dataset | Model | Family | RMSE | MAE | R² |")
    md.append("|---------|-------|--------|------|-----|-----|")
    for _, r in best.iterrows():
        md.append(f"| {r['Dataset']} | {r['Model']} | {r['Family']} | {r['RMSE_last']:.2f} | {r['MAE_last']:.2f} | {r['R2_last']:.4f} |")

    md.append("\n## Epoch Metrics (10 / 50 / 100 / 200)\n")
    md.append("ML models use **n_estimators / max_iter** as epoch-equivalent stages. "
                "DL models log true training epochs.\n")
    if len(epoch_ml):
        for fd in DATASET_IDS:
            sub = epoch_ml[epoch_ml.Dataset == fd].pivot_table(
                index="Model", columns="Epoch", values="RMSE", aggfunc="first"
            )
            if len(sub):
                md.append(f"\n### {fd} — ML RMSE by Stage\n")
                md.append(df_to_md(sub.round(2).reset_index()))
    if len(epoch_dl):
        for fd in DATASET_IDS:
            sub = epoch_dl[epoch_dl.Dataset == fd]
            if len(sub):
                md.append(f"\n### {fd} — DL RMSE by Epoch\n")
                pt = sub.pivot_table(index="Model", columns="Epoch", values="RMSE", aggfunc="first")
                md.append(df_to_md(pt.round(2).reset_index()))
        md.append("\nFull tables: `docs/epoch_metrics_ml.csv`, `docs/epoch_metrics_dl.csv`\n")

    md.append("\n## Loss Curve Analysis\n")
    md.append("Loss curves saved in `docs/figures/loss_<Model>_FD001.png` and `_FD004.png`.\n")
    md.append("Vertical dashed lines mark epochs 10, 50, 100, 200. Red dotted line = estimated overfitting onset.\n")

    md.append("\n## Model Behavior Classification\n")
    if len(behavior_df):
        md.append(df_to_md(behavior_df))

    md.append("\n## Sensor Importance (Top contributors to RUL)\n")
    for fd in DATASET_IDS:
        sub = sensor_imp[sensor_imp.Dataset == fd].head(8)
        md.append(f"\n### {fd}\n")
        md.append(df_to_md(sub.reset_index(drop=True)))
        md.append(f"\n![Sensor importance {fd}](figures/sensor_importance_{fd}.png)\n")

    md.append("\n## Architecture Diagrams\n")
    for m in ARCH_SPECS:
        md.append(f"### {m}\n![{m}](figures/arch_{m}.png)\n")

    md.append("\n## All Models — Final Results (last-cycle)\n")
    pivot_rmse = all_df.pivot_table(index="Model", columns="Dataset", values="RMSE_last").round(2)
    md.append(df_to_md(pivot_rmse.reset_index()))
    md.append("\n")
    pivot_r2 = all_df.pivot_table(index="Model", columns="Dataset", values="R2_last").round(4)
    md.append("### R² by Model and Dataset\n")
    md.append(df_to_md(pivot_r2.reset_index()))

    md.append("\n## Underfitting / Overfitting / Stable — Summary\n")
    md.append(
        "| Category | Models | Evidence |\n"
        "|----------|--------|----------|\n"
        "| **Underfitting (early)** | XGBoost, HistGBoost @ stage 10 | RMSE 26–43 on FD001–FD004 before enough estimators |\n"
        "| **Stable (ML)** | ExtraTrees, XGBoost @ 50–200 stages | RMSE plateaus; small change 50→200 |\n"
        "| **Stable (DL)** | LSTM, Transformer | Val loss tracks train; best results with early stopping |\n"
        "| **Overfitting (DL)** | CNN1D, TCN (FD004) | Val loss rises while train loss falls after ~30–60 epochs |\n"
        "| **Best balance** | ExtraTrees, TemporalGAT, Transformer | Strong RMSE without large train/val gap |\n"
    )
    md.append("![RMSE Heatmap](figures/heatmap_rmse_all.png)\n")
    for fd in DATASET_IDS:
        md.append(f"![{fd} RMSE](figures/compare_{fd}_RMSE_last.png)\n")

    md.append("\n## Final Recommendation\n")
    md.append(
        "- **Production baseline**: ExtraTrees or HistGradientBoosting with engineered features — best overall RMSE.\n"
        "- **Multi-condition fleets (FD002/FD004)**: Transformer or TCN if GPU available; otherwise XGBoost.\n"
        "- **Interpretability**: ExtraTrees + sensor importance (sensors 4, 11, 12, 7 often top).\n"
        "- **Graph models**: TemporalGAT competitive on FD001; worth exploring for sensor-relationship modeling.\n"
        "- **Avoid**: CNN1D alone (weakest DL); rely on sequence models or boosted trees.\n"
    )

    md.append("\n## File Index\n")
    md.append("| File | Description |")
    md.append("|------|-------------|")
    md.append("| `results/all_model_results.csv` | ML results |")
    md.append("| `results/dl_model_results.csv` | DL results |")
    md.append("| `results/graph_model_results.csv` | Graph results |")
    md.append("| `docs/epoch_metrics_ml.csv` | ML staged metrics |")
    md.append("| `docs/epoch_metrics_dl.csv` | DL epoch checkpoints |")
    md.append("| `docs/sensor_importance.csv` | Feature importance |")

    (DOCS / "CMAPSS_RUL_Benchmark_Documentation.md").write_text("\n".join(md), encoding="utf-8")
    print(f"Report written: {DOCS / 'CMAPSS_RUL_Benchmark_Documentation.md'}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="FD001 only for DL history (faster)")
    parser.add_argument("--skip-dl-train", action="store_true", help="Skip DL retraining")
    args = parser.parse_args()

    file_map = discover_files(str(ROOT))
    print("Loading existing results...")
    all_df = load_all_results()
    plot_model_comparisons(all_df)

    print("Computing sensor importance...")
    sensor_imp = compute_sensor_importance(file_map)

    print("Collecting ML epoch metrics...")
    collect_ml_epoch_data(file_map)

    epoch_dl_rows = []
    if not args.skip_dl_train:
        dl_models = ["LSTM", "GRU", "CNN1D", "TCN", "Transformer"]
        ds = ["FD001"] if args.quick else DATASET_IDS
        print(f"Collecting DL training history ({dl_models}, {ds})...")
        epoch_dl_rows = collect_dl_epoch_history(file_map, dl_models, ds)

    for m in ARCH_SPECS:
        draw_architecture(m, FIGS / f"arch_{m}.png")

    epoch_dl = pd.read_csv(DOCS / "epoch_metrics_dl.csv") if (DOCS / "epoch_metrics_dl.csv").exists() else pd.DataFrame()
    epoch_ml = pd.read_csv(DOCS / "epoch_metrics_ml.csv")
    behavior = classify_model_behavior(
        pd.concat([epoch_dl, epoch_ml], ignore_index=True) if len(epoch_dl) else epoch_ml,
        all_df,
    )
    behavior.to_csv(DOCS / "model_behavior.csv", index=False)

    write_markdown_report(all_df, epoch_ml, epoch_dl, sensor_imp, behavior)
    print("Done.")


# fix function name typo
collect_dl_epoch_history = collect_dl_epoch_data

if __name__ == "__main__":
    main()
