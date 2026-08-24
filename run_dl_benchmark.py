"""
C-MAPSS deep-learning regression benchmark (PyTorch + CUDA).
Runs models one-by-one on FD001-FD004 with last-cycle RMSE / MAE / R2.
"""
from __future__ import annotations

import argparse
import glob
import math
import os
import time
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")

RUL_CAP = 125
SEQ_LEN = 30
DATASET_IDS = ["FD001", "FD002", "FD003", "FD004"]
from paths import DATA_DIR, RESULTS

OUTPUT_DIR = str(RESULTS)
RESULTS_CSV = os.path.join(OUTPUT_DIR, "dl_model_results.csv")
os.makedirs(OUTPUT_DIR, exist_ok=True)
DATA_DIR = str(DATA_DIR)

COL_NAMES = (
    ["unit", "cycle", "op_setting_1", "op_setting_2", "op_setting_3"]
    + [f"sensor_{i+1}" for i in range(21)]
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DL_MODELS = ["LSTM", "GRU", "CNN1D", "TCN", "Transformer"]


# ---------------------------------------------------------------------------
# Data utilities (shared with sklearn benchmark)
# ---------------------------------------------------------------------------
def discover_files(data_dir: str) -> dict:
    mapping: dict = {}
    for path in glob.glob(os.path.join(data_dir, "*")):
        name = os.path.basename(path).upper()
        for fd in DATASET_IDS:
            if fd in name:
                mapping.setdefault(fd, {})
                if "TRAIN" in name:
                    mapping[fd]["train"] = path
                if "TEST" in name:
                    mapping[fd]["test"] = path
                if "RUL" in name:
                    mapping[fd]["rul"] = path
    return mapping


def load_dataset(train_path: str, test_path: str, rul_path: str):
    train = pd.read_csv(train_path, sep=r"\s+", header=None, names=COL_NAMES)
    train["RUL"] = train.groupby("unit")["cycle"].transform("max") - train["cycle"]
    test = pd.read_csv(test_path, sep=r"\s+", header=None, names=COL_NAMES)
    rul = pd.read_csv(rul_path, sep=r"\s+", header=None, names=["RUL_final"])
    max_cycle = test.groupby("unit")["cycle"].max().reset_index().sort_values("unit")
    max_cycle["RUL_final"] = rul["RUL_final"].values
    test = test.merge(max_cycle[["unit", "RUL_final"]], on="unit", how="left")
    test["RUL"] = (
        test.groupby("unit")["cycle"].transform("max") - test["cycle"] + test["RUL_final"]
    )
    return train, test.drop(columns=["RUL_final"])


def select_features(train_df: pd.DataFrame, cols: list[str], min_std: float = 1e-6) -> list[str]:
    stds = train_df[cols].std()
    return stds[stds > min_std].index.tolist()


def unit_relative(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    grouped = out.groupby("unit", sort=False)
    for col in cols:
        out[col] = grouped[col].transform(lambda s: s - s.iloc[0])
    return out


def create_sequences(df: pd.DataFrame, feature_cols: list[str], seq_len: int, cap_rul: bool):
    """Return X (N,T,F), y (N,), meta list of (unit, cycle)."""
    xs, ys, meta = [], [], []
    for uid, unit_df in df.groupby("unit", sort=True):
        unit_df = unit_df.sort_values("cycle")
        feats = unit_df[feature_cols].values.astype(np.float32)
        y = unit_df["RUL"].values.astype(np.float32)
        if cap_rul:
            y = np.minimum(y, RUL_CAP)
        if len(unit_df) <= seq_len:
            continue
        for i in range(seq_len, len(unit_df)):
            xs.append(feats[i - seq_len : i])
            ys.append(y[i])
            meta.append((int(uid), int(unit_df["cycle"].iloc[i])))
    return np.asarray(xs), np.asarray(ys), meta


def last_cycle_indices(meta: list[tuple[int, int]]) -> np.ndarray:
    best: dict[int, int] = {}
    for idx, (uid, cycle) in enumerate(meta):
        if uid not in best or cycle > meta[best[uid]][1]:
            best[uid] = idx
    return np.array(sorted(best.values()), dtype=int)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_pred = np.clip(y_pred, 0, None)
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
    }


def prepare_sequence_data(train_df, test_df, feature_cols, seq_len):
    train_rel = unit_relative(train_df, feature_cols)
    test_rel = unit_relative(test_df, feature_cols)
    scaler = StandardScaler()
    scaler.fit(train_rel[feature_cols])

    def scale_df(df):
        out = df.copy()
        out[feature_cols] = scaler.transform(df[feature_cols])
        return out

    train_scaled = scale_df(train_rel)
    test_scaled = scale_df(test_rel)

    X_train, y_train, train_meta = create_sequences(
        train_scaled, feature_cols, seq_len, cap_rul=True
    )
    X_test, y_test, test_meta = create_sequences(
        test_scaled, feature_cols, seq_len, cap_rul=False
    )
    return X_train, y_train, train_meta, X_test, y_test, test_meta


def engine_split(meta: list[tuple[int, int]], val_frac: float = 0.2, seed: int = 42):
    units = sorted({uid for uid, _ in meta})
    rng = np.random.default_rng(seed)
    n_val = max(1, int(len(units) * val_frac))
    val_units = set(rng.choice(units, size=n_val, replace=False))
    train_idx = np.array([i for i, (u, _) in enumerate(meta) if u not in val_units])
    val_idx = np.array([i for i, (u, _) in enumerate(meta) if u in val_units])
    return train_idx, val_idx


# ---------------------------------------------------------------------------
# PyTorch models
# ---------------------------------------------------------------------------
class LSTMRegressor(nn.Module):
    def __init__(self, n_features: int, hidden: int = 64, layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.rnn = nn.LSTM(
            n_features, hidden, num_layers=layers, batch_first=True, dropout=dropout if layers > 1 else 0.0
        )
        self.head = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, 1))

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.head(out[:, -1, :]).squeeze(-1)


class GRURegressor(nn.Module):
    def __init__(self, n_features: int, hidden: int = 64, layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.rnn = nn.GRU(
            n_features, hidden, num_layers=layers, batch_first=True, dropout=dropout if layers > 1 else 0.0
        )
        self.head = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, 1))

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.head(out[:, -1, :]).squeeze(-1)


class CNN1DRegressor(nn.Module):
    def __init__(self, n_features: int, channels: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_features, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(channels, channels * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(channels * 2, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        x = x.transpose(1, 2)
        return self.head(self.net(x)).squeeze(-1)


class TCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, dilation, dropout):
        super().__init__()
        pad = (kernel - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel, padding=pad, dilation=dilation)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.down = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None

    def forward(self, x):
        out = self.drop(self.relu(self.conv(x)))
        out = out[..., : x.size(-1)]
        res = x if self.down is None else self.down(x)
        return out + res


class TCNRegressor(nn.Module):
    def __init__(self, n_features: int, channels: int = 48, dropout: float = 0.2):
        super().__init__()
        blocks = []
        in_ch = n_features
        for i, dilation in enumerate([1, 2, 4, 8]):
            out_ch = channels if i == 0 else channels
            blocks.append(TCNBlock(in_ch, channels, kernel=3, dilation=dilation, dropout=dropout))
            in_ch = channels
        self.tcn = nn.Sequential(*blocks)
        self.head = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(channels, 1))

    def forward(self, x):
        x = x.transpose(1, 2)
        return self.head(self.tcn(x)).squeeze(-1)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 200):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class TransformerRegressor(nn.Module):
    def __init__(self, n_features: int, d_model: int = 64, nhead: int = 4, layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos = PositionalEncoding(d_model, max_len=SEQ_LEN + 5)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=128, dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.head = nn.Sequential(nn.Linear(d_model, 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, 1))

    def forward(self, x):
        x = self.pos(self.input_proj(x))
        x = self.encoder(x)
        return self.head(x[:, -1, :]).squeeze(-1)


def build_model(name: str, n_features: int) -> nn.Module:
    builders = {
        "LSTM": LSTMRegressor,
        "GRU": GRURegressor,
        "CNN1D": CNN1DRegressor,
        "TCN": TCNRegressor,
        "Transformer": TransformerRegressor,
    }
    return builders[name](n_features)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_model(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    train_meta: list,
    batch_size: int = 64,
    epochs: int = 60,
    patience: int = 10,
    lr: float = 1e-3,
) -> nn.Module:
    tr_idx, va_idx = engine_split(train_meta)
    X_tr, y_tr = X_train[tr_idx], y_train[tr_idx]
    X_va, y_va = X_train[va_idx], y_train[va_idx]

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr)),
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
    )
    X_va_t = torch.from_numpy(X_va).to(DEVICE)
    y_va_t = torch.from_numpy(y_va).to(DEVICE)

    model = model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    best_state = None
    best_val = float("inf")
    stale = 0

    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(X_va_t)
            val_loss = loss_fn(val_pred, y_va_t).item()

        if val_loss < best_val - 1e-4:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(DEVICE)
    return model


@torch.no_grad()
def predict(model: nn.Module, X: np.ndarray, batch_size: int = 256) -> np.ndarray:
    model.eval()
    preds = []
    for i in range(0, len(X), batch_size):
        xb = torch.from_numpy(X[i : i + batch_size]).to(DEVICE)
        preds.append(model(xb).cpu().numpy())
    return np.concatenate(preds)


def run_one_model(model_name: str, file_map: dict, existing_df: pd.DataFrame | None) -> pd.DataFrame:
    rows = []
    print(f"\n{'#' * 70}\n# MODEL: {model_name}  |  device={DEVICE}\n{'#' * 70}")

    for fd in DATASET_IDS:
        t0 = time.time()
        train_df, test_df = load_dataset(
            file_map[fd]["train"], file_map[fd]["test"], file_map[fd]["rul"]
        )
        candidate = [c for c in train_df.columns if c.startswith("sensor_") or c.startswith("op_setting_")]
        feature_cols = select_features(train_df, candidate)
        X_train, y_train, train_meta, X_test, y_test, test_meta = prepare_sequence_data(
            train_df, test_df, feature_cols, SEQ_LEN
        )
        n_features = X_train.shape[2]
        batch_size = 32 if len(X_train) > 45000 else 64

        print(
            f"\n[{model_name}] {fd}: seq_train={len(X_train):,} seq_test={len(X_test):,} "
            f"features={n_features} batch={batch_size}"
        )

        torch.cuda.empty_cache()
        model = build_model(model_name, n_features)
        model = train_model(model, X_train, y_train, train_meta, batch_size=batch_size)
        y_pred = predict(model, X_test)

        metrics_all = regression_metrics(y_test, y_pred)
        last_idx = last_cycle_indices(test_meta)
        metrics_last = regression_metrics(y_test[last_idx], y_pred[last_idx])

        elapsed = time.time() - t0
        print(
            f"  last-cycle -> RMSE={metrics_last['RMSE']:.2f}  MAE={metrics_last['MAE']:.2f}  "
            f"R2={metrics_last['R2']:.4f}  ({elapsed/60:.1f} min)"
        )

        rows.append(
            {
                "Model": model_name,
                "Dataset": fd,
                "RMSE_last": metrics_last["RMSE"],
                "MAE_last": metrics_last["MAE"],
                "R2_last": metrics_last["R2"],
                "RMSE_all": metrics_all["RMSE"],
                "MAE_all": metrics_all["MAE"],
                "R2_all": metrics_all["R2"],
                "seq_len": SEQ_LEN,
                "n_features": n_features,
                "device": str(DEVICE),
                "train_minutes": round(elapsed / 60, 2),
            }
        )

        del model
        torch.cuda.empty_cache()

    new_df = pd.DataFrame(rows)
    if existing_df is not None and len(existing_df):
        combined = pd.concat([existing_df, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["Model", "Dataset"], keep="last")
    else:
        combined = new_df
    combined.to_csv(RESULTS_CSV, index=False)
    return combined


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=DL_MODELS, default=None, help="Run a single model")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    file_map = discover_files(DATA_DIR)

    print(f"Data: {DATA_DIR}")
    print(f"Device: {DEVICE}", end="")
    if torch.cuda.is_available():
        print(f" ({torch.cuda.get_device_name(0)})")
    else:
        print()

    existing = pd.read_csv(RESULTS_CSV) if os.path.isfile(RESULTS_CSV) else None
    models_to_run = [args.model] if args.model else DL_MODELS

    for model_name in models_to_run:
        existing = run_one_model(model_name, file_map, existing)

    print(f"\nSaved: {RESULTS_CSV}")
    if existing is not None:
        print("\n=== ALL DL RESULTS (last-cycle) ===")
        pivot = existing.pivot(index="Dataset", columns="Model", values="RMSE_last")
        print(pivot.round(2).to_string())
        print("\nBest per dataset:")
        for fd in DATASET_IDS:
            sub = existing[existing["Dataset"] == fd]
            best = sub.loc[sub["RMSE_last"].idxmin()]
            print(f"  {fd}: {best['Model']}  RMSE={best['RMSE_last']:.2f}  R2={best['R2_last']:.4f}")


if __name__ == "__main__":
    main()
