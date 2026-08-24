"""
C-MAPSS graph neural network benchmark (PyTorch Geometric + CUDA).
Models: TemporalGCN, TemporalGAT
"""
from __future__ import annotations

import argparse
import glob
import os
import time
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.nn import GATConv, GCNConv, SAGEConv

warnings.filterwarnings("ignore")

RUL_CAP = 125
SEQ_LEN = 30
DATASET_IDS = ["FD001", "FD002", "FD003", "FD004"]
from paths import DATA_DIR, RESULTS

OUTPUT_DIR = str(RESULTS)
RESULTS_CSV = os.path.join(OUTPUT_DIR, "graph_model_results.csv")
os.makedirs(OUTPUT_DIR, exist_ok=True)
DATA_DIR = str(DATA_DIR)

COL_NAMES = (
    ["unit", "cycle", "op_setting_1", "op_setting_2", "op_setting_3"]
    + [f"sensor_{i+1}" for i in range(21)]
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GRAPH_MODELS = ["TemporalGCN", "TemporalGAT"]


# ---------------------------------------------------------------------------
# Data (same as DL benchmark)
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
    X_train, y_train, train_meta = create_sequences(train_scaled, feature_cols, seq_len, True)
    X_test, y_test, test_meta = create_sequences(test_scaled, feature_cols, seq_len, False)
    return X_train, y_train, train_meta, X_test, y_test, test_meta


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


def engine_split(meta: list[tuple[int, int]], val_frac: float = 0.2, seed: int = 42):
    units = sorted({uid for uid, _ in meta})
    rng = np.random.default_rng(seed)
    n_val = max(1, int(len(units) * val_frac))
    val_units = set(rng.choice(units, size=n_val, replace=False))
    train_idx = np.array([i for i, (u, _) in enumerate(meta) if u not in val_units])
    val_idx = np.array([i for i, (u, _) in enumerate(meta) if u in val_units])
    return train_idx, val_idx


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------
def chain_edge_index(seq_len: int) -> torch.Tensor:
    rows, cols = [], []
    for i in range(seq_len - 1):
        rows.extend([i, i + 1])
        cols.extend([i + 1, i])
    return torch.tensor([rows, cols], dtype=torch.long)


def build_sensor_edge_index(X_train: np.ndarray, top_k: int = 4) -> torch.Tensor:
    """Correlation graph over sensors/ops (node = feature dimension)."""
    flat = X_train.reshape(-1, X_train.shape[-1])
    corr = np.corrcoef(flat.T)
    np.fill_diagonal(corr, 0)
    rows, cols = [], []
    n = corr.shape[0]
    for i in range(n):
        neighbors = np.argsort(-np.abs(corr[i]))[:top_k]
        for j in neighbors:
            if i != j:
                rows.extend([i, j])
                cols.extend([j, i])
    return torch.tensor([rows, cols], dtype=torch.long)


class SequenceGraphDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, mode: str, edge_index: torch.Tensor):
        self.X = X
        self.y = y
        self.mode = mode
        self.edge_index = edge_index

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx: int) -> Data:
        window = torch.from_numpy(self.X[idx])
        if self.mode == "temporal":
            x = window
        else:
            x = window.T
        return Data(x=x, y=torch.tensor(self.y[idx], dtype=torch.float32), edge_index=self.edge_index)


# ---------------------------------------------------------------------------
# GNN models
# ---------------------------------------------------------------------------
class TemporalGCN(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 48):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.head = nn.Linear(hidden, 1)

    def forward(self, batch: Batch) -> torch.Tensor:
        x, edge_index = batch.x, batch.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        last_idx = batch.ptr[1:] - 1
        return self.head(x[last_idx]).squeeze(-1)


class TemporalGAT(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 32, heads: int = 2):
        super().__init__()
        self.conv1 = GATConv(in_dim, hidden, heads=heads, concat=False, dropout=0.2)
        self.conv2 = GATConv(hidden, hidden, heads=1, concat=False, dropout=0.2)
        self.head = nn.Linear(hidden, 1)

    def forward(self, batch: Batch) -> torch.Tensor:
        x, edge_index = batch.x, batch.edge_index
        x = F.elu(self.conv1(x, edge_index))
        x = F.elu(self.conv2(x, edge_index))
        last_idx = batch.ptr[1:] - 1
        return self.head(x[last_idx]).squeeze(-1)


class TemporalGraphSAGE(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 48):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden)
        self.conv2 = SAGEConv(hidden, hidden)
        self.head = nn.Linear(hidden, 1)

    def forward(self, batch: Batch) -> torch.Tensor:
        x, edge_index = batch.x, batch.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        last_idx = batch.ptr[1:] - 1
        return self.head(x[last_idx]).squeeze(-1)


class SensorGCN(nn.Module):
    def __init__(self, seq_len: int, hidden: int = 48):
        super().__init__()
        self.conv1 = GCNConv(seq_len, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.head = nn.Linear(hidden, 1)

    def forward(self, batch: Batch) -> torch.Tensor:
        x, edge_index = batch.x, batch.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        # mean pool nodes per graph
        out = []
        for g in range(batch.num_graphs):
            mask = batch.batch == g
            out.append(x[mask].mean(dim=0))
        return self.head(torch.stack(out)).squeeze(-1)


def build_model(name: str, in_dim: int, seq_len: int) -> nn.Module:
    if name == "TemporalGCN":
        return TemporalGCN(in_dim)
    if name == "TemporalGAT":
        return TemporalGAT(in_dim)
    if name == "TemporalGraphSAGE":
        return TemporalGraphSAGE(in_dim)
    if name == "SensorGCN":
        return SensorGCN(seq_len)
    raise ValueError(name)


def train_gnn(
    model: nn.Module,
    train_ds: Dataset,
    val_ds: Dataset,
    batch_size: int = 64,
    epochs: int = 50,
    patience: int = 8,
    lr: float = 1e-3,
) -> nn.Module:
    train_loader = PyGDataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = PyGDataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    best_state = None
    best_val = float("inf")
    stale = 0

    for _ in range(epochs):
        model.train()
        for batch in train_loader:
            batch = batch.to(DEVICE)
            opt.zero_grad()
            pred = model(batch)
            loss = loss_fn(pred, batch.y)
            loss.backward()
            opt.step()

        model.eval()
        vals = []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(DEVICE)
                pred = model(batch)
                vals.append(loss_fn(pred, batch.y).item())
        val_loss = float(np.mean(vals))

        if val_loss < best_val - 1e-4:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    if best_state:
        model.load_state_dict(best_state)
    return model.to(DEVICE)


@torch.no_grad()
def predict_gnn(model: nn.Module, dataset: Dataset, batch_size: int = 128) -> np.ndarray:
    loader = PyGDataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    preds = []
    for batch in loader:
        batch = batch.to(DEVICE)
        preds.append(model(batch).cpu().numpy())
    return np.concatenate(preds)


def run_one_model(model_name: str, file_map: dict, existing: pd.DataFrame | None) -> pd.DataFrame:
    rows = []
    print(f"\n{'#' * 70}\n# GRAPH MODEL: {model_name}  |  {DEVICE}\n{'#' * 70}")

    temporal = model_name.startswith("Temporal")
    graph_mode = "temporal" if temporal else "sensor"

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

        if temporal:
            edge_index = chain_edge_index(SEQ_LEN)
            in_dim = X_train.shape[2]
        else:
            edge_index = build_sensor_edge_index(X_train)
            in_dim = SEQ_LEN

        tr_idx, va_idx = engine_split(train_meta)
        train_ds = SequenceGraphDataset(X_train[tr_idx], y_train[tr_idx], graph_mode, edge_index)
        val_ds = SequenceGraphDataset(X_train[va_idx], y_train[va_idx], graph_mode, edge_index)
        test_ds = SequenceGraphDataset(X_test, y_test, graph_mode, edge_index)

        batch_size = 16 if len(X_train) > 45000 else 32
        if model_name == "TemporalGAT":
            batch_size = min(batch_size, 16)
        print(
            f"\n[{model_name}] {fd}: train={len(train_ds):,} test={len(test_ds):,} "
            f"nodes_per_graph={SEQ_LEN if temporal else len(feature_cols)} batch={batch_size}"
        )

        torch.cuda.empty_cache()
        model = build_model(model_name, in_dim, SEQ_LEN)
        model = train_gnn(model, train_ds, val_ds, batch_size=batch_size)
        y_pred = predict_gnn(model, test_ds, batch_size=batch_size)

        metrics_all = regression_metrics(y_test, y_pred)
        last_idx = last_cycle_indices(test_meta)
        metrics_last = regression_metrics(y_test[last_idx], y_pred[last_idx])
        elapsed = (time.time() - t0) / 60

        print(
            f"  last-cycle -> RMSE={metrics_last['RMSE']:.2f}  MAE={metrics_last['MAE']:.2f}  "
            f"R2={metrics_last['R2']:.4f}  ({elapsed:.1f} min)"
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
                "graph_type": "temporal_chain" if temporal else "sensor_correlation",
                "device": str(DEVICE),
                "train_minutes": round(elapsed, 2),
            }
        )
        del model
        torch.cuda.empty_cache()

    new_df = pd.DataFrame(rows)
    if existing is not None and len(existing):
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["Model", "Dataset"], keep="last")
    else:
        combined = new_df
    combined.to_csv(RESULTS_CSV, index=False)
    return combined


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=GRAPH_MODELS, default=None)
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    file_map = discover_files(DATA_DIR)
    print(f"Data: {DATA_DIR}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    existing = pd.read_csv(RESULTS_CSV) if os.path.isfile(RESULTS_CSV) else None
    for name in ([args.model] if args.model else GRAPH_MODELS):
        existing = run_one_model(name, file_map, existing)

    print(f"\nSaved: {RESULTS_CSV}")
    if existing is not None:
        print("\n=== GRAPH MODELS (last-cycle RMSE) ===")
        print(existing.pivot(index="Dataset", columns="Model", values="RMSE_last").round(2).to_string())


if __name__ == "__main__":
    main()
