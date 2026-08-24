# Saved models

Best model per dataset from the verified benchmark:

| Dataset | Model | File |
|---------|-------|------|
| FD001 | ExtraTrees | `FD001_ExtraTrees.pkl` *(not in git — ~217 MB)* |
| FD002 | Transformer | `FD002_Transformer.pth` + `FD002_Transformer_meta.pkl` |
| FD003 | HistGradientBoosting | `FD003_HistGradientBoosting.pkl` |
| FD004 | HistGradientBoosting | `FD004_HistGradientBoosting.pkl` |

## Regenerate

From the repo root:

```bash
python save_best_models.py
```

Or run `notebooks/05_save_best_models.ipynb`.

Requires the C-MAPSS files under `data/` and packages from `requirements.txt`.
