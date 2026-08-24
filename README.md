# NASA C-MAPSS Remaining Useful Life (RUL) Benchmark

Comprehensive benchmark of **13 models** (6 classical ML + 5 deep learning + 2 graph neural networks) on the NASA C-MAPSS turbofan engine datasets **FD001–FD004**.

**Repository:** [dakshagarwal76/nasa-cmapps](https://github.com/dakshagarwal76/nasa-cmapps)

## Best results (last-cycle RMSE)

| Dataset | Best model | Family | RMSE | MAE | R² |
|---------|------------|--------|------|-----|-----|
| FD001 | ExtraTrees | ML | 13.89 | 10.20 | 0.888 |
| FD002 | Transformer | DL | 26.57 | 19.12 | 0.753 |
| FD003 | HistGradientBoosting | ML | 17.13 | 12.85 | 0.829 |
| FD004 | HistGradientBoosting | ML | 29.39 | 21.66 | 0.710 |

Primary metric: **last observed cycle per test engine**. Training RUL is capped at **125** cycles.

## Models

- **ML:** Ridge, ElasticNet, RandomForest, ExtraTrees, HistGradientBoosting, XGBoost  
- **DL:** LSTM, GRU, CNN1D, TCN, Transformer  
- **Graph:** TemporalGCN, TemporalGAT  

## Repository layout

```
data/                 C-MAPSS train / test / RUL text files
notebooks/            01–05 runnable Jupyter notebooks
results/              CSV metrics (ML, DL, graph, master)
docs/                 Markdown + Word report, figures, epoch metrics
models/               Saved best models (see models/README.md)
*.py                  Benchmark & documentation scripts
requirements.txt      Python dependencies
```

## Quick start

```bash
# 1. Create environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt

# 2. Data is already in ./data/
#    Or download NASA C-MAPSS and place train_FD*.txt, test_FD*.txt, RUL_FD*.txt in ./data/

# 3. Run benchmarks (optional — results/ already includes verified CSVs)
python run_regression_benchmark.py
python run_dl_benchmark.py          # GPU recommended
python run_graph_benchmark.py       # GPU + torch-geometric

# 4. Regenerate documentation / Word report
python generate_documentation.py
python generate_word_documentation.py
python verify_and_package.py
```

Or open notebooks in order:

1. `notebooks/01_ml_regression_benchmark.ipynb`
2. `notebooks/02_dl_benchmark.ipynb`
3. `notebooks/03_graph_benchmark.ipynb`
4. `notebooks/04_documentation_and_analysis.ipynb`
5. `notebooks/05_save_best_models.ipynb`

## Documentation

- **Word report:** [`docs/CMAPSS_RUL_Benchmark_Documentation.docx`](docs/CMAPSS_RUL_Benchmark_Documentation.docx)
- **Markdown:** [`docs/CMAPSS_RUL_Benchmark_Documentation.md`](docs/CMAPSS_RUL_Benchmark_Documentation.md)
- Figures under `docs/figures/`

## Dataset reference

| Code | Conditions | Fault modes | Difficulty |
|------|------------|-------------|------------|
| FD001 | 1 (sea level) | 1 (HPC) | Easy |
| FD002 | 6 | 1 (HPC) | Medium |
| FD003 | 1 (sea level) | 2 (HPC + Fan) | Medium |
| FD004 | 6 | 2 (HPC + Fan) | Hard |

Source: A. Saxena et al., “Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation,” PHM08, 2008.

## Notes

- Paths are relative to the repo root (`paths.py`); no machine-specific absolute paths.
- `models/FD001_ExtraTrees.pkl` is excluded from Git (≈217 MB). Regenerate with `python save_best_models.py` or notebook `05`.
- Graph models: TemporalGCN and TemporalGAT.

## License

Research / educational use of the NASA C-MAPSS dataset. Code in this repo is provided as-is for benchmarking and learning.
