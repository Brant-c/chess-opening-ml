# Source Code

`baseline.py` is the reproducible Milestone 3 evaluation pipeline. It reads
the cleaned M2 Parquet file (or a CSV with the same schema), validates the
required columns/outcomes, selects only the four pre-game predictors, makes
the fixed stratified split, fits a majority-class benchmark and the agreed
decision tree, and writes validation metrics.

## Run on Windows

From the repository root, create and activate a virtual environment, install
the pinned runtime dependencies, then run the pipeline:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m src.baseline
```

By default, input is
`data/processed/lichess_clean_2016_08.parquet`. Override it with
`--input path\to\cleaned-data.csv` or `--input path\to\cleaned-data.parquet`.
The runner writes `baseline_table.csv`, `classification_report.csv`, and
`run_metadata.json` to `reports/generated/`. It does not evaluate the reserved
test split.