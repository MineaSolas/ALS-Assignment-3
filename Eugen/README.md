# Meta-Learning for Learning Curves

This project is for the Kaggle Competition: Meta-Learning for Learning Curves (DSAIT4025).

## Getting Started

1. Set up a Python virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the competition data (requires Kaggle API configured):
   ```bash
   kaggle competitions download -c performance-prediction-via-learning-curves
   unzip performance-prediction-via-learning-curves.zip -d data/
   ```

4. Launch Jupyter Notebook:
   ```bash
   jupyter notebook
   ```

Open `LCDB_Learning_Curves_Experiment.ipynb` to begin.

## Running the Best Experiment

Our best-performing submission (`sub_07_ridge_stats_and_params.csv`, local median NLL: **4.94**) used a Ridge regression meta-model combining both curve statistics and parametric fit parameters as features.

To reproduce it, open `LCDB_Learning_Curves_Experiment.ipynb` and set the **`CFG` dictionary at the very top of the notebook** to the following:

```python
CFG = {
    'experiment_name': '07_ridge_stats_and_params',
    'data': {
        'train_path': 'data/LCDB11_ER_train.hdf5',
        'eval_path':  'data/LCDB11_ER_eval.hdf5',
        'val_split': 0.1,
        'random_seed': 42
    },
    'features': {
        'use_params': True,
        'use_stats':  True,
    },
    'preprocessing': {
        'nan_handling': 'forward'
    },
    'fitting': {
        'method':  'pow2',
        'n_inits':  5,
        'n_jobs':   4
    },
    'model': {
        'type': 'ridge'
    },
    'tuning': {
        'method':   'grid',
        'cv_folds':  3,
        'n_jobs':    4
    },
    'uncertainty': {
        'method': 'gaussian'
    },
    'submission': {
        'demo_path': 'data/demo_submission.csv',
        'out_dir':   'submissions'
    }
}
```

Then **Run All** cells. The submission CSV will be saved to `submissions/sub_07_ridge_stats_and_params.csv` and the run will be appended to `experiment_logs.csv`.

---

## Notebook Structure

The notebook is divided into 9 sections, each controlled by `CFG`. **Do not hardcode values outside Section 0.**

| Section | Name | What it does |
| :--- | :--- | :--- |
| **0** | Configuration (`CFG`) | Single source of truth - every experimental variable lives here. Change only this block to run a new experiment. |
| **1** | Data Ingestion | Loads the HDF5 train and eval files, extracts valid curve observations and targets, and computes the log-spaced anchor array. |
| **2** | Preprocessing & NaN Handling | Forward-fills (or drops) missing values in raw curves. Ensures no NaN reaches the feature extractor. |
| **3** | Parametric Curve Fitting | Fits a mathematical function (e.g. `pow2`, `exp3`, `mmf4`) to each curve using `scipy.curve_fit` with multiple random initialisations for numerical stability, parallelised via `joblib`. |
| **4** | Feature Engineering | Assembles a tabular feature matrix from: curve length, fit parameters, and/or raw curve statistics. Applies `StandardScaler` before model input. |
| **5** | Meta-Model Training | Trains a point-prediction model (Ridge / Random Forest / HistGradientBoosting) on the feature matrix. Optionally searches hyperparameters via `GridSearchCV` or `HalvingGridSearchCV`. |
| **6** | Uncertainty Estimation | Converts each point prediction into a 1000-bin probability distribution using a Gaussian fitted to validation residuals. |
| **7** | Local Evaluation | Computes the **median NLL** on a held-out validation set using the exact Kaggle metric formula - a safety check before submitting. |
| **8** | Submission & Diagnostics | Writes the formatted `submission.csv` to `submissions/sub_<experiment_name>.csv`, appends results to `experiment_logs.csv`, and plots the NLL histogram. |
