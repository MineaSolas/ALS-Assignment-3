# Meta-Learning for Learning Curves

This project is for the Kaggle Competition: Meta-Learning for Learning Curves (DSAIT4025).

## Project Structure

For the code to run, ensure the `data/` folder is at the root with these files:
- `LCDB11_ER_train.hdf5`: Training curves and targets.
- `LCDB11_ER_eval.hdf5`: Evaluation curves to predict.
- `demo_submission.csv`: Template for submission format.

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

### Command Line Execution
Alternatively, run the notebook directly using `papermill`:
```bash
pip install papermill
papermill LCDB_Learning_Curves_Experiment.ipynb output.ipynb --kernel python3
```

Open `LCDB_Learning_Curves_Experiment.ipynb` to begin.

## Running the Best Experiment

Our best-performing submission (`sub_07_ridge_stats_and_params.csv`, local median NLL: **4.94**) used a Ridge regression meta-model combining both curve statistics and parametric fit parameters as features.

To reproduce it, open `LCDB_Learning_Curves_Experiment.ipynb` and set the **`CFG` dictionary at the very top of the notebook** to the following:

```python
CFG = {
    'experiment_name': 'baseline_ridge_scaled_v1',
    'data': {
        'train_path': 'data/LCDB11_ER_train.hdf5',
        'eval_path': 'data/LCDB11_ER_eval.hdf5',
        'val_split': 0.1,  
        'random_seed': 42
    },
    'features': {
        'use_params': True,
        'use_stats': True # options: True or false
    },
    'preprocessing': {
        'nan_handling': 'forward'
    },
    'fitting': {
        'method': 'pow2', # 'pow2', 'pow3', 'exp3', 'mmf4', 'wbl4'
        'n_inits': 5,
        'n_jobs': 4
    },
    'model': {
        # options: 'ridge', 'rf', 'hgb' (HistGradientBoosting)
        'type': 'ridge', 
    },
    'tuning': {
        # 'none', 'grid' (GridSearchCV), 'halving' (HalvingGridSearchCV)
        'method': 'grid', 
        'cv_folds': 3,
        'n_jobs': 4
    },
    'uncertainty': {
        'method': 'gaussian', # 'gaussian', 'kde', 'conformal', 'quantile'
    },
    'submission': {
        'demo_path': 'data/demo_submission.csv',
        'out_dir': 'submissions'
    }
}
```

Then **Run All** cells. The submission CSV will be saved to `submissions/sub_07_ridge_stats_and_params.csv` and the run will be appended to `experiment_logs.csv`.

---

## Notebook Structure

The notebook is divided into 9 modular sections. **Configuration in Section 0** controls the entire pipeline.

| Section | Name | Description |
| :--- | :--- | :--- |
| **0** | Configuration | Single source of truth for all parameters (paths, model type, feature flags). |
| **1** | Data Ingestion | Loads HDF5 datasets and extracts valid learning curves (length > 16). |
| **2** | Preprocessing | Handles NaNs via forward-filling to ensure curve continuity. |
| **3** | Curve Fitting | Modular fitting (Power Laws, Exponential, MMF) via `scipy.optimize`. |
| **4** | Feature Engineering | Assembles feature matrix from fit parameters and curve statistics. |
| **5** | Meta-Model Training | Trains the predictor (Ridge/RF/HGB) with automated hyperparameter tuning. |
| **6** | Uncertainty | Estimates error distribution using Gaussian modeling of residuals. |
| **7** | Local Evaluation | Computes Local Median NLL on a 10% validation split. |
| **8** | Submission | Generates the 1000-bin submission CSV and logs numerical results. |
