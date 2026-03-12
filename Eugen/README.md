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
