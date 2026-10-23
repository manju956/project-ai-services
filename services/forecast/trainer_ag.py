import pandas as pd
from autogluon.tabular import TabularDataset, TabularPredictor
import os
import shutil

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# AutoGluon saves models to a directory, not a single file
MODEL_DIR = os.path.join(BASE_DIR, "ag_forecast_model")
CSV_PATH = os.path.join(BASE_DIR, "UCI_Credit_Card.csv")

def train_model(algorithms: list[str] | None = None):
    """
    Train AutoGluon predictor.
    
    :param algorithms: Optional list of algorithm keys to train (e.g. ['GBM', 'XGB', 'CAT']).
                       If None, checks ALGORITHMS env var or trains full preset suite.
    """
    print("Fetching data...")
    train_path = os.getenv("TRAIN_DATA_PATH")
    if train_path:
        # Data pulled from IBM i Db2 (see poc_ibmi/pull_from_ibmi.py)
        print(f"Loading training data from {train_path}")
        df = pd.read_parquet(train_path)
    else:
        # Load the Kaggle dataset from a local CSV
        df = pd.read_csv(CSV_PATH if os.path.exists(CSV_PATH) else 'UCI_Credit_Card.csv')
    if 'ID' in df.columns:
        df = df.drop(columns=['ID'])
    
    # AutoGluon handles data preparation (no need to manually separate X and y)
    train_data = TabularDataset(df)
    
    # Target column
    label_column = 'default.payment.next.month'
    
    # If a previous model directory exists, clear its contents (without deleting mount point itself)
    if os.path.exists(MODEL_DIR):
        for item in os.listdir(MODEL_DIR):
            item_path = os.path.join(MODEL_DIR, item)
            try:
                if os.path.isdir(item_path):
                    shutil.rmtree(item_path, ignore_errors=True)
                else:
                    os.remove(item_path)
            except Exception as e:
                print(f"Warning clearing {item_path}: {e}")
        
    print("Initializing AutoGluon pipeline...")

    # Check if specific algorithms are requested
    env_algos = os.getenv("ALGORITHMS")
    if algorithms is None and env_algos:
        algorithms = [a.strip() for a in env_algos.split(",") if a.strip()]

    fit_kwargs = {
        "train_data": train_data,
        "time_limit": 120, # 2 minutes for POC
    }

    if algorithms:
        print(f"Training specifically requested algorithms: {algorithms}")
        # Map list of algorithm keys to AutoGluon hyperparameters dict
        fit_kwargs["hyperparameters"] = {algo: {} for algo in algorithms}
    else:
        # Default preset training all standard tabular models
        fit_kwargs["presets"] = 'medium_quality'
    
    # Train the predictor
    predictor = TabularPredictor(
        label=label_column,
        path=MODEL_DIR,
        eval_metric='roc_auc', # Best for imbalanced binary classification
    ).fit(**fit_kwargs)
    
    print("\n" + "="*80)
    print("                MODEL TRAINING & EVALUATION SUMMARY")
    print("="*80)
    
    # Generate leaderboard
    lb = predictor.leaderboard(train_data, silent=True)
    
    # Filter and format only the most relevant columns
    relevant_cols = [c for c in ['model', 'score_val', 'score_test', 'fit_time', 'pred_time_val'] if c in lb.columns]
    summary_df = lb[relevant_cols].copy()
    
    # Sort by validation score descending
    if 'score_val' in summary_df.columns:
        summary_df = summary_df.sort_values(by='score_val', ascending=False)
    
    # Format metrics for clean display
    if 'score_val' in summary_df.columns:
        summary_df['score_val'] = summary_df['score_val'].apply(lambda x: f"{x:.4f}")
    if 'score_test' in summary_df.columns:
        summary_df['score_test'] = summary_df['score_test'].apply(lambda x: f"{x:.4f}")
    if 'fit_time' in summary_df.columns:
        summary_df['fit_time'] = summary_df['fit_time'].apply(lambda x: f"{x:.2f}s")
    if 'pred_time_val' in summary_df.columns:
        summary_df['pred_time_val'] = summary_df['pred_time_val'].apply(lambda x: f"{x*1000:.2f}ms")
    
    # Rename columns for clarity
    column_renames = {
        'model': 'Model Name',
        'score_val': 'Val ROC-AUC',
        'score_test': 'Train Score',
        'fit_time': 'Training Time',
        'pred_time_val': 'Val Latency'
    }
    summary_df = summary_df.rename(columns={k: v for k, v in column_renames.items() if k in summary_df.columns})
    
    print(summary_df.to_string(index=False))
    print("="*80)
    
    best_model = predictor.model_best
    print(f"Best Performing Model : {best_model}")
    print(f"Model Saved Location   : {MODEL_DIR}\n")

if __name__ == "__main__":
    train_model()
