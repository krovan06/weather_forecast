import io
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from datetime import datetime
from clickhouse_driver import Client as ClickHouseClient
from airflow.providers.amazon.aws.hooks.s3 import S3Hook

from scripts.ml.ml_model import WeatherMLP


BUCKET = 'weather-data'
REPORTS_PREFIX = 'ml/reports'
RANDOM_SEED = 42


class MLEvaluator:

    def __init__(self):
        self.client = ClickHouseClient(
            host='clickhouse', port=9000,
            user='default', password='default',
            database='weather_db'
        )
        self.s3_hook = S3Hook(aws_conn_id='s3_conn')
        np.random.seed(RANDOM_SEED)

    def _save_plot_to_s3(self, fig, key: str):
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png', bbox_inches='tight')
        buffer.seek(0)
        s3_key = f"{REPORTS_PREFIX}/{key}"
        self.s3_hook.load_bytes(
            bytes_data=buffer.getvalue(),
            key=s3_key,
            bucket_name=BUCKET,
            replace=True,
        )
        plt.close(fig)

    def _save_json_to_s3(self, data: dict, key: str):
        s3_key = f"{REPORTS_PREFIX}/{key}"
        self.s3_hook.load_string(
            string_data=json.dumps(data, indent=2, default=str),
            key=s3_key,
            bucket_name=BUCKET,
            replace=True,
        )

    def _get_latest_model_and_config(self) -> tuple:
        all_keys = self.s3_hook.list_keys(bucket_name=BUCKET, prefix='ml/models/')
        
        model_keys = sorted([k for k in all_keys if k.endswith('.pt')])
        if not model_keys:
            raise ValueError("No models found in S3")
        model_key = model_keys[-1]
        
        timestamp = model_key.split('model_')[1].split('.pt')[0]
        config_key = f"ml/models/config_{timestamp}.json"
        
        config_str = self.s3_hook.read_key(key=config_key, bucket_name=BUCKET)
        config = json.loads(config_str)
        
        return model_key, config

    def _load_model(self, model_key: str, config: dict) -> WeatherMLP:
        s3_object = self.s3_hook.get_key(key=model_key, bucket_name=BUCKET)
        model_bytes = s3_object.get()['Body'].read()

        model = WeatherMLP(
            input_dim=len(config['feature_cols']),
            hidden_dims=config['hidden_dims'],
            output_dim=1,
        )
        
        buffer = io.BytesIO(model_bytes)
        model.load_state_dict(torch.load(buffer, map_location='cpu'))
        model.eval()
        
        return model

    def load_data(self) -> pd.DataFrame:
        result = self.client.execute(
            "SELECT * FROM weather_db.ml_features",
            with_column_types=True,
        )
        columns = [col[0] for col in result[1]]
        return pd.DataFrame(result[0], columns=columns)

    def split_data(self, df: pd.DataFrame) -> pd.DataFrame:
        unique_locs = df['location_id'].unique()
        np.random.shuffle(unique_locs)
        
        test_locs = unique_locs[:10]
        return df[df['location_id'].isin(test_locs)].copy()

    def plot_results(self, test_df: pd.DataFrame, y_true_c: np.ndarray, y_pred_c: np.ndarray):
        residuals = y_true_c - y_pred_c
        
        # 1. Prediction vs Actual
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(y_true_c, y_pred_c, alpha=0.5, edgecolors='k')
        ax.plot([y_true_c.min(), y_true_c.max()], [y_true_c.min(), y_true_c.max()], 'r--', lw=2)
        ax.set_xlabel('Actual Temperature (°C)')
        ax.set_ylabel('Predicted Temperature (°C)')
        ax.set_title('Prediction vs Actual (Test Set)')
        self._save_plot_to_s3(fig, 'eval/prediction_vs_actual.png')

        # 2. Residuals Histogram
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(residuals, bins=50, alpha=0.7, color='purple')
        ax.axvline(0, color='red', linestyle='--')
        ax.set_xlabel('Residual (Actual - Predicted) in °C')
        ax.set_ylabel('Count')
        ax.set_title('Residuals Distribution')
        self._save_plot_to_s3(fig, 'eval/residuals_hist.png')

        # 3. Error Map
        fig, ax = plt.subplots(figsize=(12, 8))
        scatter = ax.scatter(
            test_df['longitude'], test_df['latitude'], 
            c=np.abs(residuals), cmap='Reds', alpha=0.8, edgecolors='k'
        )
        fig.colorbar(scatter, label='Absolute Error (°C)')
        ax.set_title('Absolute Error Map (Test Points)')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        self._save_plot_to_s3(fig, 'eval/error_map.png')

    def run(self):
        model_key, config = self._get_latest_model_and_config()
        model = self._load_model(model_key, config)

        df = self.load_data()
        test_df = self.split_data(df)

        feature_cols = config['feature_cols']
        target_col = 'temp_max_norm'
        temp_mean = config['norm_stats']['temp_mean']
        temp_std = config['norm_stats']['temp_std']

        X_test = torch.tensor(test_df[feature_cols].values, dtype=torch.float32)
        y_true_norm = test_df[target_col].values

        with torch.no_grad():
            y_pred_norm = model(X_test).numpy().flatten()

        # Денормализация
        y_true_c = y_true_norm * temp_std + temp_mean
        y_pred_c = y_pred_norm * temp_std + temp_mean

        # Метрики
        mae = np.mean(np.abs(y_true_c - y_pred_c))
        rmse = np.sqrt(np.mean((y_true_c - y_pred_c) ** 2))
        ss_res = np.sum((y_true_c - y_pred_c) ** 2)
        ss_tot = np.sum((y_true_c - np.mean(y_true_c)) ** 2)
        r2 = 1 - (ss_res / ss_tot)

        metrics = {
            'timestamp': datetime.now().isoformat(),
            'model_key': model_key,
            'test_mae_celsius': float(mae),
            'test_rmse_celsius': float(rmse),
            'test_r2': float(r2),
            'test_rows': len(test_df)
        }
        
        self._save_json_to_s3(metrics, 'eval/eval_metrics.json')
        self.plot_results(test_df, y_true_c, y_pred_c)

        return metrics


if __name__ == "__main__":
    evaluator = MLEvaluator()
    evaluator.run()