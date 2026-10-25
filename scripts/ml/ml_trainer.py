import io
import json
import copy
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from datetime import datetime
from clickhouse_driver import Client as ClickHouseClient
from airflow.providers.amazon.aws.hooks.s3 import S3Hook

from scripts.ml.ml_model import WeatherMLP


FEATURE_COLS = [
    'lat_norm', 'lon_norm',
    'month_sin', 'month_cos',
    'weekday_sin', 'weekday_cos',
]
TARGET_COL = 'temp_max_norm'

HIDDEN_DIMS = [64, 128, 64]
EPOCHS = 100
LR = 1e-3
PATIENCE = 10
RANDOM_SEED = 42


class MLTrainer:

    def __init__(self):
        self.client = ClickHouseClient(
            host='clickhouse', port=9000,
            user='default', password='default',
            database='weather_db'
        )
        self.s3_hook = S3Hook(aws_conn_id='s3_conn')
        torch.manual_seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)
        self.hidden_dims = self._get_best_config()

    def _load_features(self) -> pd.DataFrame:
        result = self.client.execute(
            "SELECT * FROM weather_db.ml_features",
            with_column_types=True,
        )
        columns = [col[0] for col in result[1]]
        return pd.DataFrame(result[0], columns=columns)

    def _split_data(self, df: pd.DataFrame) -> tuple:
        unique_locs = df['location_id'].unique()
        np.random.shuffle(unique_locs)

        test_locs = unique_locs[:10]
        valid_locs = unique_locs[10:15]
        train_locs = unique_locs[15:]

        train = df[df['location_id'].isin(train_locs)].copy()
        valid = df[df['location_id'].isin(valid_locs)].copy()
        test = df[df['location_id'].isin(test_locs)].copy()
        
        return train, valid, test

    def _get_best_config(self) -> list:
        default_dims = HIDDEN_DIMS
        try:
            best_config_key = 'ml/reports/best_config.json'
            config_str = self.s3_hook.read_key(key=best_config_key, bucket_name='weather-data')
            config = json.loads(config_str)
            hidden_dims = config.get('hidden_dims')
            if hidden_dims:
                return hidden_dims
        except Exception:
            pass
        
        return default_dims

    def _train_model(self, train: pd.DataFrame, valid: pd.DataFrame) -> tuple:
        X_train = torch.tensor(train[FEATURE_COLS].values, dtype=torch.float32)
        y_train = torch.tensor(train[TARGET_COL].values, dtype=torch.float32).reshape(-1, 1)
        X_valid = torch.tensor(valid[FEATURE_COLS].values, dtype=torch.float32)
        y_valid = torch.tensor(valid[TARGET_COL].values, dtype=torch.float32).reshape(-1, 1)

        model = WeatherMLP(input_dim=len(FEATURE_COLS), hidden_dims=self.hidden_dims, output_dim=1)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        loss_fn = nn.MSELoss()

        best_valid_loss = float('inf')
        best_model_state = None
        patience_counter = 0
        history = {'train_loss': [], 'valid_loss': []}

        for epoch in range(EPOCHS):
            model.train()
            optimizer.zero_grad()
            y_pred = model(X_train)
            loss = loss_fn(y_pred, y_train)
            loss.backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                valid_loss = loss_fn(model(X_valid), y_valid).item()

            history['train_loss'].append(loss.item())
            history['valid_loss'].append(valid_loss)

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                best_model_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= PATIENCE:
                    break

        model.load_state_dict(best_model_state)
        return model, history

    def _compute_metrics(self, model: WeatherMLP, test: pd.DataFrame, norm_stats: dict) -> dict:
        model.eval()
        X_test = torch.tensor(test[FEATURE_COLS].values, dtype=torch.float32)
        y_test = torch.tensor(test[TARGET_COL].values, dtype=torch.float32).reshape(-1, 1)

        with torch.no_grad():
            y_pred_norm = model(X_test)

        mse = torch.mean((y_test - y_pred_norm) ** 2).item()
        mae = torch.mean(torch.abs(y_test - y_pred_norm)).item()
        ss_res = torch.sum((y_test - y_pred_norm) ** 2)
        ss_tot = torch.sum((y_test - y_test.mean()) ** 2)
        r2 = (1 - ss_res / ss_tot).item()

        temp_std = norm_stats['temp_std']
        temp_mean = norm_stats['temp_mean']
        y_pred_c = y_pred_norm.numpy() * temp_std + temp_mean
        y_test_c = y_test.numpy() * temp_std + temp_mean
        mae_celsius = np.mean(np.abs(y_pred_c - y_test_c))

        return {
            'mse_norm': mse,
            'mae_norm': mae,
            'r2': r2,
            'mae_celsius': float(mae_celsius),
            'test_rows': len(test),
        }

    def _save_artifacts(self, model, metrics, norm_stats, bucket: str = 'weather-data') -> dict:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        model_key = f"ml/models/model_{timestamp}.pt"
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        buffer.seek(0)
        self.s3_hook.load_bytes(bytes_data=buffer.getvalue(), key=model_key, bucket_name=bucket, replace=True)

        config = {
            'feature_cols': FEATURE_COLS,
            'hidden_dims': self.hidden_dims,
            'norm_stats': norm_stats,
        }
        config_key = f"ml/models/config_{timestamp}.json"
        self.s3_hook.load_string(string_data=json.dumps(config, indent=2), key=config_key, bucket_name=bucket, replace=True)

        metrics['timestamp'] = timestamp
        metrics_key = f"ml/models/metrics_{timestamp}.json"
        self.s3_hook.load_string(string_data=json.dumps(metrics, indent=2), key=metrics_key, bucket_name=bucket, replace=True)

        return {'model_key': model_key, 'config_key': config_key, 'metrics_key': metrics_key}

    def train(self, bucket: str = 'weather-data') -> dict:
        df = self._load_features()

        norm_stats = {
            'temp_mean': df[TARGET_COL].mean(),
            'temp_std': df[TARGET_COL].std(),
        }

        train, valid, test = self._split_data(df)
        model, _ = self._train_model(train, valid)
        metrics = self._compute_metrics(model, test, norm_stats)
        artifacts = self._save_artifacts(model, metrics, norm_stats, bucket)

        return {
            'artifacts': artifacts,
            'metrics': metrics
        }