import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from airflow.models import Variable
from clickhouse_driver import Client as ClickHouseClient
from airflow.providers.amazon.aws.hooks.s3 import S3Hook

from scripts.ml.ml_model import WeatherMLP


RANDOM_SEED = 42
EPOCHS = 50
BUCKET = 'weather-data'
REPORTS_PREFIX = 'ml/reports'


class MLResearch:

    def __init__(self):
        self.client = ClickHouseClient(
            host='clickhouse', port=9000,
            user='default', password='default',
            database='weather_db'
        )
        self.s3_hook = S3Hook(aws_conn_id='s3_conn')
        torch.manual_seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)

    def _save_json_to_s3(self, data: dict, key: str):
        s3_key = f"{REPORTS_PREFIX}/{key}"
        self.s3_hook.load_string(
            string_data=json.dumps(data, indent=2, default=str),
            key=s3_key,
            bucket_name=BUCKET,
            replace=True,
        )

    def load_data(self) -> pd.DataFrame:
        result = self.client.execute(
            "SELECT * FROM weather_db.ml_features",
            with_column_types=True,
        )
        columns = [col[0] for col in result[1]]
        return pd.DataFrame(result[0], columns=columns)

    def split_data(self, df: pd.DataFrame) -> tuple:
        unique_locs = df['location_id'].unique()
        np.random.shuffle(unique_locs)

        test_locs = unique_locs[:10]
        valid_locs = unique_locs[10:15]
        train_locs = unique_locs[15:]

        train = df[df['location_id'].isin(train_locs)].copy()
        valid = df[df['location_id'].isin(valid_locs)].copy()
        test = df[df['location_id'].isin(test_locs)].copy()

        return train, valid, test

    def experiment_depth_width(self, train, valid) -> pd.DataFrame:
        feature_cols = [
            'lat_norm', 'lon_norm',
            'month_sin', 'month_cos',
            'weekday_sin', 'weekday_cos',
        ]
        target_col = 'temp_max_norm'

        X_train = torch.tensor(train[feature_cols].values, dtype=torch.float32)
        y_train = torch.tensor(train[target_col].values, dtype=torch.float32).reshape(-1, 1)
        X_valid = torch.tensor(valid[feature_cols].values, dtype=torch.float32)
        y_valid = torch.tensor(valid[target_col].values, dtype=torch.float32).reshape(-1, 1)

        default_configs = [
            {'depth': 1, 'width': 32},
            {'depth': 1, 'width': 64},
            {'depth': 2, 'width': 64},
            {'depth': 2, 'width': 128},
            {'depth': 3, 'width': 64},
            {'depth': 3, 'width': 128},
            {'depth': 4, 'width': 64},
        ]
        
        configs_str = Variable.get("ml_grid_configs", default_var=json.dumps(default_configs))
        configs = json.loads(configs_str)

        results = []

        for config in configs:
            hidden_dims = [config['width']] * config['depth']

            model = WeatherMLP(
                input_dim=len(feature_cols),
                hidden_dims=hidden_dims,
                output_dim=1,
            )
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            loss_fn = nn.MSELoss()

            for epoch in range(EPOCHS):
                model.train()
                optimizer.zero_grad()
                y_pred = model(X_train)
                loss = loss_fn(y_pred, y_train)
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                y_valid_pred = model(X_valid)
                valid_loss = loss_fn(y_valid_pred, y_valid).item()
                mae = torch.mean(torch.abs(y_valid_pred - y_valid)).item()
                ss_res = torch.sum((y_valid - y_valid_pred) ** 2)
                ss_tot = torch.sum((y_valid - y_valid.mean()) ** 2)
                r2 = (1 - ss_res / ss_tot).item()

            results.append({
                'depth': config['depth'],
                'width': config['width'],
                'params': sum(p.numel() for p in model.parameters()),
                'valid_loss': valid_loss,
                'mae': mae,
                'r2': r2,
            })

        df_results = pd.DataFrame(results)

        csv_str = df_results.to_csv(index=False)
        s3_key = f"{REPORTS_PREFIX}/experiments/depth_width_results.csv"
        self.s3_hook.load_string(
            string_data=csv_str,
            key=s3_key,
            bucket_name=BUCKET,
            replace=True,
        )

        return df_results

    def run(self):
        df = self.load_data()
        train, valid, _ = self.split_data(df)

        df_results = self.experiment_depth_width(train, valid)

        best_config = df_results.loc[df_results['valid_loss'].idxmin()].to_dict()
        
        best_config_to_save = {
            'hidden_dims': [int(best_config['width'])] * int(best_config['depth']),
            'valid_loss': best_config['valid_loss'],
            'r2': best_config['r2']
        }
        
        self._save_json_to_s3(best_config_to_save, 'best_config.json')


if __name__ == "__main__":
    research = MLResearch()
    research.run()