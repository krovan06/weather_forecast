import pandas as pd
from datetime import datetime
import json
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from clickhouse_driver import Client as ClickHouseClient
import logging


logger = logging.getLogger(__name__)


class MLDataSaver:
    
    GRID_LAT_MIN = 42.0
    GRID_LAT_MAX = 45.0
    GRID_LON_MIN = 130.0
    GRID_LON_MAX = 134.0
    GRID_STEP = 0.5

    def __init__(self):
        self.s3_hook = S3Hook(aws_conn_id='s3_conn')
        self.client = ClickHouseClient(
            host='clickhouse',
            port=9000,
            user='default',
            password='default',
            database='weather_db'
        )

    def _read_from_s3(
        self,
        bucket: str = 'weather-data',
        prefix: str = 'ml/archive/',
    ) -> list:
        s3_files = self.s3_hook.list_keys(bucket_name=bucket, prefix=prefix)

        if not s3_files:
            raise ValueError(f"No files found in s3://{bucket}/{prefix}")

        all_data = []
        for key in s3_files:
            raw = self.s3_hook.read_key(key=key, bucket_name=bucket)
            all_data.extend(json.loads(raw))

        return all_data

    def _compute_location_id(self, lat: float, lon: float) -> int:
        n_lon = round((self.GRID_LON_MAX - self.GRID_LON_MIN) / self.GRID_STEP) + 1
        lat_idx = round((lat - self.GRID_LAT_MIN) / self.GRID_STEP)
        lon_idx = round((lon - self.GRID_LON_MIN) / self.GRID_STEP)
        return int(lat_idx * n_lon + lon_idx)

    def _parse_json_to_dataframe(self, json_data: list) -> pd.DataFrame:
        all_rows = []

        for point in json_data:
            lat = point['latitude']
            lon = point['longitude']
            daily = point['daily']
            dates = daily['time']
            location_id = self._compute_location_id(lat, lon)

            variables = {k: v for k, v in daily.items() if k != 'time'}

            for i in range(len(dates)):
                row = {
                    'date': dates[i],
                    'latitude': lat,
                    'longitude': lon,
                    'location_id': location_id,
                }
                for var_name, values in variables.items():
                    row[var_name] = values[i]

                all_rows.append(row)

        df = pd.DataFrame(all_rows)
        df['date'] = pd.to_datetime(df['date'])

        return df

    def _create_table_if_not_exists(
        self,
        df: pd.DataFrame,
        table_name: str = 'weather_archive',
    ):
        type_map = {
            'float64': 'Float32',
            'float32': 'Float32',
            'int64': 'Int64',
            'int32': 'Int32',
            'datetime64[ns]': 'Date',
            'object': 'String',
            'bool': 'UInt8',
        }

        columns_sql = []
        for col in df.columns:
            if col == 'date':
                columns_sql.append(f'"{col}" Date')
            elif col in ('latitude', 'longitude'):
                columns_sql.append(f'"{col}" Float32')
            elif col == 'location_id':
                columns_sql.append(f'"{col}" UInt8')
            else:
                dtype = str(df[col].dtype)
                ch_type = type_map.get(dtype, 'String')
                columns_sql.append(f'"{col}" Nullable({ch_type})')

        columns_str = ', '.join(columns_sql)

        self.client.execute(f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                {columns_str}
            ) ENGINE = MergeTree()
            ORDER BY (date, location_id)
        """)

    def _evolve_schema(
        self,
        df: pd.DataFrame,
        table_name: str = 'weather_archive',
    ):
        existing = self.client.execute(f"DESCRIBE TABLE {table_name}")
        existing_cols = {row[0] for row in existing}
        new_cols = set(df.columns) - existing_cols

        for col in new_cols:
            if col == 'location_id':
                ch_type = 'UInt8'
            else:
                dtype = str(df[col].dtype)
                type_map = {
                    'float64': 'Float32',
                    'float32': 'Float32',
                    'int64': 'Int64',
                    'int32': 'Int32',
                    'object': 'String',
                }
                ch_type = type_map.get(dtype, 'String')

            self.client.execute(f"""
                ALTER TABLE {table_name}
                ADD COLUMN IF NOT EXISTS "{col}" Nullable({ch_type})
            """)
            logger.info(f"Added column: {col} ({ch_type})")

    def _load_to_clickhouse(
        self,
        df: pd.DataFrame,
        table_name: str = 'weather_archive',
    ) -> int:
        if df.empty:
            logger.warning("DataFrame is empty")
            return 0

        self._create_table_if_not_exists(df, table_name)
        self._evolve_schema(df, table_name)

        df = df.copy()
        df['date'] = df['date'].dt.date

        columns = list(df.columns)
        columns_str = ', '.join([f'"{c}"' for c in columns])
        data_tuples = df.to_records(index=False).tolist()

        self.client.execute(
            f'INSERT INTO {table_name} ({columns_str}) VALUES',
            data_tuples,
            types_check=True
        )

        logger.info(f"Loaded {len(df)} rows into {table_name}")
        return len(df)

    def process(
        self,
        bucket: str = 'weather-data',
        prefix: str = 'ml/archive/',
        table_name: str = 'weather_archive',
    ) -> dict:
        json_data = self._read_from_s3(bucket, prefix)
        df = self._parse_json_to_dataframe(json_data)
        rows_count = self._load_to_clickhouse(df, table_name)

        result = {
            'table_name': table_name,
            'rows_count': rows_count,
            'columns': list(df.columns),
        }

        return result