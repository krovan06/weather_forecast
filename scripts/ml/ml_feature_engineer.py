from clickhouse_driver import Client as ClickHouseClient
import logging


logger = logging.getLogger(__name__)


class MLFeatureEngineer:
    """
    Feature Engineer для ML-ветки.
    Читает weather_archive_clean -> создает фичи -> ml_features.
    """

    def __init__(
        self,
        host: str = 'clickhouse',
        port: int = 9000,
        user: str = 'default',
        password: str = 'default',
        database: str = 'weather_db',
        source_table: str = 'weather_archive_clean',
        target_table: str = 'ml_features',
    ):
        self.client = ClickHouseClient(
            host=host, port=port, user=user,
            password=password, database=database
        )
        self.database = database
        self.source_table = source_table
        self.target_table = target_table

    def _get_stats(self) -> dict:
        query = f"""
            SELECT
                min(latitude) AS lat_min,
                max(latitude) AS lat_max,
                min(longitude) AS lon_min,
                max(longitude) AS lon_max,
                avg(temperature_2m_max) AS temp_mean,
                stddevPop(temperature_2m_max) AS temp_std,
                avg(temperature_2m_min) AS temp_min_mean,
                stddevPop(temperature_2m_min) AS temp_min_std,
                avg(precipitation_sum) AS precip_mean,
                stddevPop(precipitation_sum) AS precip_std,
                avg(wind_speed_10m_max) AS wind_mean,
                stddevPop(wind_speed_10m_max) AS wind_std
            FROM {self.database}.{self.source_table}
        """

        result = self.client.execute(query)
        row = result[0]

        return {
            'lat_min': row[0], 'lat_max': row[1],
            'lon_min': row[2], 'lon_max': row[3],
            'temp_mean': row[4], 'temp_std': row[5],
            'temp_min_mean': row[6], 'temp_min_std': row[7],
            'precip_mean': row[8], 'precip_std': row[9],
            'wind_mean': row[10], 'wind_std': row[11],
        }

    def _create_features_table(self):
        self.client.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.database}.{self.target_table} (
                date Date,
                location_id UInt8,
                latitude Float32,
                longitude Float32,
                lat_norm Float32,
                lon_norm Float32,
                month_sin Float32,
                month_cos Float32,
                weekday_sin Float32,
                weekday_cos Float32,
                temp_max_norm Float32,
                temp_min_norm Float32,
                precip_norm Float32,
                wind_norm Float32
            ) ENGINE = MergeTree()
            ORDER BY (date, location_id)
        """)

    def _populate_features(self, stats: dict):
        self.client.execute(f"""
            INSERT INTO {self.database}.{self.target_table}
            SELECT
                date,
                location_id,
                latitude,
                longitude,
                (latitude - {stats['lat_min']}) 
                    / ({stats['lat_max']} - {stats['lat_min']}) AS lat_norm,
                (longitude - {stats['lon_min']}) 
                    / ({stats['lon_max']} - {stats['lon_min']}) AS lon_norm,
                sin(2 * pi() * toMonth(date) / 12) AS month_sin,
                cos(2 * pi() * toMonth(date) / 12) AS month_cos,
                sin(2 * pi() * toDayOfWeek(date) / 7) AS weekday_sin,
                cos(2 * pi() * toDayOfWeek(date) / 7) AS weekday_cos,
                (temperature_2m_max - {stats['temp_mean']}) 
                    / {stats['temp_std']} AS temp_max_norm,
                (temperature_2m_min - {stats['temp_min_mean']}) 
                    / {stats['temp_min_std']} AS temp_min_norm,
                (precipitation_sum - {stats['precip_mean']}) 
                    / {stats['precip_std']} AS precip_norm,
                (wind_speed_10m_max - {stats['wind_mean']}) 
                    / {stats['wind_std']} AS wind_norm
            FROM {self.database}.{self.source_table}
        """)

    def build_features(self) -> dict:
        stats = self._get_stats()
        self._create_features_table()
        self.client.execute(f"TRUNCATE TABLE {self.database}.{self.target_table}")
        self._populate_features(stats)

        rows_count = self.client.execute(
            f"SELECT count(*) FROM {self.database}.{self.target_table}"
        )[0][0]

        return {
            'table_name': self.target_table,
            'rows_count': rows_count,
            'normalization_stats': stats,
        }