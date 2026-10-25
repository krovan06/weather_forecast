from clickhouse_driver import Client as ClickHouseClient
import logging


logger = logging.getLogger(__name__)


class WeatherMapView:

    def __init__(self, database: str = 'weather_db'):
        self.client = ClickHouseClient(
            host='clickhouse', port=9000,
            user='default', password='default',
            database=database
        )
        self.database = database

    def _view_exists(self) -> bool:
        result = self.client.execute(f"""
            SELECT count(*) 
            FROM system.tables 
            WHERE database = '{self.database}' 
            AND name = 'weather_map'
        """)
        return result[0][0] > 0

    def _check_sources(self):
        required = ['fact_weather', 'dim_location', 'weather_detailed']
        
        result = self.client.execute(f"""
            SELECT name 
            FROM system.tables 
            WHERE database = '{self.database}'
        """)
        existing = {row[0] for row in result}

        missing = [t for t in required if t not in existing]

        if missing:
            raise ValueError(f"Missing tables: {missing}")

    def create_view(self) -> str:
        self._check_sources()
        existed_before = self._view_exists()

        self.client.execute(f"""
            CREATE OR REPLACE VIEW {self.database}.weather_map AS
            SELECT
                f.date AS date,
                dl.latitude AS latitude,
                dl.longitude AS longitude,
                f.temperature_max AS temperature,
                f.precipitation AS precipitation,
                f.wind_speed_max AS wind_speed,
                'api' AS source
            FROM {self.database}.fact_weather f
            JOIN {self.database}.dim_location dl 
                ON f.location_id = dl.location_id

            UNION ALL

            SELECT
                date AS date,
                latitude AS latitude,
                longitude AS longitude,
                temperature_max AS temperature,
                CAST(NULL AS Nullable(Float32)) AS precipitation,
                CAST(NULL AS Nullable(Float32)) AS wind_speed,
                'mlp' AS source
            FROM {self.database}.weather_detailed
        """)

        return 'updated' if existed_before else 'created'

    def get_stats(self) -> dict:
        result = self.client.execute(f"""
            SELECT 
                source,
                count(*) AS cnt
            FROM {self.database}.weather_map
            GROUP BY source
        """)

        return {row[0]: row[1] for row in result}