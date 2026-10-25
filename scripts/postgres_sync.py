from clickhouse_driver import Client as ClickHouseClient
import psycopg2
import os
import logging


logger = logging.getLogger(__name__)


class PostgresSync:

    def __init__(self, database: str = 'weather_db'):
        self.ch = ClickHouseClient(
            host='clickhouse',
            port=9000,
            user='default',
            password='default',
            database=database
        )
        self.database = database

        self.pg = psycopg2.connect(
            host=os.getenv('PG_HOST', 'postgres_stream'),
            port=int(os.getenv('PG_PORT', 5432)),
            user=os.getenv('PG_USER', 'stream'),
            password=os.getenv('PG_PASSWORD', 'stream'),
            database=os.getenv('PG_DATABASE', 'weather_stream')
        )

    def _create_weather_map_table(self):
        cursor = self.pg.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS weather_map (
                date DATE,
                latitude FLOAT,
                longitude FLOAT,
                temperature FLOAT,
                precipitation FLOAT,
                wind_speed FLOAT,
                source VARCHAR(10),
                PRIMARY KEY (date, latitude, longitude, source)
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_weather_map_date 
            ON weather_map (date DESC)
        """)

        self.pg.commit()
        cursor.close()

    def _sync_weather_map(self) -> int:
        rows = self.ch.execute("""
            SELECT 
                date, latitude, longitude,
                temperature, precipitation, wind_speed,
                source
            FROM weather_db.weather_map
        """)

        if not rows:
            return 0

        cursor = self.pg.cursor()

        upsert_query = """
            INSERT INTO weather_map 
                (date, latitude, longitude, temperature, 
                 precipitation, wind_speed, source)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (date, latitude, longitude, source)
            DO UPDATE SET
                temperature = EXCLUDED.temperature,
                precipitation = EXCLUDED.precipitation,
                wind_speed = EXCLUDED.wind_speed
        """

        for row in rows:
            cursor.execute(upsert_query, row)

        self.pg.commit()
        cursor.close()

        return len(rows)

    def sync_all(self) -> int:
        self._create_weather_map_table()
        total = self._sync_weather_map()
        return total