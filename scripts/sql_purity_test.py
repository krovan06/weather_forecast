from clickhouse_driver import Client as ClickHouseClient
import logging


logger = logging.getLogger(__name__)


class DataQualityChecker:

    def __init__(
        self,
        host: str = 'clickhouse',
        port: int = 9000,
        user: str = 'default',
        password: str = 'default',
        database: str = 'weather_db',
        table_name: str = 'weather_raw',
        target_table: str = 'weather_clean',
    ):
        self.client = ClickHouseClient(
            host=host, port=port, user=user,
            password=password, database=database
        )
        self.database = database
        self.table_name = table_name
        self.target_table = target_table

    def check_quality(self) -> dict:
        query = f"""
            SELECT
                count(*) AS total_rows,
                countIf(date IS NULL) AS null_dates,
                countIf(latitude IS NULL OR longitude IS NULL) AS null_coords,
                countIf(latitude < -90 OR latitude > 90 
                    OR longitude < -180 OR longitude > 180) AS invalid_coords,
                countIf(date < '1900-01-01' OR date > today() + 16) AS bad_dates,
                countIf(temperature_2m_max < -60 OR temperature_2m_max > 60) AS bad_temp_max,
                countIf(temperature_2m_min < -60 OR temperature_2m_min > 60) AS bad_temp_min,
                countIf(precipitation_sum < 0 OR precipitation_sum > 1000) AS bad_precip,
                countIf(wind_speed_10m_max < 0 OR wind_speed_10m_max > 100) AS bad_wind,
                count(*) - count(distinct (date, latitude, longitude)) AS duplicates
            FROM {self.database}.{self.table_name}
        """

        result = self.client.execute(query)
        row = result[0]

        return {
            'total_rows': row[0],
            'null_dates': row[1],
            'null_coords': row[2],
            'invalid_coords': row[3],
            'bad_dates': row[4],
            'bad_temp_max': row[5],
            'bad_temp_min': row[6],
            'bad_precip': row[7],
            'bad_wind': row[8],
            'duplicates': row[9],
        }

    def _create_clean_table(self):
        existing = self.client.execute(
            f"DESCRIBE TABLE {self.database}.{self.table_name}"
        )

        columns_sql = []
        for row in existing:
            col_name = row[0]
            col_type = row[1]
            columns_sql.append(f'"{col_name}" {col_type}')

        columns_str = ', '.join(columns_sql)

        self.client.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.database}.{self.target_table} (
                {columns_str}
            ) ENGINE = ReplacingMergeTree()
            ORDER BY (date, latitude, longitude)
        """)

    def clean(self) -> dict:
        quality_before = self.check_quality()
        self._create_clean_table()
        self.client.execute(f"TRUNCATE TABLE {self.database}.{self.target_table}")

        self.client.execute(f"""
            INSERT INTO {self.database}.{self.target_table}
            SELECT *
            FROM {self.database}.{self.table_name}
            WHERE
                date IS NOT NULL
                AND latitude IS NOT NULL AND longitude IS NOT NULL
                AND latitude BETWEEN -90 AND 90
                AND longitude BETWEEN -180 AND 180
                AND date >= '1900-01-01' AND date <= today() + 16
                AND (temperature_2m_max IS NULL OR temperature_2m_max BETWEEN -60 AND 60)
                AND (temperature_2m_min IS NULL OR temperature_2m_min BETWEEN -60 AND 60)
                AND (precipitation_sum IS NULL OR precipitation_sum BETWEEN 0 AND 1000)
                AND (wind_speed_10m_max IS NULL OR wind_speed_10m_max BETWEEN 0 AND 100)
        """)

        self.client.execute(f"""
            OPTIMIZE TABLE {self.database}.{self.target_table} FINAL
        """)

        clean_count = self.client.execute(
            f"SELECT count(*) FROM {self.database}.{self.target_table} FINAL"
        )[0][0]

        return {
            'raw_rows': quality_before['total_rows'],
            'clean_rows': clean_count,
            'removed_rows': quality_before['total_rows'] - clean_count,
            'quality_before': quality_before,
        }

    def check_and_decide(self, auto_clean: bool = True) -> dict:
        if not auto_clean:
            return {
                'check_result': self.check_quality(),
                'cleaned': False,
                'status': 'checked_only',
            }

        return self.clean()