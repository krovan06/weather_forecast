from kafka import KafkaConsumer
import psycopg2
import json
import os
import logging


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


class WeatherConsumer:

    def __init__(self):
        kafka_bootstrap = os.getenv('KAFKA_BOOTSTRAP', 'kafka:9092')
        topic = os.getenv('KAFKA_TOPIC', 'weather_hourly_raw')
        group_id = os.getenv('KAFKA_GROUP', 'weather_stream_group')

        self.consumer = KafkaConsumer(
            topic,
            bootstrap_servers=kafka_bootstrap,
            auto_offset_reset='latest',
            enable_auto_commit=True,
            group_id=group_id,
            value_deserializer=lambda x: json.loads(x.decode('utf-8'))
        )

        self.conn = psycopg2.connect(
            host=os.getenv('PG_HOST', 'postgres_stream'),
            port=int(os.getenv('PG_PORT', 5432)),
            user=os.getenv('PG_USER', 'stream'),
            password=os.getenv('PG_PASSWORD', 'stream'),
            database=os.getenv('PG_DATABASE', 'weather_stream')
        )

        logger.info(f"Connected to Kafka: {kafka_bootstrap}, topic: {topic}")

    def _parse_timeseries(self, data: dict) -> list:
        rows = []
        timeseries = data.get('properties', {}).get('timeseries', [])

        for entry in timeseries:
            ts = entry.get('time')
            details = entry.get('data', {}).get('instant', {}).get('details', {})
            next_1h = entry.get('data', {}).get('next_1_hours', {})
            precipitation = next_1h.get('details', {}).get('precipitation_amount', 0)

            rows.append({
                'timestamp': ts,
                'latitude': 43.0,
                'longitude': 131.0,
                'temperature': details.get('air_temperature'),
                'wind_speed': details.get('wind_speed'),
                'humidity': details.get('relative_humidity'),
                'precipitation': precipitation
            })

        return rows

    def _upsert(self, rows: list):
        cursor = self.conn.cursor()

        query = """
            INSERT INTO weather_realtime
                (timestamp, latitude, longitude, temperature, wind_speed, humidity, precipitation)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (timestamp, latitude, longitude)
            DO UPDATE SET
                temperature = EXCLUDED.temperature,
                wind_speed = EXCLUDED.wind_speed,
                humidity = EXCLUDED.humidity,
                precipitation = EXCLUDED.precipitation,
                ingested_at = NOW()
        """

        for row in rows:
            cursor.execute(query, (
                row['timestamp'],
                row['latitude'],
                row['longitude'],
                row['temperature'],
                row['wind_speed'],
                row['humidity'],
                row['precipitation']
            ))

        self.conn.commit()
        cursor.close()

    def run(self):
        logger.info("Consumer started. Waiting for Kafka messages...")

        try:
            for message in self.consumer:
                rows = self._parse_timeseries(message.value)
                if not rows:
                    continue

                self._upsert(rows)

        except KeyboardInterrupt:
            logger.info("Consumer stopped by user")
        except Exception as e:
            logger.error(f"Consumer error: {e}")
            raise
        finally:
            self.consumer.close()
            self.conn.close()


if __name__ == "__main__":
    consumer = WeatherConsumer()
    consumer.run()