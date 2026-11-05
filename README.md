# Weather Data Pipeline

Pet-проект по Data Engineering: end-to-end пайплайн для сбора, обработки и визуализации погодных данных Приморского края, а также интерполяции температуры с помощью нейросетей.

## Архитектура

Проект построен по паттерну Medallion (Bronze -> Silver -> Gold) с разделением на Batch и Streaming ветки

Проект разделен на 3 независимые ветки.
  - Batch ETL (ежедневно). API -> S3 -> ClickHouse -> PostgreSQL
  - Streaming (каждые 30 минут). API -> Kafka -> PostgreSQL
  - ML Pipline (Ручной запуск). API архив за 10 лет -> S3 -> Feature Engineering -> обучение PyTorch -> Инференс (интерполяция сетки) -> СlickHouse -> PostgreSQL

## Стек технологий

  - Оркестрация: Apache Airflow 
  - Хранение: MinIO (S3), ClickHouse, PostgreSQL
  - Стриминг: Apache Kafka 
  - ML: PyTorch, Pandas, NumPy
  - Запуск: Docker Compose 

## Как запустить локально

1. Клонируйте репозиотрий
```
  git clone https://github.com/krovan06/weather_forecast.git
  cd weather_forecast
```

2. Создайте файл окружения .env в корне проекта и заполните его (пример есть в .env.example)
PostgreSQLS_USER=PostgreSQLs
PostgreSQLS_PASSWORD=PostgreSQLs
PostgreSQLS_DB=weather_db
STREAM_DB_USER=stream
STREAM_DB_PASSWORD=stream
STREAM_DB_NAME=weather_stream
_AIRFLOW_WWW_USER_USERNAME=admin
_AIRFLOW_WWW_USER_PASSWORD=admin

3. Поднимите инфраструктуру через Docker Compose
```
docker compose up -d --build
```

4. Откройте интерфейсы (они будут онаписаны в терминале или приложении docker desktop)
  - Airflow http://localhost:8080 (admin / admin)
  - MinIO (S3) http://localhost:9001 (minioadmin / minioadmin)
  - Adminer (PostgreSQLs) http://localhost:8081

5.  В Airflow включите и запустите DAG-и
  - weather_data_pipline (Batch ETL)
  - ml_data_pipeline (Подготовока данных для ML)
  - ml_training_pipeline (Обучение модели (research + train))
  - assembly-inspection (Инференс)

