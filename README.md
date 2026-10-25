# Weather Data Pipeline

Pet-проект по Data Engineering. end-to-end пайплайн для сбора, обработки и визуализации погодных данных Приморского края, а также интерполяции температуры с помощью нейросетей.

## Архитектура

Проект построен по паттерну Medallion (Bronze -> Silver -> Gold) с разделением на Batch и Streaming ветки

Проект разделен на 3 независимые ветки.
  - Batch ETL (ежедневно). API -> S3 -> ClickHouse -> Postgre
  - Streaming (каждые 30 минут). API -> Kafka -> Postgre
  - ML Pipline (Ручной запуск). API архив за 10 лет -> S3 -> Feature Engineering -> обучение PyTorch -> Инференс (интерполяция сетки) -> СlickHouse -> Postgre

## Стек технологий

  - Оркестрация: Apache Airflow 
  - Хранение: MinIO (S3), ClickHouse, PostgreSQL
  - Стриминг: Apache Kafka 
  - ML: PyTorch, Pandas, NumPy
  - Запуск: Docker Compose 

## Как запустить локально

1. Клонируйте репозиотрий
  git clone https://github.com/твой_ник/MLde_project.git
  cd MLde_project

2. Создайте файл окружения .env в корне проекта и заполните его (пример есть в .env.example)
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres
POSTGRES_DB=weather_db
STREAM_DB_USER=stream
STREAM_DB_PASSWORD=stream
STREAM_DB_NAME=weather_stream
_AIRFLOW_WWW_USER_USERNAME=admin
_AIRFLOW_WWW_USER_PASSWORD=admin

3. Поднимите инфраструктуру через Docker Compose
docker compose up -d --build

4. Откройте интерфейсы (они будут онаписаны в терминале или приложении docker desktop)
  - Airflow http://localhost:8080 (admin / admin)
  - MinIO (S3) http://localhost:9001 (minioadmin / minioadmin)
  - Adminer (Postgres) http://localhost:8081

5.  В Airflow включите и запустите DAG-и
  - weather_data_pipline (Batch ETL)
  - ml_data_pipeline (Подготовока данных для ML)
  - ml_training_pipeline (Обучение модели (research + train))
  - assembly-inspection (Инференс)

