from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.dummy import DummyOperator
from airflow.models import Variable
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from datetime import datetime, timedelta

from scripts.sql_purity_test import DataQualityChecker
from scripts.star_schema_builder import StarBuilder
from scripts.showcase_builder import ShowcaseBuilder
from scripts.postgres_sync import PostgresSync
from scripts.create_weather_map_view import WeatherMapView
from scripts.ml.ml_inference import MLInference


default_args = {
    'owner': 'oleg',
    'depends_on_past': False,
    'start_date': datetime(2026, 1, 1),
    'max_active_runs': 1,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}


def check_and_clean_data(**context):
    auto_clean = Variable.get("auto_clean_weather", default_var="false").lower() == "true"
    checker = DataQualityChecker(table_name='weather_raw')
    result = checker.check_and_decide(auto_clean=auto_clean)
    context['ti'].xcom_push(key='clean_result', value=result)
    return result


def build_star(**context):
    builder = StarBuilder()
    result = builder.build_all()
    context['ti'].xcom_push(key='star_result', value=result)
    return result


def _get_latest_model_key(bucket: str = 'weather-data') -> str:
    s3_hook = S3Hook(aws_conn_id='s3_conn')
    all_keys = s3_hook.list_keys(bucket_name=bucket, prefix='ml/models/')
    if not all_keys:
        raise ValueError("No models found in S3 ml/models/")
    model_keys = [k for k in all_keys if k.endswith('.pt')]
    if not model_keys:
        raise ValueError("No .pt files found in ml/models/")
    return sorted(model_keys)[-1]


def ml_inference(**context):
    model_key = _get_latest_model_key(bucket='weather-data')
    context['ti'].xcom_push(key='model_key', value=model_key)
    
    inference = MLInference()
    result = inference.run(
        model_key=model_key,
        dates=None,
        bucket='weather-data',
    )
    return result


def build_daily_stats():
    return ShowcaseBuilder().build_daily_stats()

def build_warmest_locations():
    return ShowcaseBuilder().build_warmest_locations()

def build_precipitation_trend():
    return ShowcaseBuilder().build_precipitation_trend()

def build_season_stats():
    return ShowcaseBuilder().build_season_stats()

def build_region_stats():
    return ShowcaseBuilder().build_region_stats()


def create_weather_map(**context):
    view = WeatherMapView()
    status = view.create_view()
    context['ti'].xcom_push(key='view_status', value=status)
    return status


def sync_to_postgres(**context):
    sync = PostgresSync()
    total = sync.sync_all()
    return total


with DAG(
    dag_id='assembly-inspection',
    default_args=default_args,
    description='Silver -> Star -> ML -> weather_map -> PostgreSQL',
    schedule='@daily',
    catchup=False,
    tags=['data', 'ml', 'clickhouse', 'postgres'],
) as dag:

    start = DummyOperator(task_id='start')
    end = DummyOperator(task_id='end')

    check_clean = PythonOperator(
        task_id='check_and_clean_data',
        python_callable=check_and_clean_data,
    )
    star_builder = PythonOperator(
        task_id='build_star',
        python_callable=build_star,
    )
    ml_inference_task = PythonOperator(
        task_id='ml_inference',
        python_callable=ml_inference,
    )
    weather_map = PythonOperator(
        task_id='create_weather_map',
        python_callable=create_weather_map,
    )

    daily_stats = PythonOperator(task_id='build_daily_stats', python_callable=build_daily_stats)
    warmest_locations = PythonOperator(task_id='build_warmest_locations', python_callable=build_warmest_locations)
    precipitation_trend = PythonOperator(task_id='build_precipitation_trend', python_callable=build_precipitation_trend)
    season_stats = PythonOperator(task_id='build_season_stats', python_callable=build_season_stats)
    region_stats = PythonOperator(task_id='build_region_stats', python_callable=build_region_stats)
    
    sync_postgres = PythonOperator(
        task_id='sync_to_postgres',
        python_callable=sync_to_postgres,
    )

    start >> check_clean >> star_builder >> ml_inference_task >> weather_map
    weather_map >> [daily_stats, warmest_locations, precipitation_trend, season_stats, region_stats, sync_postgres] >> end