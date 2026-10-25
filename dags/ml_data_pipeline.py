from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.dummy import DummyOperator
from datetime import datetime, timedelta

from scripts.ml.ml_data_loader import MLDataLoader
from scripts.ml.ml_data_saver import MLDataSaver
from scripts.ml.ml_data_cleaner import MLDataCleaner
from scripts.ml.ml_feature_engineer import MLFeatureEngineer


default_args = {
    'owner': 'oleg',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}


def load_archive(**context):
    loader = MLDataLoader()
    keys = loader.load(start_year=2015, end_year=2025, step=0.5)
    return {'files_loaded': len(keys), 'keys': keys}


def save_to_clickhouse(**context):
    saver = MLDataSaver()
    return saver.process()


def clean_data(**context):
    cleaner = MLDataCleaner()
    return cleaner.clean()


def build_features(**context):
    engineer = MLFeatureEngineer()
    return engineer.build_features()


with DAG(
    dag_id='ml_data_pipeline',
    default_args=default_args,
    description='ML data preparation (manual trigger)',
    schedule=None,
    catchup=False,
    tags=['ml', 'data', 'archive'],
) as dag:

    start = DummyOperator(task_id='start')
    end = DummyOperator(task_id='end')

    load = PythonOperator(task_id='load_archive', python_callable=load_archive)
    save = PythonOperator(task_id='save_to_clickhouse', python_callable=save_to_clickhouse)
    clean = PythonOperator(task_id='clean_data', python_callable=clean_data)
    features = PythonOperator(task_id='build_features', python_callable=build_features)

    start >> load >> save >> clean >> features >> end