from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.dummy import DummyOperator
from datetime import datetime, timedelta

from scripts.ml.ml_trainer import MLTrainer
from scripts.ml.ml_research import MLResearch


default_args = {
    'owner': 'oleg',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}


def train_model_task(**context):
    trainer = MLTrainer()
    result = trainer.train(bucket='weather-data')
    
    model_key = result['artifacts']['model_key']
    metrics = result['metrics']
    
    context['ti'].xcom_push(key='model_key', value=model_key)
    context['ti'].xcom_push(key='metrics', value=metrics)
    
    return result


def research_model():
    researcher = MLResearch()
    return researcher.run()


with DAG(
    dag_id='ml_training_pipeline',
    default_args=default_args,
    description='ML model training (manual trigger)',
    schedule=None,
    catchup=False,
    tags=['ml', 'training', 's3'],
) as dag:

    start = DummyOperator(task_id='start')
    end = DummyOperator(task_id='end')

    researcher = PythonOperator(
        task_id='research_model',
        python_callable=research_model
    )
    train = PythonOperator(
        task_id='train_model',
        python_callable=train_model_task,
    )

    start >> researcher >> train >> end