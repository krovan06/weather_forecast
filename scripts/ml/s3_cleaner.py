from airflow.providers.amazon.aws.hooks.s3 import S3Hook

class S3MLCleaner:
    def __init__(self, bucket: str = 'weather-data', prefix: str = 'ml/models/'):
        self.bucket = bucket
        self.prefix = prefix
        self.s3_hook = S3Hook(aws_conn_id='s3_conn')

    def clean_old_models(self, keep_last: int = 3):
        all_keys = self.s3_hook.list_keys(bucket_name=self.bucket, prefix=self.prefix)
        
        models = sorted([k for k in all_keys if k.endswith('.pt')])
        
        if len(models) <= keep_last:
            return 0
            
        to_delete = models[:-keep_last]
        
        for model_key in to_delete:
            self.s3_hook.delete_objects(bucket=self.bucket, keys=[model_key])

            timestamp = model_key.split('model_')[1].split('.pt')[0]
            config_key = f"{self.prefix}config_{timestamp}.json"
            metrics_key = f"{self.prefix}metrics_{timestamp}.json"
            
            self.s3_hook.delete_objects(bucket=self.bucket, keys=[config_key, metrics_key])
            
        return len(to_delete)