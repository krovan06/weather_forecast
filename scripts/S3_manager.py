from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from datetime import datetime, timedelta
import logging


logger = logging.getLogger(__name__)


class S3Manager:

    def __init__(
        self,
        bucket: str = 'weather-data',
        staging_prefix: str = 'staging',
        history_prefix: str = 'history',
        days_to_keep: int = 3,
    ):
        self.bucket = bucket
        self.staging_prefix = staging_prefix
        self.history_prefix = history_prefix
        self.days_to_keep = days_to_keep
        self.s3_hook = S3Hook(aws_conn_id='s3_conn')

    def archive_staging(self) -> dict:
        staging_files = self.s3_hook.list_keys(
            bucket_name=self.bucket,
            prefix=self.staging_prefix
        )

        if not staging_files:
            return {'archived': 0, 'cleared': 0}

        now = datetime.now()
        history_folder = (
            f"{self.history_prefix}/"
            f"year={now.year}/"
            f"month={now.month:02d}/"
            f"day={now.day:02d}/"
        )

        archived = 0
        for staging_key in staging_files:
            data = self.s3_hook.read_key(key=staging_key, bucket_name=self.bucket)
            filename = staging_key.split('/')[-1]
            history_key = f"{history_folder}{filename}"

            self.s3_hook.load_string(
                string_data=data,
                key=history_key,
                bucket_name=self.bucket,
                replace=True
            )
            self.s3_hook.delete_objects(
                bucket=self.bucket,
                keys=[staging_key]
            )
            archived += 1

        return {'archived': archived, 'cleared': archived}

    def clean_history(self) -> dict:
        cutoff_date = datetime.now() - timedelta(days=self.days_to_keep)

        all_keys = self.s3_hook.list_keys(
            bucket_name=self.bucket,
            prefix=self.history_prefix
        )

        if not all_keys:
            return {'deleted': 0, 'kept': 0}

        folders = {}
        for key in all_keys:
            folder = '/'.join(key.split('/')[:-1])
            if folder not in folders:
                folders[folder] = []
            folders[folder].append(key)

        deleted = 0
        kept = 0

        for folder, files in folders.items():
            try:
                parts = folder.split('/')
                if len(parts) < 4 or not parts[1].startswith('year='):
                    kept += len(files)
                    continue

                year = int(parts[1].split('=')[1])
                month = int(parts[2].split('=')[1])
                day = int(parts[3].split('=')[1])
                folder_date = datetime(year, month, day)

                if folder_date < cutoff_date:
                    self.s3_hook.delete_objects(bucket=self.bucket, keys=files)
                    deleted += len(files)
                else:
                    kept += len(files)
            except (IndexError, ValueError):
                kept += len(files)

        return {'deleted': deleted, 'kept': kept}

    def run(self) -> dict:
        archive_result = self.archive_staging()
        clean_result = self.clean_history()
        return {
            'archive': archive_result,
            'clean': clean_result
        }