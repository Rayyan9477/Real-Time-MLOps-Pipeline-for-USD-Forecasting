"""
Storage utilities for MinIO integration.
Handles uploading and downloading processed data to/from object storage.
"""

from minio import Minio
from minio.error import S3Error
from pathlib import Path
from datetime import datetime

from config import MINIO_CONFIG
from src.utils.logger import get_logger

logger = get_logger("storage")


class MinIOClient:
    """Client for MinIO object storage."""

    def __init__(self):
        """Initialize MinIO client."""
        self.client = Minio(
            MINIO_CONFIG["endpoint"],
            access_key=MINIO_CONFIG["access_key"],
            secret_key=MINIO_CONFIG["secret_key"],
            secure=MINIO_CONFIG["secure"],
        )
        self.bucket = MINIO_CONFIG["bucket"]

        # Ensure bucket exists
        self._ensure_bucket_exists()

        logger.info(f"Initialized MinIO client for bucket '{self.bucket}'")

    def _ensure_bucket_exists(self):
        """Create bucket if it doesn't exist."""
        try:
            if not self.client.bucket_exists(self.bucket):
                self.client.make_bucket(self.bucket)
                logger.info(f"Created bucket '{self.bucket}'")
            else:
                logger.info(f"Bucket '{self.bucket}' already exists")
        except S3Error as e:
            logger.error(f"Error checking/creating bucket: {e}")
            raise

    def upload_file(self, local_path: str, object_name: str = None) -> str:
        """
        Upload file to MinIO.

        Args:
            local_path: Path to local file
            object_name: Name for object in MinIO (uses filename if not provided)

        Returns:
            Object name in MinIO
        """
        if object_name is None:
            object_name = Path(local_path).name

        try:
            self.client.fput_object(self.bucket, object_name, local_path)
            logger.info(f"Uploaded {local_path} to {self.bucket}/{object_name}")
            return object_name

        except S3Error as e:
            logger.error(f"Error uploading file: {e}")
            raise

    def download_file(self, object_name: str, local_path: str) -> str:
        """
        Download file from MinIO.

        Args:
            object_name: Name of object in MinIO
            local_path: Path to save file locally

        Returns:
            Local path to downloaded file
        """
        try:
            self.client.fget_object(self.bucket, object_name, local_path)
            logger.info(f"Downloaded {self.bucket}/{object_name} to {local_path}")
            return local_path

        except S3Error as e:
            logger.error(f"Error downloading file: {e}")
            raise

    def list_objects(self, prefix: str = None) -> list:
        """
        List objects in bucket.

        Args:
            prefix: Filter objects by prefix

        Returns:
            List of object names
        """
        try:
            objects = self.client.list_objects(self.bucket, prefix=prefix)
            object_names = [obj.object_name for obj in objects]
            logger.info(f"Found {len(object_names)} objects in bucket")
            return object_names

        except S3Error as e:
            logger.error(f"Error listing objects: {e}")
            raise

    def get_latest_object(self, prefix: str = "processed_data") -> str:
        """
        Get the most recently uploaded object with given prefix.

        Args:
            prefix: Object name prefix to filter by

        Returns:
            Name of latest object
        """
        objects = self.list_objects(prefix=prefix)

        if not objects:
            raise ValueError(f"No objects found with prefix '{prefix}'")

        # Sort by name (timestamp is in filename)
        latest = sorted(objects)[-1]
        logger.info(f"Latest object: {latest}")

        return latest
