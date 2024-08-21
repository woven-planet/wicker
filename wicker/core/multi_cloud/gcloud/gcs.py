from typing import Set

from google.cloud import storage


def gcloud_file_exists(
        gcloud_bucket: storage.Bucket,
        gcloud_client: storage.Client,
        gcloud_file_path: str
    ) -> bool:
    """Test existance of file at path on gcloud storage under current account.

    bucket & client passed around for gcloud as thread safe
    https://github.com/googleapis/google-cloud-dotnet/blob/main/apis/Google.Cloud.Storage.V1/docs/index.md 
    
    Args:
        gcloud_bucket (storage.Bucket): GCloud bucket to use for data storage.
        gcloud_client (storage.Client): GCloud client to use for existance checking.
        gcloud_file_path (str): Path on GCS to file
    
    Returns:
        bool: Existance or non existance
    """
    gcloud_blob = storage.Blob(bucket=gcloud_bucket, name=gcloud_file_path)
    return gcloud_blob.exists(gcloud_client)


def get_non_existant_file_set(
        file_set: Set[str]
    ) -> Set[str]:
    """Get subset of files that do not exist on gcs.

    Uses the definitions in the gcloud config to get the files
    subset that do not exist on gcloud. Used for generating manifest
    file for gcloud transfer service in general.

    Args:
        file_set (Set[str]): set of files to parse
    """

