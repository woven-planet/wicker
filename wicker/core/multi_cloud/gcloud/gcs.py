import csv
import datetime
from dataclasses import dataclass
from typing import List, Tuple

from google.cloud import storage, storage_transfer

from wicker.core.config import get_config
from wicker.core.parsing import (
    chunk_data_for_split,
    multiproc_file_parse,
    thread_file_parse,
)


@dataclass
class S3FileToTransferSourceDest:
    """Data required to move from s3 to gcp"""

    source_bucket: str
    source_key: str
    gcloud_path: str


def gcloud_file_exists(gcloud_bucket: storage.Bucket, gcloud_client: storage.Client, gcloud_file_path: str) -> bool:
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


def generate_manifest_file(files_to_move: List[S3FileToTransferSourceDest], manifest_dest_path: str) -> str:
    """Generate the manifest file and save at dest path.

    Generates a manifest file for running gcloud transfer service.

    Args:
        files_to_move (List[str]): list of files in s3 to move to gcp.
        manifest_dest_path (str): path to save the manifest file.
    """
    files_to_move_paths = [[file_to_move.source_key] for file_to_move in files_to_move]
    with open(manifest_dest_path, "w+") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(files_to_move_paths)


def get_non_existant_s3_file_set(file_list: List[Tuple[str, str]]) -> List[S3FileToTransferSourceDest]:
    """Get subset of files that do not exist on gcs.

    Uses the definitions in the gcloud config to get the files
    subset that do not exist on gcloud. Used for generating manifest
    file for gcloud transfer service in general.

    Args:
        file_list (List[str]): list of files to parse where indices correspond
            to 0 - bucket, 1 - key
    """
    # create the gcloud client and bucket. Slightly non optimal in
    # a multi threaded env as each thread running this creates it's own
    # instance but it simplifies the code.
    gcloud_client = storage.Client()
    config = get_config()
    gcloud_bucket = gcloud_client.bucket(config.gcloud_storage_config.bucket)
    gcloud_wicker_root_path = config.gcloud_storage_config.bucket_wicker_data_head_path
    aws_transfer_cut_prefix = config.gcloud_storage_config.aws_transfer_cut_prefix
    non_existant_files = []
    idx = 0
    for file in file_list:
        bucket, key = file
        gcloud_key = key.replace(aws_transfer_cut_prefix, gcloud_wicker_root_path)
        if not gcloud_file_exists(
            gcloud_bucket=gcloud_bucket, gcloud_client=gcloud_client, gcloud_file_path=gcloud_key
        ):
            non_existant_file = S3FileToTransferSourceDest(
                gcloud_path=gcloud_key,
                source_bucket=bucket,
                source_key=key,
            )
            non_existant_files.append(non_existant_file)
        idx += 1
        if idx == 20:
            break
    return non_existant_files


def launch_gcs_transfer_job(
    aws_access_key_id: str,
    aws_secret_access_key: str,
    description: str,
    manifest_location: str,
    project_id: str,
):
    """Launch a gcloud transfer job from s3 to gcs.

    Args:
        aws_access_key_id (str): AWS access key
        aws_secret_access_key (str): AWS secret access key
        description (str): Description of the transfer job
        manifest_location (str): Location of the manifest file, specifies
            what files to transfer
        project_id (str): Id of the project in which to store data
        sink_bucket (str): Bucket in which to store data
        source_bucket (str): Bucket from which data is taken

    Returns:
        bool: Success or failure flag for launching job
    """

    client = storage_transfer.StorageTransferServiceClient()

    source_ds_path = get_config().aws_s3_config.s3_datasets_path
    source_bucket, _ = source_ds_path.replace("s3://", "").split("/", 1)

    sink_bucket = get_config().gcloud_storage_config.bucket

    now = datetime.datetime.utcnow()
    one_time_schedule = {"day": now.day, "month": now.month, "year": now.year}
    transfer_job_request = storage_transfer.CreateTransferJobRequest(
        {
            "transfer_job": {
                "description": description,
                "schedule": {
                    "schedule_start_date": one_time_schedule,
                    "schedule_end_date": one_time_schedule,
                },
                "status": storage_transfer.TransferJob.Status.ENABLED,
                "project_id": project_id,
                "transfer_spec": {
                    "aws_s3_data_source": {
                        "bucket_name": source_bucket,
                        "aws_access_key": {
                            "access_key_id": aws_access_key_id,
                            "secret_access_key": aws_secret_access_key,
                        },
                    },
                    "gcs_data_sink": {
                        "bucket_name": sink_bucket,
                    },
                    "transfer_manifest": {"location": "gs://ads-ml-data/manifest.csv"},
                },
            }
        }
    )

    result = client.create_transfer_job(transfer_job_request)
    return result
