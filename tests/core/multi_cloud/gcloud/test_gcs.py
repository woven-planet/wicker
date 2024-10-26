import tempfile
from unittest.mock import patch

import wicker.core.multi_cloud.gcloud.gcs as gcs
from tests.core.multi_cloud.gcloud import mocked_classes


def test_gcloud_file_exists():
    with tempfile.TemporaryDirectory() as temp_dir:
        with patch("wicker.core.multi_cloud.gcloud.gcs.storage.Blob", mocked_classes.MockedGCSBlob):
            mocked_bucket_name = "mocked_wicker_bucket"
            mocked_client = mocked_classes.MockedGCSClient(mock_file_root=temp_dir)

            mocked_client.create_bucket(mock_bucket_name=mocked_bucket_name)
            mocked_bucket = mocked_classes.MockedGCSBucket(bucket_name=mocked_bucket_name)

            # mock a file path that looks like it would be on gcloud
            file_path = "wicker_head_path/__COLUMN_CONCATENATED_FILES__/test-col-file"
            # mock some data that will constitute the file
            mock_data = b"test-data-string"

            mocked_bucket.write_file(client=mocked_client, file_path=file_path, data=mock_data)

            gcs.gcloud_file_exists(gcloud_bucket=mocked_bucket, gcloud_client=mocked_client, gcloud_file_path=file_path)
