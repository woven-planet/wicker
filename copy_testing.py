from wicker.core.datasets import S3Dataset

ds = S3Dataset(
    dataset_name="nvidia_hyperion_kb",
    dataset_version="0.0.2f",
    dataset_partition_name="train"
)

print('testing the copy read')
ds.dataset_size
#ds.copy_partition_to_gcloud()
