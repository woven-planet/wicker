# wicker-internal
Internal pre-open source release Wicker repository

# Getting Started

Create a Wicker configuration JSON file. This file can either be at the default location (`~/.wickerconfig`) or at a location specified by an environment variable `WICKER_CONFIG_PATH`.

```
{
    "aws_s3_config": {
        "s3_datasets_path": <base path to aws bucket and prefix for all data>,
        "s3_buckets_region": <region of bucket>
    }
}
```
