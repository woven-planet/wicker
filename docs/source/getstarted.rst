Getting Started
===============

Wicker is an open source framework for Machine Learning dataset storage and serving developed at Woven Planet L5.

Wicker leverages other open source technologies such as Apache Arrow and Apache Parquet to store and serve data. Operating
Wicker mainly requires users to provide an object store (currently Wicker is only compatible with AWS S3, but integrations with
other cloud object stores are a work-in-progress).

Out of the box, Wicker provides integrations with several widely used technologies such as Spark, Flyte and DynamoDB to allow users
to write Wicker datasets from these data infrastructures. However, Wicker was built with a high degree of extensibility in mind, and
allows users to build and use their own implementations to easily integrate with their own infrastructure.

Installation
------------

``pip install wicker``

Additionally, in order to use some of the provided integrations with other open-source tooling such as Spark, Flyte and Kubernetes,
users may optionally add these options as extra install arguments:

``pip install wicker[spark,flyte,kubernetes,...]``

Configuration
-------------

By default, Wicker searches for a configurations file at ``~/wickerconfig.json``. Users may also change this path by setting the
``WICKER_CONFIG_PATH`` variable to point to their configuration JSON file.

.. code-block:: json

    {
        "aws_s3_config": {
            "s3_datasets_path": "s3://my-bucket/somepath",  // Path to the AWS bucket + prefix to use
            "region": "us-west-2",  // Region of your bucket
            "store_concatenated_bytes_files_in_dataset": true // (Optional) Whether to store concatenated bytes files in the dataset
        }
    }

Writing your first Dataset
--------------------------

Wicker allows users to work both locally and in the cloud, by leveraging different compute and storage backends.

Note that every dataset must have a defined schema. We define schemas using Wicker's schema library:

.. code-block:: python3

    from wicker import schema

    MY_SCHEMA = schema.DatasetSchema(
        primary_keys=["foo"],
        fields=[
            schema.StringField("foo", description="This is an optional human-readable description of the field"),
            schema.NumpyField("arr", shape=(4, 4), dtype="float64"),
        ]
    )

The above schema defines a dataset that consists of data that looks like:

.. code-block:: python3

    {
        "foo": "some_string",
        "arr": np.array([
            [1., 1., 1., 1.],
            [1., 1., 1., 1.],
            [1., 1., 1., 1.],
            [1., 1., 1., 1.],
        ])
    }

We have the guarantee that the dataset will be:

1. Sorted by each examples's `"foo"` field as this is the only primary_key of the dataset
2. Each example's `"arr"` field contains a 4-by-4 numpy array of float64 values

After defining a schema, we can then start to write data conforming to this schema to a dataset

Using Spark
^^^^^^^^^^^

Spark is a common data engine and Wicker provides integrations to write datasets from Spark.

.. code-block:: python3

    from wicker.plugins.spark import SparkPersistor

    examples = [
        (
            "train",  # Wicker dataset partition that this row belongs to
            {
                "foo": f"foo{i}",
                "arr": np.ones((4, 4)),
            }
        ) for i in range(1000)
    ]

    rdd = spark_context.parallelize(examples)
    persistor = SparkPersistor()
    persistor.persist_wicker_dataset(
        "my_dataset_name",
        "0.0.1",
        MY_SCHEMA,
        rdd,
    )

And that's it! Wicker will handle all the sorting and persisting of the data for you under the hood.

Using Non-Data Engine Infrastructures
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Not all users have access to infrastructure like Spark, or want to fire up something quite as heavyweight for
maybe a smaller dataset or use-case. For these users, Wicker exposes a ``DatasetWriter`` API for adding and committing
examples from any environment.

To make this work, Wicker needs an intermediate ``MetadataDatabase`` to store and index information about each row before
it commits the dataset. We provide a default integration with DynamoDB, but users can implement their own integrations easily
by implementing the abstract interface ``wicker.core.writer.AbstractDatasetWriterMetadataDatabase``, and use their own
MetadataDatabases as intermediate storage for persisting their data. Integrations with other databases as to use as a Wicker-compatible
MetadataDatabase is a work-in-progress.

Below, we provide an example of how we can use `Flyte <https://flyte.org/>`_ to commit our datasets, using DynamoDB as our
MetadataDatabase. More plugins are being written for other commonly used cloud infrastructure such as AWS Batch, Kubernetes etc.

.. code-block:: python3

    from wicker.schema import serialization
    from wicker.core.definitions import DatasetDefinition, DatasetID
    from wicker.core.writer import DatasetWriter
    from wicker.plugins import dynamodb, flyte

    # First, add the following to our ~/.wickerconfig.json file to enable Wicker's DynamoDB integrations
    #
    # "dynamodb_config": { // only if users need to use DynamoDB for writing datasets
    #     "table_name": "my-table",  // name of the table to use in dynamodb
    #     "region": "us-west-2"  // region of your table
    # }

    metadata_database = dynamodb.DynamodbMetadataDatabase()
    dataset_definition = DatasetDefinition(DatasetID(name="my_dataset", version="0.0.1"), MY_SCHEMA)

    # (1): Add examples to your dataset
    #
    # Note that this can be called from anywhere asynchronously, e.g. in different Flyte workers, from
    # a Jupyter notebook, a local Python script etc - as long as the same metadata_database config is used
    with DatasetWriter(dataset_definition, metadata_database) as writer:
        writer.add_example(
            "train",  # Name of your Wicker dataset partition (e.g. train, test, eval, unittest, ...)
            {
                "foo": "foo1",
                "arr": np.eye(4).astype("float64"),
            },  # Raw data for a single example that conforms to your schema
        )

    # (2): When ready, commit the dataset.
    #
    # Trigger the Flyte workflow to commit the dataset, either from the Flyte UI, Flyte CLI or from a Python script
    flyte.WickerDataShufflingWorkflow(
        dataset_id=str(dataset_definition.dataset_id),
        schema_json_str=serialization.dumps(MY_SCHEMA),
    )

1. Start adding examples to your dataset. Note:
    a) Here we use a ``DynamodbMetadataDatabase`` as the metadata storage for this dataset, but users can use other Metadata Database implementations here as well if they do not have an accessible DynamoDB instance.
    b) The ``.add_example(...)`` call writes a single example to the ``"train"`` partition, and can potentially throw a ``WickerSchemaException`` error if the data provided does not conform to the schema.
2. Commit your dataset. Note here that we use the committing functionality provided by ``wicker.plugins.flyte``, but more plugins for other data infrastructures are a work-in-progress (e.g. Kubernetes, AWS Batch)


Reading from your Dataset
-------------------------

.. code-block:: python3

    from wicker.core.datasets import S3Dataset

    ds = S3Dataset("my_new_dataset", "0.0.1", "train", columns_to_load=["foo", "arr"])

    # Check the size of your "train" partition
    len(ds)

    # Retrieve a single item, initial access is slow (O(seconds))
    x0 = ds[0]

    # Subsequent data accesses are fast (O(us)), data is cached in page buffers
    x0_ = ds[0]

    # Access to contiguous indices is also fast (O(ms)), data is cached on disk/in page buffers
    x1 = ds[1]

Reading from your dataset is as simple as indexing on an ``S3Dataset`` handle. Note:

1. Wicker is built for high-throughput and initial access times are amortized by accessing contiguous chunks of indices. Sampling for distributed ML training should take this into account and provide each worker with a contiguous chunk of indices as its working set for good performance.

2. Wicker allows users to select columns that they are interested in using, using the ``columns_to_load`` keyword argument
