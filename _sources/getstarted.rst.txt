Getting Started
===============

Wicker is an open source framework for Machine Learning dataset storage and serving developed at Woven Planet L5.

Wicker leverages other open source technologies such as Apache Arrow and Apache Parquet to store and serve data. Operating
Wicker requires users to provide an object store, a metadata database and a data engine. Out of the box, Wicker provides 
integrations with several widely used technologies, but also allows users to build and use their own implementations to 
easily integrate with their own infrastructure.

Installation
------------

``pip install wicker``

Additionally, in order to use some of the provided integrations with other open-source tooling such as Spark, Flyte and Kubernetes,
users may optionally add these options as extra install arguments:

``pip install wicker[spark,flyte,kubernetes,...]``

Configuration
-------------

By default, Wicker searches for a configurations file at ``~/.wickerconfig.json``. Users may also change this path by setting the 
``WICKER_CONFIG_PATH`` variable to point to their configuration JSON file.

.. code-block:: json

    {
        "aws_s3_config": {
            "s3_datasets_path": "s3://my-bucket/somepath",  // Path to the AWS bucket + prefix to use
            "region": "us-west-2"  // Region of your bucket
        },
        "dynamodb_config": {
            "table_name": "my-table",  // name of the table to use in dynamodb
            "region": "us-west-2"  // region of your table
        }
    }

Writing your first Dataset
--------------------------

Wicker allows users to work both locally and in the cloud, by leveraging different compute and storage backends.
When working locally, users can use the plugins found in ``wicker.plugins.local`` to use their local machine's resources
as a backend for writing datasets.

.. code-block:: python3

    from wicker import schema
    from wicker.core.definitions import DatasetID
    from wicker.core.writer import DatasetWriter
    from wicker.plugins import local, dynamodb
    from wicker.core.storage import S3PathFactory, S3DataStorage

    # (1): Give your dataset a name and a version
    dataset_id = DatasetID(name="my_new_dataset", version="0.0.1")

    # (2): Define a schema for your dataset
    my_schema = schema.DatasetSchema(
        primary_keys=["foo", "bar"],
        fields=[
            schema.StringField("foo", description="This is an optional human-readable description of the field"),
            schema.IntField("bar"),
            schema.FloatField("baz", required=False),
            schema.NumpyField("arr", shape=(4, 4), dtype="float64")
        ]
    )

    # (3): Add examples to your Dataset
    dataset_definition = DatasetDefinition(dataset_id, my_schema)
    metadata_database = dynamodb.DynamodbMetadataDatabase()
    with DatasetWriter(dataset_definition, metadata_database) as writer:
        writer.add_example(
            "train",  # Name of your partition (e.g. train, test, eval, unittest, ...)
            {
                "foo": "some_string",
                "bar": 1,
                "baz": 1.0,
                "arr": np.eye(4).astype("float64"),
            },  # Raw data for a single example that conforms to your schema
        )

    # (4): Commit your dataset when ready (can be called from separate process)
    local.commit_dataset(
        dataset_definition,
        metadata_database,
    )

1. Every dataset must have a name and a version, and we create one explicitly using a ``DatasetID``
2. Every dataset also has an associated schema, which we create using the ``Wicker.schema`` library.
    a) This particular schema consists 4 columns:
        i) Column ``foo`` is guaranteed to be a String. ``foo`` is also a primary key.
        ii) Column ``bar`` is guaranteed to be an integer. ``bar`` is also a primary key.
        iii) Column ``baz`` is guaranteed to be an float. It is also designated ``required=False`` which means that users may choose to omit this field when writing data.
        iv) Column ``arr`` is Numpy array field. It is guaranteed to have a shape of ``(4, 4)`` and each element is a ``float64``.
    b) Note that the primary key here is going to be a tuple of ``(foo, bar)``, and clients reading/writing from this dataset should assume that all rows will contain a unique tuple of their ``foo`` and ``bar`` columns. The ordering of the keys matter as well, as rows in a datasets are sorted based on these tuples and keys must be listed in order of precedence. Primary keys can only be of types ``StringField``, ``IntField``, ``LongField``, ``BoolField``.
3. Start adding examples to your dataset. Note:
    a) Here we use a ``DynamodbMetadataDatabase`` as the metadata storage for this dataset, but users can use other Metadata Database implementations here as well if they do not have an accessible DynamoDB instance.
    b) The ``.add_example(...)`` call writes a single example to the ``"train"`` partition, and can potentially throw a ``WickerSchemaException`` error if the data provided does not conform to the schema.
4. Commit your dataset. Note here that we use the committing functionality provided by ``wicker.plugins.local.commit_dataset``, but users may also choose to use other plugins such as the Spark or Flyte plugins for committing much larger datasets.


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
