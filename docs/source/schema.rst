Wicker Schemas
==============

Every Wicker dataset has an associated schema which is declared at schema write-time.

Wicker schemas are Python objects which are serialized in storage as Avro-compatible JSON files.
When declaring schemas, we use the ``wicker.schema.DatasetSchema`` object:

.. code-block:: python3

    from wicker.schema import DatasetSchema

    my_schema = DatasetSchema(
        primary_keys=["foo", "bar"],
        fields=[...],
    )

Your schema must be defined with a set of primary_keys. Your primary keys must be the names of 
string, float, int or bool fields in your schema, and will be used to order your dataset.

Schema Fields
-------------

Here is a list of Schema fields that Wicker provides. Most notably, users can implement custom fields
by implementing their own codecs and using the ``ObjectField``.

.. automodule:: wicker.schema.schema
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: DatasetSchema, *._accept_visitor
