from abc import ABC


class DataPersistor(ABC):
    """
    Abstract class for persisting data onto a user defined cloud or local instance.

    Only s3 is supported right now but plan to support other data stores
    (BigQuery, Whatever the hell Azure is called, Postgres)
    """

    def __init__(self) -> None:
        super().__init__()
