"""
Gcloud unfortunately has no moto equivalent. This is the location where
we create a few mocked classes similar to moto to ensure we are at the
least calling functions correctly.

It is important to remember that while we can mock the exact calls we are
making we cannot mock the underlying behavior 1-1 reducing the use of these
tests dramatically.
"""

class MockedGCSBucket:
    """
    Mocked bucket object for communicating with gcs, only mocks connection calls right
    now and gives correct returns.
    """

    def __init__():
        pass

class MockedGCSClient:
    """
    Mocked client object for communicating with gcs, only mocks conenction calls right
    now and gives correct returns.
    """

    def __init__():
        pass
