import logging
from multiprocessing import Pool, cpu_count
from multiprocessing.pool import ThreadPool
from typing import Any, List, Tuple


def chunk_data_for_split(chunkable_data: List[Any], chunk_number: int = 500) -> List[List[Any]]:
    """Chunk data into a user specified number of chunks.

    Args:
        chunkable_data (List[Any]): Data to be chunked into smaller pieces.
        chunk_number (int): Number of chunks to form.

    Returns:
        List[List[Any]]: List of subsets of input data.
    """
    local_chunks = []
    local_chunk_size = len(chunkable_data) // chunk_number
    for i in range(0, chunk_number - 1):
        chunk = chunkable_data[i * local_chunk_size : (i + 1) * local_chunk_size]
        local_chunks.append(chunk)

    last_chunk_size = len(chunkable_data) - (chunk_number * local_chunk_size)
    if last_chunk_size > 0:
        last_chunk = chunkable_data[-last_chunk_size:]
        local_chunks.append(last_chunk)

    return local_chunks


def multiproc_file_parse(
    buckets_keys: List[Tuple[str, str]], function_for_process: Any, result_collapse_func: Any = None
) -> Any:
    """Get file size of s3 files, most often column files.

    This works on any list of buckets and keys but is generally only
    used for column files as those are the majority of what is stored on
    s3 for Wicker. Wicker also stores parquet files on s3 but those are limited
    to one file per dataset and one schema file.

    This splits your buckets_keys_list across multiple processes on your local host
    where each process is then multi threaded further. This reduces the i/o wait by
    parellelizing across all available procs and threads on a single machine.

    Args:
        buckets_keys: (List[Tuple[str, str]]): A list of buckets and keys for which
        to multi process. Tuple index 0 is bucket and index 1 is key of the file.
        function_for_process (Any): The process to run for each of the bucket key chunks
        result_collapse_func (Any): The process to run on the results of proc to collapse.
            Defaults to None

    Returns:
        int size of file list in bytes.
    """
    buckets_keys_chunks = chunk_data_for_split(chunkable_data=buckets_keys, chunk_number=200)

    logging.info("Grabbing file information from s3 heads")
    pool = Pool(cpu_count() - 2)
    results = list(pool.map(function_for_process, buckets_keys_chunks))
    if result_collapse_func is not None:
        return result_collapse_func(results)
    return results


def thread_file_parse(
    buckets_keys_chunks_local: List[Tuple[str, str]], function_for_thread: Any, result_collapse_func: Any = None
) -> Any:
    """Get file size of a list of s3 paths.

    Args:
        buckets_keys_chunks_local (List[Tuple[str, str]]):
            The list of tuples denoting bucket and key of files on s3 to
            parse. Generally column files but will work with any data.
        function_for_thread (Any): The function to run on each thread
        result_collapse_func (Any): The process to run on the results of proc to collapse.
            Defaults to None

    Returns:
        int: size of the set of files in bytes
    """
    local_chunks = chunk_data_for_split(chunkable_data=buckets_keys_chunks_local, chunk_number=200)
    thread_pool = ThreadPool()

    results = list(thread_pool.map(function_for_thread, local_chunks))  # type: ignore
    if result_collapse_func is not None:
        return result_collapse_func(results)
    return results
