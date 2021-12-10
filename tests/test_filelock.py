import multiprocessing
import os
import re
import sys
import tempfile
import time
import unittest
from typing import Optional

from wicker.core.filelock import SimpleUnixFileLock


def write_index_to_file(filepath: str, index: int) -> None:
    # If we don't protect this writing operation with a lock, we will get interleaved
    # writing in the file instead of lines that have "foo-[index]"
    with SimpleUnixFileLock(f"{filepath}.lock"):
        for c in f"foo-{index}\n":
            with open(filepath, "a") as f:
                f.write(c)


class TestFileLock(unittest.TestCase):
    def test_simple_acquire_no_race_conditions(self) -> None:
        with tempfile.NamedTemporaryFile() as lockfile:
            placeholder: Optional[str] = None
            with SimpleUnixFileLock(lockfile.name):
                placeholder = "foo"
            self.assertEqual(placeholder, "foo")

    def test_synchronized_concurrent_writes(self) -> None:
        with tempfile.NamedTemporaryFile("w+") as write_file:
            with multiprocessing.Pool(4) as pool:
                pool.starmap(write_index_to_file, [(write_file.name, i) for i in range(1000)])
            for line in write_file.readlines():
                self.assertTrue(re.match(r"foo-[0-9]+", line))

    def test_timeout(self) -> None:
        with tempfile.NamedTemporaryFile() as lockfile:

            def run_acquire_and_sleep() -> None:
                with SimpleUnixFileLock(lockfile.name):
                    time.sleep(2)

            proc = multiprocessing.Process(target=run_acquire_and_sleep)
            proc.start()

            # Give the other process some time to start up
            time.sleep(1)

            with self.assertRaises(TimeoutError):
                with SimpleUnixFileLock(lockfile.name, timeout_seconds=1):
                    assert False, "Lock acquisition should time out"

    def test_process_dies(self) -> None:
        with tempfile.NamedTemporaryFile() as lockfile:

            def run_acquire_and_die() -> None:
                """Simulate a process dying on some exception while the lock is acquired"""
                # Mute itself to prevent too much noise in stdout/stderr
                with open(os.devnull, "w") as devnull:
                    sys.stdout = devnull
                    sys.stderr = devnull
                    lock = SimpleUnixFileLock(lockfile.name)
                    lock.__enter__()
                    raise ValueError("die")

            # When the child process dies, the fd (and the exclusive lock) should be closed automatically
            proc = multiprocessing.Process(target=run_acquire_and_die)
            proc.start()

            # Give the other process some time to start up
            time.sleep(1)

            placeholder: Optional[str] = None
            with SimpleUnixFileLock(lockfile.name, timeout_seconds=1):
                placeholder = "foo"
            self.assertEqual(placeholder, "foo")
