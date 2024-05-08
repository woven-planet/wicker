import contextlib
import signal


@contextlib.contextmanager
def time_limit(seconds, error_message: str):
    def signal_handler(signum, frame):
        raise TimeoutError(f"Timed out!. {error_message}")

    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
