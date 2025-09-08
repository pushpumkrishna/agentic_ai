import time
from functools import wraps
from backend.config.logging_lib import logger


def measure_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time_seconds = end_time - start_time
        elapsed_time_minutes = elapsed_time_seconds / 60
        logger.info(
            f"Function '{func.__name__}' executed in ({elapsed_time_seconds:.2f}) "
            f"seconds or ({elapsed_time_minutes:.4f} minutes)"
        )
        return result

    return wrapper
