import inspect
import logging
import datetime
import os
import sys
from typing import Any

LOG_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "ERROR": logging.ERROR,
}

LOG_DIR = "./logs"
LOG_FILE = "running_logs.log"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE_PATH = os.path.join(LOG_DIR, LOG_FILE)

# Basic logging config — message only, no wrapping
logging.basicConfig(
    format='%(message)s',
    # encoding='utf-8',
    level=LOG_LEVELS["INFO"],
    handlers=[
        logging.FileHandler(LOG_FILE_PATH, encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)


class Logging:
    @staticmethod
    def format_message(msg: Any, level: str) -> str:
        # Caller frame info: 2 steps back in stack
        frame = inspect.stack()[2]
        filename = os.path.basename(frame.filename)
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]
        return f"[{timestamp}] [{level}] [{filename}:{frame.lineno} - {frame.function}()] ['message': {msg}]"

    @staticmethod
    def info(message: Any) -> None:
        logger_ = logging.getLogger("S")
        logger_.info(Logging.format_message(message, "INFO"))

    @staticmethod
    def debug(message: Any) -> None:
        logger_ = logging.getLogger("S")
        logger_.debug(Logging.format_message(message, "DEBUG"))

    @staticmethod
    def warning(message: Any) -> None:
        logger_ = logging.getLogger("S")
        logger_.warning(Logging.format_message(message, "WARNING"))

    @staticmethod
    def error(message: Any) -> None:
        logger_ = logging.getLogger("S")
        logger_.error(Logging.format_message(message, "ERROR"))

    @staticmethod
    def exception(message: Any) -> None:
        logger_ = logging.getLogger("S")
        logger_.exception(Logging.format_message(message, "EXCEPTION"))


# Create a global instance named `logger` to keep your calls unchanged
logger = Logging()


# Usage examples:
if __name__ == "__main__":
    Logging.info("Service started")
    Logging.debug("This is a debug message")
    Logging.warning("Watch out!")
    Logging.error("Something went wrong")
    try:
        1 / 0
    except Exception as e:
        Logging.exception(f"Exception caught: {e}")
