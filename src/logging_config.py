import logging
import os

try:
    from .config import LOG_DIR
except ImportError:
    from config import LOG_DIR


def configure_logging(
    log_filename: str,
    logger_name: str | None = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """Create the log directory and attach console and file handlers to a named logger."""
    os.makedirs(LOG_DIR, exist_ok=True)
    log_file = os.path.join(LOG_DIR, log_filename)

    target_logger = logging.getLogger(logger_name)
    if not target_logger.handlers:
        target_logger.setLevel(level)
        target_logger.propagate = False
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)

        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)

        target_logger.addHandler(stream_handler)
        target_logger.addHandler(file_handler)

    return target_logger