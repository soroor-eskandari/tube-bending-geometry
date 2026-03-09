import logging
from functools import wraps

if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

logger = logging.getLogger(__name__)


def log_function(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger.info("Started %s", func.__qualname__)
        result = func(*args, **kwargs)
        logger.info("Finished %s", func.__qualname__)
        return result

    return wrapper
