import logging
from functools import wraps

logger = logging.getLogger(__name__)


def log_function(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger.info("Started")
        result = func(*args, **kwargs)
        logger.info("Finished")
        return result

    return wrapper
