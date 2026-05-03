import logging
import sys

def setupLogger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.propagate = False
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        '{"time":"%(asctime)s","level":"%(levelname)s","module":"%(name)s","message":"%(message)s"}'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger