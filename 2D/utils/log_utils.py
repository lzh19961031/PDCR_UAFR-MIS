import os
import logging
import time
from datetime import timedelta
import pandas as pd


class LogFormatter:
    def __init__(self):
        self.start_time = time.time()

    def format(self, record):
        elapsed_seconds = round(record.created - self.start_time)

        prefix = "%s - %s - %s" % (
            record.levelname,
            time.strftime("%x %X"),
            timedelta(seconds=elapsed_seconds),
        )
        message = record.getMessage()
        message = message.replace("\n", "\n" + " " * (len(prefix) + 3))
        return "%s - %s" % (prefix, message) if message else ""


def create_logger(filepath, rank):
    """
    Create a logger.
    Use a different log file for each process.
    """

    # # setting level for different rank
    # logging.basicConfig(level=logging.INFO if rank in [-1, 0] else logging.ERROR)

    # create log formatter
    log_formatter = LogFormatter()

    # create file handler and set level to debug
    if filepath is not None:
        file_handler = logging.FileHandler(filepath, "a")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(log_formatter)

    # create console handler and set level to info
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(log_formatter)

    # create logger and set level to debug
    logger = logging.getLogger()
    logger.handlers = []
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if filepath is not None:
        logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # reset logger elapsed time
    def reset_time():
        log_formatter.start_time = time.time()

    logger.reset_time = reset_time

    return logger


class pd_stats(object):
    """
    Log stuff with pandas library
    """

    def __init__(self, path, columns):
        self.path = path
        # reload path stats
        if os.path.isfile(self.path):
            self.stats = pd.read_csv(self.path)
            # # check that columns are the same
            # assert list(self.stats.columns) == list(columns), print('header not match!')
        else:
            self.stats = pd.DataFrame(columns=columns)
            self.stats.to_csv(self.path, index=False)

    def update(self, row, save=True):
        self.stats.loc[len(self.stats.index)] = row
        # save the statistics
        if save:
            self.stats.to_csv(self.path, index=False)


if __name__ == '__main__':
    pass
