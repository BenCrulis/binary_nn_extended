from abc import ABC, abstractmethod


class Logger():
    @abstractmethod
    def log(self, logs, step=None):
        pass


class MultiLogger(Logger):
    def __init__(self, loggers):
        self.loggers = loggers

    def log(self, logs, step=None):
        for logger in self.loggers:
            logger.log(logs, step=step)