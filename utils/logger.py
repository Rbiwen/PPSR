import logging
from paddle import distributed as dist
import os

def log_at_train0(fun):

    def warpper(*args, **kargs):
        if dist.get_rank() == 0:
            fun(*args, **kargs)

    return warpper

class Logger:
    def __init__(self, name='ppsr', logger_file='./output/train.log'):
        _dir = os.path.dirname(logger_file)
        os.makedirs(_dir, exist_ok=True)
        self._logger = logging.getLogger(name)
        self._logger.setLevel(level=logging.DEBUG)
        handler = logging.FileHandler(logger_file, 'a', encoding="UTF-8")
        handler.setLevel(level=logging.DEBUG)
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(message)s")
        handler.setFormatter(formatter)
        self._logger.addHandler(handler)
    
    @log_at_train0
    def info(self, fmt):
        self._logger.info(fmt)

    @log_at_train0
    def debug(self, fmt):
        self._logger.debug(fmt)

    @log_at_train0
    def warning(self, fmt):
        self._logger.warning(fmt)

    @log_at_train0
    def error(self, fmt):
        self._logger.error(fmt)

    def print_config(self, cfg):
        self.info(cfg)
