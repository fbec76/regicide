import logging
import os


LOG_LEVEL = os.environ.get("REGICIDE_LOG_LEVEL", "INFO").upper()

DATA_LEVEL_NUM = 15  # Between DEBUG (10) and INFO (20)
logging.addLevelName(DATA_LEVEL_NUM, "DATA")

def data(self, message, *args, **kws):
    if self.isEnabledFor(DATA_LEVEL_NUM):
        self._log(DATA_LEVEL_NUM, message, args, **kws)
logging.Logger.data = data

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("regicide.log", encoding="utf-8")
    ]
)

def get_logger(name=None):
    return logging.getLogger(name)
