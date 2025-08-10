import logging, sys, time, uuid, contextvars
from pythonjsonlogger import jsonlogger

request_id_ctx: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")

class UTCJsonFormatter(jsonlogger.JsonFormatter):
    def formatTime(self, record, datefmt=None):
        return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(record.created))

def get_logger(name: str="api"):
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(UTCJsonFormatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
        logger.addHandler(handler)
        logger.propagate = False
    return logger

def bind_request_id(rid: str):
    request_id_ctx.set(rid)

def get_request_id() -> str:
    return request_id_ctx.get()
