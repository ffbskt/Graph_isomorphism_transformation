import logging
from pythonjsonlogger import jsonlogger
import json


# Structured JSON Logging Setup
# class JSONLogger:
#     def __init__(self, name="GraphCollection", filename="Log_graph.json"):
#         self.logger = logging.getLogger(name)
#         self.logger.setLevel(logging.DEBUG)
#         handler = logging.FileHandler(filename)
#         formatter = jsonlogger.JsonFormatter('%(asctime)s %(levelname)s %(message)s')
#         handler.setFormatter(formatter)
#         self.logger.addHandler(handler)

#     def info(self, message, **kwargs):
#         self.logger.info(message, extra=kwargs)

#     def debug(self, message, **kwargs):
#         self.logger.debug(message, extra=kwargs)

#     def error(self, message, **kwargs):
#         self.logger.error(message, extra=kwargs)


import logging
from pythonjsonlogger import jsonlogger

class JSONLogger:
    _instance = None  # Store a single instance

    def __new__(cls, name="GraphCollection", filename="Log_graph.json"):
        if cls._instance is None:
            cls._instance = super(JSONLogger, cls).__new__(cls)
            cls._instance.logger = logging.getLogger(name)
            cls._instance.logger.setLevel(logging.DEBUG)

            # Ensure only one handler is attached
            if not cls._instance.logger.hasHandlers():
                handler = logging.FileHandler(filename)
                formatter = jsonlogger.JsonFormatter('%(asctime)s %(levelname)s %(message)s')
                handler.setFormatter(formatter)
                cls._instance.logger.addHandler(handler)

        return cls._instance  # Always return the same instance

    def info(self, message, **kwargs):
        self.logger.info(message, extra=kwargs)

    def debug(self, message, **kwargs):
        self.logger.debug(message, extra=kwargs)

    def error(self, message, **kwargs):
        self.logger.error(message, extra=kwargs)



    




