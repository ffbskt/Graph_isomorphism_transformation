import logging
import inspect
import threading
from pythonjsonlogger import jsonlogger

class JSONLogger2:
    _instance = None
    _thread_local = threading.local()  # Store per-thread data
    current_id = 0

    def __new__(cls, name="GraphCollection", filename="Log_graph.json"):
        if cls._instance is None:
            cls._instance = super(JSONLogger, cls).__new__(cls)
            cls._instance.logger = logging.getLogger(name)
            cls._instance.logger.setLevel(logging.DEBUG)
            cls._instance.current_id = 0

            if not cls._instance.logger.hasHandlers():
                handler = logging.FileHandler(filename)
                formatter = jsonlogger.JsonFormatter(
                    'id: %(id)s %(asctime)s %(levelname)s caller=%(caller)s message=%(message)s'
                )
                handler.setFormatter(formatter)
                cls._instance.logger.addHandler(handler)

        return cls._instance

    @classmethod
    def reset_instance(cls):
        """Reset the singleton instance (for testing)."""
        if cls._instance is not None:

            # Remove handlers properly
            for handler in cls._instance.logger.handlers[:]:
                cls._instance.logger.removeHandler(handler)
                handler.close()  # Close handler to free the file

            cls._instance.logger.handlers.clear()

        cls._instance = None  # Finally reset the instance
        cls.current_id = 0
 

    @classmethod
    def set_caller(cls, caller_name):
        """Set the calling class/module name."""
        cls._thread_local.caller = caller_name

    def _get_caller(self):
        """Automatically get the caller class/module."""
        return getattr(self._thread_local, "caller", "Unknown")

    def info(self, message, **kwargs):
        caller = self._get_caller()
        self.logger.info(message, extra={**kwargs, 'id': self.current_id, 'caller': caller})
        self.current_id += 1

    def debug(self, message, **kwargs):
        caller = self._get_caller()
        self.logger.debug(message, extra={**kwargs, 'id': self.current_id, 'caller': caller})
        self.current_id += 1

    def error(self, message, **kwargs):
        caller = self._get_caller()
        self.logger.error(message, extra={**kwargs, 'id': self.current_id, 'caller': caller})
        self.current_id += 1


import logging
import threading
from pythonjsonlogger import jsonlogger

class JSONLogger:
    _instance = None
    _thread_local = threading.local()  # Store per-thread data
    current_id = 0

    def __new__(cls, name="GraphCollection", filename="Log_graph.json"):
        if cls._instance is None:
            cls._instance = super(JSONLogger, cls).__new__(cls)
            cls._instance.logger = logging.getLogger(name)
            cls._instance.logger.setLevel(logging.DEBUG)
            cls._instance.current_id = 0
            cls._instance.enabled = True  # Add toggle flag

            if not cls._instance.logger.hasHandlers():
                handler = logging.FileHandler(filename)
                formatter = jsonlogger.JsonFormatter(
                    'id: %(id)s %(asctime)s %(levelname)s caller=%(caller)s message=%(message)s'
                )
                handler.setFormatter(formatter)
                cls._instance.logger.addHandler(handler)

        return cls._instance

    @classmethod
    def reset_instance(cls):
        if cls._instance is not None:
            for handler in cls._instance.logger.handlers[:]:
                cls._instance.logger.removeHandler(handler)
                handler.close()

            cls._instance.logger.handlers.clear()
        cls._instance = None
        cls.current_id = 0

    @classmethod
    def set_caller(cls, caller_name):
        cls._thread_local.caller = caller_name

    def _get_caller(self):
        return getattr(self._thread_local, "caller", "Unknown")

    def enable_logging(self):
        self.enabled = True

    def disable_logging(self):
        self.enabled = False

    def info(self, message, **kwargs):
        if self.enabled:
            caller = self._get_caller()
            self.logger.info(message, extra={**kwargs, 'id': self.current_id, 'caller': caller})
            self.current_id += 1

    def debug(self, message, **kwargs):
        if self.enabled:
            caller = self._get_caller()
            self.logger.debug(message, extra={**kwargs, 'id': self.current_id, 'caller': caller})
            self.current_id += 1

    def error(self, message, **kwargs):
        if self.enabled:
            caller = self._get_caller()
            self.logger.error(message, extra={**kwargs, 'id': self.current_id, 'caller': caller})
            self.current_id += 1
