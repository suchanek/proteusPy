import logging
import tempfile
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from proteusPy import (
    configure_master_logger,
    create_logger,
    disable_stream_handlers_for_namespace,
    list_all_loggers,
    list_handlers,
    set_logger_level,
    set_logger_level_for_module,
    set_logging_level_for_all_handlers,
    toggle_stream_handler,
)


class TestLoggingUtilities(unittest.TestCase):

    def setUp(self):
        """Setup before each test."""
        self.test_logger_name = "test_logger"
        self.test_logger = create_logger(self.test_logger_name, logging.DEBUG)
        # Use a unique log file name for each test to avoid contention
        self.test_log_file = f"test_{self._testMethodName}.log"
        # Use tempfile to get a platform-independent temporary directory
        self.test_log_path = Path(tempfile.gettempdir()) / "proteuspy_test_logs"
        self.temp_dir_obj = TemporaryDirectory(prefix="proteusPy_")

    def tearDown(self):
        """Cleanup after each test."""
        # Close all handlers to release file handles
        for handler in self.test_logger.handlers[:]:
            if isinstance(handler, logging.FileHandler):
                handler.close()
            self.test_logger.removeHandler(handler)

        # Also close any handlers on the root logger that might be using our test files
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            if isinstance(handler, logging.FileHandler):
                if str(self.test_log_path) in str(handler.baseFilename):
                    handler.close()
                    root_logger.removeHandler(handler)

        # Now try to clean up the files
        # Remove the temporary directory via the TemporaryDirectory object's cleanup method.
        self.temp_dir_obj.cleanup()
        return

    def test_set_logging_level_for_all_handlers(self):
        set_logging_level_for_all_handlers(logging.ERROR)
        self.assertEqual(self.test_logger.level, logging.ERROR)

    def test_disable_stream_handlers_for_namespace(self):
        self.test_logger.addHandler(logging.StreamHandler())
        disable_stream_handlers_for_namespace("test_logger")
        self.assertFalse(
            any(isinstance(h, logging.StreamHandler) for h in self.test_logger.handlers)
        )

    def test_configure_master_logger(self):
        configure_master_logger(
            self.test_log_file, str(self.test_log_path), logging.INFO
        )
        root_logger = logging.getLogger()
        self.assertEqual(root_logger.level, logging.INFO)
        self.assertTrue(
            any(isinstance(h, logging.FileHandler) for h in root_logger.handlers)
        )

    def test_create_logger(self):
        logger = create_logger("test_logger2", logging.WARNING)
        self.assertEqual(logger.level, logging.WARNING)

    def test_set_logger_level(self):
        set_logger_level(self.test_logger_name, "DEBUG")
        self.assertEqual(self.test_logger.level, logging.DEBUG)

    def test_list_all_loggers(self):
        loggers = list_all_loggers()
        self.assertIn(self.test_logger_name, loggers)

    def test_list_handlers(self):
        handlers = list_handlers(self.test_logger_name)
        self.assertTrue(isinstance(handlers, list))
        self.assertTrue(all("type" in h for h in handlers))

    def test_set_logger_level_for_module(self):
        set_logger_level_for_module("test_logger", "INFO")
        self.assertEqual(self.test_logger.level, logging.INFO)


if __name__ == "__main__":
    unittest.main()
