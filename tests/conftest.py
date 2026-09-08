"""Keep test-generated failures out of application runtime logs."""

import os
import tempfile
import pytest


_test_log_dir = tempfile.mkdtemp(prefix="vie-pytest-logs-")
os.environ["LOG_DIR"] = _test_log_dir


def pytest_report_header():
    return f"测试日志目录: {_test_log_dir}"


@pytest.fixture
def log_records():
    from utils import vision_logger

    records = []
    sink = vision_logger.add(lambda message: records.append(message.record.copy()), level="DEBUG")
    try:
        yield records
    finally:
        vision_logger.remove(sink)
