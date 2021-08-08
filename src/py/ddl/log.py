from ddl.tensorflow.cpp_backend import CPPBackend
from ctypes import create_string_buffer
from enum import Enum

LOG_CONFIG_PIPELINE_IO_GRAD_LOG = True


class TimeUnit(Enum):
    SEC = 0  # 秒
    MS = 1  # 毫秒
    US = 2  # 微秒
    NS = 3  # 纳秒


def time_log(msg: str, unit: TimeUnit) -> None:
    if unit == TimeUnit.SEC:
        CPPBackend.c_api().py_sec_time_log(
            create_string_buffer(bytes(msg, encoding='UTF-8')),
        )
    elif unit == TimeUnit.MS:
        CPPBackend.c_api().py_ms_time_log(
            create_string_buffer(bytes(msg, encoding='UTF-8')),
        )
    elif unit == TimeUnit.US:
        CPPBackend.c_api().py_us_time_log(
            create_string_buffer(bytes(msg, encoding='UTF-8')),
        )
    elif unit == TimeUnit.NS:
        CPPBackend.c_api().py_ns_time_log(
            create_string_buffer(bytes(msg, encoding='UTF-8')),
        )
    else:
        raise ValueError(f'unknown time unit: {unit}')


def info(msg: str) -> None:
    CPPBackend.c_api().py_info(
        create_string_buffer(bytes(msg, encoding='UTF-8')),
    )


def debug(msg: str) -> None:
    CPPBackend.c_api().py_debug(
        create_string_buffer(bytes(msg, encoding='UTF-8')),
    )


def error(msg: str) -> None:
    CPPBackend.c_api().py_error(
        create_string_buffer(bytes(msg, encoding='UTF-8')),
    )
