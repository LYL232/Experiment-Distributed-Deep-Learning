from ctypes import create_string_buffer
from ddl.tensorflow.cpp_backend import CPPBackend


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
