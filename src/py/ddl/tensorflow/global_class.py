import ctypes
import os
from tensorflow import load_op_library
from ddl.tensorflow.message import _CMessage, Message


class Global:
    """
    全局对象, 负责加载C++后台和提供接口
    """
    __initialized = False
    __c_api = None
    __tf_lib = None

    path_to_lib = os.path.abspath(
        os.path.join(
            __file__,
            '../../../../../cmake-build-dir/lib.so'
        )
    )

    @classmethod
    def processes(cls) -> int:
        return cls.c_api().processes()

    @classmethod
    def process_rank(cls) -> int:
        return cls.c_api().process_rank()

    @classmethod
    def c_api(cls):
        if not cls.__initialized:
            cls.initialize()
        return cls.__c_api

    @classmethod
    def send_message(cls, msg: str, receiver: int):
        if not cls.__initialized:
            cls.initialize()
        cls.__c_api.send_message(
            ctypes.create_string_buffer(bytes(msg, encoding='UTF-8')),
            receiver
        )

    @classmethod
    def listen_message(cls) -> Message:
        if not cls.__initialized:
            cls.initialize()
        msg_ptr = cls.__c_api.listen_message()
        message = Message(msg_ptr.contents)
        # 释放new出来的内存
        # todo: 或许可以由python端申请: ctypes.create_string_buffer, 这样会自动释放
        cls.__c_api.destroy_message(msg_ptr)
        return message

    @classmethod
    def tf_lib(cls):
        if not cls.__initialized:
            cls.initialize()
        return cls.__tf_lib

    @classmethod
    def initialize(cls, path_to_lib: str = None):
        if path_to_lib is not None:
            cls.path_to_lib = path_to_lib
        # tf_lib 必须先于c_api加载, 否则tensorflow会找不到op
        cls.__tf_lib = load_op_library(cls.path_to_lib)
        cls.__c_api = ctypes.CDLL(cls.path_to_lib, mode=ctypes.RTLD_GLOBAL)
        cls.__c_api.listen_message.restype = ctypes.POINTER(_CMessage)
        cls.__c_api.destroy_message.argtypes = [ctypes.POINTER(_CMessage)]
        cls.__c_api.send_message.argtypes = [ctypes.c_char_p, ctypes.c_int]
        cls.__initialized = True
