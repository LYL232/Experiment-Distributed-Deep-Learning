import ctypes
import os
from tensorflow import load_op_library


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
        cls.__initialized = True
