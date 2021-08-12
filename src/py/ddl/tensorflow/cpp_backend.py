from ctypes import *
import os
from tensorflow import load_op_library
import tensorflow as tf


class CMessage(Structure):
    _fields_ = [
        ('msg', c_void_p),
        ('sender', c_int),
        ('length', c_size_t)
    ]


class CPPBackend:
    """
    管理c++后端的类
    """
    __path_to_lib = os.path.abspath(
        os.path.join(
            __file__,
            '../../../../../cmake-build-dir/lib.so'
        )
    )

    __initialized = False
    __c_api = None
    __tf_lib = None

    @classmethod
    def __initialize(cls, path_to_lib: str = None):
        if path_to_lib is None:
            try:
                path_to_lib = os.environ['ddl_lib']
                print(f'found ddl_lib path: {path_to_lib}')
            except KeyError:
                pass
        if path_to_lib is not None:
            cls.__path_to_lib = path_to_lib
        else:
            print(f'environment variable ddl_lib not found,'
                  f' using: {cls.__path_to_lib}')
        # tf_lib 必须先于c_api加载, 否则tensorflow会找不到op
        cls.__tf_lib = load_op_library(cls.__path_to_lib)
        cls.__c_api = CDLL(cls.__path_to_lib, mode=RTLD_GLOBAL)

        communicator_id_type = c_longlong

        cls.__c_api.communicator_rank.argtypes = [communicator_id_type]
        cls.__c_api.communicator_rank.restype = c_int

        cls.__c_api.communicator_size.argtypes = [communicator_id_type]
        cls.__c_api.communicator_size.restype = c_int

        cls.__c_api.world_communicator.restype = communicator_id_type

        cls.__c_api.destroy_message.argtypes = [POINTER(CMessage)]

        cls.__c_api.listen_message.argtypes = [communicator_id_type]
        cls.__c_api.listen_message.restype = POINTER(CMessage)

        cls.__c_api.send_message.argtypes = [
            c_char_p, c_int, communicator_id_type, c_size_t
        ]

        cls.__c_api.broadcast_message.restype = POINTER(CMessage)
        cls.__c_api.broadcast_message.argtypes = [
            c_char_p, c_int, communicator_id_type, c_size_t
        ]

        cls.__c_api.split_communicator.argtypes = [
            communicator_id_type, c_int, c_int
        ]
        cls.__c_api.split_communicator.restype = communicator_id_type

        cls.__c_api.detach_communicator.argtypes = [communicator_id_type]

        cls.__c_api.communicator_rank.restype = c_int

        cls.__initialized = True

    @classmethod
    def c_api(cls):
        if not cls.__initialized:
            cls.__initialize()
        return cls.__c_api

    @classmethod
    @tf.autograph.experimental.do_not_convert
    def tf_lib(cls):
        if not cls.__initialized:
            cls.__initialize()
        return cls.__tf_lib


@tf.RegisterGradient('TimeLog')
def time_log(_unused_op, grad):
    return grad


@tf.RegisterGradient('SendTensor')
def send_tensor(_unused_op, grad):
    return grad


@tf.RegisterGradient('ReceiveTensor')
def receive_tensor(_unused_op, grad):
    return grad
