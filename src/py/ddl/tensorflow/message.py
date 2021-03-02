from ctypes import *


class _CMessage(Structure):
    _fields_ = [
        ('msg', c_char_p),
        ('sender', c_int)
    ]


class Message:
    def __init__(self, c_msg: _CMessage):
        # 复制出来一份, 因为原c api中的Message对象会被delete
        self.__msg = ''.join(bytes.decode(c_msg.msg, encoding='UTF-8'))
        self.__sender = c_msg.sender

    @property
    def msg(self) -> str:
        return self.__msg

    @property
    def sender(self) -> int:
        return self.__sender

    def __str__(self) -> str:
        return f'Message{{msg: {self.__msg}, sender: {self.__sender}}}'
