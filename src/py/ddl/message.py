from ctypes import *
from ddl.tensorflow.cpp_backend import CPPBackend, CMessage
from ddl.tensorflow.communicator import Communicator


class Message:

    def __init__(self, c_msg: CMessage, is_bytes: bool = False):
        self.__is_bytes = is_bytes
        self.__msg = string_at(c_msg.msg, c_msg.length)
        if not is_bytes:
            self.__msg = bytes.decode(self.__msg, encoding='UTF-8')
        self.__sender = c_msg.sender

    @property
    def is_bytes(self) -> bool:
        return self.__is_bytes

    @property
    def msg(self) -> str or bytes:
        return self.__msg

    @property
    def sender(self) -> int:
        return self.__sender

    def __str__(self) -> str:
        return f'Message{{msg: {self.__msg}, sender: {self.__sender}}}'

    @staticmethod
    def broadcast(
            msg: str, root: int,
            communicator: Communicator = Communicator.world()) -> 'Message':
        sending_bytes = bytes(msg, encoding='UTF-8')
        msg_ptr = CPPBackend.c_api().broadcast_message(
            create_string_buffer(sending_bytes), root,
            communicator.id, len(sending_bytes)
        )
        message = Message(msg_ptr.contents)
        # 释放new出来的内存
        CPPBackend.c_api().destroy_message(msg_ptr)
        return message

    @staticmethod
    def send(
            msg: str, receiver: int,
            communicator: Communicator = Communicator.world()):
        """
        在communicator内发送信息至receiver
        @param msg: 信息字符串或者字节串
        @param receiver: 接受者
        @param communicator: 通信域
        @return: None
        """
        sending_bytes = bytes(msg, encoding='UTF-8')
        CPPBackend.c_api().send_message(
            create_string_buffer(sending_bytes),
            receiver, communicator.id, len(sending_bytes)
        )

    @staticmethod
    def send_bytes(
            data: bytes, receiver: int,
            communicator: Communicator = Communicator.world()):
        """
        在communicator内发送信息至receiver, bytes版本, 主要用于对象的传输
        @param data: 信息字符串或者字节串
        @param receiver: 接受者
        @param communicator: 通信域
        @return: None
        """
        CPPBackend.c_api().send_message(
            create_string_buffer(data),
            receiver, communicator.id, len(data)
        )

    @staticmethod
    def listen(communicator: Communicator = Communicator.world()) -> 'Message':
        """
        在通信域内收听信息
        @param communicator: 通信域
        @return:Message
        """
        msg_ptr = CPPBackend.c_api().listen_message(communicator.id)
        message = Message(msg_ptr.contents)
        # 释放new出来的内存
        # todo: 或许可以由python端申请: ctypes.create_string_buffer, 这样会自动释放
        CPPBackend.c_api().destroy_message(msg_ptr)
        return message

    @staticmethod
    def listen_bytes(communicator: Communicator = Communicator.world()) \
            -> 'Message':
        """
        在通信域内收听信息, bytes版本, 主要用于对象的传输
        @param communicator: 通信域
        @return:Message
        """
        msg_ptr = CPPBackend.c_api().listen_message(communicator.id)
        message = Message(msg_ptr.contents, True)
        # 释放new出来的内存
        # todo: 或许可以由python端申请: ctypes.create_string_buffer, 这样会自动释放
        CPPBackend.c_api().destroy_message(msg_ptr)
        return message
