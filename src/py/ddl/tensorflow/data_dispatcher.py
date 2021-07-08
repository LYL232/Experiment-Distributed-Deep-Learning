from ddl.tensorflow.communicator import Communicator
from ddl.message import Message
from ddl.log import info
import numpy as np
import pickle
import json
import os


class DataDispatcher:
    """
    进行数据并行时需要将数据分发到其他进程上进行读取

    目前的实现很简单, 有较大改进空间:
    1. 主从式
    2. 非组通信
    3. 只使用.npz进行缓存
    """

    def __init__(
            self, root_rank: int,
            communicator: Communicator = Communicator.world(),
            root_data: np.ndarray or tuple = None,
            data_save_path: str = None,
            name: str = None
    ):
        """
        @param root_rank: 进行数据分发的根节点
        @param communicator: 通信域, 默认为全局通信域, 请保证通过此类对象调用
        @param root_data: 进行数据分发的根节点所拥有的数据, 可以是数个np数组组成的元组,
         如果是单个np数组, 则将其视为只有一个元素的元组
         PipelineModel.fit的所有进程都在此通信域内, 不能多也不能少, 否则无法进行通信域的分割
        @param data_save_path: 缓存路径, 可选, 如果为None表明没有缓冲, 需要时会向指定节点请求数据
        @param name: 数据的标签等, 用于区分分发的数据
         如果定义, 则要求以.npz结尾
        """
        self.__name = name if name else 'unnamed'
        if communicator.rank == root_rank:
            assert root_data is not None
            if isinstance(root_data, np.ndarray):
                self.__data = (root_data,)
                self.__data_count = 1
                self.__samples = len(self.__data[0])
            elif isinstance(root_data, (tuple, list)):
                assert all(isinstance(each, np.ndarray) for each in root_data)
                for i in range(1, len(root_data)):
                    assert len(root_data[i - 1]) == len(root_data[i]), \
                        'every data must have same len'
                self.__samples = len(root_data[0])
                self.__data_count = len(root_data)
                self.__data = root_data
            else:
                raise Exception(f'unsupported data format: {type(root_data)}')

            self.__begin = 0
            self.__end = self.__samples
        else:
            self.__samples = None
            self.__data_count = None
            self.__data = None
            self.__begin = None
            self.__end = None

            self.__data_save_path = data_save_path
            if data_save_path is not None:
                assert data_save_path.endswith('.npz'), \
                    'data_save_path must ends with .npz'
                if os.path.exists(data_save_path):
                    load = np.load(data_save_path)
                    assert 'meta-info' in load.keys(), \
                        'wrong data format'
                    self.__data_count = int(load['meta-info'][0])
                    self.__begin = int(load['meta-info'][1])
                    self.__end = int(load['meta-info'][2])
                    self.__data = []
                    for i in range(self.__data_count):
                        key = f'data-{i}'
                        # 没有的数据直接设为None
                        self.__data.append(
                            load[key] if key in load.keys() else None)
                    assert any(each is not None for each in self.__data)
                    self.__data = tuple(self.__data)
        self.__root_rank = root_rank
        self.__comm = communicator
        self.__dispatched = False

    @property
    def data(self) -> tuple:
        assert self.__dispatched, 'call dispatch() before call data()'
        assert self.__data is not None, 'bug exists'
        return self.__data

    @property
    def communicator(self) -> Communicator:
        return self.__comm

    @property
    def root_rank(self) -> int:
        return self.__root_rank

    def dispatch(
            self, communicator: Communicator,
            root_rank: int, data_indexes: tuple or list,
            root_keep_data: bool = True,
    ) -> None:
        """
        在全局通信域内进行数据分发, 并且将结果保存在self.__data里
        分发的数据如果不能整除需要数据的进程, 则最后一个进程收到的数据会比其他进程稍少
        @param communicator: 进行数据分配的通信域
        @param root_rank: 拥有数据的进程号
        @param data_indexes: 所需数据的下标, 非根节点有效
        @param root_keep_data: 非根节点无效, 根节点是否需要持有数据,
         如果根节点不需要持有数据, 则此项为False,
        @return: None
        """
        info(f'communicator: {communicator.id} dispatching data,'
             f' size: {communicator.size}')

        if communicator.rank == root_rank:
            assert self.__data is not None

            replica = communicator.size

            if not root_keep_data:
                replica -= 1
                assert replica >= 1, f'{self.__name} has nothing todo'

            samples_per_replica = self.__samples // replica
            samples_remain = self.__samples - samples_per_replica * replica

            for i in range(1, communicator.size):
                # 主进程先收听其他进程拥有数据情况
                msg = Message.listen(communicator)
                sender = msg.sender
                msg_obj = json.loads(msg.msg)

                begin = samples_per_replica * (
                    sender if root_keep_data else sender - 1)

                if begin != 0:
                    begin += samples_remain

                end = min(begin + samples_per_replica, self.__samples)
                # 多余的样例总是分给负责第一份数据的流水线

                if begin == 0:
                    end += samples_remain

                data_indexes = msg_obj['data_indexes']
                assert all(each < self.__data_count for each in data_indexes)

                info(f'sender: {sender}, begin: {begin}, end: {end}')
                if begin == msg_obj['begin'] and end == msg_obj['end']:
                    # sender进程已经拥有了所需的数据
                    info(f'communicator: {communicator.id}'
                         f' rank {sender} already got data no dispatching')
                    Message.send(
                        json.dumps({'set': False}),
                        sender, communicator
                    )
                else:
                    info(f'communicator: {communicator.id}'
                         f' rank {sender} have not got'
                         f' data do dispatch')
                    Message.send(
                        json.dumps({
                            'set': True, 'begin': begin, 'end': end,
                            'data_count': self.__data_count
                        }),
                        sender, communicator
                    )
                    # 将数据字节发送过去
                    for each in data_indexes:
                        Message.send_bytes(
                            self.__data[each][begin:end].dumps(),
                            sender, communicator
                        )
            if root_keep_data:
                keep_data = []
                for i in range(len(self.__data)):
                    keep_data.append(self.__data[i][
                                     0:samples_per_replica + samples_remain])
                self.__data = tuple(keep_data)
        else:
            # 从进程先发送自己所拥有的数据的部分给主进程, -1 代表没有
            if self.__begin is None or self.__end is None:
                info(f'missing require data, request data from root process')
                Message.send(
                    json.dumps({
                        'begin': -1, 'end': -1, 'data_indexes': data_indexes}),
                    root_rank, communicator
                )
            else:
                info(f'got require data, notify root process')
                Message.send(
                    json.dumps({
                        'begin': self.__begin, 'end': self.__end,
                        'data_indexes': data_indexes
                    }),
                    root_rank, communicator
                )
            # 再收听主进程发来的数据补充
            msg = Message.listen(communicator)
            assert msg.sender == root_rank
            msg_obj = json.loads(msg.msg)
            assert 'set' in msg_obj.keys()
            if msg_obj['set']:
                # 如果set为True, 那么表明本进程所拥有的数据需要被重置, 或者写入缓存路径中
                self.__begin = msg_obj['begin']
                self.__end = msg_obj['end']
                data_count = msg_obj['data_count']
                # 接收发送字节
                self.__data = [None for _ in range(data_count)]
                for each in data_indexes:
                    msg = Message.listen_bytes(communicator)
                    self.__data[each] = pickle.loads(msg.msg)
                if self.__data_save_path is not None:
                    # 需要更新缓存数据
                    kwargs = {
                        'meta-info': np.array([
                            data_count, self.__begin, self.__end]),
                    }
                    for each in data_indexes:
                        kwargs[f'data-{each}'] = self.__data[each]
                    np.savez(self.__data_save_path, **kwargs)
                info(f'got require data')

        self.__dispatched = True
