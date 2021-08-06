from ddl.tensorflow.communicator import Communicator
from ddl.log import info
from ddl.message import Message
from abc import ABCMeta, abstractmethod
from typing import Union
import numpy as np
import json


class Data(metaclass=ABCMeta):
    @property
    def samples(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def get(self, begin: int, end: int = None, step: int = None):
        pass

    def __getitem__(self, item):
        if isinstance(item, int):
            return self.get(item)
        elif isinstance(item, slice):
            return self.get(item.start, item.stop, item.step)
        else:
            raise ValueError(f'not supported item type {type(item)}')


class DistributedData(Data, metaclass=ABCMeta):
    def __init__(
            self,
            single_sample_shape: tuple,
            communicator: Communicator = Communicator.world()
    ):
        self.__base = 0
        self.__communicator = communicator
        self.__work = False
        self.__distributed = False
        self.__distributed_comm = None
        self.__samples = None
        self.__single_sample_shape = single_sample_shape
        self.__has_data = False

    @property
    def communicator(self) -> Communicator:
        return self.__communicator

    @property
    def shape(self) -> tuple:
        return (self.samples, *self.__single_sample_shape)

    @property
    def has_data(self) -> bool:
        return self.__has_data

    @property
    def samples(self) -> int:
        assert self.__distributed
        assert self.__samples is not None
        return self.__samples

    def distribute(self, need_data: bool, key: int):
        assert not self.__distributed
        self.__distributed_comm = self.__communicator.split_communicator(
            1 if need_data else 0, key=key)
        if need_data:
            total_samples = self._initialize_data()
            self.__has_data = True
            size = self.__distributed_comm.size
            rank = self.__distributed_comm.rank

            info(f'need data: found initialized data total samples: '
                 f'{total_samples} '
                 f'do data total_samples allgather')
            if size > 1:
                # todo: 懒得写Message版的allgather，先用broadcast代替
                samples_allgathered = []
                for root in range(size):
                    msg = Message.broadcast(json.dumps({
                        'samples': total_samples
                    }), 0, self.__distributed_comm
                    )
                    obj = json.loads(msg.msg)
                    if root != rank:
                        samples = obj['samples']
                        samples = 0 if samples is None else samples
                    else:
                        samples = total_samples
                    samples_allgathered.append(samples)
                samples_allgathered = np.array(samples_allgathered)

                info(f'total_samples allgathered: {samples_allgathered}')
                assert isinstance(samples_allgathered, np.ndarray)
                info(
                    f'total_samples allgathered shape: '
                    f'{samples_allgathered.shape}')
                assert len(samples_allgathered) == size

                for i in range(size):
                    if samples_allgathered[i] != 0:
                        if total_samples is None:
                            total_samples = samples_allgathered[i]
                        else:
                            assert \
                                total_samples == samples_allgathered[i], \
                                f'not the same total data samples: \n' \
                                f'local: {total_samples}, ' \
                                f'rank {i} in distributed communicator: ' \
                                f'{samples_allgathered[i]}'
            else:
                info('distributed communicator size == 1 no allgather')
            assert total_samples is not None, 'all processes cannot find' \
                                              ' total data samples'
            assert 0 <= rank < size
            self.__samples = total_samples // size

            self.__base = self.__samples * rank
            if rank == size - 1:
                remain = total_samples - self.__samples * size
                self.__samples += remain
        self.__distributed = True

    def _initialize_data(self) -> Union[None, int]:
        """
        数据初始化回调函数: 当节点确认需要这个数据时会调用这个函数，可以用来加载数据
        @return:
        """
        return None

    def __getitem__(self, item):
        assert self.__distributed
        if isinstance(item, int):
            return self.get(item + self.__base)
        elif isinstance(item, slice):
            start = None if item.start is None else item.start + self.__base
            stop = None if item.start is None else item.stop + self.__base
            return self.get(start, stop, item.step)
        else:
            raise ValueError(f'not supported item type {type(item)}')

    def __len__(self):
        return self.samples
