from ddl.tensorflow.communicator import Communicator
from ddl.log import info
from ddl.message import Message
from abc import ABCMeta, abstractmethod
import json


class Data(metaclass=ABCMeta):
    def __init__(self, name: str = None):
        if name is None:
            name = f'{self.__class__.__name__}-{id(self)}'
        self.__name = name

    @property
    def name(self) -> str:
        return self.__name

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
            communicator: Communicator = Communicator.world(),
            total_sample_compute_process_rank: int = 0,
            name: str = None
    ):
        self.__base = 0
        self.__end = 0
        self.__total_samples = 0
        self.__communicator = communicator
        self.__work = False
        self.__distributed = False
        self.__distributed_comm = None
        self.__samples = None
        self.__single_sample_shape = single_sample_shape
        self.__has_data = False
        assert total_sample_compute_process_rank < communicator.size
        self.__total_sample_compute_process_rank = \
            total_sample_compute_process_rank
        if name is None:
            name = f'{self.__class__.__name__}-{id(self)}'
        super().__init__(name=name)

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
        """
        注意，这个属性返回的是该进程所分配到的数据的个数，而不是所有数据的个数，
        所有数据的个数是total_samples
        """
        assert self.__distributed
        return self.__samples if self.has_data else 0

    @property
    def in_charge_of(self) -> (int, int):
        assert self.__distributed
        if self.has_data:
            return self.__base, self.__end
        return -1, -1

    @property
    def total_samples(self) -> int:
        """
        在需要数据的进程的数据被实际加载前需要知道的所有样例的数量
        """
        assert self.__distributed
        assert self.has_data
        return self.__total_samples

    def distribute(self, need_data: bool, key: int = None):
        assert not self.__distributed
        if key is None:
            key = self.__communicator.rank
        assert key >= 0
        is_total_samples_load_process = \
            self.__total_sample_compute_process_rank == self.__communicator.rank
        # 保证负责广播所有样例数的节点在新的通信域下的rank是0
        key = 0 if is_total_samples_load_process else key + 1
        self.__distributed_comm = self.__communicator.split_communicator(
            1 if need_data else 0, key=key)
        if need_data:
            if self.__distributed_comm.rank == 0:
                total_samples = self._total_sample_getter()
                Message.broadcast(json.dumps({
                    'samples': total_samples
                }), 0, self.__distributed_comm)
            else:
                msg = Message.broadcast('', 0, self.__distributed_comm)
                total_samples = json.loads(msg.msg)['samples']
            self.__total_samples = total_samples

            size = self.__distributed_comm.size
            rank = self.__distributed_comm.rank

            info(f'need data, total samples: {total_samples}')

            assert 0 <= rank < size
            self.__samples = total_samples // size

            self.__base = self.__samples * rank
            if rank == size - 1:
                remain = total_samples - self.__samples * size
                self.__samples += remain
            self.__end = self.__base + self.__samples
            self._initialize_data(self.__base, self.__end)
            self.__has_data = True
        self.__distributed = True

    @abstractmethod
    def _total_sample_getter(self) -> int:
        pass

    def _initialize_data(self, sample_begin: int, sample_end: int) -> None:
        """
        数据初始化回调函数: 当节点确认需要这个数据时会调用这个函数，可以用来加载数据，保证
        在这个进程中只会访问[sample_begin, sample_end)的部分
        @param sample_begin: 该进程中所有get请求访问的sample的下标的下界
        @param sample_end: 该进程中所有get请求访问的sample的下标的上届（不包含）
        @return:
        """
        return None

    def __getitem__(self, item):
        assert self.__distributed
        assert self.has_data, \
            f'{self.name} in rank {self.__communicator.rank} has not got data'
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
