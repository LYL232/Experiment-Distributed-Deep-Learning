from ddl.tensorflow.cpp_backend import CPPBackend


class Communicator:
    """
    表示一个通信域的类
    """
    # 缓存全体进程所在通信域的信息
    __world = None

    def __init__(self, communicator_id: int):
        """
        @param communicator_id: 其实是一个整数, 传入c_api中, C后端将会将其转换成指针再处理
         python端将其看作一个整数即可
        """
        self.__id = communicator_id
        # 懒加载
        self.__rank = None
        self.__size = None

    @property
    def id(self):
        return self.__id

    @property
    def rank(self) -> int:
        if self.__rank is None:
            self.__rank = CPPBackend.c_api().communicator_rank(self.id)
        return self.__rank

    @property
    def size(self) -> int:
        if self.__size is None:
            self.__size = CPPBackend.c_api().communicator_size(self.id)
        return self.__size

    def split_communicator(self, color: int, key: int = None) -> 'Communicator':
        """
        通过着色分割本通信域, 需要通信域内所有进程参与, 并且给出自己的颜色, 颜色相同的在同一个
        通信域
        @param color: 颜色
        @param key: 控制本进程在新通信域内的rank, 按key的顺序从0开始到新通信域的大小 - 1,
         如果为None, 则以本进程在本通信域内的rank作为key
        @return: 本进程所属的新的通信域
        """
        if key is None:
            key = self.rank
        return Communicator(
            CPPBackend.c_api().split_communicator(self.id, color, key))

    @classmethod
    def world(cls) -> 'Communicator':
        """
        获取所有进程所在的通信域
        @return:
        """
        if cls.__world is None:
            cls.__world = cls(CPPBackend.c_api().world_communicator())
        return cls.__world
