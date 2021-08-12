from ddl.tensorflow.cpp_backend import CPPBackend
from ctypes import create_string_buffer
from enum import Enum
from tensorflow.keras.callbacks import Callback


class TimeUnit(Enum):
    SEC = 0  # 秒
    MS = 1  # 毫秒
    US = 2  # 微秒
    NS = 3  # 纳秒


class LogType:
    __default = None
    __must = None

    def __init__(self, level: int, desc: str = None):
        if desc is not None:
            desc = ': ' + desc
        else:
            desc = ''
        self.__desc = desc
        self.__level = level

    @property
    def level(self) -> int:
        return self.__level

    @property
    def tag(self) -> int:
        return id(self)

    @classmethod
    def default(cls) -> 'LogType':
        if cls.__default is None:
            cls.__default = cls(0, 'Default')
        return cls.__default

    @classmethod
    def must(cls) -> 'LogType':
        if cls.__must is None:
            cls.__must = cls(0, 'Must')
        return cls.__must

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return f'LogType-{self.tag}{self.__desc}'


class Log:
    __config = {
        LogType.default().tag: True,
        LogType.must().tag: True
    }
    __level = 0

    @classmethod
    def get_level(cls) -> int:
        return cls.__level

    @classmethod
    def set_level(cls, level: int) -> None:
        assert isinstance(level, int)
        cls.__level = level

    @classmethod
    def get_type_visible(cls, t: LogType):
        return cls.__config[t.tag]

    @classmethod
    def set_type_visible(cls, t: LogType, visible: bool):
        cls.__config[t.tag] = visible

    @classmethod
    def new_log_type(cls, level: int, default: bool = True, desc: str = None) \
            -> LogType:
        new_t = LogType(level=level, desc=desc)
        cls.__config[new_t.tag] = default
        return new_t

    @classmethod
    def is_logging_type(cls, t: LogType) -> bool:
        return t.tag == LogType.must().tag or (
                cls.__config[t.tag] and cls.__level >= t.level)

    @classmethod
    def time_log(
            cls, msg: str,
            unit: TimeUnit,
            log_type: LogType = LogType.default()
    ) -> None:
        if not cls.is_logging_type(log_type):
            return
        if unit == TimeUnit.SEC:
            CPPBackend.c_api().py_sec_time_log(
                create_string_buffer(bytes(msg, encoding='UTF-8')),
            )
        elif unit == TimeUnit.MS:
            CPPBackend.c_api().py_ms_time_log(
                create_string_buffer(bytes(msg, encoding='UTF-8')),
            )
        elif unit == TimeUnit.US:
            CPPBackend.c_api().py_us_time_log(
                create_string_buffer(bytes(msg, encoding='UTF-8')),
            )
        elif unit == TimeUnit.NS:
            CPPBackend.c_api().py_ns_time_log(
                create_string_buffer(bytes(msg, encoding='UTF-8')),
            )
        else:
            raise ValueError(f'unknown time unit: {unit}')

    @classmethod
    def info(cls, msg: str, log_type: LogType = LogType.default()) -> None:
        if not cls.is_logging_type(log_type):
            return
        CPPBackend.c_api().py_info(
            create_string_buffer(bytes(msg, encoding='UTF-8')),
        )

    @classmethod
    def debug(cls, msg: str, log_type: LogType = LogType.default()) -> None:
        if not cls.is_logging_type(log_type):
            return
        CPPBackend.c_api().py_debug(
            create_string_buffer(bytes(msg, encoding='UTF-8')),
        )

    @classmethod
    def error(cls, msg: str, log_type: LogType = LogType.default()) -> None:
        if not cls.is_logging_type(log_type):
            return
        CPPBackend.c_api().py_error(
            create_string_buffer(bytes(msg, encoding='UTF-8')),
        )


class TimeLogCallback(Callback):
    """
    记录时间回调函数
    """

    def on_train_batch_begin(self, batch, logs=None):
        Log.time_log(f'batch {batch} begin', TimeUnit.MS)

    def on_train_batch_end(self, batch, logs=None):
        Log.time_log(f'batch {batch} end', TimeUnit.MS)

    def on_epoch_begin(self, epoch, logs=None):
        Log.time_log(f'epoch {epoch} begin', TimeUnit.MS)

    def on_epoch_end(self, epoch, logs=None):
        Log.time_log(f'epoch {epoch} end', TimeUnit.MS)


def exception_with_world_rank_info(fun):
    def wrapper(*args, **kwargs):
        try:
            return fun(*args, **kwargs)
        except Exception as e:
            from ddl.tensorflow.communicator import Communicator
            import sys
            import traceback
            exc_type, exc_value, exc_traceback = sys.exc_info()
            sys.stderr.write(
                f'Exception in world-{Communicator.world().rank}\n'
            )
            Log.error(f'Unhandled Exception {exc_type}: {exc_value}\n'
                      f'{traceback.format_exc()}', log_type=LogType.must())
            raise e

    return wrapper
