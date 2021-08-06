from ddl.tensorflow.communicator import Communicator
from ddl.tensorflow.tensor_communicate import allreduce
from ddl.tensorflow.util import executing_eagerly
from tensorflow.keras.callbacks import Callback
from tensorflow.python.keras import backend
import tensorflow as tf


class MetricAverageCallback(Callback):
    """
    每个进程之间的Metric的求平均回调函数, 参考Horovod
    """

    def __init__(
            self, communicator: Communicator = Communicator.world()
    ):
        super().__init__()
        self.variables = {}
        self.allreduce_ops = {}
        self.__communicator = communicator

    def _make_variable(self, metric, value):
        with tf.name_scope('MetricAverageCallback'):
            # todo: 这个name_scope没用，name还是只有一个Allreduce
            var = backend.variable(value, name=metric)
            backend.get_session().run(var.initializer)
            allreduce_op = allreduce(var, communicator=self.__communicator)
            return var, allreduce_op

    def _average_metrics_in_place(self, logs):
        logs = logs or {}
        reduced_logs = {}
        # Reduce every metric among workers. Sort metrics by name
        # to ensure consistent order.
        for metric, value in sorted(logs.items()):
            if executing_eagerly():
                reduced_logs[metric] = \
                    allreduce(
                        backend.constant(value, name=metric),
                        self.__communicator
                    ).numpy()
            else:
                if metric not in self.variables:
                    self.variables[metric], self.allreduce_ops[metric] = \
                        self._make_variable(metric, value)
                else:
                    backend.set_value(self.variables[metric], value)
                reduced_logs[metric] = \
                    backend.get_session().run(self.allreduce_ops[metric])
        # Override the reduced values back into logs dictionary
        # for other callbacks to use.
        for metric, value in reduced_logs.items():
            logs[metric] = value / self.__communicator.size

    def on_epoch_end(self, epoch, logs=None):
        if self.__communicator.size > 1:
            # 仅在需要通信域大小大于1时进行平均运算
            self._average_metrics_in_place(logs)
