from ddl.tensorflow import Global, util, communicate
from tensorflow.keras.optimizers import Optimizer
from abc import ABC
import tensorflow as tf


class DataParallelismDistributedOptimizer(Optimizer, ABC):
    """
    数据并行分布式优化器: 每个节点的优化器都对数据进行计算, 计算出梯度后提交到后台Op请求进行全规约
    """

    def __init__(self, **kwargs):
        self.__name = f'LYL232/experiment/' \
                      f'DataParallelismDistributedOptimizer/' \
                      f'{self.__class__.__base__.__name__}'

        super(self.__class__, self).__init__(**kwargs)

    def get_gradients(self, loss, params):
        return self.__allreduce(
            super(self.__class__, self).get_gradients(loss, params)
        )

    def _aggregate_gradients(self, grads_and_vars):
        grads, variables = list(zip(*grads_and_vars))
        return self.__allreduce(grads)

    def apply_gradients(self, *args, **kwargs):
        results = super(self.__class__, self).apply_gradients(*args, **kwargs)

        if not self.__gradients_allreduced:
            raise Exception(
                '`apply_gradients()` was called without a call to '
                '`get_gradients()` or `_aggregate_gradients`. If you\'re '
                'using TensorFlow 2.0, please specify '
                '`experimental_run_tf_function=False` in `compile()`.'
            )

        return results

    def __allreduce(self, grads):
        def allreduce_grads():
            with tf.name_scope(self.__name + "Allreduce"):
                return [
                    tf.cond(
                        tf.convert_to_tensor(Global.processes() > 1),
                        lambda: communicate.allreduce(grad),
                        lambda: grad
                    )
                    if grad is not None else grad
                    for grad in grads
                ]

        if util.executing_eagerly():
            allreduce_grads = util.make_tf_function(allreduce_grads)

        self.__gradients_allreduced = True
        return allreduce_grads()


def data_parallelism_distributed_optimizer_wrapper(optimizer: Optimizer):
    cls = type(optimizer.__class__.__name__, (optimizer.__class__,),
               dict(DataParallelismDistributedOptimizer.__dict__))
    assert issubclass(cls, Optimizer)
    return cls.from_config(optimizer.get_config())
