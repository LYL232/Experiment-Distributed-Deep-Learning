from ddl.tensorflow import util
from ddl.tensorflow.tensor_communicate import allreduce_gradient
from ddl.tensorflow.communicator import Communicator
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
        self.__gradients_allreduced = False
        self.communicator: Communicator = Communicator.world()
        self.__allreduce_fn = None

    def get_gradients(self, loss, params):
        return self.__allreduce(self.original_get_gradients(loss, params))

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
        if len(grads) == 0:
            return grads
        self.__gradients_allreduced = False
        for each in grads:
            if each is not None:
                if util.executing_eagerly():
                    allreduced = getattr(each, 'done_allreduced', False)
                    self.__gradients_allreduced = allreduced
                else:
                    self.__gradients_allreduced = 'Allreduce' in each.name
                if self.__gradients_allreduced:
                    return grads

        opt_name = self.__name

        def allreduce_grads(gradients):
            with tf.name_scope(opt_name + 'Allreduce'):
                return [
                    tf.cond(
                        tf.convert_to_tensor(
                            self.communicator.size > 1
                        ),
                        lambda: allreduce_gradient(grad, self.communicator),
                        lambda: tf.convert_to_tensor(grad)
                        # allgather尚有bug，目前不支持
                        if isinstance(grad, tf.IndexedSlices) else grad,
                        name='if-do-allreduce'
                    )
                    if grad is not None else grad
                    for grad in gradients
                ]

        if util.executing_eagerly():
            if self.__allreduce_fn is None:
                self.__allreduce_fn = util.make_tf_function(allreduce_grads)
            # 因为eager模式下不能通过name来判断是否已经规约，所以设置一个属性
            for each in grads:
                setattr(each, 'done_allreduced', True)
        else:
            self.__allreduce_fn = allreduce_grads

        self.__gradients_allreduced = True
        return self.__allreduce_fn(grads)

    @property
    def is_distributed_optimizer(self) -> bool:
        """
        用来判断是否是分布式优化器
        @return:
        """
        return True

    def original_get_gradients(self, loss, params):
        """
        返回原本优化器被复写的get_gradients方法
        @param loss:
        @param params:
        @return:
        """
        return super(self.__class__, self).get_gradients(loss, params)

    def original_aggregate_gradients(self, grads_and_vars):
        """
        返回原本优化器被复写的_aggregate_gradients方法
        @param grads_and_vars:
        @return:
        """
        return super(self.__class__, self)._aggregate_gradients(grads_and_vars)


def data_parallelism_distributed_optimizer_wrapper(
        optimizer: Optimizer,
        communicator: Communicator = Communicator.world()) -> Optimizer:
    opt_cls = optimizer.__class__
    assert issubclass(opt_cls, Optimizer)
    cls = type(optimizer.__class__.__name__, (opt_cls,),
               dict(DataParallelismDistributedOptimizer.__dict__))
    assert issubclass(cls, Optimizer)
    config = optimizer.get_config()

    res = cls.from_config(config)

    res.communicator = communicator

    return res
