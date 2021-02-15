from tensorflow.keras.optimizers import Optimizer
from abc import ABC


class PipelineDistributedOptimizer(Optimizer, ABC):
    """
    流水线分布式优化器, 拦截最后一个梯度的变量交给
    """

    def __init__(self, pipeline_gradients_back_propagation=None, **kwargs):
        self.__name = f'LYL232/experiment/' \
                      f'PipelineDistributedOptimizer/' \
                      f'{self.__class__.__base__.__name__}'
        self.__pgbp = pipeline_gradients_back_propagation
        self.__propagated = False
        super(self.__class__, self).__init__(**kwargs)

    def get_gradients(self, loss, params):
        return self.__pipeline_gradients_back_propagation(
            super(self.__class__, self).get_gradients(loss, params)
        )

    def _aggregate_gradients(self, grads_and_vars):
        grads, variables = list(zip(*grads_and_vars))
        return self.__pipeline_gradients_back_propagation(grads)

    def apply_gradients(self, *args, **kwargs):
        results = super(self.__class__, self).apply_gradients(*args, **kwargs)

        if self.__pgbp and not self.__propagated:
            raise Exception(
                '`apply_gradients()` was called without a call to '
                '`get_gradients()` or `_aggregate_gradients`. If you\'re '
                'using TensorFlow 2.0, please specify '
                '`experimental_run_tf_function=False` in `compile()`.'
            )

        return results

    def set_pipeline_gradients_back_propagation(self, func):
        self.__pgbp = func

    def __pipeline_gradients_back_propagation(self, gradients):
        if self.__pgbp is not None:
            self.__pgbp(gradients)
            self.__propagated = True
        return gradients


def pipeline_distributed_optimizer_wrapper(
        optimizer: Optimizer,
        pipeline_gradients_back_propagation=None
):
    cls = type(optimizer.__class__.__name__, (optimizer.__class__,),
               dict(PipelineDistributedOptimizer.__dict__))
    assert issubclass(cls, Optimizer)
    res = cls.from_config(optimizer.get_config())
    res.set_pipeline_gradients_back_propagation(
        pipeline_gradients_back_propagation
    )
    return res
