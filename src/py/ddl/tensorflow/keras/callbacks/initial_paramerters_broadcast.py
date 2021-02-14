from tensorflow.keras.callbacks import Callback
from ddl.tensorflow.util import executing_eagerly
from ddl.tensorflow.communicate import broadcast_by_group, \
    broadcast_global_variables
import tensorflow as tf


class InitialParametersBroadcastCallBack(Callback):
    """
    参数初始化回调函数: 将根节点的参数广播至其余节点
    """

    def __init__(self, root_rank: int):
        super().__init__()
        self.__root_rank = root_rank
        self.__initial_broadcast_done = False

    def on_batch_begin(self, batch, logs=None):
        if self.__initial_broadcast_done:
            return

        with tf.device(''):
            if executing_eagerly() and hasattr(self.model, 'variables'):
                # TensorFlow 2.0 or TensorFlow eager
                broadcast_by_group(
                    self.model.variables, root_rank=self.__root_rank
                )
                broadcast_by_group(
                    self.model.optimizer.variables(),
                    root_rank=self.__root_rank
                )
            else:
                broadcast_global_variables(self.__root_rank)

        self.__initial_broadcast_done = True
