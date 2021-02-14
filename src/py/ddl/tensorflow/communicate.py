from ddl.tensorflow import Global
from ddl.tensorflow import util
from tensorflow.python.framework.ops import Tensor
from collections.abc import Iterable
import tensorflow as tf


def allreduce(tensor: Tensor):
    """
    对Tensor进行Allreduce操作
    :param tensor:
    :return:
    """
    return Global.tf_lib().allreduce(tensor)


def broadcast(tensor: Tensor, root_rank: int):
    """
    对tensor进行广播操作
    :param tensor:
    :param root_rank: 广播根节点
    :return:
    """
    return Global.tf_lib().broadcast(tensor, root_rank=root_rank)


@tf.function
def broadcast_by_group_eagerly(tensors: Iterable, root_rank):
    for tensor in tensors:
        tensor.assign(broadcast(tensor, root_rank))


def broadcast_by_group(tensors: Iterable, root_rank):
    if util.executing_eagerly():
        # Eager mode will parallelize independent control flow
        return broadcast_by_group_eagerly(tensors, root_rank)
    else:
        # Graph mode requires an Op
        return tf.group(
            *[tensor.assign(
                broadcast(tensor, root_rank))
                for tensor in tensors]
        )


try:
    # noinspection PyUnresolvedReferences
    __global_variables = tf.compat.v1.global_variables
except AttributeError:
    try:
        # noinspection PyUnresolvedReferences
        __global_variables = tf.global_variables
    except AttributeError:
        __global_variables = None


def broadcast_global_variables(root_rank: int):
    if util.executing_eagerly():
        raise RuntimeError(
            'broadcast_global_variables(root_rank)'
            ' does not support eager execution. '
            'Please use `broadcast_by_group(tensor, root_rank)` instead.'
        )
    if __global_variables is None:
        raise RuntimeError(
            'broadcast_global_variables(root_rank):'
            ' tensorflow global variables not found'
        )
    tf.compat.v1.keras.backend.get_session().run(broadcast_by_group(
        __global_variables(), root_rank
    ))
