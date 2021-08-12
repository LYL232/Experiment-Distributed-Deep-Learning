from ddl.tensorflow.cpp_backend import CPPBackend
from ddl.tensorflow import util
from ddl.tensorflow.communicator import Communicator
from tensorflow.python.framework.ops import Tensor
from collections.abc import Iterable
import tensorflow as tf


def allreduce_gradient(
        tensor: Tensor or tf.IndexedSlices,
        communicator: Communicator,
        return_mean: bool = True,
        key: str = ''
):
    """
    对梯度张量进行全规约操作, 根据tensor是否是稀疏张量选择进行allreduce还是allgather操作
    @param tensor: 梯度张量
    @param communicator: 通信域
    @param return_mean: 是否返回平均值
    @param key: 张量的关键字，用于区分不同的allreduce请求
    @return:
    """
    if isinstance(tensor, Tensor):
        summed = allreduce(tensor, communicator, key=key)
        if return_mean:
            return summed / communicator.size
        return summed
    if isinstance(tensor, tf.IndexedSlices):
        summed = allreduce(tf.convert_to_tensor(tensor), communicator, key=key)
        if return_mean:
            return summed / communicator.size
        return summed
        # todo: 现有bug，暂不支持tf.IndexedSlices
        # values = allgather(tensor.values, communicator)
        # if return_mean:
        #     values = values / communicator.size
        # indices = allgather(tensor.indices, communicator)
        # return tf.IndexedSlices(values, indices, dense_shape=tensor.dense_shape)
    else:
        raise ValueError(f'not supported gradient tensor type: {type(tensor)}')


def allreduce(tensor: Tensor, communicator: Communicator, key: str = ''):
    """
    对Tensor进行Allreduce操作
    @param tensor:
    @param communicator: 进行allreduce操作的通信域
    @param key: 张量的关键字，用于区分不同的allreduce请求
    @return:
    """
    return CPPBackend.tf_lib().allreduce(
        tensor, communicator_id=communicator.id, key=key
    )


def allgather(tensor: Tensor, communicator: Communicator, key: str = ''):
    """
    对Tensor进行allgather操作
    @param tensor:
    @param communicator: 进行操作的通信域
    @param key: 张量的关键字，用于区分不同的allgather请求
    @return:
    """
    return CPPBackend.tf_lib().allgather(
        tensor, communicator_id=communicator.id, key=key
    )


def broadcast(
        tensor: Tensor, root_rank: int, communicator: Communicator,
        key: str = ''):
    """
    对tensor进行广播操作
    @param tensor:
    @param root_rank: 广播根节点
    @param communicator: 广播通信域
    @param key: 张量的关键字，用于区分不同的broadcast请求
    @return:
    """
    return CPPBackend.tf_lib().broadcast(
        tensor, root_rank=root_rank, communicator_id=communicator.id,
        key=key
    )


@tf.function
def broadcast_by_group_eagerly(
        tensors: Iterable, root_rank: int,
        communicator: Communicator):
    for tensor in tensors:
        tensor.assign(
            broadcast(
                tensor, root_rank, communicator, key=f'Broadcast/{tensor.name}')
        )


def broadcast_by_group(
        tensors: Iterable, root_rank: int,
        communicator: Communicator):
    if util.executing_eagerly():
        # Eager mode will parallelize independent control flow
        return broadcast_by_group_eagerly(tensors, root_rank, communicator)
    else:
        # Graph mode requires an Op
        return tf.group(
            *[tensor.assign(
                broadcast(
                    tensor, root_rank, communicator,
                    key=f'Broadcast/{tensor.name}'
                )
            )
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


def broadcast_global_variables(root_rank: int, communicator: Communicator):
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
        __global_variables(), root_rank, communicator
    ))
