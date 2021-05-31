import tensorflow as tf
import numpy as np


def main():
    from ddl.tensorflow.communicator import Communicator
    from ddl.tensorflow.tensor_communicate import allgather

    tf.compat.v1.disable_eager_execution()

    # 进行数据分发(可选)
    world = Communicator.world()
    tensor = tf.IndexedSlices(
        tf.constant(np.arange(4 + world.rank) + world.rank, dtype=tf.float32),
        tf.constant(np.array(
            [[0, 0], [1, 1], [2, 2], [3, 3]]) + world.rank, dtype=tf.float32),
        dense_shape=tf.constant([16, 16], dtype=tf.int64)
    )
    values = allgather(tensor.values, world)
    indices = allgather(tensor.indices, world)
    res = tf.IndexedSlices(values, indices, dense_shape=tensor.dense_shape)

    with tf.compat.v1.Session() as session:
        res = session.run(res)
    print(f'rank: {world.rank}, {res}')


if __name__ == '__main__':
    import sys
    from os.path import abspath, join

    sys.path.append(abspath(join(__file__, '../../../')))

    main()
