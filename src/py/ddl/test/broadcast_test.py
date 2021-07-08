import tensorflow as tf
import numpy as np


def main():
    from ddl.tensorflow.communicator import Communicator
    from ddl.tensorflow.tensor_communicate import broadcast

    tf.compat.v1.disable_eager_execution()

    # 进行数据分发(可选)
    world = Communicator.world()
    tensor = tf.constant(np.zeros((16,)) + world.rank + 1, dtype=tf.float32)
    res = broadcast(tensor, 3, world)
    with tf.compat.v1.Session() as session:
        res = session.run(res)
    print(f'rank: {world.rank}, {res}')


if __name__ == '__main__':
    import sys
    from os.path import abspath, join

    sys.path.append(abspath(join(__file__, '../../../')))

    main()
