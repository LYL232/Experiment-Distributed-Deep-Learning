from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
import tensorflow as tf


def main():
    from ddl.tensorflow.keras.parallelism.pipeline.model import PipelineModel, \
        PipelineStage
    from ddl.tensorflow.keras.parallelism.pipeline import \
        DensePipelineInputLayer
    from ddl.tensorflow.data_dispatcher import DataDispatcher
    from ddl.tensorflow.communicator import Communicator

    tf.compat.v1.disable_eager_execution()

    # 原模型定义:
    # [
    #   Flatten(input_shape=(28, 28)),
    #   Dense(784, activation='relu'),
    #   Dense(196, activation='relu'),
    #   Dense(128, activation='relu', name='dense-0'),
    #   Dense(256, activation='relu', name='dense-1'),
    #   Dense(10, activation='softmax', name='dense-2')
    # ]
    model = PipelineModel([
        PipelineStage(
            lambda: Sequential([
                Flatten(input_shape=(28, 28)),
                Dense(784, activation='relu', name='dense-0'),
            ])),
        PipelineStage(
            lambda: Sequential([
                DensePipelineInputLayer(
                    196, activation='relu', name='dense-0',
                    input_shape=(784,)
                ),
                Dense(128, activation='relu', name='dense-1'),
            ])),
        PipelineStage(
            lambda: Sequential([
                # 这里需要手动输入上一Stage的输出shape, 后期可以考虑由程序自行推断
                # 但是涉及通信, 需要进行模型定义之后才能传输各自模型的输出模型
                DensePipelineInputLayer(
                    256, activation='relu', name='dense-0',
                    input_shape=(128,)
                ),
                Dense(10, activation='softmax', name='dense-1')
            ]))
    ])

    world = Communicator.world()
    if world.rank == 0:
        (data, label), _ = tf.keras.datasets.mnist.load_data(
            path='original-mnist.npz')
        data = data / 255.0
        data = DataDispatcher(0, world, root_data=data)
        label = DataDispatcher(0, world, root_data=label)
    else:
        data = DataDispatcher(
            0, world, data_save_path=f'data-cache-{world.rank}.npz')
        label = DataDispatcher(
            0, world, data_save_path=f'label-cache-{world.rank}.npz')

    # 设置`experimental_run_tf_function=False` 让TensorFlow
    # 使用opt计算梯度
    # noinspection PyTypeChecker
    model.compile(
        loss=tf.losses.SparseCategoricalCrossentropy(),
        optimizer=tf.optimizers.Adam(0.001),
        metrics=['accuracy'],
        try_data_parallelism=True
    )

    model.fit(
        x=data, y=label,
        batch_size=1000, micro_batch_size=100,
        epochs=5, verbose=1
    )


if __name__ == '__main__':
    import sys
    from os.path import abspath, join

    sys.path.append(abspath(join(__file__, '../../../')))

    from ddl.tensorflow.data_dispatcher import DataDispatcher
    from ddl.tensorflow.communicator import Communicator

    main()
