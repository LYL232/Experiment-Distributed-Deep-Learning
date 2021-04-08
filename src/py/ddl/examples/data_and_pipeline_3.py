from tensorflow.keras.layers import Dense, Flatten
import tensorflow as tf


def main():
    from ddl.tensorflow.keras.parallelism.pipeline.model import PipelineModel, \
        PipelineStage
    from ddl.tensorflow.data_dispatcher import DataDispatcher
    from ddl.tensorflow.communicator import Communicator
    from ddl.tensorflow.keras.models.model import Sequential

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
            Sequential([
                Flatten(input_shape=(28, 28)),
                Dense(784, activation='relu', name='dense-0'),
            ])),
        PipelineStage(
            Sequential([
                Dense(196, activation='relu', name='dense-0',),
                Dense(128, activation='relu', name='dense-1'),
            ])),
        PipelineStage(
            Sequential([
                Dense(256, activation='relu', name='dense-0'),
                Dense(10, activation='softmax', name='dense-1')
            ]))
    ])

    # 进行数据分发(可选)
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
        # 尝试进行数据并行
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

    main()
