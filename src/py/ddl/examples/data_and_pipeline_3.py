from tensorflow.keras.layers import Dense, Flatten, Conv2D, \
    MaxPooling2D, Dropout, Reshape
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
    #     Reshape(input_shape=(28, 28), target_shape=(28, 28, 1)),
    #     Conv2D(32, [3, 3], activation='relu'),
    #     Conv2D(64, [3, 3], activation='relu'),
    #     MaxPooling2D(pool_size=(2, 2)),
    #     Dropout(0.25),
    #     Flatten(),
    #     Dense(128, activation='relu'),
    #     Dropout(0.5),
    #     Dense(10, activation='softmax')
    # ]

    model = PipelineModel([
        PipelineStage(
            Sequential([
                Reshape(input_shape=(28, 28), target_shape=(28, 28, 1)),
                Conv2D(32, [3, 3], activation='relu'),

            ])),
        PipelineStage(
            Sequential([
                Conv2D(64, [3, 3], activation='relu'),
                MaxPooling2D(pool_size=(2, 2)),
                Dropout(0.25),
                Flatten()
            ])
        ),
        PipelineStage(
            Sequential([
                Dense(128, activation='relu'),
                Dropout(0.5),
                Dense(10, activation='softmax')
            ])
        )
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
