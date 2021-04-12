from tensorflow.keras.layers import Dense, Flatten, Reshape, Conv2D, \
    MaxPooling2D, Dropout, concatenate
import tensorflow as tf


def main():
    from ddl.tensorflow.keras.parallelism.pipeline.model import PipelineModel, \
        PipelineStage
    from ddl.tensorflow.data_dispatcher import DataDispatcher
    from ddl.tensorflow.communicator import Communicator
    from ddl.tensorflow.keras.models.model import Model

    tf.compat.v1.disable_eager_execution()

    # 原模型定义:
    # inputs = Input(shape=(28, 28))
    # branch_0 = Reshape((28, 28, 1))(inputs)
    # branch_0 = Conv2D(8, [3, 3], activation='relu')(branch_0)
    # branch_0 = Conv2D(16, [3, 3], activation='relu')(branch_0)
    # branch_0 = MaxPooling2D(pool_size=(2, 2))(branch_0)
    # branch_0 = Dropout(0.25)(branch_0)
    # branch_0 = Flatten()(branch_0)
    #
    # branch_1 = Dense(32, activation='relu')(inputs)
    # branch_1 = Dense(64, activation='relu')(branch_1)
    # branch_1 = Dropout(0.5)(branch_1)
    # branch_1 = Flatten()(branch_1)
    #
    # merged = concatenate([branch_0, branch_1])
    # outputs = Dense(10, activation='softmax')(merged)
    #
    # model = Model(inputs=inputs, outputs=outputs)

    def first_stage_model(inputs):
        branch_0 = Reshape((28, 28, 1))(inputs)
        branch_1 = Dense(32, activation='relu')(inputs)
        return branch_0, branch_1

    def second_stage_model(branch_0, branch_1):
        branch_0 = Conv2D(8, [3, 3], activation='relu')(branch_0)
        branch_0 = Conv2D(16, [3, 3], activation='relu')(branch_0)
        branch_0 = MaxPooling2D(pool_size=(2, 2))(branch_0)
        branch_0 = Dropout(0.25)(branch_0)
        branch_0 = Flatten()(branch_0)

        branch_2 = Dense(32, activation='relu')(branch_0)

        branch_1 = Dense(64, activation='relu')(branch_1)
        branch_1 = Dropout(0.5)(branch_1)
        branch_1 = Flatten()(branch_1)

        return branch_0, branch_1, branch_2

    def third_stage_model(branch_0, branch_1, branch_2):
        merged = concatenate([branch_0, branch_1, branch_2])
        outputs = Dense(10, activation='softmax')(merged)
        return outputs

    model = PipelineModel([
        # 第一层需要有Input张量
        PipelineStage(Model(first_stage_model, input_shape=(28, 28))),
        PipelineStage(Model(second_stage_model)),
        PipelineStage(Model(third_stage_model))
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
