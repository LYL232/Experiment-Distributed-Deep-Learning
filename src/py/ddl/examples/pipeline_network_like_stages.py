from tensorflow.keras.layers import Dense, Flatten, Reshape, Conv2D, \
    MaxPooling2D, Dropout, concatenate
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint


def main():
    from ddl.tensorflow.keras.parallelism.pipeline.model import \
        PipelineModel
    from ddl.tensorflow.keras.parallelism.pipeline.stage import PipelineStage
    from ddl.tensorflow.keras.parallelism.pipeline.pipe import PipelineInput

    tf.compat.v1.disable_eager_execution()

    class Stage0(PipelineStage):
        def __init__(self):
            super().__init__(output_num=2)

        def call(self, inputs):
            branch_0 = Reshape((28, 28, 1))(inputs)
            branch_1 = Flatten()(inputs)
            branch_1 = Dense(32, activation='relu')(branch_1)
            return branch_0, branch_1

    class Stage1(PipelineStage):
        def __init__(self):
            super().__init__(output_num=2)

        def call(self, branch_0, branch_1):
            branch_0 = Conv2D(4, [3, 3], activation='relu')(branch_0)
            branch_0 = Conv2D(8, [3, 3], activation='relu')(branch_0)
            branch_0 = MaxPooling2D(pool_size=(2, 2))(branch_0)
            branch_0 = Dropout(0.25)(branch_0)
            branch_0 = Flatten()(branch_0)

            branch_1 = Dense(64, activation='relu')(branch_1)
            branch_1 = Dropout(0.5)(branch_1)
            branch_1 = Flatten()(branch_1)
            return branch_0, branch_1

    class Stage2(PipelineStage):
        def __init__(self):
            super().__init__(output_num=3)

        def call(self, branch_0, branch_1):
            branch_2 = Dense(48, activation='relu')(branch_0)
            branch_3 = Dense(56, activation='relu')(branch_1)
            branch_4 = Dense(60, activation='relu')(branch_1)
            return branch_2, branch_3, branch_4

    class Stage3(PipelineStage):
        def __init__(self):
            super().__init__(output_num=1)

        def call(self, branch_0, branch_1, branch_2, branch_3, branch_4):
            merged = concatenate(
                [branch_0, branch_1, branch_2, branch_3, branch_4], axis=1)
            outputs = Dense(10, activation='softmax')(merged)
            return outputs

    input_tensor = PipelineInput(shape=(28, 28))

    # noinspection PyCallingNonCallable,PyTypeChecker
    outputs_0 = Stage0()(input_tensor)
    # noinspection PyCallingNonCallable
    outputs_1 = Stage1()(*outputs_0)
    # noinspection PyCallingNonCallable
    outputs_2 = Stage2()(*outputs_1)
    # noinspection PyCallingNonCallable
    outputs_3 = Stage3()(*outputs_1, *outputs_2)

    model = PipelineModel(input_tensor, outputs_3)

    data = MnistDistributedTrainData()
    label = MnistDistributedTrainLabel()

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
        batch_size=batch_size, micro_batch_size=micro_batch_size,
        epochs=epochs, verbose=1,
        callbacks=[
            ModelCheckpoint(
                # todo: monitor
                filepath='./network_like_stages_checkpoints/',
                # 目前只支持只保存权重
                save_weights_only=True,
            )
        ]
    )

    model.save_weights('./network_like_stages_finished')

    evaluate(model)


if __name__ == '__main__':
    import sys
    from os.path import abspath, join

    sys.path.append(abspath(join(__file__, '../../../')))

    from ddl.examples.pipeline_common import MnistDistributedTrainData, \
        MnistDistributedTrainLabel, evaluate, batch_size, micro_batch_size, \
        epochs

    main()
