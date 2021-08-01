from tensorflow.keras.layers import Dense, Flatten, Conv2D, \
    MaxPooling2D, Dropout, Reshape
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow as tf


def main():
    from ddl.tensorflow.keras.parallelism.pipeline.model import \
        PipelineSequentialModel
    from ddl.tensorflow.keras.parallelism.pipeline.stage import PipelineStage

    # 原模型定义:
    # [
    #     Reshape((28, 28, 1)),
    #     Conv2D(32, [3, 3], activation='relu'),
    #     Conv2D(64, [3, 3], activation='relu'),
    #     MaxPooling2D(pool_size=(2, 2)),
    #     Dropout(0.25),
    #     Flatten(),
    #     Dense(128, activation='relu'),
    #     Dropout(0.5),
    #     Dense(10, activation='softmax')
    # ]

    class Stage0(PipelineStage):
        def __init__(self):
            super().__init__(output_num=1)

        def call(self, inputs):
            res = Reshape((28, 28, 1))(inputs)
            res = Conv2D(32, [3, 3], activation='relu')(res)
            return res

    class Stage1(PipelineStage):
        def __init__(self):
            super().__init__(output_num=1)

        def call(self, inputs):
            res = Conv2D(64, [3, 3], activation='relu')(inputs)
            res = MaxPooling2D(pool_size=(2, 2))(res)
            res = Dropout(0.25)(res)
            res = Flatten()(res)
            return res

    class Stage2(PipelineStage):
        def __init__(self):
            super().__init__(output_num=1)

        def call(self, inputs):
            res = Dense(128, activation='relu')(inputs)
            res = Dropout(0.5)(res)
            res = Dense(10, activation='softmax')(res)
            return res

    model = PipelineSequentialModel(
        [Stage0(), Stage1(), Stage2()],
        input_shape=(28, 28))

    data = MnistDistributedTrainData()
    label = MnistDistributedTrainLabel()

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
                filepath='./3stages_checkpoints/',
                # 目前只支持只保存权重
                save_weights_only=True,
            )
        ]
    )

    model.save_weights('./3stages_finished')

    evaluate(model)


if __name__ == '__main__':
    import sys
    from os.path import abspath, join

    sys.path.append(abspath(join(__file__, '../../../')))

    from ddl.examples.pipeline_common import MnistDistributedTrainData, \
        MnistDistributedTrainLabel, evaluate, batch_size, micro_batch_size, \
        epochs

    main()
