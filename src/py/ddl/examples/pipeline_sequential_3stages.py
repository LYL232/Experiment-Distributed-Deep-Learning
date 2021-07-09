import numpy as np
from tensorflow.keras.layers import Dense, Flatten, Conv2D, \
    MaxPooling2D, Dropout, Reshape
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow as tf
import pickle


def main():
    from ddl.tensorflow.keras.parallelism.pipeline.model import \
        PipelineSequentialModel
    from ddl.tensorflow.keras.parallelism.pipeline.stage import PipelineStage

    from ddl.tensorflow.data_dispatcher import DataDispatcher
    from ddl.tensorflow.communicator import Communicator
    from ddl.message import Message

    tf.compat.v1.disable_eager_execution()

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
        def call(self, inputs):
            res = Reshape((28, 28, 1))(inputs)
            res = Conv2D(32, [3, 3], activation='relu')(res)
            return res

        @property
        def input_shape(self):
            return 28, 28

        @property
        def output_shape(self):
            return 26, 26, 32

    class Stage1(PipelineStage):
        def call(self, inputs):
            res = Conv2D(64, [3, 3], activation='relu')(inputs)
            res = MaxPooling2D(pool_size=(2, 2))(res)
            res = Dropout(0.25)(res)
            res = Flatten()(res)
            return res

        @property
        def input_shape(self):
            return 26, 26, 32

        @property
        def output_shape(self):
            return 9216

    class Stage2(PipelineStage):
        def call(self, inputs):
            res = Dense(128, activation='relu')(inputs)
            res = Dropout(0.5)(res)
            res = Dense(10, activation='softmax')(res)
            return res

        @property
        def input_shape(self):
            return 9216

        @property
        def output_shape(self):
            return 10

    model = PipelineSequentialModel([Stage0(), Stage1(), Stage2()])

    # 进行数据分发(可选)
    world = Communicator.world()
    if world.rank == 0:
        (data, label), (test_data, _) = \
            tf.keras.datasets.mnist.load_data(path='original-mnist.npz')
        data = data / 255.0
        test_data = test_data / 255.0
        data = DataDispatcher(0, world, root_data=data)
        label = DataDispatcher(0, world, root_data=label)
        test_data = DataDispatcher(0, world, root_data=test_data)
    else:
        data = DataDispatcher(
            0, world, data_save_path=f'data-cache-{world.rank}.npz')
        label = DataDispatcher(
            0, world, data_save_path=f'label-cache-{world.rank}.npz')
        test_data = DataDispatcher(
            0, world, data_save_path=f'test-data-cache-{world.rank}.npz')

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
        epochs=5, verbose=1,
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

    result = model.predict(
        test_data, batch_size=1000, micro_batch_size=100, verbose=1)

    if model.pipeline_communicator.rank == \
            model.pipeline_communicator.size - 1:
        # 流水线的最后一个阶段的结果才是需要的分类结果
        predict = np.argmax(result, axis=-1)
        Message.send_bytes(predict.dumps(), 0, world)

    # 最后将预测信息发送到 world 0节点进行精确度计算
    if world.rank == 0:
        pipeline_counts = model.model_communicator.size // \
                          model.processes_require
        predicts = [None for _ in range(pipeline_counts)]
        for _ in range(pipeline_counts):
            msg = Message.listen_bytes(world)
            pipeline_rank = msg.sender // model.processes_require
            predicts[pipeline_rank] = pickle.loads(msg.msg)
        predict = np.concatenate(predicts, axis=0)
        _, (_, test_label) = \
            tf.keras.datasets.mnist.load_data(path='original-mnist.npz')
        assert len(test_label) == len(predict)
        corrects = 0
        for i in range(len(test_label)):
            if test_label[i] == predict[i]:
                corrects += 1
        print(f'test accuracy: {corrects / float(len(test_label))}')


if __name__ == '__main__':
    import sys
    from os.path import abspath, join

    sys.path.append(abspath(join(__file__, '../../../')))

    main()
