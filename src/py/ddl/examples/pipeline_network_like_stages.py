from tensorflow.keras.layers import Dense, Flatten, Reshape, Conv2D, \
    MaxPooling2D, Dropout, concatenate
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
import numpy as np
import pickle


def main():
    from ddl.tensorflow.keras.parallelism.pipeline.model import \
        PipelineModel
    from ddl.tensorflow.keras.parallelism.pipeline.stage import PipelineStage
    from ddl.tensorflow.keras.parallelism.pipeline.pipe import PipelineInput
    from ddl.tensorflow.data_dispatcher import DataDispatcher
    from ddl.tensorflow.communicator import Communicator
    from ddl.message import Message

    tf.compat.v1.disable_eager_execution()

    class Stage0(PipelineStage):
        def call(self, inputs):
            branch_0 = Reshape((28, 28, 1))(inputs)
            branch_1 = Flatten()(inputs)
            branch_1 = Dense(32, activation='relu')(branch_1)
            return branch_0, branch_1

        @property
        def input_shape(self):
            return 28, 28

        @property
        def output_shape(self):
            return (28, 28, 1), (32,)

    class Stage1(PipelineStage):
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

        @property
        def input_shape(self):
            return (28, 28, 1), (32,)

        @property
        def output_shape(self):
            return (1152,), (64,)

    class Stage2(PipelineStage):
        def call(self, branch_0, branch_1):
            branch_2 = Dense(48, activation='relu')(branch_0)
            branch_3 = Dense(56, activation='relu')(branch_1)
            branch_4 = Dense(60, activation='relu')(branch_1)
            return branch_2, branch_3, branch_4

        @property
        def input_shape(self):
            return (1152,), (64,)

        @property
        def output_shape(self):
            return (48,), (56,), (60,)

    class Stage3(PipelineStage):
        def call(self, branch_0, branch_1, branch_2, branch_3, branch_4):
            merged = concatenate(
                [branch_0, branch_1, branch_2, branch_3, branch_4], axis=1)
            outputs = Dense(10, activation='softmax')(merged)
            return outputs

        @property
        def input_shape(self):
            return (1152,), (64,), (48,), (56,), (60,)

        @property
        def output_shape(self):
            return (10,)

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

    # 进行数据分发(可选)
    world = Communicator.world()
    if world.rank == 0:
        (data, label), (test_data, _) = tf.keras.datasets.mnist.load_data(
            path='original-mnist.npz')
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
                filepath='./network_like_stages_checkpoints/',
                # 目前只支持只保存权重
                save_weights_only=True,
            )
        ]
    )

    model.save_weights('./network_like_stages_finished')

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
