from ddl.data import DistributedData
from ddl.tensorflow.communicator import Communicator
from ddl.tensorflow.keras.parallelism.pipeline.model import PipelineModel
from ddl.message import Message
import tensorflow as tf
import pickle
import numpy as np

world = Communicator.world()
batch_size = 1000
micro_batch_size = 100
epochs = 5


class MnistDistributedTrainData(DistributedData):
    def __init__(self):
        super().__init__(60000, (28, 28), world)
        self.__data = None

    def get(self, begin: int, end: int = None, step: int = None):
        assert self.__data is not None
        return self.__data[begin:end:step, ...]

    def _initialize_data(self):
        (self.__data, _), _ = \
            tf.keras.datasets.mnist.load_data(path='original-mnist.npz')
        self.__data = self.__data / 255.0


class MnistDistributedTrainLabel(DistributedData):
    def __init__(self):
        super().__init__(60000, tuple(), world)
        self.__data = None

    def get(self, begin: int, end: int = None, step: int = None):
        assert self.__data is not None
        return self.__data[begin:end:step, ...]

    def _initialize_data(self):
        (_, self.__data), _ = \
            tf.keras.datasets.mnist.load_data(path='original-mnist.npz')


class MnistDistributedTestData(DistributedData):
    def __init__(self):
        super().__init__(10000, (28, 28), world)
        self.__data = None

    def get(self, begin: int, end: int = None, step: int = None):
        assert self.__data is not None
        return self.__data[begin:end:step, ...]

    def _initialize_data(self):
        _, (self.__data, _) = \
            tf.keras.datasets.mnist.load_data(path='original-mnist.npz')
        self.__data = self.__data / 255.0


def evaluate(model: PipelineModel):
    result = model.predict(
        MnistDistributedTestData(),
        batch_size=batch_size, micro_batch_size=micro_batch_size, verbose=1)

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
