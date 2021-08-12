from ddl.data import DistributedData
from ddl.tensorflow.communicator import Communicator
from ddl.tensorflow.keras.parallelism.pipeline.model import PipelineModel
from ddl.message import Message
import tensorflow as tf
import numpy as np
import random
import json
from argparse import ArgumentParser

parser = ArgumentParser()

parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--micro_batch_size', type=int, default=32)
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--in_graph_mode', action='store_true')
parser.add_argument('--lr_warm_up_epochs', type=int, default=5)
parser.add_argument('--lr', type=float, default=0.001)

arguments = parser.parse_args()

world = Communicator.world()
batch_size = arguments.batch_size
micro_batch_size = arguments.micro_batch_size
epochs = arguments.epochs
seed = arguments.seed
lr_warm_up_epochs = arguments.lr_warm_up_epochs
lr = arguments.lr

if arguments.in_graph_mode:
    print('execute in graph')
    tf.compat.v1.disable_eager_execution()
else:
    print('execute in eager')

tf.random.set_seed(seed)
np.random.seed(seed)
random.seed(seed)


class MnistDistributedData(DistributedData):
    def __init__(self, test: bool, label: bool):
        shape = () if label else (28, 28)
        super().__init__(single_sample_shape=shape, communicator=world)
        self.__data = None
        self.__label = label
        self.__test = test

    def _total_sample_getter(self) -> int:
        return 10000 if self.__test else 60000

    def get(self, begin: int, end: int = None, step: int = None):
        assert self.__data is not None
        # 由于我们在_initialized的实现中只保留了这个进程只会访问到的部分，
        # 而get的begin和end是所有数据的下标，所以需要减去sample_begin, 也就是
        # self.in_charge_of[0]来映射到该进程所保留的数据部分
        begin -= self.in_charge_of[0]
        end -= self.in_charge_of[0]
        return self.__data[begin:end:step, ...]

    def _initialize_data(self, sample_begin: int, sample_end: int) -> None:
        (train_data, train_label), (test_data, test_label) = \
            tf.keras.datasets.mnist.load_data(path='original-mnist.npz')
        if self.__test:
            self.__data = test_label if self.__label else test_data
        else:
            self.__data = train_label if self.__label else train_data
        if not self.__label:
            self.__data = self.__data / 255.0
        self.__data = self.__data[sample_begin:sample_end]


def evaluate(model: PipelineModel):
    result = model.predict(
        MnistDistributedData(test=True, label=False),
        batch_size=batch_size, verbose=1,
        micro_batch_size=micro_batch_size
    )

    label = MnistDistributedData(test=True, label=True)

    pipeline_counts = model.model_communicator.size // model.processes_require

    total_acc = 0

    if model.pipeline_communicator.rank == \
            model.pipeline_communicator.size - 1:
        # 流水线的最后一个阶段的结果才是需要的分类结果
        predict = np.argmax(result, axis=-1)
        label.distribute(need_data=True)
        # 这一步主要是为了转换类型, np.where不能识别我们自己定义的类
        local_samples = len(label)
        total_samples = label.total_samples
        label_numpy = label[0:local_samples]
        assert len(label_numpy) == len(predict)

        corrects = len(np.where(label_numpy == predict)[0])

        if world.rank != 0:
            Message.send(json.dumps({
                'acc': corrects / total_samples,
                'pipeline': model.pipeline_model_rank
            }), 0, world)
        else:
            # 如果 world 0是流水线的最后一个阶段（一般并不会这样设计），
            # 那么就不用接收自己的数据，只需要接收其他流水线最后阶段发来的数据
            total_acc = corrects / total_samples
            print(f'pipeline {model.pipeline_model_rank}'
                  f' accuracy contribution: {total_acc}')
            for _ in range(pipeline_counts - 1):
                msg = json.loads(Message.listen(world).msg)
                total_acc += msg['acc']
                print(f'pipeline {msg["pipeline"]}'
                      f' accuracy contribution: {msg["acc"]}')
    else:
        # 注意distribute方法需要label的所有通信域内都要调用，否则会无法完成该动作
        label.distribute(need_data=False)
        if world.rank == 0:
            # 如果 world 0不是流水线的最后一个阶段，
            # 需要接受所有流水线最后阶段发来的准确率数据，包括自身所处的流水线
            for _ in range(pipeline_counts):
                msg = json.loads(Message.listen(world).msg)
                total_acc += msg['acc']
                print(f'pipeline {msg["pipeline"]}'
                      f' accuracy contribution: {msg["acc"]}')
    if world.rank == 0:
        print(f'test accuracy: {total_acc}')
