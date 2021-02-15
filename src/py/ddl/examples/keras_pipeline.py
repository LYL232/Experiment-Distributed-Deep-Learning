# 通过多进程模拟分布式环境并实现分布式流水线并行训练tensorflow2模型

import tensorflow as tf
import numpy as np
from multiprocessing import Process, Pipe
from multiprocessing.connection import Connection
from threading import Condition

batch_size = 200

samples = 60000

GET_FORWARD_PROPAGATION = 0
FORWARD_PROPAGATION_RESULT = 1
GRADIENTS_BACK_PROPAGATION = 2
DONE = 3

verbose = 0


def log(msg: str):
    if verbose != 0:
        print(msg, flush=True)


class FirstStageProcess(Process):
    def __init__(self, pipe: Connection):
        super().__init__()
        self.__pipe = pipe
        self.__model = None
        self.__data = None
        self.__label = None
        self.__last_forward_propagation_inputs = None
        self.__done = False

    def run(self) -> None:
        log('FirstStage: started, initializing')
        self.__init()
        log('FirstStage: started, initialized')
        while not self.__done:
            log(
                'FirstStage: waiting for getting forward propagation'
                ' or done message'
            )
            msg = self.__pipe.recv()
            log(
                f'FirstStage: got for getting forward propagation'
                f' or done message {msg}'
            )
            assert len(msg) == 3 or len(msg) == 1
            if msg[0] == GET_FORWARD_PROPAGATION:
                begin = msg[1]
                end = msg[2]
                self.__send_forward_propagation_result(begin, end)
            elif msg == DONE:
                return
            else:
                raise Exception(f'unexpected msg type: {msg[0]}')
            self.__wait_back_propagation_msg()
        log('FirstStage: returning')

    def __init(self):
        gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

        self.__model = tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(784, activation='relu'),
            tf.keras.layers.Dense(196, activation='relu'),
        ])

        def intermediate_loss(diff, output_tensor):
            return tf.reduce_sum(tf.multiply(output_tensor, diff), axis=None)

        (self.__data, self.__label), _ = tf.keras.datasets.mnist.load_data(
            path='mnist.npz')

        self.__data = self.__data / 255.0

        self.__model.compile(
            loss=intermediate_loss,
            optimizer=tf.optimizers.Adam(0.001),
            metrics=[intermediate_loss],
            experimental_run_tf_function=False
        )
        log('first stage model:')
        self.__model.summary()

    def __send_forward_propagation_result(self, begin: int, end: int):
        inputs = self.__data[begin:end, ...]
        self.__last_forward_propagation_inputs = inputs
        inputs = self.__model.predict(inputs)
        labels = self.__label[begin:end, ...]
        log('FirstStage: sending forward propagation result')

        # print('last_activation:')
        # print(inputs)
        # print('labels:')
        # print(labels)

        self.__pipe.send(
            (FORWARD_PROPAGATION_RESULT, inputs, labels)
        )

    def __wait_back_propagation_msg(self):
        log('FirstStage: waiting back propagation result')
        msg = self.__pipe.recv()
        log(f'FirstStage: got back propagation result')
        assert len(msg) == 2 or len(msg) == 1
        if len(msg) == 1:
            # 当fit结束时, 只会调用一次前向传播
            # (个人猜测应该是最后一次更新模型梯度后再进行一次前向传播来计算准确率)
            assert msg[0] == DONE
            self.__done = True
        else:
            assert msg[0] == GRADIENTS_BACK_PROPAGATION
            assert isinstance(msg[1], np.ndarray)
            assert self.__last_forward_propagation_inputs is not None
            diff = msg[1]
            self.__model.fit(
                self.__last_forward_propagation_inputs, diff,
                batch_size=batch_size,
                epochs=1,
                verbose=0
            )


class SecondStageProcess(Process):
    STATUS_INIT = 0
    STATUS_READY = 1
    STATUS_GOT_FWD = 2
    STATUS_SENT_BP = 3
    STATUS_DONE = 3

    def __init__(self, pipe: Connection):
        super().__init__()
        self.__pipe = pipe
        self.__model = None
        self.__last_activation = None
        self.__pipe_cond = None
        self.__status = self.STATUS_INIT

    def run(self) -> None:
        log('SecondStage: started, initializing')
        self.__init()
        log('SecondStage: started, initialized')
        self.__model.fit(
            self.__second_stage_data_generator(),
            steps_per_epoch=samples // batch_size,
            epochs=1, verbose=1
        )

        self.__pipe_cond.acquire()
        # 当fit结束后, 只会调用一次前向传播
        # (个人猜测应该是最后一次更新模型梯度后再进行一次前向传播来计算准确率)
        while self.__status != self.STATUS_GOT_FWD:
            self.__pipe_cond.wait()
        self.__pipe.send((DONE,))
        self.__status = self.STATUS_DONE
        log('SecondStage: done, sending Done to FirstStage')
        self.__pipe_cond.notify_all()
        self.__pipe_cond.release()
        log('SecondStage returning')

    def __init(self):
        import sys
        from os.path import abspath, join

        self.__pipe_cond: Condition = Condition()

        sys.path.append(abspath(join(__file__, '../../../')))

        from ddl.tensorflow.keras import pipeline_distributed_optimizer_wrapper

        gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

        self.__model = tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape=(196,)),
            tf.keras.layers.Dense(128, activation='relu', name='need_weights'),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax')
        ])

        def pipeline_gradients_back_propagation(gradients):
            assert self.__last_activation is not None
            assert isinstance(self.__last_activation, np.ndarray)
            weights = self.__model.get_layer('need_weights').get_weights()[0]
            diff = tf.keras.backend.dot(
                tf.constant(self.__last_activation),
                tf.constant(gradients[0])
            )
            diff = tf.keras.backend.dot(
                diff, tf.transpose(tf.constant(weights))
            ).numpy()

            log(
                'SecondStage: sending back propagation gradients to FirstStage'
            )

            # print('self.__last_activation:')
            # print(self.__last_activation)
            # print('gradients:')
            # print(gradients[0])
            # print('weights:')
            # print(weights)
            # print('diff:')
            # print(diff)
            # exit(0)
            self.__last_activation = None

            self.__pipe_cond.acquire()
            while self.__status != self.STATUS_GOT_FWD:
                self.__pipe_cond.wait()

            self.__pipe.send((
                GRADIENTS_BACK_PROPAGATION,
                diff
            ))

            self.__status = self.STATUS_SENT_BP
            self.__pipe_cond.notify_all()
            self.__pipe_cond.release()

        self.__model.compile(
            loss=tf.losses.SparseCategoricalCrossentropy(),
            optimizer=pipeline_distributed_optimizer_wrapper(
                tf.optimizers.Adam(0.001),
                pipeline_gradients_back_propagation
            ),
            metrics=['accuracy'],
            experimental_run_tf_function=False,
            run_eagerly=True
        )

        log('second stage model:')
        self.__model.summary()

        self.__pipe_cond.acquire()
        self.__status = self.STATUS_READY
        self.__pipe_cond.release()

    def __get_forward_propagation(self, begin, end) -> \
            (np.ndarray, np.ndarray):
        self.__pipe_cond.acquire()
        while self.__status != self.STATUS_READY and \
                self.__status != self.STATUS_SENT_BP:
            log(
                'SecondStage: '
                'waiting to forward propagation batch index'
                f'current status: {self.__status}'
            )
            self.__pipe_cond.wait()
        log('SecondStage: sending forward propagation batch index')
        self.__pipe.send((GET_FORWARD_PROPAGATION, begin, end))
        log('SecondStage: getting forward propagation result')
        msg = self.__pipe.recv()
        assert len(msg) == 3
        assert msg[0] == FORWARD_PROPAGATION_RESULT
        assert isinstance(msg[1], np.ndarray)
        assert isinstance(msg[2], np.ndarray)
        self.__status = self.STATUS_GOT_FWD
        self.__pipe_cond.notify_all()
        self.__pipe_cond.release()
        log(f'SecondStage: got forward propagation result')
        return msg[1], msg[2]

    def __second_stage_data_generator(self):
        begin = 0
        while True:
            end = min(begin + batch_size, samples)
            last_activation, labels = self.__get_forward_propagation(begin, end)
            begin = 0 if end >= samples else begin + batch_size
            self.__last_activation = last_activation
            yield last_activation, labels


def main():
    first_stage_pipe, second_stage_pipe = Pipe()
    first_stage_process = FirstStageProcess(first_stage_pipe)
    second_stage_process = SecondStageProcess(second_stage_pipe)
    first_stage_process.start()
    second_stage_process.start()
    first_stage_process.join()
    second_stage_process.join()
    first_stage_pipe.close()
    second_stage_pipe.close()


if __name__ == '__main__':
    main()
