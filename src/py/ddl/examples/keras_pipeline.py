# 通过多进程模拟分布式环境并实现分布式流水线并行训练tensorflow2模型

import tensorflow as tf
import numpy as np
from multiprocessing import Process, Pipe
from multiprocessing.connection import Connection
from threading import Condition

batch_size = 200

samples = 60000

lr = 0.001

epochs = 24

GET_FORWARD_PROPAGATION = 0
FORWARD_PROPAGATION_RESULT = 1
GRADIENTS_BACK_PROPAGATION = 2
DONE = 3

verbose = 0

test_correct_loss = True


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

        def correct_intermediate_loss(diff, output):
            return tf.reduce_mean(
                tf.multiply(diff, output),
                axis=None
            )

        def wrong_intermediate_loss(diff, output):
            return tf.reduce_mean(
                tf.multiply(
                    tf.reduce_mean(diff, axis=0),
                    tf.reduce_mean(output, axis=0)),
                axis=None
            )

        (self.__data, self.__label), _ = tf.keras.datasets.mnist.load_data(
            path='mnist.npz')

        self.__data = self.__data / 255.0

        loss = correct_intermediate_loss if test_correct_loss else \
                   wrong_intermediate_loss,

        self.__model.compile(
            loss=loss,
            optimizer=tf.optimizers.Adam(lr),
            metrics=[loss],
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
            if test_correct_loss:
                diff = msg[1].T
            else:
                # 将diff按照输入的样例个数(input.shape[0])复制到维度0, 因为
                # keras的fit会检查输入和输出的shape[0]是否一致
                diff = np.expand_dims(msg[1], 0).repeat(
                    self.__last_forward_propagation_inputs.shape[0],
                    0
                )
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
        self.__pipe_cond = None
        self.__status = self.STATUS_INIT
        self.__first_layer_biased_gradients = None

    def run(self) -> None:
        log('SecondStage: started, initializing')
        self.__init()
        log('SecondStage: started, initialized')
        self.__model.fit(
            self.__second_stage_data_generator(),
            steps_per_epoch=samples // batch_size,
            epochs=epochs, verbose=1
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
            tf.keras.layers.Dense(128, activation='relu', name='dense-0'),
            tf.keras.layers.Dense(256, activation='relu', name='dense-1'),
            tf.keras.layers.Dense(10, activation='softmax', name='dense-2')
        ])

        def pipeline_gradients_back_propagation(gradients):

            if test_correct_loss:
                # 在loss函数手动对每个样例计算得到的梯度, shape[0]=输入的样例个数
                biased_gradients = self.__first_layer_biased_gradients
            else:
                # 优化器传过来的梯度为该batch所有样例产生的梯度平均之后的梯度
                biased_gradients = gradients[1]
            weights = self.__model.get_layer('dense-0').get_weights()[0]
            diff = tf.matmul(
                tf.constant(weights),
                tf.transpose(biased_gradients) if test_correct_loss else
                tf.expand_dims(biased_gradients, axis=1)
            ).numpy()
            log(
                'SecondStage: sending back propagation gradients to FirstStage'
            )

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

        opt = tf.optimizers.Adam(lr)

        def loss(target, output):
            from tensorflow.keras.losses import sparse_categorical_crossentropy
            if not test_correct_loss:
                return tf.reduce_mean(
                    sparse_categorical_crossentropy(target, output), axis=None
                )

            from tensorflow.keras.backend import relu, softmax

            inputs = self.__last_inputs

            weights0 = self.__model.get_layer('dense-0').weights
            weights1 = self.__model.get_layer('dense-1').weights
            weights2 = self.__model.get_layer('dense-2').weights

            first_layer_biased_gradients = []

            for i in range(target.shape[0]):
                # 暴力计算每个样例对变量的权重, 保存到中
                # self.__first_layer_biased_gradients
                var_list = []

                def __loss():
                    sample = inputs[i, ...].reshape((1, -1))
                    t = tf.reshape(target[i, ...], (1, -1))
                    z0 = tf.matmul(sample, weights0[0]) + weights0[1]
                    a0 = relu(z0)
                    z1 = tf.matmul(a0, weights1[0]) + weights1[1]
                    a1 = relu(z1)
                    z2 = tf.matmul(a1, weights2[0]) + weights2[1]
                    a2 = softmax(z2)
                    var_list.append(weights0[1])
                    return sparse_categorical_crossentropy(t, a2)

                gs = opt._compute_gradients(__loss, var_list)
                var_list.clear()
                first_layer_biased_gradients.append(gs[0][0])
            self.__first_layer_biased_gradients = np.array(
                first_layer_biased_gradients
            )
            return tf.reduce_mean(
                sparse_categorical_crossentropy(target, output), axis=None
            )

        self.__model.compile(
            loss=loss,
            optimizer=pipeline_distributed_optimizer_wrapper(
                opt,
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
            inputs, labels = self.__get_forward_propagation(begin, end)
            self.__last_inputs = inputs
            begin = 0 if end >= samples else begin + batch_size
            yield inputs, labels


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
