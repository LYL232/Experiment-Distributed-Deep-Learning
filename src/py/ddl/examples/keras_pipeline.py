"""
通过多进程模拟分布式环境并实现分布式流水线并行训练tensorflow2模型

通信方式: 进程间通过Pipe管道进行通信, 与Mpi通信方式相似

实现思路: 将keras_pipeline_baseline.py 中定义的模型拆分到两个独立的进程上运行,
    这两个模型分别称为FirstStage(包含原始模型的输入)和SecondStage(包含原始模型的输出)
    这两个模型使用同一种优化器,
    当前向传播时, SecondStage请求FirstStage按照batch_size取出数据,
    完成FirstStage的前向传播计算, 将其输出作为输入, 完成前向传播,
    当后向传播时, SecondStage的优化器即将应用(apply)梯度时, 也即模型的权重即将改变时, 将
    计算出的偏差(其实不止偏差)传输到FirstStage, FirstStage获得偏差, 使用偏差计算其loss值,
    再调用其优化器对FirstStage模型进行优化
"""

import tensorflow as tf
import numpy as np
from multiprocessing import Process, Pipe
from multiprocessing.connection import Connection
from threading import Condition

batch_size = 200

# 总共的样例个数, 写到这里是因为SecondStage模型并不需要载入数据
samples = 60000

lr = 0.001

epochs = 5

# Pipe发送的消息类型枚举定义
# SecondStage请求FirstStage取出数据计算前向传播, 并等待其传输结果
GET_FORWARD_PROPAGATION = 0
# FirstStage返回其前项传播的结果给SecondStage
FORWARD_PROPAGATION_RESULT = 1
# SecondStage将后向传播给FirstStage的偏差数据
GRADIENTS_BACK_PROPAGATION = 2
# 完成训练
DONE = 3

# 是否打印log到标准输出
verbose = False

# FirstStage模型是否使用正确的loss函数
first_stage_with_correct_loss = True


def log(msg: str):
    if verbose:
        print(msg, flush=True)


class FirstStageProcess(Process):
    def __init__(self, pipe: Connection):
        """
        虚假的初始化函数: 因为Process在start的时候会调用pickle序列化self,
        然而有很多成员变量无法被初始化, 所以在run的时候才进行初始化
        :param pipe: 进行通信的管道对象
        """
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
        """
        真正的初始化函数: 因为Process在start的时候会调用pickle序列化self,
        然而有很多成员变量无法被初始化, 所以在run的时候才进行初始化
        """
        gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

        self.__model = tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(784, activation='relu'),
            tf.keras.layers.Dense(196, activation='relu'),
        ])

        def correct_intermediate_loss(diff, output):
            """
            正确的FirstStage loss函数, 此函数的训练效果理论上与原模型的训练效果一致,
            其实就是偏差与输出的Hadamard积, 即按对应元素相乘
            :param diff: 从SecondStage后向传播得到的偏差数组
            :param output: 模型的输出
            :return: loss
            """
            return tf.reduce_mean(
                tf.multiply(diff, output),
                axis=None
            )

        def wrong_intermediate_loss(diff, output):
            """
            错误的FirstStage loss函数, 此函数的训练效果会导致模型loss先减后增,
            因为先进行平均会导致过多的信息损失, 从而计算出错误的优化方向
            :param diff: 从SecondStage后向传播得到的偏差数组
            :param output: 模型的输出
            :return: loss
            """
            return tf.reduce_mean(
                tf.multiply(
                    tf.reduce_mean(diff, axis=0),
                    tf.reduce_mean(output, axis=0)),
                axis=None
            )

        (self.__data, self.__label), _ = tf.keras.datasets.mnist.load_data(
            path='mnist.npz')

        self.__data = self.__data / 255.0

        loss = correct_intermediate_loss if first_stage_with_correct_loss else \
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
        """
        完成前向传播并将结果传输给SecondStage
        :param begin: 数据batch在整个数据库的开始索引
        :param end: 数据batch在整个数据库的结束索引
        :return: None
        """
        inputs = self.__data[begin:end, ...]
        self.__last_forward_propagation_inputs = inputs
        inputs = self.__model.predict(inputs)
        labels = self.__label[begin:end, ...]
        log('FirstStage: sending forward propagation result')

        self.__pipe.send(
            (FORWARD_PROPAGATION_RESULT, inputs, labels)
        )

    def __wait_back_propagation_msg(self):
        """
        等待SecondStage完成后向传播并使用后向传播得到的偏差进行模型的训练
        :return: None
        """
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
            if first_stage_with_correct_loss:
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
        """
        虚假的初始化函数: 因为Process在start的时候会调用pickle序列化self,
        然而有很多成员变量无法被初始化, 所以在run的时候才进行初始化
        :param pipe: 进行通信的管道对象
        """
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
        """
        真正的初始化函数: 因为Process在start的时候会调用pickle序列化self,
        然而有很多成员变量无法被初始化, 所以在run的时候才进行初始化
        """
        import sys
        from os.path import abspath, join

        self.__pipe_cond: Condition = Condition()

        sys.path.append(abspath(join(__file__, '../../../')))

        from ddl.tensorflow.keras import pipeline_distributed_optimizer_wrapper

        gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

        self.__model = tf.keras.Sequential([
            # SecondStage的输入是FirstStage的输出
            tf.keras.layers.Flatten(input_shape=(196,)),
            tf.keras.layers.Dense(128, activation='relu', name='dense-0'),
            tf.keras.layers.Dense(256, activation='relu', name='dense-1'),
            tf.keras.layers.Dense(10, activation='softmax', name='dense-2')
        ])

        def pipeline_gradients_back_propagation(gradients):
            """
            优化器应用梯度前的回调函数, 需要将第一层的偏置的梯度传输给FirstStage
            :param gradients: 所有变量的梯度列表
            :return: None
            """
            if first_stage_with_correct_loss:
                # 在loss函数手动对每个样例计算得到的梯度, shape[0]=输入的样例个数
                biased_gradients = self.__first_layer_biased_gradients
            else:
                # 优化器传过来的梯度为该batch所有样例产生的梯度平均之后的梯度
                biased_gradients = gradients[1]
            weights = self.__model.get_layer('dense-0').get_weights()[0]
            # 需要与第一层的权重模型进行一次矩阵乘法
            diff = tf.matmul(
                tf.constant(weights),
                tf.transpose(
                    biased_gradients) if first_stage_with_correct_loss else
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
            if not first_stage_with_correct_loss:
                # 如果不是测试正确的loss函数, 直接返回即可
                return tf.reduce_mean(
                    sparse_categorical_crossentropy(target, output), axis=None
                )

            # FirstStage正确的loss需要所有样例产生的梯度, 而不是平均值

            from tensorflow.keras.backend import relu, softmax

            # tensorflow中没有提供接口, 只能手动暴力实现前向传播和后向传播的梯度计算

            inputs = self.__last_inputs

            # 获取模型的参数
            weights0 = self.__model.get_layer('dense-0').weights
            weights1 = self.__model.get_layer('dense-1').weights
            weights2 = self.__model.get_layer('dense-2').weights

            # 只需要记录第一层的偏置的梯度
            first_layer_biased_gradients = []

            for i in range(target.shape[0]):
                # 暴力计算每个样例对变量的梯度, 保存到
                # self.__first_layer_biased_gradients中
                var_list = []

                def __loss():
                    # 一个个样例计算前向传播, 优化器会根据这些计算自动求导
                    sample = inputs[i, ...].reshape((1, -1))
                    t = tf.reshape(target[i, ...], (1, -1))
                    z0 = tf.matmul(sample, weights0[0]) + weights0[1]
                    a0 = relu(z0)
                    z1 = tf.matmul(a0, weights1[0]) + weights1[1]
                    a1 = relu(z1)
                    z2 = tf.matmul(a1, weights2[0]) + weights2[1]
                    a2 = softmax(z2)
                    # a2的输出与output相等, 已经测试过
                    # 只需要监控第一层的偏置权重
                    var_list.append(weights0[1])
                    return sparse_categorical_crossentropy(t, a2)

                gs = opt._compute_gradients(__loss, var_list)
                var_list.clear()
                # 取出结果
                first_layer_biased_gradients.append(gs[0][0])
            # 保存结果等待优化器回调函数取出结果
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
        """
        从FirstStage获取前向传播的结果
        :param begin: 数据batch在整个数据库的开始索引
        :param end: 数据batch在整个数据库的结束索引
        :return: 模型batch的输入, batch的label
        """
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
        """
        模型训练的数据生成器
        :return: 数据获取迭代器
        """
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
