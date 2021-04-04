import os
import sys
import tensorflow as tf
import json
import numpy as np

log_file = open('pylog-0.log', 'w')

next_stage_rank = 1


def log(msg: str, flush: bool = True):
    from ddl.examples.keras_pipeline_mpi.main import verbose
    if verbose:
        log_file.write(msg + '\n')
        if flush:
            log_file.flush()


class FirstStageModel:
    def __init__(self):
        from ddl.examples.keras_pipeline_mpi.main import lr
        from ddl.tensorflow.cpp_backend import CPPBackend
        from ddl.tensorflow.communicator import Communicator

        world = Communicator.world()

        log('FirstStage: started, initializing')

        self.__model = tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(784, activation='relu'),
            tf.keras.layers.Dense(196, activation='relu'),
        ])

        def loss(error, output):
            """
            loss函数, 此函数的训练效果理论上与原模型的训练效果一致,
            其实就是偏差与输出的Hadamard积, 即按对应元素相乘
            :@param diff: 从SecondStage后向传播得到的偏差数组
            :@param output: 模型的输出
            :@return: loss
            """
            return tf.reduce_mean(
                tf.multiply(error, output),
                axis=None
            )

        (self.__data, self.__label), _ = tf.keras.datasets.mnist.load_data(
            path='mnist.npz')

        self.__data = self.__data / 255.0

        self.__model.compile(
            loss=loss,
            optimizer=tf.optimizers.Adam(lr),
            metrics=[loss],
            experimental_run_tf_function=False
        )
        log('first stage model:')
        self.__model.summary(print_fn=log)

        self.__done = False

        self.__session = tf.compat.v1.Session()

        # 定义静态计算图
        self.__send_fwd_outputs_placeholder = tf.compat.v1.placeholder(
            dtype=tf.float32, shape=(None, 196))
        self.__send_fwd_outputs = CPPBackend.tf_lib().send_tensor(
            self.__send_fwd_outputs_placeholder,
            receiver=next_stage_rank,
            name='0-forward-input-to-1',
            communicator_id=world.id
        )

        self.__send_fwd_labels_placeholder = tf.compat.v1.placeholder(
            dtype=tf.float32, shape=None)
        with tf.control_dependencies([self.__send_fwd_outputs]):
            self.__send_fwd_labels = CPPBackend.tf_lib().send_tensor(
                self.__send_fwd_labels_placeholder,
                receiver=next_stage_rank,
                name='0-forward-label-to-1',
                communicator_id=world.id
            )

        self.__receive_diff_placeholder = tf.compat.v1.placeholder(
            dtype=tf.float32, shape=(None, 196))
        self.__receive_diff = CPPBackend.tf_lib().receive_tensor(
            self.__receive_diff_placeholder,
            sender=next_stage_rank,
            communicator_id=world.id
        )

        log('FirstStage: started, initialized')

    def run(self) -> None:
        from ddl.examples.keras_pipeline_mpi.main import \
            MSG_GET_FORWARD_PROPAGATION, MSG_DONE
        from ddl.tensorflow.message import Message

        while not self.__done:
            log(
                'FirstStage: waiting for getting forward propagation'
                ' or done message'
            )
            msg: Message = Message.listen()
            log(
                f'FirstStage: got for getting forward propagation'
                f' or done message {msg.msg}'
            )
            msg_obj = json.loads(msg.msg)
            if msg_obj['type'] == MSG_GET_FORWARD_PROPAGATION:
                begin = msg_obj['begin']
                end = msg_obj['end']
                self.__send_forward_propagation_result(begin, end)
            elif msg_obj['type'] == MSG_DONE:
                return
            else:
                raise Exception(
                    f'receive unexpected message: msg: {msg.msg}, '
                    f'sender: {msg.sender}'
                )
            self.__wait_back_propagation_msg()
        log('FirstStage: returning')

    def __send_forward_propagation_result(self, begin: int, end: int):
        """
        完成前向传播并将结果传输给SecondStage
        :@param begin: 数据batch在整个数据库的开始索引
        :@param end: 数据batch在整个数据库的结束索引
        :@return: None
        """
        from ddl.examples.keras_pipeline_mpi.main import \
            MSG_FORWARD_PROPAGATION_RESULT
        from ddl.tensorflow.message import Message
        from ddl.tensorflow.cpp_backend import CPPBackend
        from ddl.tensorflow.util import executing_eagerly
        from ddl.tensorflow.communicator import Communicator

        world = Communicator.world()

        inputs = self.__data[begin:end, ...]
        self.__last_fwd_inputs = inputs
        outputs = self.__model.predict(inputs)
        labels = self.__label[begin:end, ...]

        log('FirstStage: sending forward propagation message')
        # todo: 可以写到op里包装到一个层里
        Message.send(
            json.dumps({'type': MSG_FORWARD_PROPAGATION_RESULT}),
            next_stage_rank
        )
        log(f'FirstStage: sending forward propagation result: [{begin}, {end}]')
        if executing_eagerly():
            CPPBackend.tf_lib().send_tensor(
                tf.constant(outputs, dtype=tf.float32),
                receiver=next_stage_rank,
                name='0-forward-input-to-1',
                communicator_id=world.id
            )
            CPPBackend.tf_lib().send_tensor(
                tf.constant(labels, dtype=tf.float32),
                receiver=next_stage_rank,
                name='0-forward-label-to-1',
                communicator_id=world.id
            )
        else:
            self.__session.run(
                [self.__send_fwd_outputs, self.__send_fwd_labels],
                feed_dict={
                    self.__send_fwd_outputs_placeholder: outputs,
                    self.__send_fwd_labels_placeholder: labels
                }
            )

        log(f'FirstStage: sent forward propagation result: [{begin}, {end}]')

    def __wait_back_propagation_msg(self):
        """
        等待SecondStage完成后向传播并使用后向传播得到的偏差进行模型的训练
        :@return: None
        """
        from ddl.examples.keras_pipeline_mpi.main import MSG_DONE, \
            MSG_GRADIENTS_BACK_PROPAGATION, batch_size
        from ddl.tensorflow.cpp_backend import CPPBackend
        from ddl.tensorflow.communicator import Communicator
        from ddl.tensorflow.message import Message
        from ddl.tensorflow.util import executing_eagerly

        world = Communicator.world()

        log('FirstStage: waiting back propagation result or done message')
        msg = Message.listen()

        msg_obj = json.loads(msg.msg)
        if msg_obj['type'] == MSG_DONE:
            # 当fit结束时, 只会调用一次前向传播而没有后向传播
            # (个人猜测应该是最后一次更新模型梯度后再进行一次前向传播来计算准确率)
            self.__done = True
        elif msg_obj['type'] == MSG_GRADIENTS_BACK_PROPAGATION:
            assert self.__last_fwd_inputs is not None

            if executing_eagerly():
                diff = CPPBackend.tf_lib().receive_tensor(tf.zeros(shape=(
                    self.__last_fwd_inputs.shape[0],
                    196
                )),
                    sender=next_stage_rank,
                    communicator_id=world.id
                )
            else:
                diff = self.__session.run(
                    self.__receive_diff,
                    feed_dict={
                        self.__receive_diff_placeholder: np.zeros(
                            shape=(
                                self.__last_fwd_inputs.shape[0],
                                196
                            )
                        )
                    }
                )

            log(f'FirstStage: got back propagation result, shape={diff.shape}')

            self.__model.fit(
                self.__last_fwd_inputs, diff,
                batch_size=batch_size,
                epochs=1,
                verbose=0
            )
        else:
            raise Exception(f'got unexpected msg: {msg}')

    def __del__(self):
        if self.__session is not None:
            self.__session.close()


def main():
    sys.path.append(
        os.path.abspath(
            os.path.join(
                __file__,
                '../../../../'
            )
        )
    )

    tf.compat.v1.disable_eager_execution()
    try:
        FirstStageModel().run()
    finally:
        log_file.flush()
        log_file.close()
