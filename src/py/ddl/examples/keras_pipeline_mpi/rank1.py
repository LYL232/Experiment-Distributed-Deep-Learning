import os
import sys
import numpy as np
import tensorflow as tf
import json
from threading import Condition
from tensorflow.python.framework.ops import Tensor
from tensorflow.python.framework.ops import RegisterGradient
# noinspection PyProtectedMember
from tensorflow.python.ops.nn_grad import _BiasAddGrad
from tensorflow.keras.callbacks import Callback

log_file = open('pylog-1.log', 'w')

last_stage_rank = 0


def log(msg: str, flush: bool = True):
    from ddl.examples.keras_pipeline_mpi.main import verbose
    if verbose:
        log_file.write(msg + '\n')
        if flush:
            log_file.flush()


class PipelineBackPropagationDense(tf.keras.layers.Dense):
    """
    分布式流水线并行中向上一层中进行误差后向传播的全连接层
    """

    def __init__(
            self, units, activation=None,
            kernel_initializer='glorot_uniform', bias_initializer='zeros',
            kernel_regularizer=None, bias_regularizer=None,
            activity_regularizer=None, kernel_constraint=None,
            bias_constraint=None,
            **kwargs):
        """
        @param 见Dense的文档
        """
        super(PipelineBackPropagationDense, self).__init__(
            units, activation, True, kernel_initializer,
            bias_initializer, kernel_regularizer,
            bias_regularizer, activity_regularizer,
            kernel_constraint, bias_constraint, **kwargs)

    def call(self, inputs):
        """
        替换原有的偏置加法OP, 让其正常完成加添置的同时, 将批次内的所有样例的偏差传递到上一个节点
        @param inputs: 见父类文档
        @return: 见父类文档
        """

        @RegisterGradient('DenseForwardAndSendGrad')
        def dense_forward_and_send_grad(op, received_grad: Tensor):
            """
            替换原有OP, 将其传给ForwardAndSendOp, 将上一阶段
            @param op: _BiasAddGrad的第一函数参数
            @param received_grad: 偏置应收到的每个样例产生的梯度
            @return: _BiasAddGrad returns
            """
            from ddl.tensorflow.global_class import Global
            from ddl.examples.keras_pipeline_mpi.main import \
                MSG_GRADIENTS_BACK_PROPAGATION
            # 需要进行一次矩阵乘法
            last_stage_errors = tf.transpose(tf.matmul(
                self.kernel,
                tf.transpose(received_grad)
            ))

            received_grad = Global.tf_lib().forward_and_send(
                received_grad, last_stage_errors, receiver=last_stage_rank,
                msg=json.dumps({
                    'type': MSG_GRADIENTS_BACK_PROPAGATION
                })
            )
            return _BiasAddGrad(op, received_grad)

        # 替换OP
        with tf.compat.v1.get_default_graph().gradient_override_map({
            'BiasAdd': 'DenseForwardAndSendGrad'
        }):
            return super(PipelineBackPropagationDense, self).call(inputs)


class TrainingEndCallBack(Callback):
    def __init__(self, model: 'SecondStageModel'):
        super().__init__()
        self.__model = model

    def on_train_batch_end(self, batch, logs=None):
        # 通知模型已经张量已经传输完毕, 或许可以通过op和C api进行通知, 但是有点复杂
        self.__model.wait_and_set_status(
            SecondStageModel.STATUS_GOT_FWD,
            SecondStageModel.STATUS_SENT_BP
        )


class SecondStageModel:
    STATUS_INIT = 0
    STATUS_READY = 1
    STATUS_GOT_FWD = 2
    STATUS_SENT_BP = 3
    STATUS_DONE = 3

    def __init__(self):
        from ddl.examples.keras_pipeline_mpi.main import lr
        from ddl.tensorflow.global_class import Global
        log('SecondStage: started, initializing')

        self.__communicate_cond: Condition = Condition()

        self.__model = tf.keras.Sequential([
            # SecondStage的输入是FirstStage的输出
            tf.keras.layers.Flatten(input_shape=(196,)),
            PipelineBackPropagationDense(
                128, activation='relu', name='dense-0'),
            tf.keras.layers.Dense(256, activation='relu', name='dense-1'),
            tf.keras.layers.Dense(10, activation='softmax', name='dense-2')
        ])

        self.__model.compile(
            loss='sparse_categorical_crossentropy',
            optimizer=tf.optimizers.Adam(lr),
            metrics=['accuracy'],
            experimental_run_tf_function=False,
            run_eagerly=False
        )

        log('second stage model:')
        self.__model.summary(print_fn=log)
        self.__communicate_cond.acquire()
        self.__status = self.STATUS_READY
        self.__communicate_cond.release()

        self.__session = tf.compat.v1.Session()

        self.__receive_fwd_outputs_placeholder = tf.compat.v1.placeholder(
            dtype=tf.float32, shape=(None, 196))
        self.__receive_fwd_outputs = Global.tf_lib().receive_tensor(
            self.__receive_fwd_outputs_placeholder,
            sender=last_stage_rank,
            name='0-forward-input-to-1'
        )
        self.__receive_fwd_labels_placeholder = tf.compat.v1.placeholder(
            dtype=tf.float32, shape=(None, 1))
        with tf.control_dependencies([self.__receive_fwd_outputs]):
            self.__receive_fwd_labels = Global.tf_lib().receive_tensor(
                self.__receive_fwd_labels_placeholder,
                sender=last_stage_rank,
                name='0-forward-label-to-1'
            )

        log('SecondStage: started, initialized')

    def run(self) -> None:
        from ddl.examples.keras_pipeline_mpi.main import samples, batch_size, \
            epochs, MSG_DONE
        from ddl.tensorflow.global_class import Global

        self.__model.fit(
            self.__second_stage_data_generator(),
            steps_per_epoch=samples // batch_size,
            epochs=epochs, verbose=1,
            callbacks=[TrainingEndCallBack(self)]
        )

        self.__communicate_cond.acquire()
        # 当fit结束后, 只会调用一次前向传播
        # (个人猜测应该是最后一次更新模型梯度后再进行一次前向传播来计算准确率)
        while self.__status != self.STATUS_GOT_FWD:
            self.__communicate_cond.wait()

        Global.send_message(json.dumps({
            'type': MSG_DONE
        }), last_stage_rank)
        self.__status = self.STATUS_DONE
        log('SecondStage: done, sending Done to FirstStage')
        self.__communicate_cond.notify_all()
        self.__communicate_cond.release()
        log('SecondStage returning')

    def wait_and_set_status(self, wait_status: int, set_status: int):
        self.__communicate_cond.acquire()
        while self.__status != wait_status:
            self.__communicate_cond.wait()
        self.__status = set_status
        self.__communicate_cond.notify_all()
        self.__communicate_cond.release()

    def __get_forward_propagation(self, begin, end) -> \
            (np.ndarray, np.ndarray):
        """
        从FirstStage获取前向传播的结果
        :param begin: 数据batch在整个数据库的开始索引
        :param end: 数据batch在整个数据库的结束索引
        :return: 模型batch的输入, batch的label
        """
        from ddl.examples.keras_pipeline_mpi.main import \
            MSG_GET_FORWARD_PROPAGATION, MSG_FORWARD_PROPAGATION_RESULT
        from ddl.tensorflow.global_class import Global
        from ddl.tensorflow.util import executing_eagerly
        from ddl.tensorflow.message import Message
        self.__communicate_cond.acquire()
        while self.__status != self.STATUS_READY and \
                self.__status != self.STATUS_SENT_BP:
            log(
                'SecondStage: '
                'waiting to forward propagation batch index '
                f'current status: {self.__status}'
            )
            self.__communicate_cond.wait()

        log('SecondStage: sending forward propagation batch index')

        Global.send_message(json.dumps({
            'type': MSG_GET_FORWARD_PROPAGATION,
            'begin': begin,
            'end': end
        }), last_stage_rank)

        msg: Message = Global.listen_message()
        msg_obj = json.loads(msg.msg)
        assert msg_obj['type'] == MSG_FORWARD_PROPAGATION_RESULT
        log('SecondStage: getting forward propagation result')

        input_shape = (end - begin, 196)
        label_shape = (end - begin, 1)

        if executing_eagerly():
            inputs_value = Global.tf_lib().receive_tensor(
                tf.zeros(shape=input_shape, dtype=tf.float32),
                sender=last_stage_rank,
                name='0-forward-input-to-1'
            )
            labels_value = Global.tf_lib().receive_tensor(
                tf.zeros(shape=label_shape, dtype=tf.float32),
                sender=last_stage_rank,
                name='0-forward-label-to-1'
            )
        else:
            inputs_value, labels_value = self.__session.run(
                [self.__receive_fwd_outputs, self.__receive_fwd_labels],
                feed_dict={
                    self.__receive_fwd_outputs_placeholder: np.zeros(
                        shape=input_shape
                    ),
                    self.__receive_fwd_labels_placeholder: np.zeros(
                        shape=label_shape
                    )
                }
            )

        self.__status = self.STATUS_GOT_FWD
        self.__communicate_cond.notify_all()
        self.__communicate_cond.release()
        log(f'SecondStage: got forward propagation result')
        return inputs_value, labels_value

    def __second_stage_data_generator(self):
        """
        模型训练的数据生成器
        :return: 数据获取迭代器
        """
        from ddl.examples.keras_pipeline_mpi.main import batch_size, samples
        begin = 0
        while True:
            end = min(begin + batch_size, samples)
            inputs, labels = self.__get_forward_propagation(begin, end)
            self.__last_inputs = inputs
            begin = 0 if end >= samples else begin + batch_size
            yield inputs, labels

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

    try:
        tf.compat.v1.disable_eager_execution()
        SecondStageModel().run()
    finally:
        log_file.flush()
        log_file.close()
