import os
import sys
import numpy as np
import tensorflow as tf
import json
from threading import Condition
from tensorflow.python import ops
# noinspection PyProtectedMember
from tensorflow.python.ops.nn_grad import _BiasAddGrad
from tensorflow.keras.callbacks import Callback

name_need_grad = 'need_bias'

# noinspection PyProtectedMember
tf.python.ops._gradient_registry._registry.pop('BiasAdd', None)

handler_dict = {}


@ops.RegisterGradient('BiasAdd')
def us_bias_add_grad(op, received_grad: ops.Tensor):
    for key, handler in handler_dict.items():
        if key in received_grad.name:
            received_grad = handler(received_grad)
    return _BiasAddGrad(op, received_grad)


log_file = open('pylog-1.log', 'w')

last_stage_rank = 0


def log(msg: str, flush: bool = True):
    from ddl.examples.keras_pipeline_mpi.main import verbose
    if verbose:
        log_file.write(msg + '\n')
        if flush:
            log_file.flush()


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
        from ddl.examples.keras_pipeline_mpi.main import lr, \
            MSG_GRADIENTS_BACK_PROPAGATION
        log('SecondStage: started, initializing')

        self.__communicate_cond: Condition = Condition()

        self.__model = tf.keras.Sequential([
            # SecondStage的输入是FirstStage的输出
            tf.keras.layers.Flatten(input_shape=(196,)),
            tf.keras.layers.Dense(128, activation='relu', name=name_need_grad),
            tf.keras.layers.Dense(256, activation='relu', name='dense-1'),
            tf.keras.layers.Dense(10, activation='softmax', name='dense-2')
        ])

        def first_layer_bias_grad_handler(grad: ops.Tensor):
            from ddl.tensorflow.global_class import Global
            weights = self.__model.get_layer(name_need_grad).weights[0]
            # 需要与第一层的权重模型进行一次矩阵乘法
            diff = tf.transpose(tf.matmul(
                weights,
                tf.transpose(grad)
            ))

            return Global.tf_lib().forward_and_send(
                grad, diff, receiver=last_stage_rank,
                msg=json.dumps({
                    'type': MSG_GRADIENTS_BACK_PROPAGATION
                })
            )

        handler_dict[name_need_grad] = first_layer_bias_grad_handler

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
        inputs = Global.tf_lib().receive_tensor(
            tf.zeros(shape=input_shape, dtype=tf.float32),
            sender=last_stage_rank,
            name='0-forward-input-to-1'
        )
        labels = Global.tf_lib().receive_tensor(
            tf.zeros(shape=label_shape, dtype=tf.float32),
            sender=last_stage_rank,
            name='0-forward-label-to-1'
        )

        if not executing_eagerly():
            with tf.compat.v1.Session() as session:
                inputs_value = inputs.eval(session=session)
                with tf.control_dependencies([inputs]):
                    labels_value = labels.eval(session=session)

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
