from threading import Condition
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import Callback, History
from ddl.tensorflow.cpp_backend import CPPBackend
from ddl.tensorflow.communicator import Communicator
from ddl.tensorflow.util import executing_eagerly
from ddl.message import Message
from ddl.log import info
from ddl.tensorflow.keras.parallelism.data import \
    InitialParametersBroadcastCallBack
from tensorflow.python.keras import backend
from tensorflow.python.ops import control_flow_ops
import json  # 暂时用json传送stage与stage之间的信息
import numpy as np
import abc
from enum import Enum


class BaseTrainingStage(metaclass=abc.ABCMeta):
    """
    本类对象实现了训练过程时负责每个Stage的进程执行的主函数, 每个进程执行的具体分支代码在此类实现
    生命周期是从整体模型的训练开始至结束, 也即每次训练都要创建一次本类对象, 进行静态图构建等资源的初始化
    """

    class MessageCode(Enum):
        REQUEST_FPROP = 0
        FPROP_RESULT = 1
        BPROP_GRAD = 2
        DONE = 3

    def __init__(
            self,
            model: Model,
            pipeline_model_id: int,
            pipeline_communicator: Communicator,
            stage_communicator: Communicator,
            log_level: int,
            log_stream=None,
            **kwargs
    ):
        self.__log_stream = log_stream
        self.__log_level = log_level
        self.__pipeline_model_id = pipeline_model_id
        self.__pipeline_communicator = pipeline_communicator
        self.__stage_communicator = stage_communicator
        # todo: 改进: 判断是否是即时执行模式后选择
        #  是否构建静态图
        self.__session = tf.compat.v1.keras.backend.get_session()

        # 第一次fit需要进行初始的变量广播, 这个变量用于记录是否需要进行初始变量广播
        self._do_initial_params_broadcast = stage_communicator.size > 1

        self.__micro_batch_gradients = []
        with tf.name_scope(f'pipeline-{pipeline_model_id}-vars'):
            self.__tf_var_micro_batch_counter = \
                tf.Variable(0, name='current_micro_batch', dtype=tf.int32)
            # 总批次数
            self.__tf_var_micro_batches = tf.Variable(
                0, name='micro_batches', dtype=tf.int32
            )
            # 每个批次的大小: 第一批大小和最后一批大小
            self.__tf_var_first_micro_batch_size = tf.Variable(
                0, name='first_micro_batch_size', dtype=tf.int32
            )
            self.__tf_var_last_micro_batch_size = tf.Variable(
                0, name='last_micro_batch_size', dtype=tf.int32
            )
            self.__tf_var_grads = []
            for param in model.weights:
                self.__tf_var_grads.append(
                    tf.Variable(
                        tf.zeros(shape=param.shape),
                        name='micro-batch-saved-grads-' + param.name.replace(
                            '/', '-').replace(':', '-'),
                        dtype=param.dtype
                    )
                )
        for v in [
            self.__tf_var_micro_batch_counter,
            self.__tf_var_first_micro_batch_size,
            self.__tf_var_last_micro_batch_size,
            self.__tf_var_micro_batches,
            *self.__tf_var_grads
        ]:
            backend.track_variable(v)

        self.__micro_batch_first_size_placeholder = \
            tf.compat.v1.placeholder(shape=(), dtype=tf.int32)

        self.__init_micro_first_batch_size = \
            self.__tf_var_first_micro_batch_size.assign(
                self.__micro_batch_first_size_placeholder
            )

        self.__micro_batch_last_size_placeholder = \
            tf.compat.v1.placeholder(shape=(), dtype=tf.int32)
        self.__init_micro_last_batch_size = \
            self.__tf_var_last_micro_batch_size.assign(
                self.__micro_batch_last_size_placeholder
            )

        self.__micro_batch_sizes_placeholder = \
            tf.compat.v1.placeholder(shape=(), dtype=tf.int32)
        self.__init_micro_batches = self.__tf_var_micro_batches.assign(
            self.__micro_batch_sizes_placeholder
        )

        self.__init_current_micro_batch = \
            self.__tf_var_micro_batch_counter.assign(0)
        self.__init_micro_batch_grads = []

        for each in self.__tf_var_grads:
            self.__init_micro_batch_grads.append(
                each.assign(tf.zeros_like(each))
            )

        self.__fit_args = kwargs

        # 记录上一个批次下一阶段请求获取前向传播的结果, 由子类赋值
        # 这里其实有一些浪费计算资源, 因为传输给下一个阶段计算了一次前向传播, 后面进行fit的时候
        # 也计算了一次前向传播, todo: 可优化一次前向传播的计算, 或许可以直接调用优化器的apply_gradient
        self._micro_batch_inputs = []

        opt = model.optimizer

        if hasattr(opt, 'is_distributed_optimizer'):
            # 由于数据并行分布式优化器每一批次都会进行一次allreduce,
            # 所以需要复写一下优化器的get_gradient方法和_aggregate_gradients方法, 防止不必要的allreduce

            # get_gradients方法直接使用原优化器的方法即可
            opt.get_gradients = opt.original_get_gradients

            # 记录一下原分布式的apply_gradients和_aggregate_gradients方法,
            distributed_apply_gradients = opt.apply_gradients
            distributed_aggregate_gradients = opt._aggregate_gradients

            def micro_batch_aggregate_gradients(grads_and_vars):
                """
                根据微批次计数器进行allreduce的方法, 当该批次是最后一批次时, 进行
                allreduce
                @param grads_and_vars: [(梯度, 变量)]
                @return: [梯度], todo: 注意, tf2.4版本的返回值有所不同
                """
                grads, variables = list(zip(*grads_and_vars))
                return tf.cond(
                    # 这counter是未进行自加操作的值, 所以减一
                    self.__tf_var_micro_batch_counter
                    >= self.__tf_var_micro_batches - 1,
                    lambda: distributed_aggregate_gradients(grads_and_vars),
                    lambda: list(grads),
                    name=f'if-original-aggregate-gradients'
                )

            opt._aggregate_gradients = micro_batch_aggregate_gradients

            def micro_batch_apply_gradients(
                    grads_and_vars,
                    name=None,
                    experimental_aggregate_gradients=True):
                """
                替换掉模型优化器的应用梯度方法, 其实就是一个装饰器
                @param grads_and_vars: List of (gradient, variable) pairs.
                @param name:
                @param experimental_aggregate_gradients:
                @return:
                """
                assert len(grads_and_vars) == len(self.__tf_var_grads)

                # 计数器自加1op
                current_batch_added = \
                    self.__tf_var_micro_batch_counter.assign_add(1)

                for i in range(len(grads_and_vars)):
                    # 微批次累加变量
                    cumsum_grad = self.__tf_var_grads[i]
                    # 该批次计算出的梯度
                    grad = grads_and_vars[i][0]
                    # 梯度对应的变量
                    variable = grads_and_vars[i][1]
                    assert cumsum_grad.shape == variable.shape

                    # 将该批次计算出来的梯度进行累加的op
                    assign_added = cumsum_grad.assign_add(
                        grad * tf.cast(
                            tf.cond(
                                current_batch_added <
                                self.__tf_var_micro_batches,
                                # 由于一批次的微批次大小只有两种,
                                # 第一到倒数第二微批次的大小都是一致的,
                                # 最后一微批次可能会有所不同, 所以这里要加一个判断
                                lambda: self.__tf_var_first_micro_batch_size,
                                lambda: self.__tf_var_last_micro_batch_size,
                            ),
                            dtype=grad.dtype
                        )
                    )

                    applying_gradient = tf.cond(
                        # 小于的时候就返回0梯度
                        current_batch_added < self.__tf_var_micro_batches,
                        lambda: tf.zeros_like(variable),
                        lambda: tf.divide(
                            assign_added,
                            tf.cast(
                                (self.__tf_var_micro_batches - 1) *
                                self.__tf_var_first_micro_batch_size +
                                self.__tf_var_last_micro_batch_size,
                                dtype=variable.dtype
                            )
                        ),
                    )
                    grads_and_vars[i] = (
                        applying_gradient, grads_and_vars[i][1])

                return tf.cond(
                    current_batch_added >= self.__tf_var_micro_batches,
                    lambda: distributed_apply_gradients(
                        grads_and_vars, name, experimental_aggregate_gradients),
                    # 如果没有到最后一微批次, 就什么也不做, 相当于把值赋给了
                    # 梯度累计变量
                    lambda: control_flow_ops.no_op(),
                    name=f'original-applying-grads'
                )

            opt.apply_gradients = micro_batch_apply_gradients

        self.__model = model

    @property
    def pipeline_model_id(self) -> int:
        return self.__pipeline_model_id

    @property
    def pipeline_communicator(self) -> Communicator:
        return self.__pipeline_communicator

    @property
    def stage_communicator(self) -> Communicator:
        return self.__stage_communicator

    @property
    def model(self) -> Model:
        return self.__model

    @property
    def _session(self) -> tf.compat.v1.Session:
        return self.__session

    @property
    def _fit_args(self) -> dict:
        return self.__fit_args

    @abc.abstractmethod
    def run(self) -> list or History:
        """
        执行模型训练过程
        @return: 每个batch的fit History 组成的列表
        """

    def _initialize_micro_batch_vars(self):
        """
        初始化微批次所需的变量
        @return: None
        """
        assert len(self._micro_batch_inputs) > 0

        self.__session.run([
            self.__init_micro_first_batch_size,
            self.__init_micro_last_batch_size,
            self.__init_micro_batches,
            self.__init_current_micro_batch,
            *self.__init_micro_batch_grads
        ], feed_dict={
            self.__micro_batch_first_size_placeholder:
                self._micro_batch_inputs[0].shape[0],
            self.__micro_batch_last_size_placeholder:
                self._micro_batch_inputs[-1].shape[0],
            self.__micro_batch_sizes_placeholder:
                len(self._micro_batch_inputs)
        })


class StageWithNextStage(BaseTrainingStage, metaclass=abc.ABCMeta):
    """
    非最后一个阶段的相同接口
    """

    def __init__(
            self,
            pipeline_model_id: int,
            pipeline_communicator: Communicator,
            stage_communicator: Communicator,
            model: Model,
            next_stage_input_shape: tuple,
            log_level: int,
            log_stream,
            **kwargs
    ):
        super().__init__(
            pipeline_model_id=pipeline_model_id,
            pipeline_communicator=pipeline_communicator,
            stage_communicator=stage_communicator,
            model=model,
            log_level=log_level,
            log_stream=log_stream,
            **kwargs
        )
        self.__next_stage_input_shape = next_stage_input_shape

        self.__next_stage_rank = pipeline_communicator.rank + 1

        # 定义静态计算图: 发送前向传播结果
        self._send_fwd_outputs_placeholder = tf.compat.v1.placeholder(
            dtype=tf.float32, shape=next_stage_input_shape)
        self._send_fwd_outputs = CPPBackend.tf_lib().send_tensor(
            self._send_fwd_outputs_placeholder,
            receiver=self.__next_stage_rank,
            name=f'pipeline-{pipeline_model_id}-{pipeline_communicator.rank}-'
                 f'forward-input-to-{self.__next_stage_rank}',
            communicator_id=self.pipeline_communicator.id
        )
        # 定义静态计算图: 接收下一阶段传输的后向传播结果
        self._receive_error_placeholder = tf.compat.v1.placeholder(
            dtype=tf.float32, shape=next_stage_input_shape,
        )
        self._receive_error = CPPBackend.tf_lib().receive_tensor(
            self._receive_error_placeholder,
            sender=self.__next_stage_rank,
            communicator_id=self.pipeline_communicator.id,
            name=f'pipeline-{self.pipeline_model_id}'
                 f'-{self.pipeline_communicator.rank}-'
                 f'back-result-from-{self.__next_stage_rank}',
        )

        self._done = False

        self._history = []

    @property
    def next_stage_input_shape(self) -> tuple:
        return self.__next_stage_input_shape

    @property
    def next_stage_rank(self) -> int:
        return self.__next_stage_rank

    def _back_propagation(self, micro_batch_size: int):
        """
        等待下一阶段完成后向传播并使用后向传播得到的偏差进行模型的训练
        :@param: micro_batch_size 微批次大小
        :@return: None
        """
        info(f'waiting back propagation result, micro batches: '
             f'{len(self._micro_batch_inputs)}')

        errors = []

        for i in range(len(self._micro_batch_inputs)):
            info('listening back prop msg')
            msg = Message.listen(self.pipeline_communicator)
            info(f'got back prop msg: {msg}')
            msg_obj = json.loads(msg.msg)

            if msg_obj['code'] == self.MessageCode.DONE.value:
                # 当fit结束时, 只会调用一次前向传播而没有后向传播
                # (个人猜测应该是最后一次更新模型梯度后再进行一次前向传播来计算准确率)
                self._done = True
                if isinstance(self, StageWithPreviousStage):
                    # 如果本对象有前一阶段, 那么需要将Done信息传到上一阶段
                    info('done, sending Done to previous stage')
                    Message.send(json.dumps({
                        'code': self.MessageCode.DONE.value
                    }), self.previous_stage_rank,
                        communicator=self.pipeline_communicator
                    )
                return
            elif msg_obj['code'] == self.MessageCode.BPROP_GRAD.value:
                assert len(self._micro_batch_inputs) > 0
                if executing_eagerly():
                    error = CPPBackend.tf_lib().receive_tensor(
                        tf.zeros(shape=(
                            self._micro_batch_inputs[i].shape[0],
                            *self.__next_stage_input_shape[1:]
                        )),
                        name=f'pipeline-{self.pipeline_model_id}'
                             f'-{self.pipeline_communicator.rank}-'
                             f'back-result-from-{self.__next_stage_rank}',
                        sender=self.__next_stage_rank,
                        communicator_id=self.pipeline_communicator.id
                    )
                else:
                    error = self._session.run(
                        self._receive_error,
                        feed_dict={
                            self._receive_error_placeholder: np.zeros(
                                shape=(
                                    self._micro_batch_inputs[i].shape[0],
                                    *self.__next_stage_input_shape[1:]
                                )
                            )
                        }
                    )
                info(f'got back propagation result, shape={error.shape}')
                errors.append(error)
            else:
                raise Exception(f'got unexpected msg: {msg}')

        callbacks = self._fit_args.pop('callbacks', None)
        if self._do_initial_params_broadcast:
            self._do_initial_params_broadcast = False
            callbacks = [
                InitialParametersBroadcastCallBack(
                    0, communicator=self.stage_communicator)
            ]

        self._initialize_micro_batch_vars()

        self._history.append(self.model.fit(
            np.concatenate(self._micro_batch_inputs, axis=0),
            np.concatenate(errors, axis=0),
            batch_size=micro_batch_size,
            epochs=1,
            verbose=0,
            callbacks=callbacks,
            **self._fit_args
        ))
        self._micro_batch_inputs = []

    def _send_forward_propagation_result(self, outputs: np.ndarray):
        """
        将前向传播结果传输给下一阶段
        @param outputs: 前向传播结果
        @return: None
        """
        info('sending forward propagation message')
        # todo: 可以写到op里包装到一个层里
        Message.send(
            json.dumps({
                'code': self.MessageCode.FPROP_RESULT.value}),
            self.__next_stage_rank,
            communicator=self.pipeline_communicator
        )
        if executing_eagerly():
            CPPBackend.tf_lib().send_tensor(
                tf.constant(outputs, dtype=tf.float32),
                receiver=self.__next_stage_rank,
                name=f'pipeline-{self.pipeline_model_id}'
                     f'-{self.pipeline_communicator.rank}-'
                     f'forward-input-to-{self.__next_stage_rank}',
                communicator_id=self.pipeline_communicator.id
            )
        else:
            self._session.run(
                self._send_fwd_outputs,
                feed_dict={
                    self._send_fwd_outputs_placeholder: outputs,
                }
            )


class StageWithPreviousStage(BaseTrainingStage, metaclass=abc.ABCMeta):
    """
    非第一个阶段的相同接口
    """

    def __init__(
            self,
            pipeline_model_id: int,
            pipeline_communicator: Communicator,
            stage_communicator: Communicator,
            previous_stage_output_shape: tuple,
            model: Model,
            log_level: int,
            log_stream,
            **kwargs
    ):
        super().__init__(
            pipeline_model_id=pipeline_model_id,
            pipeline_communicator=pipeline_communicator,
            stage_communicator=stage_communicator,
            model=model,
            log_level=log_level,
            log_stream=log_stream,
            **kwargs
        )
        self.__previous_stage_output_shape = previous_stage_output_shape

        self.__previous_stage_rank = self.pipeline_communicator.rank - 1

        # 接收上一阶段传来的前向传播结果op静态图
        self.__receive_fwd_outputs_placeholder = tf.compat.v1.placeholder(
            dtype=tf.float32, shape=previous_stage_output_shape)
        self.__receive_fwd_outputs = CPPBackend.tf_lib().receive_tensor(
            self.__receive_fwd_outputs_placeholder,
            sender=self.__previous_stage_rank,
            name=f'pipeline-{pipeline_model_id}-{self.__previous_stage_rank}-'
                 f'forward-input-to-{pipeline_communicator.rank}',
            # 这里要传入handle(整数值), 而不是一个python对象
            communicator_id=pipeline_communicator.id
        )

    @property
    def previous_stage_output_shape(self) -> tuple:
        return self.__previous_stage_output_shape

    @property
    def previous_stage_rank(self) -> int:
        return self.__previous_stage_rank

    def _request_forward_propagation(
            self, batch_end: int,
            micro_batch_begin: int, micro_batch_end: int
    ) -> np.ndarray:
        """
        从上一阶段获取前向传播的结果
        :@param batch_end: 数据batch在整个数据库的结束索引
        :@param micro_batch_begin: 当前微批次数据batch在整个数据库的结束索引
        :@param micro_batch_end: 当前微批次数据batch在整个数据库的结束索引
        :@return: 模型batch的输入
        """
        info(f'request forward propagation batch ['
             f'{micro_batch_begin}, {micro_batch_end}]')
        Message.send(
            json.dumps({
                'code': self.MessageCode.REQUEST_FPROP.value,
                'batch_end': batch_end,
                'micro_batch_begin': micro_batch_begin,
                'micro_batch_end': micro_batch_end,
            }),
            self.__previous_stage_rank,
            communicator=self.pipeline_communicator
        )

        msg: Message = Message.listen(self.pipeline_communicator)
        msg_obj = json.loads(msg.msg)
        assert msg_obj['code'] == self.MessageCode.FPROP_RESULT.value
        info('getting forward propagation result')

        shape = (
            micro_batch_end - micro_batch_begin,
            *self.__previous_stage_output_shape[1:]
        )

        if executing_eagerly():
            result = CPPBackend.tf_lib().receive_tensor(
                tf.zeros(shape=shape, dtype=tf.float32),
                sender=self.__previous_stage_rank,
                name=f'pipeline-{self.pipeline_model_id}-'
                     f'{self.__previous_stage_rank}-'
                     f'forward-input-to-{self.pipeline_communicator.rank}',
                # 这里要传入handle(整数值), 而不是一个python对象
                communicator_id=self.pipeline_communicator.id
            )
        else:
            result = self._session.run(
                self.__receive_fwd_outputs,
                feed_dict={
                    self.__receive_fwd_outputs_placeholder: np.zeros(
                        shape=shape
                    ),
                }
            )
        # info(f'got forward propagation result')
        return result


class FirstTrainingStage(StageWithNextStage):
    def __init__(
            self,
            pipeline_model_id: int,
            pipeline_communicator: Communicator,
            stage_communicator: Communicator,
            model: Model,
            next_stage_input_shape: tuple,
            inputs: np.ndarray,
            log_level: int,
            log_stream,
            **kwargs
    ):
        super().__init__(
            pipeline_model_id=pipeline_model_id,
            pipeline_communicator=pipeline_communicator,
            stage_communicator=stage_communicator,
            model=model,
            next_stage_input_shape=next_stage_input_shape,
            log_level=log_level,
            log_stream=log_stream,
            **kwargs
        )
        # info('started, initializing')

        self.__inputs = inputs

        # info('model:')
        # self.model.summary(print_fn=info)

        self.__next_stage_rank = self.pipeline_communicator.rank + 1

        self.__next_stage_input_shape = next_stage_input_shape

        # 记录每个batch训练的History
        self.__history = []
        # info('started, initialized')

    def run(self) -> list or History:
        # 在每个批次中的第一个微批次大小, 用来输入Model.fit的batch_size参数
        micro_batch_size = None
        while not self._done:
            info(
                'waiting for getting forward propagation'
                ' or done message'
            )
            # 其实不一定要接收信息, 可以通过全局初始化信息推断出什么时候应该进行tensor的传输
            msg: Message = Message.listen(self.pipeline_communicator)
            info(
                f'got for getting forward propagation'
                f' or done message {msg.msg}'
            )
            msg_obj = json.loads(msg.msg)
            if msg_obj['code'] == \
                    self.MessageCode.REQUEST_FPROP.value:
                batch_end = msg_obj['batch_end']
                micro_batch_begin = msg_obj['micro_batch_begin']
                micro_batch_end = msg_obj['micro_batch_end']

                if micro_batch_size is None:
                    # 说明是一个新的批次的开始
                    micro_batch_size = micro_batch_end - micro_batch_begin

                inputs = self.__inputs[micro_batch_begin:micro_batch_end, ...]

                self._micro_batch_inputs.append(inputs)
                outputs = self.model.predict(inputs)

                info(
                    f'sending forward propagation result: '
                    f'[{micro_batch_begin}, {micro_batch_end}]')
                self._send_forward_propagation_result(outputs)
                info(
                    f'sent forward propagation result: '
                    f'[{micro_batch_begin}, {micro_batch_end}]')

                if batch_end <= micro_batch_end:
                    # 进行这些批次的后向传播, 否则继续等待后一个阶段获取前向传播结果
                    self._back_propagation(micro_batch_size)
                    micro_batch_size = None
            elif msg_obj['code'] == self.MessageCode.DONE.value:
                break
            else:
                raise Exception(
                    f'receive unexpected message: msg: {msg.msg}, '
                    f'sender: {msg.sender}'
                )

        info('returning')
        return self.__history


class IntermediateTrainingStage(StageWithPreviousStage, StageWithNextStage):
    def __init__(
            self,
            pipeline_model_id: int,
            pipeline_communicator: Communicator,
            stage_communicator: Communicator,
            previous_stage_output_shape: tuple,
            model: Model,
            next_stage_input_shape: tuple,
            log_level: int,
            log_stream,
            **kwargs
    ):
        super().__init__(
            pipeline_model_id=pipeline_model_id,
            pipeline_communicator=pipeline_communicator,
            stage_communicator=stage_communicator,
            previous_stage_output_shape=previous_stage_output_shape,
            model=model,
            next_stage_input_shape=next_stage_input_shape,
            log_level=log_level,
            log_stream=log_stream,
            **kwargs
        )

    def run(self) -> list:
        micro_batch_size = None
        while not self._done:
            info(
                'waiting for getting forward propagation'
                ' or done message'
            )
            # 其实不一定要接收信息, 可以通过全局初始化信息推断出什么时候应该进行tensor的传输
            msg: Message = Message.listen(self.pipeline_communicator)
            info(
                f'got forward propagation'
                f' or done message {msg.msg}'
            )
            msg_obj = json.loads(msg.msg)
            if msg_obj['code'] == \
                    self.MessageCode.REQUEST_FPROP.value:
                batch_end = msg_obj['batch_end']
                micro_batch_begin = msg_obj['micro_batch_begin']
                micro_batch_end = msg_obj['micro_batch_end']

                if micro_batch_size is None:
                    # 说明是一个新的批次的开始
                    micro_batch_size = micro_batch_end - micro_batch_begin

                inputs = self._request_forward_propagation(
                    batch_end, micro_batch_begin, micro_batch_end
                )

                self._micro_batch_inputs.append(inputs)
                outputs = self.model.predict(inputs)

                info(
                    f'sending forward propagation result: '
                    f'[{micro_batch_begin}, {micro_batch_end}]')
                self._send_forward_propagation_result(outputs)
                info(
                    f'sent forward propagation result: '
                    f'[{micro_batch_begin}, {micro_batch_end}]')

                if batch_end <= micro_batch_end:
                    # 进行这些批次的后向传播, 否则继续等待后一个阶段获取前向传播结果
                    self._back_propagation(micro_batch_size)
                    micro_batch_size = None
            else:
                raise Exception(
                    f'receive unexpected message: msg: {msg.msg}, '
                    f'sender: {msg.sender}'
                )

        info('returning')
        return self._history


class LastTrainingStage(StageWithPreviousStage):
    """
    最后一个训练阶段, 负责请求前向传播和
    """

    class Status(Enum):
        """
        由于fit和data_generator不是同一个线程的动作, 所以需要一些线程间同步操作,
        本类是线程间同步操作的状态类
        """
        # 初始化
        INIT = 0
        # 准备就绪
        READY = 1
        # 前向传播之后
        AFTER_FPROP = 2
        # 批次结束
        BATCH_END = 3
        # 训练结束
        DONE = 4

    class TrainingEndCallBack(Callback):
        """
        后向传播结束回调函数
        """

        def __init__(self, stage: 'LastTrainingStage'):
            super().__init__()
            self.__stage = stage

        def on_train_batch_end(self, batch, logs=None):
            # 通知模型已经张量已经传输完毕, 或许可以通过op和C api进行通知, 但是有点复杂
            self.__stage.on_micro_batch_end(batch)

    def __init__(
            self,
            pipeline_model_id: int,
            pipeline_communicator: Communicator,
            stage_communicator: Communicator,
            previous_stage_output_shape: tuple,
            model: Model,
            targets: np.ndarray,
            verbose: int,
            batch_size: int, micro_batch_size: int,
            epochs: int,
            log_level: int,
            log_stream,
            **kwargs
    ):
        super().__init__(
            pipeline_model_id=pipeline_model_id,
            pipeline_communicator=pipeline_communicator,
            stage_communicator=stage_communicator,
            previous_stage_output_shape=previous_stage_output_shape,
            model=model,
            log_level=log_level,
            log_stream=log_stream,
            **kwargs
        )
        if micro_batch_size is not None:
            assert micro_batch_size <= batch_size, \
                'micro batch size can not be bigger than batch size'

        # info('started, initializing')

        self.__status_cond: Condition = Condition()

        # info('model:')
        # self.model.summary(print_fn=info)
        self.__status_cond.acquire()
        self.__status = self.Status.READY
        self.__status_cond.release()

        self.__samples = targets.shape[0]
        self.__epochs = epochs
        self.__batch_size = batch_size
        self.__micro_batch_size = micro_batch_size
        self.__verbose = verbose

        self.__targets = targets

        self.__total_micro_batches, self.__micro_batches_per_batch = \
            self.__get_total_micro_batches()

        # info('started, initialized')

    def run(self) -> History:
        callbacks = self._fit_args.pop('callbacks', None)
        if callbacks is None:
            callbacks = []
        if self._do_initial_params_broadcast:
            self._do_initial_params_broadcast = False
            callbacks.append(
                InitialParametersBroadcastCallBack(
                    0, self.stage_communicator)
            )
        callbacks.append(self.TrainingEndCallBack(self))

        info(f'total micro batches: {self.__total_micro_batches}')

        history = self.model.fit(
            self.__fit_data_generator(),
            steps_per_epoch=self.__total_micro_batches,
            epochs=self.__epochs,
            # 只有id为0的Model才输出信息
            verbose=0 if self.pipeline_model_id != 0 else self.__verbose,
            callbacks=callbacks,
            **self._fit_args
        )

        self.__status_cond.acquire()
        # 当fit结束后, 只会调用一次前向传播, 但不会计算梯度
        # (个人猜测应该是最后一次更新模型梯度后再进行一次前向传播来计算准确率)
        while self.__status != self.Status.AFTER_FPROP:
            self.__status_cond.wait()

        info('done, sending Done to previous stage')
        Message.send(json.dumps({
            'code': self.MessageCode.DONE.value
        }), self.previous_stage_rank,
            communicator=self.pipeline_communicator
        )
        self.__status = self.Status.DONE

        self.__status_cond.notify_all()
        self.__status_cond.release()
        info('returning')
        return history

    def on_micro_batch_end(self, batch: int):
        """
        留给TrainingEndCallBack通知本对象后向传播已经完成的方法
        @param batch: 当前的batch编号
        @return:
        """
        # 当batch编号是每一批次的最后一个微批次或者是所有微批次的最后一微批次时, 表示
        # 这一批次结束
        if batch % self.__micro_batches_per_batch == \
                self.__micro_batches_per_batch - 1 \
                or batch == self.__total_micro_batches - 1:
            info(f'batch end, micro batch: {batch}')
            self.__status_cond.acquire()
            while self.__status != LastTrainingStage.Status.AFTER_FPROP:
                self.__status_cond.wait()
            self.__status = LastTrainingStage.Status.BATCH_END
            self._micro_batch_inputs = []
            self.__status_cond.notify_all()
            self.__status_cond.release()

    def __get_total_micro_batches(self) -> (int, int):
        """
        根据输入的训练样数量和批次信息计算出需要的微批次总数
        @return: 需要的微批次总数, 每个批次的微批次总数(不包括最后的微批次)
        """
        micro_batches_per_batch = self.__batch_size // self.__micro_batch_size
        if micro_batches_per_batch * self.__micro_batch_size \
                < self.__batch_size:
            micro_batches_per_batch += 1

        tot_batches = self.__samples // self.__batch_size
        tot_micro_batches = tot_batches * micro_batches_per_batch
        if tot_batches * self.__batch_size < self.__samples:
            tot_batches += 1
            remain = self.__samples * tot_batches * self.__batch_size
            remain_micro_batches = remain // self.__micro_batch_size
            if remain_micro_batches * self.__micro_batch_size < remain:
                remain_micro_batches += 1
            tot_micro_batches += remain_micro_batches

        return tot_micro_batches, micro_batches_per_batch

    def __fit_data_generator(self):
        """
        模型训练的数据生成器
        :@return: 数据获取迭代器
        """
        # todo: 我想仍有一种更高效率的并行的算法: 可以类比TCP的滑动窗口发送包的过程
        #  不必一次等待前面的阶段全部发送这一批次的所有微批次的前向传播再进行后向传播,
        #  而是来一个微批次就马上进行后向传播的计算, 计算完毕后就马上传回上一阶段, 这样能达到
        #  更高的并行性, 但是, 通信方式更加复杂, 而且需要至少两个线程去维护这样的通信, 比较复杂
        #  目前先实现比较简单的流水线并行
        batch_begin = 0
        while True:
            batch_end = min(
                batch_begin + self.__batch_size,
                self.__samples
            )

            micro_batch_begin = batch_begin
            micro_batch_end = min(
                micro_batch_begin + self.__micro_batch_size,
                batch_end
            )

            self.__status_cond.acquire()
            while self.__status != self.Status.READY and \
                    self.__status != self.Status.BATCH_END:
                info(f'waiting to forward propagation batch['
                     f'{batch_begin}:{batch_end}]')
                self.__status_cond.wait()

            micro_batch_targets = []
            while micro_batch_begin < batch_end:
                # 获取所有批次的微批次输入
                self._micro_batch_inputs.append(
                    self._request_forward_propagation(
                        batch_end, micro_batch_begin, micro_batch_end
                    ))
                micro_batch_targets.append(
                    self.__targets[micro_batch_begin:micro_batch_end, ...]
                )
                micro_batch_begin = micro_batch_end
                micro_batch_end = min(
                    micro_batch_begin + self.__micro_batch_size,
                    batch_end
                )

            self.__status = self.Status.AFTER_FPROP
            self.__status_cond.notify_all()
            self.__status_cond.release()

            self._initialize_micro_batch_vars()

            # 进行后向传播
            info(f'batch [{batch_begin}, {batch_end}]')
            for i in range(len(self._micro_batch_inputs)):
                yield self._micro_batch_inputs[i], micro_batch_targets[i]

            batch_begin = 0 if batch_end >= self.__samples else batch_end
