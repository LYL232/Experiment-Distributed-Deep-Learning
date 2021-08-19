from ddl.tensorflow.cpp_backend import CPPBackend
from ddl.tensorflow.keras.parallelism.pipeline.pipe import PipelinePipe
from ddl.log import Log
from tensorflow.keras.layers import Layer
import tensorflow as tf
from tensorflow.python.ops.resource_variable_ops import ResourceVariable
from threading import Thread, Condition
from enum import Enum


class PipelineInputLayer(Layer):
    def __init__(
            self, stage, pipe: PipelinePipe,
            name=None, dtype=None, dynamic=False, input_shape=None,
            **kwargs
    ):
        from ddl.tensorflow.keras.parallelism.pipeline.stage import \
            PipelineStage
        assert isinstance(stage, PipelineStage)
        assert isinstance(pipe, PipelinePipe)
        if input_shape is not None:
            kwargs['input_shape'] = input_shape
        super().__init__(
            trainable=True, name=name, dtype=dtype, dynamic=dynamic,
            **kwargs
        )
        self._stage = stage
        self._communicator = stage.pipeline_model.pipeline_communicator
        self._pipe = pipe
        self._fake_kernel = None
        self._pipeline_model_rank = stage.pipeline_model.pipeline_model_rank
        self._convey_gradient = pipe.convey_gradient()
        self._pipeline_input = None
        if self._pipe.comes_from is not None:
            self._comes_from_output_index = pipe.index_of(pipe.comes_from)
        else:
            self._comes_from_output_index = None
        self._input_index = pipe.index_of(stage)

    def build(self, input_shape):
        if self._pipe.comes_from is not None:
            self._fake_kernel = self.add_weight(shape=(1,), trainable=True)
        self.built = True

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, inputs, **kwargs):
        assert self._communicator is not None

        if self._pipe.comes_from is None:
            return inputs

        if self._pipeline_input is not None:
            return self._pipeline_input(inputs, self._fake_kernel)

        pipe = self._pipe
        pipeline_model_rank = self._pipeline_model_rank
        stage = self._stage
        stage_rank = stage.stage_rank
        communicator_id = self._communicator.id
        comes_from_stage_rank = pipe.comes_from.stage_rank
        convey_gradient = self._convey_gradient

        @tf.custom_gradient
        def pipeline_input(x, _):
            """
            输入层的梯度, 因为一般情况下是不计算输入层的梯度的, 所以需要一个Variable去欺骗
            tensorflow这里有需要计算的梯度, 这个自定义梯度就是为了获取输入层的梯度
            @param x: 输入层的输入tensor
            @param _: 无用的变量输入, 不需要使用, 只是为了欺骗tensorflow这里有梯度要计算
            @return:
            """

            def grad(dy):
                if convey_gradient:
                    send_op = CPPBackend.tf_lib().send_tensor(
                        dy,
                        receiver=pipe.comes_from.stage_rank,
                        tag=self._input_index,
                        communicator_id=communicator_id,
                        key=f'gradients-of-stage-'
                            f'{comes_from_stage_rank}'
                            f'-pipeline-input-'
                            f'{self._comes_from_output_index}'
                    )
                    # 如果不pass_with_computed会导致
                    # send_op没有终点而被tf认为无用而剪掉分支
                    fake_grad = CPPBackend.tf_lib().pass_with_computed(
                        tf.zeros((1,)), [send_op]
                    )
                else:
                    fake_grad = tf.reshape(
                        tf.reduce_sum(dy, axis=None), shape=(1,)) * 0
                return None, fake_grad

            x = CPPBackend.tf_lib().receive_tensor(
                x,
                sender=comes_from_stage_rank,
                communicator_id=communicator_id,
                tag=self._input_index,
                key=f'pipeline-{pipeline_model_rank}-'
                    f'stage-{stage_rank}-'
                    f'input-{self._input_index}-'
                    f'receive-forward-from-stage-'
                    f'{comes_from_stage_rank}-output-'
                    f'{self._comes_from_output_index}'
            )

            return x, grad

        self._pipeline_input = pipeline_input

        return pipeline_input(inputs, self._fake_kernel)


class PipelineOutputLayer(Layer):
    def __init__(
            self, stage, pipe: PipelinePipe,
            name=None, dtype=None, dynamic=False, input_shape=None,
            **kwargs
    ):
        from ddl.tensorflow.keras.parallelism.pipeline.stage import \
            PipelineStage
        assert isinstance(stage, PipelineStage)
        assert isinstance(pipe, PipelinePipe)
        if input_shape is not None:
            kwargs['input_shape'] = input_shape
        super().__init__(
            trainable=True, name=name, dtype=dtype, dynamic=dynamic,
            **kwargs
        )
        self._stage = stage
        self._communicator = stage.pipeline_model.pipeline_communicator
        self._pipe = pipe
        self._pipeline_model_rank = stage.pipeline_model.pipeline_model_rank
        self._convey_gradient = pipe.convey_gradient()
        self._fake_kernel = None
        self._pipeline_output = None
        self._output_index = self._pipe.index_of(self._stage)
        self._communicate_op_args = [
            (pipe.index_of(each), each.stage_rank) for each in pipe.send_to
        ]

    def build(self, input_shape):
        self.built = True
        if len(self._pipe.send_to) > 0:
            self._fake_kernel = self.add_weight(shape=(1,), trainable=True)

    def compute_output_shape(self, input_shape):
        if input_shape[0] is not None:
            return (None, *input_shape)
        return input_shape

    def call(self, inputs, **kwargs):
        # 是中间输出, 就再前向传播中发送结果, 在后向传播中接收梯度
        assert self._communicator is not None

        if len(self._pipe.send_to) == 0:
            return inputs

        if isinstance(inputs, ResourceVariable):
            raise Exception('ResourceVariable '
                            'can not be direct output of a stage')

        if self._pipeline_output is not None:
            return self._pipeline_output(inputs, self._fake_kernel)

        convey_gradient = self._convey_gradient
        pipeline_model_rank = self._pipeline_model_rank
        stage_rank = self._stage.stage_rank
        communicator_id = self._communicator.id

        @tf.custom_gradient
        def pipeline_output(x, _):
            """
            如果是中间输出, 那么就在前向传播中发送前向传播结果, 在后向传播中接收来自下一阶段的梯度
            @param x: 输入层的输入tensor
            @param _: 无用的变量输入, 不需要使用, 只是为了欺骗tensorflow这里有梯度要计算
            @return:
            """

            # 定义后向传播图
            def grad(dy):
                if convey_gradient:
                    # tf.print('dy shape:', tf.shape(dy))
                    recv_grad_ops = []
                    for receive_from_input_index, send_stage_rank \
                            in self._communicate_op_args:
                        recv_op = CPPBackend.tf_lib().receive_tensor(
                            dy,
                            sender=send_stage_rank,
                            tag=receive_from_input_index,
                            key=f'pipeline-'
                                f'{pipeline_model_rank}'
                                f'-stage-{stage_rank}-'
                                f'backward-gradient-from-stage-'
                                f'{send_stage_rank}-input-'
                                f'{receive_from_input_index}',
                            # 这里要传入handle(整数值), 而不是一个python对象
                            communicator_id=communicator_id
                        )

                        recv_grad_ops.append(recv_op)

                    recv_grad = tf.add_n(recv_grad_ops)
                else:
                    recv_grad = tf.zeros_like(dy)
                # 为了欺骗tensorflow这里有梯度需要计算, 如果不加这一句直接返回tf.zeros,
                # 那么recv_grad不会被调用
                fake_grad = CPPBackend.tf_lib().pass_with_computed(
                    tf.zeros((1,)), [recv_grad]
                )
                return recv_grad, fake_grad

            def send_forward(_x):
                # 确保所有分支都会被计算到
                send_ops = []

                for sending_to_input_index, recv_stage_rank \
                        in self._communicate_op_args:
                    send_op = CPPBackend.tf_lib().send_tensor(
                        _x,
                        receiver=recv_stage_rank,
                        tag=sending_to_input_index,
                        communicator_id=communicator_id,
                        key=f'pipeline-{pipeline_model_rank}-stage-'
                            f'{stage_rank}-output-'
                            f'{self._output_index}-forward-to-stage-'
                            f'{recv_stage_rank}-input-'
                            f'{sending_to_input_index}'
                    )
                    send_ops.append(send_op)
                return CPPBackend.tf_lib().pass_with_computed(
                    inputs, send_ops)

            return send_forward(x), grad

        self._pipeline_output = pipeline_output
        return pipeline_output(inputs, self._fake_kernel)


epct = Log.new_log_type(-1, False, 'eager pipeline communicating thread')


class TensorCommunicateTaskType(Enum):
    FORWARD = 0
    BACKWARD = 1
    EXIT = 2


class TensorCommunicateTask:
    def __init__(self, task_type: TensorCommunicateTaskType, data):
        self.__task_type = task_type
        self.data = data

    @property
    def type(self) -> TensorCommunicateTaskType:
        return self.__task_type


class EagerPipelineInputLayer(Layer):
    def __init__(
            self, stage, pipe: PipelinePipe,
            name=None, dtype=None, dynamic=False, input_shape=None,
            **kwargs
    ):
        from ddl.tensorflow.keras.parallelism.pipeline.stage import \
            PipelineStage
        assert isinstance(stage, PipelineStage)
        assert isinstance(pipe, PipelinePipe)
        if input_shape is not None:
            kwargs['input_shape'] = input_shape
        super().__init__(
            trainable=True, name=name, dtype=dtype, dynamic=dynamic,
            **kwargs
        )
        self._stage = stage
        self._communicator = stage.pipeline_model.pipeline_communicator
        self._pipe = pipe
        self._index = pipe.index_of(stage)
        self._fake_kernel = None
        self._pipeline_model_rank = stage.pipeline_model.pipeline_model_rank
        self._convey_gradient = pipe.convey_gradient()
        self._last_grad = None
        self._pipeline_input = None
        self._handle_thread = None
        self._task_cond = None
        self._result_cond = None
        self._task_queue = None
        self._task_received = None
        self._result_queue = None
        self._task_completed = None

    @property
    def index(self) -> int:
        return self._index

    def build(self, input_shape):
        if self._pipe.comes_from is not None:
            self._fake_kernel = self.add_weight(shape=(1,), trainable=True)
            self._task_queue = []
            self._result_queue = []
            self._task_cond = Condition()
            self._result_cond = Condition()
            self._handle_thread = Thread(
                target=self._handle_thread_fn, daemon=True)
            self._task_received = self._task_completed = 0
            self._handle_thread.start()
        self.built = True

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, inputs, **kwargs):
        if self._pipe.comes_from is None:
            return inputs

        if self._pipeline_input is not None:
            return self._pipeline_input(inputs, self._fake_kernel)

        convey_gradient = self._convey_gradient

        @tf.custom_gradient
        def pipeline_input(x, _):
            """
            输入层的梯度, 因为一般情况下是不计算输入层的梯度的, 所以需要一个Variable去欺骗
            tensorflow这里有需要计算的梯度, 这个自定义梯度就是为了获取输入层的梯度
            @param x: 输入层的输入tensor
            @param _: 无用的变量输入, 不需要使用, 只是为了欺骗tensorflow这里有梯度要计算
            @return:
            """

            def grad(dy):
                if convey_gradient:
                    self._last_grad = dy
                return None, tf.tuple(tf.zeros((1,)), control_inputs=[dy])

            return x, grad

        self._pipeline_input = pipeline_input

        return pipeline_input(inputs, self._fake_kernel)

    def receive_forward(self, feed_input):
        if self._pipe.comes_from is None:
            return
        with self._task_cond:
            self._task_queue.append(
                TensorCommunicateTask(
                    TensorCommunicateTaskType.FORWARD,
                    feed_input
                )
            )
            self._task_received += 1
            self._task_cond.notify_all()
            Log.debug(
                f'PipelineInputLayer-{self.index} got receive forward task',
                log_type=epct
            )

    def wait_and_get_receive_forward(self, original_inputs):
        if self._pipe.comes_from is None:
            return original_inputs
        with self._result_cond:
            while len(self._result_queue) == 0:
                Log.debug(
                    f'PipelineInputLayer-{self.index} wait receive forward',
                    log_type=epct
                )
                self._result_cond.wait()
                Log.debug(
                    f'PipelineInputLayer-{self.index} receive forward await',
                    log_type=epct
                )
            Log.debug(
                f'PipelineInputLayer-{self.index} got result', log_type=epct
            )
            result = self._result_queue.pop(0)

        return result

    def send_backward(self):
        if self._pipe.comes_from is None:
            return
        with self._task_cond:
            self._task_queue.append(TensorCommunicateTask(
                TensorCommunicateTaskType.BACKWARD, self._last_grad
            ))
            self._task_received += 1
            self._task_cond.notify_all()
            Log.debug(
                f'PipelineInputLayer-{self.index} got send backward task',
                log_type=epct
            )

    def wait_all_backward_sent(self):
        if self._pipe.comes_from is None:
            return
        with self._result_cond:
            while self._task_received > self._task_completed:
                self._result_cond.wait()
        self._task_received = self._task_completed = 0
        Log.debug(
            f'PipelineInputLayer-{self.index} sent all micro batch backwards',
            log_type=epct
        )

    def _handle_thread_fn(self):
        pipe = self._pipe
        pipeline_model_rank = self._pipeline_model_rank
        stage = self._stage
        stage_rank = stage.stage_rank
        communicator_id = self._communicator.id
        comes_from_stage_rank = pipe.comes_from.stage_rank
        comes_from_output_index = pipe.index_of(pipe.comes_from)

        while True:
            with self._task_cond:
                while len(self._task_queue) == 0:
                    Log.debug(f'PipelineInputLayer-{self.index} wait task',
                              log_type=epct)
                    self._task_cond.wait()
                task: TensorCommunicateTask = self._task_queue.pop(0)
            if task.type == TensorCommunicateTaskType.FORWARD:
                Log.debug(f'PipelineInputLayer-{self.index}'
                          f' got forward task', log_type=epct)
                received = CPPBackend.tf_lib().receive_tensor(
                    task.data,
                    sender=comes_from_stage_rank,
                    communicator_id=communicator_id,
                    tag=self.index,
                    key=f'pipeline-{pipeline_model_rank}-'
                        f'stage-{stage_rank}-'
                        f'input-{self.index}-'
                        f'receive-forward-from-stage-'
                        f'{comes_from_stage_rank}-output-'
                        f'{comes_from_output_index}'
                )
                with self._result_cond:
                    self._result_queue.append(received)
                    self._task_completed += 1
                    self._result_cond.notify_all()
                Log.debug(f'PipelineInputLayer-{self.index}'
                          f' finish forward task', log_type=epct)
            elif task.type == TensorCommunicateTaskType.BACKWARD:
                Log.debug(f'PipelineInputLayer-{self.index}'
                          f' got backward task', log_type=epct)
                CPPBackend.tf_lib().send_tensor(
                    task.data,
                    receiver=pipe.comes_from.stage_rank,
                    tag=self.index,
                    communicator_id=communicator_id,
                    key=f'gradients-of-stage-'
                        f'{comes_from_stage_rank}'
                        f'-pipeline-input-'
                        f'{comes_from_output_index}'
                )
                with self._result_cond:
                    self._task_completed += 1
                    self._result_cond.notify_all()
                Log.debug(f'PipelineInputLayer-{self.index}'
                          f' finish backward task', log_type=epct)
            else:
                return

    def __del__(self):
        if self._handle_thread is not None:
            # todo: 验证一下每个线程是否都退出了
            with self._task_cond:
                self._task_queue.append(TensorCommunicateTask(
                    TensorCommunicateTaskType.EXIT, None
                ))
                self._task_cond.notify_all()
            self._handle_thread.join()


class EagerPipelineOutputLayer(Layer):
    def __init__(
            self, stage, pipe: PipelinePipe,
            name=None, dtype=None, dynamic=False, input_shape=None,
            **kwargs
    ):
        from ddl.tensorflow.keras.parallelism.pipeline.stage import \
            PipelineStage
        assert isinstance(stage, PipelineStage)
        assert isinstance(pipe, PipelinePipe)
        if input_shape is not None:
            kwargs['input_shape'] = input_shape
        super().__init__(
            trainable=True, name=name, dtype=dtype, dynamic=dynamic,
            **kwargs
        )
        self._stage = stage
        self._communicator = stage.pipeline_model.pipeline_communicator
        self._pipe = pipe
        self._index = pipe.index_of(stage)
        self._pipeline_model_rank = stage.pipeline_model.pipeline_model_rank
        self._convey_gradient = pipe.convey_gradient()
        self._task_obj = None
        self._handle_threads = None
        self._task_conds = None
        self._task_queues = None
        self._task_received = None
        self._result_conds = None
        self._result_queues = None
        self._task_completed = None

    @property
    def index(self) -> int:
        return self._index

    def build(self, input_shape):
        self.built = True
        if len(self._pipe.send_to) > 0:
            self._handle_threads = []
            self._task_conds = []
            self._task_queues = []
            self._result_conds = []
            self._result_queues = []
            self._task_received = []
            self._task_completed = []
            for i in range(len(self._pipe.send_to)):
                self._task_conds.append(Condition())
                self._task_queues.append([])
                self._result_conds.append(Condition())
                self._result_queues.append([])
                self._task_received.append(0)
                self._task_completed.append(0)
                self._handle_threads.append(
                    Thread(target=self._handle_thread_fn, args=(i,),
                           daemon=True)
                )
            for each in self._handle_threads:
                each.start()

    def compute_output_shape(self, input_shape):
        if input_shape[0] is not None:
            return (None, *input_shape)
        return input_shape

    def call(self, inputs, **kwargs):
        return inputs

    def send_forward(self, predict):
        if len(self._pipe.send_to) == 0:
            return
        for i in range(len(self._handle_threads)):
            with self._task_conds[i]:
                self._task_queues[i].append(TensorCommunicateTask(
                    TensorCommunicateTaskType.FORWARD, predict
                ))
                self._task_received[i] += 1
                self._task_conds[i].notify_all()
            Log.debug(
                f'PipelineOutputLayer-{self.index}-thread-{i}'
                f' got send forward task',
                log_type=epct
            )

    def wait_all_forward_sent(self):
        if len(self._pipe.send_to) == 0:
            return
        for i in range(len(self._handle_threads)):
            with self._result_conds[i]:
                while self._task_received[i] > self._task_completed[i]:
                    self._result_conds[i].wait()
                self._task_received[i] = self._task_completed[i] = 0
            Log.debug(
                f'PipelineOutputLayer-{self.index}-thread-{i} all forward sent',
                log_type=epct
            )

    def receive_backward(self, targets):
        if len(self._pipe.send_to) == 0:
            return targets
        for i in range(len(self._handle_threads)):
            with self._task_conds[i]:
                self._task_queues[i].append(
                    TensorCommunicateTask(
                        TensorCommunicateTaskType.BACKWARD,
                        targets
                    )
                )
                self._task_received[i] += 1
                self._task_conds[i].notify_all()
            Log.debug(
                f'PipelineOutputLayer-{self.index}-thread-{i}'
                f' got receive backward task',
                log_type=epct
            )

    def wait_and_get_backward(self, original_targets):
        if len(self._pipe.send_to) == 0:
            return original_targets
        results = []
        for i in range(len(self._handle_threads)):
            with self._result_conds[i]:
                while len(self._result_queues[i]) == 0:
                    Log.debug(
                        f'PipelineOutputLayer-{self.index}-thread-{i}'
                        f' wait receive backward',
                        log_type=epct
                    )
                    self._result_conds[i].wait()
                    Log.debug(
                        f'PipelineOutputLayer-{self.index}-thread-{i}'
                        f' receive backward await',
                        log_type=epct
                    )
                Log.debug(
                    f'PipelineOutputLayer-{self.index}-thread-{i} got result',
                    log_type=epct
                )
                results.append(self._result_queues[i].pop(0))
        return tf.add_n(results)

    def _handle_thread_fn(self, t_id):
        convey_gradient = self._convey_gradient
        pipe = self._pipe
        pipeline_model_rank = self._pipeline_model_rank
        stage_rank = self._stage.stage_rank
        communicator_id = self._communicator.id
        sending_to_input_index = pipe.index_of(self._pipe.send_to[t_id])
        connecting_stage_rank = self._pipe.send_to[t_id].stage_rank
        task_cond = self._task_conds[t_id]
        task_queue = self._task_queues[t_id]
        result_cond = self._result_conds[t_id]
        result_queue = self._result_queues[t_id]

        while True:
            with task_cond:
                while len(task_queue) == 0:
                    Log.debug(
                        f'PipelineOutputLayer-{self.index}-{t_id} wait task',
                        log_type=epct)
                    task_cond.wait()
                task: TensorCommunicateTask = task_queue.pop(0)

            if task.type == TensorCommunicateTaskType.FORWARD:
                Log.debug(f'PipelineOutputLayer-{self.index}-{t_id}'
                          f' got forward task', log_type=epct)
                CPPBackend.tf_lib().send_tensor(
                    task.data,
                    receiver=connecting_stage_rank,
                    tag=sending_to_input_index,
                    communicator_id=communicator_id,
                    key=f'pipeline-{pipeline_model_rank}-stage-'
                        f'{stage_rank}-output-'
                        f'{self.index}-forward-to-stage-'
                        f'{connecting_stage_rank}-input-'
                        f'{sending_to_input_index}'
                )
                Log.debug(f'PipelineOutputLayer-{self.index}-{t_id} '
                          f'finishes forward task', log_type=epct)
                with result_cond:
                    self._task_completed[t_id] += 1
                    result_cond.notify_all()
            elif task.type == TensorCommunicateTaskType.BACKWARD:
                Log.debug(f'PipelineOutputLayer-{self.index}-{t_id}'
                          f' got backward task', log_type=epct)
                if not convey_gradient:
                    Log.debug(f'PipelineOutputLayer-{self.index}-{t_id}'
                              f' finishes backward task', log_type=epct)
                    continue
                received = CPPBackend.tf_lib().receive_tensor(
                    task.data,
                    sender=connecting_stage_rank,
                    tag=sending_to_input_index,
                    key=f'pipeline-'
                        f'{pipeline_model_rank}'
                        f'-stage-{stage_rank}-'
                        f'backward-gradient-from-stage-'
                        f'{connecting_stage_rank}-input-'
                        f'{sending_to_input_index}',
                    # 这里要传入handle(整数值), 而不是一个python对象
                    communicator_id=communicator_id
                )
                with result_cond:
                    result_queue.append(received)
                    self._task_completed[t_id] += 1
                    result_cond.notify_all()
                Log.debug(f'PipelineOutputLayer-{self.index}-{t_id}'
                          f' finishes backward task', log_type=epct)
            else:
                Log.debug(f'PipelineOutputLayer-{self.index}-{t_id}'
                          f' notify main thread finished', log_type=epct)
                return

    def __del__(self):
        if self._handle_threads is not None:
            # todo: 验证一下每个线程是否都退出了
            for i in range(len(self._handle_threads)):
                with self._task_conds[i]:
                    self._task_queues[i].append(TensorCommunicateTask(
                        TensorCommunicateTaskType.EXIT, None
                    ))
                    self._task_conds[i].notify_all()
            for each in self._handle_threads:
                each.join()
