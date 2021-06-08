from ddl.log import info
from ddl.message import Message
from ddl.tensorflow.cpp_backend import CPPBackend
from ddl.tensorflow.keras.parallelism.pipeline.pipe import PipelinePipe
from ddl.tensorflow.keras.parallelism.data import \
    InitialParametersBroadcastCallBack
from ddl.tensorflow.keras.parallelism.pipeline.micro_batch_controller import \
    MicroBatchController

import json
from enum import Enum
import tensorflow as tf
from tensorflow.keras.callbacks import History, Callback
import numpy as np
from collections.abc import Iterable


class TrainingExecutor:
    class MessageCode(Enum):
        REQUEST_FPROP = 0
        FPROP_RESULT = 1
        BPROP_GRAD = 2
        DONE = 3

    class TrainingCallback(Callback):
        """
        后向传播结束回调函数
        """

        def __init__(self, executor: 'TrainingExecutor'):
            super().__init__()
            self.__executor = executor

        def on_train_batch_end(self, batch, logs=None):
            # 通知模型已经张量已经传输完毕, 或许可以通过op和C api进行通知, 但是有点复杂
            self.__executor.on_micro_batch_end(batch)

    def __init__(
            self, stage, micro_batch_controller: MicroBatchController,
            inputs_data: tuple, input_data_indexes: tuple or list,
            targets_data: tuple, target_data_indexes: tuple or list,
            batch_size: int, epochs: int,
            session: tf.compat.v1.Session,
            micro_batch_size: int = None,
            callbacks=None,
            verbose: int = 1, **fit_args
    ):
        from ddl.tensorflow.keras.parallelism.pipeline.stage import \
            PipelineStage
        assert isinstance(stage, PipelineStage)
        assert len(input_data_indexes) == len(stage.input_pipes)
        assert len(target_data_indexes) == len(stage.output_pipes)

        self.__input_data_indexes = input_data_indexes
        self.__target_data_indexes = target_data_indexes

        self.__session = session
        self.__graph = session.graph

        self.__stage = stage
        self.__inputs_data = inputs_data
        self.__targets_data = targets_data
        self.__pipeline_model_rank = stage.pipeline_model.pipeline_model_rank
        self.__pipeline_communicator = \
            stage.pipeline_model.pipeline_communicator

        # 接收上一阶段传来的前向传播结果op静态图
        self.__pipeline_input_placeholders = []

        self.__pipeline_input_count = len(stage.input_pipes)
        self.__fwd_prop_from_previous_count = self.__pipeline_input_count
        self.__recv_fwd_prop_ops = []

        self.__samples = None

        for i in range(self.__pipeline_input_count):
            if input_data_indexes[i] is not None:
                # 如果该张量对应着原始的数据输入, 那么不需要为其创建接收上一阶段传来数据的op
                self.__recv_fwd_prop_ops.append(None)
                self.__pipeline_input_placeholders.append(None)
                self.__fwd_prop_from_previous_count -= 1

                if self.__samples:
                    assert self.__samples == \
                           len(inputs_data[input_data_indexes[i]])
                else:
                    self.__samples = len(inputs_data[input_data_indexes[i]])

                continue

            pipe: PipelinePipe = stage.input_pipes[i]

            placeholder = tf.compat.v1.placeholder(
                dtype=tf.float32, shape=(None, *pipe.shape),
                name=f'pipeline-{self.__pipeline_model_rank}-'
                     f'receive-forward-outputs-{i}-placeholder'
            )

            self.__pipeline_input_placeholders.append(placeholder)
            self.__recv_fwd_prop_ops.append(
                CPPBackend.tf_lib().receive_tensor(
                    placeholder,
                    sender=pipe.comes_from.stage_rank,
                    name=f'pipeline-{self.__pipeline_model_rank}-'
                         f'{pipe.comes_from.stage_rank}-'
                         f'forward-input-{i}-to-stage-'
                         f'{self.__pipeline_communicator.rank}',
                    # 这里要传入handle(整数值), 而不是一个python对象
                    communicator_id=self.__pipeline_communicator.id
                )
            )

        # 由于发送和接收梯度的Tensor不会同时运行, 所以公用一个placeholder, 如有需要
        # 可以每个tensor对应一个placeholder
        self.__pipeline_output_placeholders = []
        self.__send_fwd_outputs_ops = []
        self.__receive_gradient_ops = []

        self.__pipeline_output_count = len(stage.output_pipes)
        self.__back_prop_to_next_count = self.__pipeline_output_count

        self.__if_output_recv_grads = []

        # 由于在多输出接受下一阶段输入时有可能出现某个输出接受到下一个微批次的梯度,而其他输出
        # 没有接受到梯度的情况, 所以需要在接收端缓存一下
        self.__recv_grads_cache = []

        # 定义静态计算图: 发送前向传播结果和接受后向传播梯度
        for i in range(len(stage.output_pipes)):
            self.__if_output_recv_grads.append({})
            self.__recv_grads_cache.append({})

            if target_data_indexes[i] is not None:
                # 如果该张量对应着原始的目标, 那么不需要为其创建传输前向传播的op
                self.__pipeline_output_placeholders.append(None)
                self.__send_fwd_outputs_ops.append(None)
                self.__receive_gradient_ops.append(None)

                self.__back_prop_to_next_count -= 1

                if self.__samples:
                    assert self.__samples == \
                           len(targets_data[target_data_indexes[i]])
                else:
                    self.__samples = len(targets_data[target_data_indexes[i]])
                continue
            pipe: PipelinePipe = stage.output_pipes[i]
            placeholder = tf.compat.v1.placeholder(
                dtype=tf.float32, shape=(None, *pipe.shape)
            )
            self.__pipeline_output_placeholders.append(placeholder)
            send_ops = []
            receive_ops = {}
            for j in range(len(pipe.send_to)):
                receiving_stage: PipelineStage = pipe.send_to[j]
                self.__if_output_recv_grads[i][
                    receiving_stage.stage_rank] = False

                self.__recv_grads_cache[i][receiving_stage.stage_rank] = []

                send_ops.append(
                    CPPBackend.tf_lib().send_tensor(
                        placeholder,
                        receiver=receiving_stage.stage_rank,
                        name=f'pipeline-{self.__pipeline_model_rank}-'
                             f'stage-{self.stage.stage_rank}-'
                             f'forward-input-{i}-to-stage-'
                             f'{receiving_stage.stage_rank}',
                        communicator_id=self.__pipeline_communicator.id
                    ))

                assert receiving_stage.stage_rank not in receive_ops.keys(), \
                    'a pipe can not send data to a stage more than once'

                receive_ops[receiving_stage.stage_rank] = \
                    CPPBackend.tf_lib().receive_tensor(
                        placeholder,
                        sender=receiving_stage.stage_rank,
                        communicator_id=self.__pipeline_communicator.id,
                        name=f'pipeline-{self.__pipeline_model_rank}'
                             f'-stage-{self.__pipeline_communicator.rank}-'
                             f'back-gradient-of-output-{i}-from-stage-'
                             f'{receiving_stage.stage_rank}',
                    )

            self.__send_fwd_outputs_ops.append(tuple(send_ops))
            self.__receive_gradient_ops.append(receive_ops)

        self.__batch_size = batch_size
        self.__micro_batch_size = micro_batch_size if micro_batch_size else \
            batch_size

        self.__micro_batch_inputs = []

        self.__stage_communicator = stage.pipeline_model.stage_communicator
        # 第一次fit需要进行初始的变量广播, 这个变量用于记录是否需要进行初始变量广播
        self.__do_initial_params_broadcast = self.__stage_communicator.size > 1

        self.__total_micro_batches, self.__micro_batches_per_batch = \
            self.__get_total_micro_batches()

        self.__epochs = epochs
        self.__verbose = verbose

        if callbacks is None:
            self.__callbacks = []
        else:
            if isinstance(Iterable, callbacks):
                self.__callbacks = [*callbacks]
            else:
                self.__callbacks = [callbacks]

        self.__micro_batch_controller = micro_batch_controller

        self.__fit_args = fit_args

    @property
    def stage(self):
        return self.__stage

    @property
    def session(self) -> tf.compat.v1.Session:
        return self.__session

    def run(self) -> History:
        if self.__do_initial_params_broadcast:
            self.__do_initial_params_broadcast = False
            self.__callbacks.append(
                InitialParametersBroadcastCallBack(
                    0, self.__stage_communicator)
            )
        self.__callbacks.append(self.TrainingCallback(self))

        info(f'total micro batches: {self.__total_micro_batches}')

        if self.__pipeline_model_rank == 0 and \
                self.__pipeline_communicator.rank == \
                self.__pipeline_communicator.size - 1:
            # 只有rank为0的Model且是最后一个阶段的进程才输出信息
            verbose = self.__verbose
        else:
            verbose = 0

        history = self.stage.model.fit(
            self.__fit_data_generator(),
            steps_per_epoch=self.__total_micro_batches,
            epochs=self.__epochs,
            verbose=verbose,
            callbacks=self.__callbacks,
            # 不允许使用多线程调用self.__fit_data_generator()
            workers=0,
            **self.__fit_args
        )

        info('done, sending Done to previous stage')
        for i in range(self.__pipeline_input_count):
            pipe: PipelinePipe = self.stage.input_pipes[i]
            if pipe.comes_from is None:
                # 没有上一阶段, 不发送信息
                continue
            Message.send(json.dumps({
                'code': self.MessageCode.DONE.value,
                'index': pipe.index_of(pipe.comes_from)
            }), pipe.comes_from.stage_rank,
                communicator=self.__pipeline_communicator
            )

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
            self.__micro_batch_inputs.clear()

    def __get_total_micro_batches(self) -> (int, int):
        """
        根据输入的训练样数量和批次信息计算出需要的微批次总数
        @return: 需要的微批次总数, 每个批次的微批次总数(不包括最后的微批次)
        """
        if self.__pipeline_communicator.rank == 0:
            # 因为进行了拓扑排序, 所以可以肯定0进程的模型的输入都有直接的数据, 可以知道所有的
            # 样例数
            assert self.__samples is not None
            Message.broadcast(json.dumps({
                'samples': self.__samples
            }), 0, self.__pipeline_communicator)
        else:
            init_msg = Message.broadcast('', 0, self.__pipeline_communicator)
            self.__samples = json.loads(init_msg.msg)['samples']

        assert self.__samples is not None

        info(f'samples: {self.__samples}')

        micro_batches_per_batch = self.__batch_size // self.__micro_batch_size
        if micro_batches_per_batch * self.__micro_batch_size \
                < self.__batch_size:
            micro_batches_per_batch += 1

        tot_batches = self.__samples // self.__batch_size
        tot_micro_batches = tot_batches * micro_batches_per_batch
        if tot_batches * self.__batch_size < self.__samples:
            remain = self.__samples - tot_batches * self.__batch_size
            tot_batches += 1
            remain_micro_batches = remain // self.__micro_batch_size
            if remain_micro_batches * self.__micro_batch_size < remain:
                remain_micro_batches += 1
            tot_micro_batches += remain_micro_batches

        return tot_micro_batches, micro_batches_per_batch

    def __get_micro_batch_inputs(
            self, micro_batch_begin: int, micro_batch_end: int) -> tuple:
        """
        从上一阶段获取一个微批次前向传播的结果
        :@param batch_end: 数据batch在整个数据库的结束索引
        :@param micro_batch_begin: 当前微批次数据batch在整个数据库的结束索引
        :@param micro_batch_end: 当前微批次数据batch在整个数据库的结束索引
        :@return: 模型batch的输入
        """
        result = [None for _ in range(self.__pipeline_input_count)]

        recv_counter = 0
        while recv_counter < self.__fwd_prop_from_previous_count:
            msg = Message.listen(communicator=self.__pipeline_communicator)
            msg_obj = json.loads(msg.msg)
            assert msg_obj['code'] == self.MessageCode.FPROP_RESULT.value
            assert micro_batch_begin == msg_obj['micro_batch_begin']
            assert micro_batch_end == msg_obj['micro_batch_end']
            input_index = msg_obj['index']
            info(f'getting forward propagation result:'
                 f'[{input_index}][{micro_batch_begin}:{micro_batch_end}]')
            result[input_index] = self.session.run(
                self.__recv_fwd_prop_ops[input_index],
                feed_dict={
                    self.__pipeline_input_placeholders[input_index]:
                        np.zeros((
                            micro_batch_end - micro_batch_begin,
                            *self.stage.input_pipes[input_index].shape
                        ))
                }
            )
            recv_counter += 1

        for i in range(len(self.stage.input_pipes)):
            if result[i] is not None:
                # 已经从上一阶段获取到数据, 跳过
                continue
            result[i] = self.__inputs_data[i][micro_batch_begin:micro_batch_end]

        return tuple(result)

    def __send_micro_batch_outputs(
            self, micro_batch_begin: int, micro_batch_end: int, outputs: tuple):
        from ddl.tensorflow.keras.parallelism.pipeline.stage import \
            PipelineStage

        for i in range(self.__pipeline_output_count):
            if self.__target_data_indexes[i] is not None:
                # 这是输出张量, 不需要发送给下一阶段
                continue

            pipe: PipelinePipe = self.stage.output_pipes[i]

            for j in range(len(pipe.send_to)):
                stage: PipelineStage = pipe.send_to[j]

                Message.send(json.dumps({
                    'code': self.MessageCode.FPROP_RESULT.value,
                    'micro_batch_begin': micro_batch_begin,
                    'micro_batch_end': micro_batch_end,
                    'index': pipe.index_of(stage)
                }), stage.stage_rank, self.__pipeline_communicator)

                self.session.run(
                    self.__send_fwd_outputs_ops[i][j],
                    feed_dict={
                        self.__pipeline_output_placeholders[i]: outputs[i],
                    }
                )

    def __get_micro_batch_targets(self, begin: int, end: int) -> tuple:
        from ddl.tensorflow.keras.parallelism.pipeline.stage import \
            PipelineStage
        result = []

        received_done_message = False

        # 先读取缓存, 看是否已经获取到梯度
        for i in range(self.__pipeline_output_count):
            pipe: PipelinePipe = self.stage.output_pipes[i]
            grads = None
            for j in range(len(pipe.send_to)):
                stage: PipelineStage = pipe.send_to[j]
                caches: list = self.__recv_grads_cache[i][stage.stage_rank]

                if len(caches) > 0:
                    if grads is None:
                        grads = caches.pop(0)
                    else:
                        # 由复合函数偏微分的计算公式可以得出这里应该将接收到的梯度累加起来
                        grads = grads + caches.pop(0)
                    # 读取到缓存,
                    self.__if_output_recv_grads[i][
                        stage.stage_rank] = True
            result.append(grads)

        while True:
            # 检查是否所有梯度都接收到
            all_grads_received = True

            for i in range(self.__pipeline_output_count):
                pipe: PipelinePipe = self.stage.output_pipes[i]
                for j in range(len(pipe.send_to)):
                    stage: PipelineStage = pipe.send_to[j]
                    if self.__if_output_recv_grads[i][stage.stage_rank]:
                        continue
                    all_grads_received = False
                    break
                if not all_grads_received:
                    break

            if all_grads_received:
                if received_done_message:
                    # 收到了done信息, 抛出停止迭代异常, 说明结束
                    info('received done, exiting')
                    raise StopIteration
                break

            msg = Message.listen(communicator=self.__pipeline_communicator)
            msg_obj = json.loads(msg.msg)
            index = msg_obj['index']
            sender = msg.sender
            if msg_obj['code'] == self.MessageCode.DONE.value:
                # 收到某个Pipe的Done信息, 那么就准备结束训练过程
                info(f'received done from stage-{sender} of output-{index}')
                received_done_message = True
                self.__if_output_recv_grads[index][sender] = True
                continue

            assert msg_obj['code'] == self.MessageCode.BPROP_GRAD.value

            shape = (end - begin, *self.stage.output_pipes[index].shape)

            res = self.session.run(
                self.__receive_gradient_ops[index][sender],
                feed_dict={
                    self.__pipeline_output_placeholders[index]:
                        np.zeros(shape=shape)
                }
            )

            if self.__if_output_recv_grads[index][sender]:
                # 在这一微批次中已经收到了梯度, 则存入缓存
                self.__recv_grads_cache[index][sender].append(res)
            else:
                self.__if_output_recv_grads[index][sender] = True
                if result[index] is None:
                    result[index] = res
                else:
                    # 由复合函数偏微分的计算公式可以得出这里应该将接收到的梯度累加起来,
                    # 这个可以写到op里 提高并行性
                    result[index] = result[index] + res

        for i in range(len(self.stage.output_pipes)):
            # 重置记录是否已经收到梯度的状态
            pipe: PipelinePipe = self.stage.output_pipes[i]
            for j in range(len(pipe.send_to)):
                stage: PipelineStage = pipe.send_to[j]
                self.__if_output_recv_grads[i][stage.stage_rank] = False

            # 剩余的结果需要从原始数据中获取
            if result[i] is not None:
                # 已经获取到后向传播结果, 跳过
                continue
            result[i] = self.__targets_data[i][begin:end]

        return tuple(result)

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

            micro_batch_segments = []

            while micro_batch_begin < batch_end:
                # 获取所有批次的微批次输入
                micro_batch_segments.append(
                    (micro_batch_begin, micro_batch_end))
                micro_batch_inputs = self.__get_micro_batch_inputs(
                    micro_batch_begin, micro_batch_end
                )
                self.__micro_batch_inputs.append(micro_batch_inputs)
                # todo: 由于需要得到本进程模型的输出数据, 所以这里需要执行预测步骤,
                #  但是fit时又会进行一次前向传播, 所以浪费了一次前向传播的步骤, 或许可以
                #  覆写fit的逻辑, 或者直接把微批次的操作写入op里

                # 好像是多线程的Session的问题, 如果不这样跑, 会发生Tensor不在图里的异常
                micro_batch_outputs = \
                    self.stage.model.predict(micro_batch_inputs)
                if not isinstance(micro_batch_outputs, (tuple, list)):
                    micro_batch_outputs = (micro_batch_outputs,)

                self.__send_micro_batch_outputs(
                    micro_batch_begin, micro_batch_end,
                    micro_batch_outputs
                )
                micro_batch_begin = micro_batch_end
                micro_batch_end = min(
                    micro_batch_begin + self.__micro_batch_size,
                    batch_end
                )

            self.__micro_batch_controller.initialize_micro_batch_vars(
                self.__micro_batch_inputs
            )

            # 进行后向传播
            for i in range(len(self.__micro_batch_inputs)):
                info(f'back propagation micro batch['
                     f'{micro_batch_segments[i][0]},'
                     f' {micro_batch_segments[i][1]}]')
                try:
                    targets = self.__get_micro_batch_targets(
                        *micro_batch_segments[i])
                    yield self.__micro_batch_inputs[i], targets
                except StopIteration:
                    for each in self.__recv_grads_cache:
                        for value in each.values():
                            assert len(value) == 0, \
                                'bug exists, cache not clean when finished'
                    return

            batch_begin = 0 if batch_end >= self.__samples else batch_end
