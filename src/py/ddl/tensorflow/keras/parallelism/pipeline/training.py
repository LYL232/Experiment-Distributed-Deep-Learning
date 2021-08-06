from ddl.log import info
from ddl.message import Message
from ddl.tensorflow.keras.parallelism.data import \
    InitialParametersBroadcastCallBack
from ddl.tensorflow.util import executing_eagerly
from ddl.tensorflow.keras.parallelism.pipeline.micro_batch_controller import \
    MicroBatchController
from ddl.tensorflow.keras.parallelism.data.metric_average_callback import \
    MetricAverageCallback
from ddl.tensorflow.keras.parallelism.data.lr_warm_up_callback import \
    LearningRateWarmupCallback

import json
import tensorflow as tf
from tensorflow.python.keras.engine.training_v1 import Model as ModelV1
from tensorflow.python.keras.engine.training import Model as ModelV2
from tensorflow.python.keras import backend
from tensorflow.keras.callbacks import History, Callback
import numpy as np
from collections.abc import Iterable


class TrainingExecutor:
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
            inputs_data: tuple, original_input_data_indexes: tuple or list,
            targets_data: tuple, original_target_data_indexes: tuple or list,
            batch_size: int, epochs: int,
            session: tf.compat.v1.Session,
            micro_batch_size: int = None,
            callbacks=None,
            verbose: int = 1, **fit_args
    ):
        from ddl.tensorflow.keras.parallelism.pipeline.stage import \
            PipelineStage
        assert isinstance(stage, PipelineStage)
        assert len(original_input_data_indexes) == len(stage.input_pipes)
        assert len(original_target_data_indexes) == len(stage.output_pipes)

        self.__original_input_data_indexes = original_input_data_indexes
        self.__original_target_data_indexes = original_target_data_indexes

        self.__session = session

        self.__stage = stage
        self.__inputs_data = inputs_data
        self.__targets_data = targets_data
        self.__pipeline_model_rank = stage.pipeline_model.pipeline_model_rank
        self.__pipeline_communicator = \
            stage.pipeline_model.pipeline_communicator

        self.__pipeline_input_count = len(stage.input_pipes)

        self.__samples = None

        for i in range(self.__pipeline_input_count):
            if original_input_data_indexes[i] is not None:
                if self.__samples:
                    assert self.__samples == \
                           len(inputs_data[original_input_data_indexes[i]])
                else:
                    self.__samples = len(
                        inputs_data[original_input_data_indexes[i]])

        self.__pipeline_output_count = len(stage.output_pipes)

        for i in range(len(stage.output_pipes)):
            if original_target_data_indexes[i] is not None:
                target = targets_data[original_target_data_indexes[i]]
                if target is None:
                    continue
                if self.__samples:
                    assert self.__samples == len(target)
                else:
                    self.__samples = len(target)

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
            if isinstance(callbacks, Iterable):
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

        lr_warm_up_verbose = self.__fit_args.pop('lr_warm_up_verbose', 1)
        if self.__pipeline_model_rank == 0 and \
                self.__pipeline_communicator.rank == \
                self.__pipeline_communicator.size - 1:
            # 只有rank为0的Model且是最后一个阶段的进程才输出信息
            verbose = self.__verbose
        else:
            verbose = 0
            lr_warm_up_verbose = 0

        lr_warm_up = self.__fit_args.pop(
            'lr_warm_up_epochs',
            min(self.__epochs, 5)
        )

        base_lr = float(backend.get_value(self.stage.model.optimizer.lr))

        if self.__pipeline_communicator.rank == \
                self.__pipeline_communicator.size - 1:
            # 只有每条流水线的最后一个阶段才输出metric信息，所以只在流水线的最后一个进程
            # 加入这个callback
            self.__callbacks.append(
                MetricAverageCallback(communicator=self.__stage_communicator),
            )

        self.__callbacks.extend([
            LearningRateWarmupCallback(
                warmup_epochs=lr_warm_up,
                communicator=self.__stage_communicator,
                verbose=lr_warm_up_verbose,
                initial_lr=base_lr * self.__stage_communicator.size
            ),
            self.TrainingCallback(self)
        ])

        info(f'total micro batches: {self.__total_micro_batches}')

        if executing_eagerly():
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
        else:
            with self.session.graph.as_default():
                with self.session.as_default():
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

        info('training done returning')
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

    def __get_micro_batch_inputs(self, begin: int, end: int) -> tuple:
        """
        从上一阶段获取一个微批次前向传播的结果
        :@param batch_end: 数据batch在整个数据库的结束索引
        :@param begin: 当前微批次数据batch在整个数据库的结束索引
        :@param end: 当前微批次数据batch在整个数据库的结束索引
        :@return: 模型batch的输入
        """
        result = []
        for i in range(len(self.__original_input_data_indexes)):
            if self.__original_input_data_indexes[i] is None:
                # 中间输入, 填入0数组
                result.append(
                    np.zeros((
                        end - begin,
                        *self.stage.input_shape[i]
                    ))
                )
            else:
                result.append(self.__inputs_data[i][begin:end])
        return tuple(result)

    def __get_micro_batch_targets(self, begin: int, end: int) -> dict:
        result = {}
        model = self.__stage.model
        if isinstance(model, ModelV1):
            # noinspection PyProtectedMember
            outputs_names = self.__stage.model._feed_output_names
        else:
            assert isinstance(model, ModelV2)
            outputs_names = model.output_names
        for i, name in enumerate(outputs_names):
            if self.__original_target_data_indexes[i] is None:
                # 中间输出, 填入0数组
                result[name] = \
                    np.zeros((
                        end - begin,
                        *self.stage.output_shape[i]
                    ))
            else:
                data = self.__targets_data[i]
                if data is None:
                    shape = [0 if each is None else each
                             for each in self.stage.output_shape[i]]
                    result[name] = np.zeros((end - begin, *shape))
                else:
                    result[name] = self.__targets_data[i][begin:end]
        return result

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
                targets = self.__get_micro_batch_targets(
                    *micro_batch_segments[i])
                yield self.__micro_batch_inputs[i], targets

            batch_begin = 0 if batch_end >= self.__samples else batch_end
