from ddl.tensorflow.communicator import Communicator
from ddl.data import DistributedData
from ddl.tensorflow.keras.parallelism.data import \
    data_parallelism_distributed_optimizer_wrapper
from ddl.tensorflow.keras.parallelism.pipeline.pipe import PipelinePipe, \
    PipelineInput
from ddl.tensorflow.keras.parallelism.pipeline.training import \
    TrainingExecutor
from ddl.tensorflow.keras.parallelism.pipeline.predict import \
    PredictExecutor
from ddl.tensorflow.keras.parallelism.pipeline.micro_batch_controller import \
    MicroBatchController
from ddl.tensorflow import util

from sys import stderr
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Optimizer
from tensorflow.keras.callbacks import History, ModelCheckpoint
from tensorflow.python.keras import backend
import tensorflow as tf
import numpy as np
from typing import Tuple
import os


def intermediate_loss(grad, output):
    """
    中间loss函数, 此函数的训练效果理论上与原模型的训练效果一致,
    其实就是偏差与输出的Hadamard积, 即按对应元素相乘
    :@param grad: 从下一阶段后向传播得到的梯度
    :@param output: 模型的输出
    :@return: loss
    """
    return tf.reduce_mean(
        tf.multiply(grad, output), axis=None,
        name='intermediate_loss'
    )


def zero_intermediate_loss(grad, output):
    """
    中间loss函数, 此函数只是为了引导tensorflow不要进行某些没有梯度的中间输出的剪枝
    :@param grad: 从下一阶段后向传播得到的梯度
    :@param output: 输出恒为0
    :@return: 0
    """
    # todo: 用passby来减少计算量
    return tf.reduce_mean(
        tf.multiply(grad, output),
        name='zero_intermediate_loss',
        axis=None
    ) * 0


class PipelineModel(Model):
    """
    流水线通用模型
    """

    def __init__(
            self,
            inputs: tuple or list or PipelinePipe,
            outputs: tuple or list or PipelinePipe,
            *args, **kwargs
    ):
        super().__init__(*args, **kwargs)

        if isinstance(inputs, PipelinePipe):
            inputs = (inputs,)
        else:
            assert isinstance(inputs, (tuple, list))
            assert len(inputs) > 0
            assert all([isinstance(each, PipelinePipe) for each in inputs])
        if isinstance(outputs, PipelinePipe):
            outputs = (outputs,)
        else:
            assert isinstance(outputs, (tuple, list))
            assert len(outputs) > 0
            assert all([isinstance(each, PipelinePipe) for each in outputs])

        self.__original_inputs_index = {}
        self.__original_outputs_index = {}
        for i in range(len(inputs)):
            self.__original_inputs_index[id(inputs[i])] = i
        for i in range(len(outputs)):
            self.__original_outputs_index[id(outputs[i])] = i

        # inputs和outputs不能有重复元素
        assert len(self.__original_inputs_index) == len(inputs), \
            'duplicate inputs'
        assert len(self.__original_outputs_index) == len(outputs), \
            'duplicate outputs'

        # inputs和outputs交集必须为空
        assert len(set(self.__original_inputs_index.keys()).intersection(
            set(self.__original_outputs_index))) == 0, \
            'inputs and outputs must not have the same tensor'

        # 检查所有inputs没有来源
        for stage in inputs:
            assert stage.comes_from is None
        # 检查所有的outputs没有接收方
        for stage in outputs:
            assert len(stage.send_to) == 0

        # 先跑一遍图定义, 用id作为唯一键, 遍历所有tensor
        self.__total_stages = {}

        queue = []
        self.__input_stages_id_set = set()
        # 找到所有input张量连接的阶段, 找到输入阶段, 要求其所有输入张量必须没有comes_from
        for pipe in inputs:
            for stage in pipe.send_to:
                # 如果已经是输入阶段, 或者其存在任何输入来自于其他阶段, 那么它肯定不是输入
                # 阶段, 跳过
                if id(stage) in self.__input_stages_id_set or any(
                        each.comes_from is not None
                        for each in stage.input_pipes
                ):
                    continue
                queue.append(stage)
                self.__input_stages_id_set.add(id(stage))

        # 如果没有输入阶段, 那么整个图定义就是无效的
        assert len(queue) > 0, \
            'there are no input stage in the graph definition'

        while len(queue) > 0:
            stage = queue.pop(0)
            self.__total_stages[id(stage)] = stage
            for pipe in stage.output_pipes:
                if len(pipe.send_to) == 0:
                    assert id(pipe) in self.__original_outputs_index.keys(), \
                        'there exists tensor that does not connect to next' \
                        ' tensor but not in outputs'
                else:
                    for stage in pipe.send_to:
                        if id(stage) in self.__total_stages.keys():
                            continue
                        self.__total_stages[id(stage)] = stage
                        queue.append(stage)

        # 记录每个stage的入度
        in_degree = {}
        queue = []
        for stage in self.__total_stages.values():
            if all(pipe.comes_from is None for pipe in stage.input_pipes):
                assert id(stage) in self.__input_stages_id_set, \
                    'there exists a PipelineTensor not in inputs, but' \
                    ' also dose not comes from other PipelineTensor'
                queue.append(stage)
                in_degree[id(stage)] = 0
            else:
                in_degree[id(stage)] = len(stage.input_pipes)

        self.__stages_rank = {}
        self.__stages = []
        # 从inputs到outputs跑一遍拓扑排序, 得出一个拓扑序
        while len(queue) > 0:
            stage = queue.pop(0)
            stage.attach_pipeline_model(self, len(self.__stages_rank))
            self.__stages_rank[id(stage)] = len(self.__stages_rank)
            self.__stages.append(stage)
            for output in stage.output_pipes:
                for next_stage in output.send_to:
                    in_degree[id(next_stage)] -= 1
                    if in_degree[id(next_stage)] > 0:
                        continue
                    queue.append(next_stage)

        # 如果入度为0的节点数不等于图中所有的PipelineTensor, 说明有回路
        assert len(self.__stages_rank) == len(self.__total_stages), \
            'can not define a pipeline tensor graph with a ring sub graph'

        # todo: 在此可以进行各个阶段间的输入输出形状匹配检查

        self.__processes_required = len(self.__total_stages)

        # 以下属性由compile赋值
        # 整体模型运行的通信域
        self.__model_comm = None
        # 正在执行的流水线并行模型总数
        self.__pipeline_models = None
        # 本节点所属PipelineModel的rank, 应用数据并行时可能不为0, 否则为0
        self.__pipeline_model_rank = None
        # 此Pipeline模型运行的通信域
        self.__pipeline_comm = None
        # 本进程所在Stage的通信域, 由所有pipeline模型的相同Stage组成
        self.__stage_comm = None
        # 当分配给self.__pipeline_comm的进程数大于所需进程数时, 只使用rank小的进程
        # 这些进程的self.__work标记为True, 否则为False, 代表不工作
        self.__work = None

        self._is_compiled = False

        self.__session = None if util.executing_eagerly() \
            else backend.get_session()

        self.__micro_batch_controller = None

    @property
    def processes_require(self) -> int:
        return self.__processes_required

    @property
    def pipeline_communicator(self) -> Communicator:
        assert self.__work is not None, \
            'access pipeline model communicator before compiling'
        assert self.__work, 'this process does not work'
        return self.__pipeline_comm

    @property
    def model_communicator(self) -> Communicator:
        assert self.__work is not None, \
            'access pipeline model communicator before compiling'
        assert self.__work, 'this process does not work'
        return self.__model_comm

    @property
    def stage_communicator(self) -> Communicator:
        assert self.__work is not None, \
            'access pipeline model communicator before compiling'
        assert self.__work, 'this process does not work'
        return self.__stage_comm

    @property
    def pipeline_model_rank(self) -> int:
        assert self.__work is not None, \
            'access pipeline model communicator before compiling'
        assert self.__work, 'this process does not work'
        return self.__pipeline_model_rank

    def compile(
            self, optimizer: str or Optimizer = 'rmsprop', loss=None,
            metrics=None,
            loss_weights=None, sample_weight_mode=None, weighted_metrics=None,
            **kwargs):
        from ddl.tensorflow.keras.parallelism.pipeline.stage import \
            PipelineStage
        if loss is not None and not isinstance(loss, (tuple, list)):
            loss = (loss,)
        if metrics is not None and not isinstance(metrics, (tuple, list)):
            metrics = (metrics,)

        self.__pipeline_model_rank = 0
        self.__model_comm = kwargs.pop('communicator', Communicator.world())
        # 是否尝试数据并行
        try_data_parallelism = kwargs.pop('try_data_parallelism', True)

        assert isinstance(self.__model_comm, Communicator)

        # 本进程所在流水线模型默认执行的通信域是整体通信域
        self.__pipeline_comm = self.__model_comm

        optimizer = self._get_optimizer(optimizer)

        # 如果是分布式数据并行, 那么需要先判断启动的进程数是否足够进行数据并行,
        # 获取通信域, 如果不需要数据并行, 则通信域为全体进程
        if try_data_parallelism:
            processes = Communicator.world().size
            self.__pipeline_models = processes // self.processes_require
            if self.__pipeline_models < 0:
                raise Exception(
                    f'the pipeline model needs at least '
                    f'{self.processes_require} processes, but only '
                    f'{processes} provided'
                )
            elif self.__pipeline_models == 1:
                stderr.write(
                    f'WARNING: the pipeline model needs at least '
                    f'{self.processes_require} processes, but only '
                    f'{processes} provided, so no data parallelism '
                    f'applied\n')
                try_data_parallelism = False
            else:
                # 进程数量可以进行数据并行, 进行通信域分割
                self.__pipeline_model_rank = \
                    self.__model_comm.rank // self.processes_require
                self.__pipeline_comm: Communicator = \
                    self.__pipeline_comm.split_communicator(
                        self.__pipeline_model_rank
                    )

        # 是否真正执行动作
        self.__work = self.__pipeline_comm.rank < self.processes_require

        # 获得通信域后, 每个Stage在self.__stages里的下标就是执行该Stage的在该通讯域下的rank,
        # 但是由于allreduce需要通信域内的所有进程参与, 所以必须保证通信域的大小正好是stages的数量
        if self.__pipeline_comm.size > self.processes_require:
            stderr.write(
                f'WARNING: each pipeline model needs '
                f'{self.processes_require} processes, but pipeline-model-'
                f'{self.__pipeline_model_rank} got '
                f'{self.__pipeline_comm.size} processes, so the last '
                f'few unnecessary processes have nothing to do\n')

            # 继续将这个通信域分割, 会有一个通信域包含正好所需的进程数量
            self.__pipeline_comm = self.__pipeline_comm.split_communicator(
                0 if self.__work else 1
            )
            if not self.__work:
                stderr.write(
                    f'WARNING: process has no thing to do, in '
                    f'pipeline-model-{self.__pipeline_model_rank}, global rank'
                    f'{self.__model_comm.size}\n')

        if self.__work:
            stage: PipelineStage = self.__stages[self.__pipeline_comm.rank]
            # 加载本阶段模型
            stage.build()
            # 进行数据并行, 分割stage通信域
            self.__stage_comm = \
                self.__model_comm.split_communicator(
                    self.__pipeline_comm.rank, self.__pipeline_model_rank
                )
            if try_data_parallelism:
                # 获取数据并行分布式优化器
                optimizer = data_parallelism_distributed_optimizer_wrapper(
                    optimizer, self.__stage_comm
                )

            output_loss = {}
            metrics_dict = {}
            for i, pipe in enumerate(stage.output_pipes):
                name = stage.model.output_names[i]
                if id(pipe) in self.__original_outputs_index.keys():
                    index = self.__original_outputs_index[id(pipe)]

                    the_loss = loss[index]
                    the_metric = metrics[index]

                    if the_loss is None:
                        the_loss = zero_intermediate_loss
                        the_metric = zero_intermediate_loss

                    output_loss[name] = the_loss
                    metrics_dict[name] = the_metric
                else:
                    if pipe.convey_gradient():
                        output_loss[name] = intermediate_loss
                        metrics_dict[name] = intermediate_loss
                    else:
                        output_loss[name] = zero_intermediate_loss
                        metrics_dict[name] = [zero_intermediate_loss]

            self.__micro_batch_controller = MicroBatchController(
                stage, optimizer, self.__session
            )

            stage.model.compile(
                optimizer=optimizer,
                loss=output_loss,
                metrics=metrics_dict,
                loss_weights=loss_weights,
                sample_weight_mode=sample_weight_mode,
                weighted_metrics=weighted_metrics,
                # 设置`experimental_run_tf_function=False` 让TensorFlow
                # 使用传入的opt计算梯度
                experimental_run_tf_function=False,
                **kwargs
            )

        self._is_compiled = True

    def fit(
            self, x=None, y=None,
            batch_size=None, epochs=1, verbose=1,
            callbacks=None,
            validation_split=0., validation_data=None,
            shuffle=True, class_weight=None, sample_weight=None,
            initial_epoch=0,
            steps_per_epoch=None,
            validation_steps=None, validation_batch_size=None,
            validation_freq=1,
            max_queue_size=10, workers=None,
            use_multiprocessing=False,
            micro_batch_size: int = None,
            clear_session: bool = True
    ) -> History or list:
        """
        启动训练进程, 所有执行非最后一个Stage的进程将监听消息,
        参数比较复杂, todo: 还需研究研究每个Stage的参数的意义和如何分配
        @param x: 非输入Stage无效, 如果是DispatchingData实例, 那么会进行数据分发
        @param y: 非输出Stage无效
        @param batch_size: 批次大小
        @param epochs: 训练轮次数
        @param verbose: 非最后一个Stage为0, 这里指定的是最后一个输出stage的
        @param callbacks: 回调函数
        @param validation_split:
        @param validation_data:
        @param shuffle:
        @param class_weight:
        @param sample_weight:
        @param initial_epoch:
        @param steps_per_epoch:
        @param validation_steps:
        @param validation_batch_size:
        @param validation_freq:
        @param max_queue_size:
        @param workers:
        @param use_multiprocessing:
        @param micro_batch_size: 微批次大小, 如果为None则代表不用微批次
        @param clear_session: 是否在训练结束后清除执行会话
        @return: Model.fit(...), 如果本进程不工作, 则None
        """
        from ddl.tensorflow.keras.parallelism.pipeline.stage import \
            PipelineStage

        assert self._is_compiled, f'{Communicator.world().rank} not compiled'
        if not self.__work:
            return

        assert workers is None, 'current version not support specify workers'

        batch_size = int(batch_size)
        if micro_batch_size is not None:
            micro_batch_size = int(micro_batch_size)
            micro_batch_size = min(micro_batch_size, batch_size)
        else:
            micro_batch_size = batch_size

        if not isinstance(x, (tuple, list)):
            x = (x,)
        if not isinstance(y, (tuple, list)):
            y = (y,)

        assert 0 <= self.__pipeline_comm.rank < self.processes_require
        stage: PipelineStage = self.__stages[self.__pipeline_comm.rank]

        pipeline_inputs_data_indexes = []
        pipeline_targets_data_indexes = []
        for pipes, data_indexes, original_index in [
            (stage.input_pipes, pipeline_inputs_data_indexes,
             self.__original_inputs_index),
            (stage.output_pipes, pipeline_targets_data_indexes,
             self.__original_outputs_index)
        ]:
            for each in pipes:
                if id(each) in original_index.keys():
                    data_indexes.append(original_index[id(each)])
                else:
                    data_indexes.append(None)

        for data, data_indexes in [
            (x, set(pipeline_inputs_data_indexes)),
            (y, set(pipeline_targets_data_indexes))
        ]:
            for i in range(len(data)):
                if isinstance(data[i], DistributedData):
                    need_data = True if i in data_indexes else False
                    data[i].distribute(
                        need_data,
                        data[i].communicator.rank
                    )

        # 因为不能所有的callback在分布式情况下都有效，所以需要过滤掉一些callback
        filtered_callbacks = []
        for each in callbacks:
            if isinstance(each, ModelCheckpoint):
                if self.__pipeline_model_rank == 0:
                    # 只有第一个流水线的模型执行保存动作即可
                    assert each.save_weights_only, \
                        'only save weights allowed for now'

                    dir_path = os.path.abspath(each.filepath)

                    split_type_name = dir_path.split('.')
                    assert len(split_type_name) > 0
                    if len(split_type_name) > 1:
                        type_name = split_type_name[-1]
                        dir_path = '.'.join(
                            split_type_name[:len(split_type_name) - 1])
                    else:
                        type_name = None
                        dir_path = split_type_name[0]

                    if type_name is None:
                        type_name = 'h5'

                    stage_rank = self.__pipeline_comm.rank

                    if not os.path.exists(dir_path):
                        os.makedirs(dir_path, exist_ok=True)

                    model_path = os.path.join(
                        dir_path, f'stage-{stage_rank}-model-'
                                  f'epoch-{{epoch:02d}}.{type_name}')

                    if len(stage.model.metrics_names) > 0:
                        monitor = stage.model.metrics_names[0]
                    else:
                        monitor = ''
                    filtered_callbacks.append(
                        ModelCheckpoint(
                            monitor=monitor,
                            filepath=model_path,
                            save_best_only=each.save_best_only,
                            save_freq=each.save_freq,
                            save_weights_only=True
                        ))
            else:
                filtered_callbacks.append(each)

        fit_args = {
            'callbacks': filtered_callbacks,
            'validation_split': validation_split,
            'validation_data': validation_data,
            'shuffle': shuffle,
            'class_weight': class_weight,
            'sample_weight': sample_weight,
            'initial_epoch': initial_epoch,
            # 'steps_per_epoch': steps_per_epoch,  这个参数需要在训练过程中重定义
            'validation_steps': validation_steps,
            # 'validation_batch_size': validation_batch_size, 这个参数在静态图执行模式中无法识别
            'validation_freq': validation_freq,
            'max_queue_size': max_queue_size,
            # 不允许使用worker参数, tensorflow 多线程调用session会出问题
            # 'workers': workers,
            'use_multiprocessing': use_multiprocessing,
            'verbose': verbose
        }

        res = TrainingExecutor(
            stage, self.__micro_batch_controller,
            x, pipeline_inputs_data_indexes,
            y, pipeline_targets_data_indexes,
            session=self.__session,
            batch_size=batch_size, micro_batch_size=micro_batch_size,
            epochs=epochs,
            **fit_args
        ).run()
        if clear_session:
            tf.keras.backend.clear_session()
        return res

    def predict(
            self,
            x,
            batch_size=None,
            verbose=0,
            steps=None,
            callbacks=None,
            max_queue_size=10,
            workers=None,
            use_multiprocessing=False,
            micro_batch_size=None) -> np.ndarray or Tuple[np.ndarray]:
        from ddl.tensorflow.keras.parallelism.pipeline.stage import \
            PipelineStage

        assert self._is_compiled, f'{Communicator.world().rank} not compiled'
        if not self.__work:
            return

        if micro_batch_size is not None:
            batch_size = int(batch_size)
            micro_batch_size = int(micro_batch_size)
            micro_batch_size = min(micro_batch_size, batch_size)

        assert workers is None, 'current version not support specify workers'

        if not isinstance(x, (tuple, list)):
            x = (x,)

        stage: PipelineStage = self.__stages[self.__pipeline_comm.rank]

        pipeline_inputs_data_indexes = []
        for each in stage.input_pipes:
            if id(each) in self.__original_inputs_index.keys():
                pipeline_inputs_data_indexes.append(
                    self.__original_inputs_index[id(each)]
                )
            else:
                pipeline_inputs_data_indexes.append(None)

        for each_data in x:
            if isinstance(each_data, DistributedData):
                need_data_indexes = tuple(filter(
                    lambda a: a is not None,
                    pipeline_inputs_data_indexes
                ))
                each_data.distribute(
                    len(need_data_indexes) > 0,
                    each_data.communicator.rank
                )

        predict_args = {
            'callbacks': callbacks,
            # 'steps': steps,  这个参数需要在训练过程中重定义
            # 'validation_batch_size': validation_batch_size, 这个参数在静态图执行模式中无法识别
            'max_queue_size': max_queue_size,
            # 不允许使用worker参数, tensorflow 多线程调用session会出问题
            # 'workers': workers,
            'use_multiprocessing': use_multiprocessing,
            'verbose': verbose
        }

        res = PredictExecutor(
            stage, self.__micro_batch_controller,
            x, pipeline_inputs_data_indexes,
            session=self.__session,
            batch_size=batch_size, micro_batch_size=micro_batch_size,
            **predict_args
        ).run()
        return res

    def save_weights(self, dir_path, overwrite=True, save_format=None):
        from ddl.tensorflow.keras.parallelism.pipeline.stage import \
            PipelineStage
        assert self._is_compiled
        if not self.__work or self.pipeline_model_rank != 0:
            return

        dir_path = os.path.abspath(dir_path)

        split_type_name = dir_path.split('.')
        if len(split_type_name) > 1:
            type_name = split_type_name[-1]
            dir_path = '.'.join(split_type_name[:len(split_type_name) - 1])
        else:
            type_name = None
            dir_path = split_type_name[0]

        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)

        stage_rank = self.__pipeline_comm.rank
        stage: PipelineStage = self.__stages[stage_rank]

        if type_name is None:
            type_name = 'h5'

        model_path = f'stage-{stage_rank}-model.{type_name}'
        if util.executing_eagerly():
            stage.model.save_weights(
                os.path.join(dir_path, model_path), overwrite=overwrite,
                save_format=save_format
            )
        else:
            with self.__session.graph.as_default():
                with self.__session.as_default():
                    stage.model.save_weights(
                        os.path.join(dir_path, model_path), overwrite=overwrite,
                        save_format=save_format
                    )

    def load_weights(self, dir_path, by_name=False, skip_mismatch=False):
        from ddl.tensorflow.keras.parallelism.pipeline.stage import \
            PipelineStage
        assert self._is_compiled, f'{Communicator.world().rank} not compiled'
        if not self.__work:
            return
        if not os.path.isdir(dir_path):
            raise ValueError(f'dir_path: {dir_path} must be a dir')
        stage_rank = self.__pipeline_comm.rank
        stage: PipelineStage = self.__stages[stage_rank]
        model_path = f'stage-{stage_rank}-model.h5'
        stage.model.load_weights(
            os.path.join(dir_path, model_path),
            by_name=by_name, skip_mismatch=skip_mismatch
        )


class PipelineSequentialModel(PipelineModel):
    """
    流水线序列模型
    """

    def __init__(self, stages: list, input_shape: tuple, *args, **kwargs):
        from ddl.tensorflow.keras.parallelism.pipeline.stage import \
            PipelineStage
        assert len(stages) > 0
        assert all(isinstance(each, PipelineStage) for each in stages)

        input_shape = util.formalize_shapes(input_shape)
        inputs = tuple(PipelineInput(shape=each) for each in input_shape)
        sequential = inputs
        for each in stages:
            if isinstance(sequential, (tuple, list)):
                sequential = each(*sequential)
            else:
                sequential = each(sequential)

        super().__init__(inputs=inputs, outputs=sequential, *args, **kwargs)
