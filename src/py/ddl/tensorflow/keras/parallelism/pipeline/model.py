from ddl.tensorflow.communicator import Communicator
from ddl.tensorflow.keras.parallelism.pipeline import PipelineLayer
from ddl.tensorflow.keras.parallelism.pipeline.layer import PipelineInputLayer
from ddl.tensorflow.data_dispatcher import DataDispatcher
from ddl.tensorflow.keras.parallelism.pipeline.training_stage import \
    FirstTrainingStage, IntermediateTrainingStage, LastTrainingStage
from ddl.tensorflow.keras.parallelism.data import \
    data_parallelism_distributed_optimizer_wrapper
from ddl.tensorflow.keras.models.model_prebuilder import ModelPreBuilder
from ddl.tensorflow.keras.models.model import \
    Sequential as PreBuilderSequential, \
    Model as PreBuilderModel
from ddl.message import Message
import json

from sys import stderr
from tensorflow.keras.models import Model as Model
from tensorflow.keras.optimizers import Optimizer
from tensorflow.keras.callbacks import History
import tensorflow as tf


def intermediate_loss(grad, output):
    """
    中间loss函数, 此函数的训练效果理论上与原模型的训练效果一致,
    其实就是偏差与输出的Hadamard积, 即按对应元素相乘
    :@param grad: 从下一阶段后向传播得到的梯度
    :@param output: 模型的输出
    :@return: loss
    """
    return tf.reduce_mean(tf.multiply(grad, output), axis=None)


class PipelineStage:
    """
    流水线模型的组成部分, 非第一阶段要求第一层是PipelineLayer的实例
    """

    def __init__(self, builder: ModelPreBuilder):
        """
        @param builder: 模型的预构建模型
        """
        assert isinstance(builder, ModelPreBuilder)
        self.__model = None
        self.__builder = builder
        self.__is_first_stage = None

    @property
    def pipeline_layers(self) -> tuple:
        """
        @return: 当该对象是第一Stage时, 返回None, 否则, 返回其连接上一Stage的层
        """
        return self.__builder.pipeline_layers

    @property
    def model(self) -> Model:
        return self.__builder.model

    @property
    def builder(self) -> ModelPreBuilder:
        return self.__builder

    @property
    def input_shape(self) -> tuple:
        return self.model.input_shape

    @property
    def output_shape(self) -> tuple:
        return self.model.output_shape


class PipelineModel(Model):
    """
    流水线模型
    """

    def __init__(self, stages: list, *args, **kwargs):
        """
        在PipelineModel定义时, 只在主进程上进行
        @param stages: 分配在每个机器上的阶段
        @param try_data_parallelism: 是否根据通信域进程数来尝试进行数据并行
        @param communicator: 通信域
        """
        super().__init__(*args, *kwargs)
        assert len(stages) > 1, 'pipeline model contains at least 2 stages'
        # 检查是否除了第一个阶段都有PipelineLayer作为第一层
        for i in range(1, len(stages)):
            stage = stages[i]
            assert isinstance(stage, PipelineStage)

        # 注意到外部调用后到compile之前
        # 不能修改stages, 因为是引用 可能会导致错误运行, 如果要深拷贝, 比较麻烦
        self.__stages = stages
        self.__processes_required = len(stages)

        # 以下属性由compile赋值
        # 整体模型运行的通信域
        self.__model_comm = None
        # 正在执行的流水线并行模型总数
        self.__pipeline_models = None
        # 本节点所属PipelineModel的id, 应用数据并行时可能不为0, 否则为0
        self.__pipeline_model_id = None
        # 此Pipeline模型运行的通信域
        self.__pipeline_comm = None
        # 本进程所在Stage的通信域, 由所有pipeline模型的相同Stage组成
        self.__stage_comm = None
        # 当分配给self.__pipeline_comm的进程数大于所需进程数时, 只使用rank小的进程
        # 这些进程的self.__work标记为True, 否则为False, 代表不工作
        self.__work = None
        # 是否是第一个阶段
        self.__is_first_stage = None
        # 是否是最后一个阶段
        self.__is_last_stage = None

        self._is_compiled = False
        # 本进程执行的keras.Model对象
        self.__executing_model = None

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

    def compile(
            self, optimizer: str or Optimizer = 'rmsprop', loss=None,
            metrics=None,
            loss_weights=None, sample_weight_mode=None, weighted_metrics=None,
            **kwargs):

        self.__pipeline_model_id = 0
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
                self.__pipeline_model_id = \
                    self.__model_comm.rank // self.processes_require
                self.__pipeline_comm: Communicator = \
                    self.__pipeline_comm.split_communicator(
                        self.__pipeline_model_id
                    )

        # 获得通信域后, 每个Stage在self.__stages里的下标就是执行该Stage的在该通讯域下的rank,
        # 但是由于allreduce需要通信域内的所有进程参与, 所以必须保证通信域的大小正好是stages的数量
        if self.__pipeline_comm.size > self.processes_require:
            stderr.write(
                f'WARNING: each pipeline model needs '
                f'{self.processes_require} processes, but pipeline-model-'
                f'{self.__pipeline_model_id} got '
                f'{self.__pipeline_comm.size} processes, so the last '
                f'few unnecessary processes have nothing to do\n')
            # 是否真正执行动作
            self.__work = self.__pipeline_comm.rank < self.processes_require
            # 继续将这个通信域分割, 会有一个通信域包含正好所需的进程数量
            self.__pipeline_comm = self.__pipeline_comm.split_communicator(
                0 if self.__work else 1
            )
            if not self.__work:
                stderr.write(
                    f'WARNING: process has no thing to do, in '
                    f'pipeline-model-{self.__pipeline_model_id}, global rank'
                    f'{self.__model_comm.size}\n')
                self.__is_first_stage = False
                self.__is_last_stage = False
        else:
            # 所有进程都投入工作
            self.__work = True

        if self.__work:
            self.__is_first_stage = self.__pipeline_comm.rank == 0
            self.__is_last_stage = \
                self.__pipeline_comm.rank == len(self.__stages) - 1
            if self.__pipeline_comm.rank < len(self.__stages):
                stage: PipelineStage = self.__stages[self.__pipeline_comm.rank]

                builder = stage.builder

                # 流水线前一阶段通知后一阶段自己的输出形状, 后一阶段将其作为输入形状,
                # 并插入流水线层
                if not self.__is_first_stage:
                    # 是第一阶段, 就不必收听上一阶段的输出形状
                    msg = Message.listen(self.__pipeline_comm).msg
                    input_shape = json.loads(msg)
                    # 插入流水线层
                    if isinstance(builder, PreBuilderSequential):
                        builder.insert_pipeline_input_layer(
                            PipelineInputLayer(input_shape=input_shape)
                        )
                    elif isinstance(builder, PreBuilderModel):
                        builder.set_inputs_shape(input_shape)

                if not self.__is_last_stage:
                    # 不是最后一阶段, 往下传递自己的输出形状
                    output_shape = list(stage.output_shape)

                    if isinstance(output_shape[0], tuple) or \
                            isinstance(output_shape[0], list):
                        for i in range(len(output_shape)):
                            output_shape[i] = list(output_shape[i][1:])
                    else:
                        output_shape = list(output_shape[1:])

                    msg = json.dumps(output_shape)
                    Message.send(
                        msg,
                        self.__pipeline_comm.rank + 1,
                        self.__pipeline_comm
                    )

                if not self.__is_first_stage > 0:
                    # 编译PipelineLayer
                    pipeline_layers = stage.pipeline_layers
                    assert pipeline_layers is not None, \
                        'not first stage must have pipeline layer'
                    for each in pipeline_layers:
                        assert isinstance(each, PipelineLayer)
                        each.compile_by_pipeline_model(self)

                # 进行数据并行, 分割stage通信域
                self.__stage_comm = \
                    self.__model_comm.split_communicator(
                        self.__pipeline_comm.rank, self.__pipeline_model_id
                    )
                if try_data_parallelism:
                    # 获取数据并行分布式优化器
                    optimizer = data_parallelism_distributed_optimizer_wrapper(
                        optimizer, self.__stage_comm
                    )

                self.__executing_model: Model = stage.model

                if not self.__is_last_stage:
                    loss = intermediate_loss

                self.__executing_model.compile(
                    optimizer=optimizer,
                    loss=loss,
                    metrics=metrics if self.__is_last_stage else [loss],
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
            max_queue_size=10, workers=1,
            use_multiprocessing=False,
            micro_batch_size: int = None
    ) -> History or list:
        """
        启动训练进程, 所有执行非最后一个Stage的进程将监听消息,
        参数比较复杂, todo: 还需研究研究每个Stage的参数的意义和如何分配
        @param x: 非第一个Stage无效, 如果是DispatchingData实例, 那么会
        @param y: 非最后一个Stage无效
        @param batch_size: 批次大小
        @param epochs: 训练轮次数
        @param verbose: 非最后一个Stage为0, 这里指定的是最后一个stage的
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
        @return: Model.fit(...), 如果本进程不工作, 则None
        """
        assert self._is_compiled, f'{Communicator.world().rank} not compiled'
        if not self.__work:
            return

        assert 0 <= self.__pipeline_comm.rank < self.processes_require

        if isinstance(x, DataDispatcher):
            x = self.__dispatch_data(x, 1 if self.__is_first_stage else 0)

        if isinstance(y, DataDispatcher):
            y = self.__dispatch_data(y, 1 if self.__is_last_stage else 0)

        fit_args = {
            'callbacks': callbacks,
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
            'workers': workers,
            'use_multiprocessing': use_multiprocessing,
        }

        if self.__is_first_stage:
            # 流水线模型第一个进程, 也即输入进程
            stage: PipelineStage = self.__stages[0]
            training_stage = FirstTrainingStage(
                pipeline_model_id=self.__pipeline_model_id,
                pipeline_communicator=self.pipeline_communicator,
                stage_communicator=self.stage_communicator,
                model=self.__executing_model,
                next_stage_input_shape=stage.output_shape,
                inputs=x,
                **fit_args
            )
        elif self.__is_last_stage:
            # 流水线模型最后一个进程, 也即输出进程
            stage: PipelineStage = self.__stages[-1]
            training_stage = LastTrainingStage(
                pipeline_model_id=self.__pipeline_model_id,
                pipeline_communicator=self.pipeline_communicator,
                stage_communicator=self.stage_communicator,
                previous_stage_output_shape=stage.input_shape,
                model=self.__executing_model,
                targets=y,
                verbose=verbose,
                batch_size=batch_size, epochs=epochs,
                micro_batch_size=micro_batch_size,
                **fit_args
            )
        else:
            # 流水线中间进程
            stage: PipelineStage = self.__stages[self.__pipeline_comm.rank]
            training_stage = IntermediateTrainingStage(
                pipeline_model_id=self.__pipeline_model_id,
                pipeline_communicator=self.pipeline_communicator,
                stage_communicator=self.stage_communicator,
                previous_stage_output_shape=stage.input_shape,
                model=self.__executing_model,
                next_stage_input_shape=stage.output_shape,
                **fit_args
            )
        return training_stage.run()

    def predict(
            self,
            x, batch_size=None, verbose=0,
            steps=None, callbacks=None,
            max_queue_size=10, workers=1, use_multiprocessing=False):
        if not self._is_compiled:
            raise Exception('model has not been compiled')
        if not self.__work:
            return

    @staticmethod
    def __dispatch_data(
            data_dispatcher: DataDispatcher, need_data: int,
    ):
        """
        1. 分割通信域, 将root_rank(保存有数据的进程号)和所有其他需要数据的rank分割到
        一个通信域内, 保证root_rank在新通信域内的rank是0
        2. 在需要进行广播的通信域内以0为根节点广播
        @param data_dispatcher: 进行操作的DispatchingData对象
        @param need_data: 是否需要进行数据分发, 01值, 1表示需要, 0表示不需要
        @return:
         当need_dispatch==1 时, DispatchingData.data(),
         否则 None
        """
        comm = data_dispatcher.communicator
        root = data_dispatcher.root_rank

        if comm.rank == root:
            key = 0
            # 根节点肯定需要参与数据分发的
            root_keep_data = True if need_data != 0 else False
            dispatch_data = 1
        else:
            key = 1 + comm.rank
            root_keep_data = False
            dispatch_data = need_data

        dispatch_comm = comm.split_communicator(dispatch_data, key=key)

        if dispatch_data == 1:
            data_dispatcher.dispatch(dispatch_comm, 0, root_keep_data)
            return data_dispatcher.data
