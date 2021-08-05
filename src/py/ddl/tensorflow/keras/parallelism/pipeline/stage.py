from ddl.tensorflow.keras.parallelism.pipeline.model import PipelineModel
from ddl.tensorflow.keras.parallelism.pipeline.pipe import PipelinePipe, \
    PipelineInput
from ddl.tensorflow.cpp_backend import CPPBackend
from ddl.tensorflow.keras.parallelism.pipeline.layer import \
    PipelineInputLayer, PipelineOutputLayer
from ddl.tensorflow import util
from ddl.message import Message

from tensorflow import control_dependencies
from tensorflow.keras.models import Model
from tensorflow.keras import Input
from abc import ABCMeta, abstractmethod
from typing import Tuple
import json


class PipelineStage(metaclass=ABCMeta):
    def __init__(self, output_num: int):
        """
        @param output_num: 由于需要在模型加载前知道模型的输出个数，所以需要先将这个数通知给Stage
        """
        self.__model = None
        self.__output_pipes = []
        self.__input_pipes = []

        self.__pipeline_model = None
        self.__stage_rank = -1

        self.__input_tensors = None
        self.__output_tensors = None
        self.__pipeline_input_tensors = None
        self.__built = False
        self.__called = False

        self.__input_shape = None
        self.__output_shape = None

        assert output_num > 0
        self.__output_num = output_num

    @abstractmethod
    def call(self, *args, **kwargs):
        """
        @param args:
        @param kwargs:
        @return:
        """

    @property
    def output_num(self) -> int:
        return self.__output_num

    @property
    def input_shape(self) -> Tuple[Tuple[int]]:
        assert self.__built
        return self.__input_shape

    @property
    def output_shape(self) -> Tuple[Tuple[int]]:
        assert self.__built
        return self.__output_shape

    @property
    def model(self) -> Model:
        assert self.built
        return self.__model

    @property
    def built(self) -> bool:
        return self.__built

    @property
    def output_pipes(self) -> tuple:
        return tuple(self.__output_pipes)

    @property
    def input_pipes(self) -> tuple:
        return tuple(self.__input_pipes)

    # 以下属性只有在attach_pipeline_model之后才能调用
    @property
    def pipeline_model(self) -> PipelineModel:
        assert self.__pipeline_model, \
            'access stage\' pipeline_model before defining the PipelineModel'
        return self.__pipeline_model

    @property
    def stage_rank(self) -> int:
        assert self.__pipeline_model, \
            'access stage\' stage_rank before defining the PipelineModel'
        return self.__stage_rank

    def attach_pipeline_model(self, pipeline_model, stage_rank: int) -> None:
        """
        将PipelineStage与PipelineModel绑定起来
        @param pipeline_model
        @param stage_rank
        @return: None
        """
        from ddl.tensorflow.keras.parallelism.pipeline.model import \
            PipelineModel
        assert isinstance(pipeline_model, PipelineModel)
        self.__pipeline_model = pipeline_model
        self.__stage_rank = stage_rank

    def build(self) -> None:
        self.__input_shape = self.__prepare_input_shape()

        self.__pipeline_input_tensors = []
        self.__input_tensors = []
        for i in range(len(self.__input_shape)):
            shape = self.__input_shape[i]
            this_inputs = Input(shape=shape)
            self.__input_tensors.append(this_inputs)
            self.__pipeline_input_tensors.append(
                PipelineInputLayer(
                    self, self.input_pipes[i],
                    name=f'pipeline-input-{i}'
                )(this_inputs)
            )
        outputs = self.call(*self.__pipeline_input_tensors)

        if not isinstance(outputs, (tuple, list)):
            outputs = (outputs,)

        self.__output_tensors = []
        for i in range(len(self.__output_pipes)):
            self.__output_tensors.append(
                PipelineOutputLayer(
                    self, self.__output_pipes[i],
                    name=f'pipeline-output-{i}'
                )(outputs[i])
            )
        self.__model = Model(
            inputs=self.__input_tensors,
            outputs=self.__output_tensors
        )

        # 去掉output_shape中的第一维，也即批次维度
        model_output_shape = self.__model.output_shape
        assert len(model_output_shape) > 0
        if not isinstance(model_output_shape[0], (tuple, list)):
            model_output_shape = (model_output_shape,)

        output_shape = []
        for each in model_output_shape:
            if len(each) > 1:
                if each[0] is None:
                    output_shape.append((*each[1:],))
                else:
                    output_shape.append(each)
            else:
                output_shape.append((1,))

        self.__output_shape = util.formalize_shapes(output_shape)

        assert len(self.__output_shape) == self.output_num
        self.__notify_output_shape()

        self.__built = True

    def __call__(self, *args, **kwargs) -> tuple or PipelinePipe:
        assert not self.__called, 'can not call PipelineStage more than once'

        for i in range(len(args)):
            pipe = args[i]
            assert isinstance(pipe, PipelinePipe)
            self.__input_pipes.append(pipe)
            pipe.send_to_stage(self, i)

        self.__called = True

        for i in range(self.output_num):
            self.__output_pipes.append(PipelinePipe(self, i))

        if self.output_num > 1:
            return self.output_pipes
        else:

            return self.output_pipes[0]

    def __str__(self):
        return f'PipelineStage-{id(self)}'

    def __repr__(self):
        return self.__str__()

    def __prepare_input_shape(self):
        input_shape = []
        receive_required = 0
        for i in range(len(self.__input_pipes)):
            pipe: PipelinePipe = self.__input_pipes[i]
            if isinstance(pipe, PipelineInput):
                # 是直接输入，可以获取形状
                input_shape.append(pipe.shape)
            else:
                input_shape.append(None)
                receive_required += 1

        pipeline_comm = self.pipeline_model.pipeline_communicator

        while receive_required > 0:
            msg = json.loads(Message.listen(pipeline_comm).msg)
            shape = tuple(msg['shape'])
            index = int(msg['input_index'])
            assert input_shape[index] is None, \
                f'input-{index} of stage-{self.stage_rank} receive shape twice'
            input_shape[index] = shape
            receive_required -= 1

        return tuple(input_shape)

    def __notify_output_shape(self):
        pipeline_communicator = self.pipeline_model.pipeline_communicator
        for i in range(len(self.__output_pipes)):
            pipe: PipelinePipe = self.__output_pipes[i]
            for stage in pipe.send_to:
                stage: PipelineStage
                Message.send(json.dumps({
                    'shape': self.__output_shape[i],
                    'input_index': pipe.index_of(stage)
                }), stage.stage_rank, pipeline_communicator)

    def __get_output_layer_send_ops(self, outputs: tuple or list):
        # 为了防止阶段与阶段间的死锁，需要按照接收阶段的拓扑序来安排每个发送张量的Op顺序
        forward_send_to = {}
        for i, pipe in enumerate(self.__output_pipes):
            for j, stage in enumerate(pipe.send_to):
                stage_rank = stage.stage_rank
                if stage.stage_rank not in forward_send_to.keys():
                    forward_send_to[stage_rank] = []
                forward_send_to[stage_rank].append((i, j, stage))

        communicator = self.pipeline_model.pipeline_communicator
        pipeline_model_rank = self.pipeline_model.pipeline_model_rank

        output_layer_send_ops = [[] for _ in range(len(self.__output_pipes))]
        last_stage_send_ops = []
        for stage_rank in sorted(list(forward_send_to.keys())):
            current_stage_send_ops = []
            for i, j, recv_stage in forward_send_to[stage_rank]:
                assert i < len(output_layer_send_ops)
                send_ops = output_layer_send_ops[i]
                output = outputs[i]
                sending_to_input_index = \
                    self.output_pipes[i].index_of(recv_stage)
                while j >= len(output_layer_send_ops[i]):
                    send_ops.append(j)
                with control_dependencies(last_stage_send_ops):
                    send_op = CPPBackend.tf_lib().forward_and_send(
                        output, output,
                        receiver=recv_stage.stage_rank,
                        tag=sending_to_input_index,
                        communicator_id=communicator.id,
                        name=f'pipeline-{pipeline_model_rank}-stage-'
                             f'{self.stage_rank}-output-'
                             f'{i}-forward-to-stage-'
                             f'{recv_stage.stage_rank}-input-'
                             f'{sending_to_input_index}'
                    )
                    send_ops[j] = send_op
                    current_stage_send_ops.append(send_op)
            last_stage_send_ops = current_stage_send_ops
        return output_layer_send_ops
