from ddl.tensorflow.keras.parallelism.pipeline.model import PipelineModel
from ddl.tensorflow.keras.parallelism.pipeline.pipe import PipelinePipe
from ddl.tensorflow.keras.parallelism.pipeline.layer import \
    PipelineInputLayer, PipelineOutputLayer
from ddl.tensorflow import util

from tensorflow.keras.models import Model
from tensorflow.keras import Input
from abc import ABCMeta


class PipelineStage(metaclass=ABCMeta):
    def __init__(self):
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

    def call(self, *args, **kwargs):
        """
        @param args:
        @param kwargs:
        @return:
        """
        raise NotImplementedError

    @property
    def input_shape(self) -> tuple:
        raise NotImplementedError

    @property
    def output_shape(self) -> tuple:
        raise NotImplementedError

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
        input_shape = util.formalize_shapes(self.input_shape)
        self.__pipeline_input_tensors = []
        self.__input_tensors = []
        for i in range(len(input_shape)):
            shape = input_shape[i]
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
        output_shape = util.formalize_shapes(self.output_shape)
        self.__output_tensors = []
        for i in range(len(output_shape)):
            self.__output_tensors.append(
                PipelineOutputLayer(
                    self, self.__output_pipes[i],
                    name=f'pipeline-output-{i}'
                )(outputs[i])
            )
        self.__model = Model(
            inputs=self.__input_tensors, outputs=self.__output_tensors
        )
        self.__built = True

    def __call__(self, *args, **kwargs) -> tuple or PipelinePipe:
        assert not self.__called, 'can not call PipelineStage more than once'

        input_shape = util.formalize_shapes(self.input_shape)
        input_count = len(input_shape)

        assert len(args) == input_count

        for i in range(input_count):
            pipe = args[i]
            assert isinstance(pipe, PipelinePipe)
            assert pipe.shape == input_shape[i], \
                f'not the same input shape, incoming: {args},' \
                f' required: {input_shape}'
            self.__input_pipes.append(pipe)
            pipe.send_to_stage(self, i)

        self.__called = True

        output_shape = util.formalize_shapes(self.output_shape)
        for i in range(len(output_shape)):
            self.__output_pipes.append(PipelinePipe(output_shape[i], self, i))

        if len(self.output_pipes) > 1:
            return self.output_pipes
        else:
            assert len(self.output_pipes) > 0
            return self.output_pipes[0]

    def __str__(self):
        return f'PipelineStage-{id(self)}'

    def __repr__(self):
        return self.__str__()
