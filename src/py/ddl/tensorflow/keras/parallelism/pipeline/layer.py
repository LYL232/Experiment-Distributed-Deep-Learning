from ddl.tensorflow.communicator import Communicator
from ddl.tensorflow.cpp_backend import CPPBackend
from ddl.tensorflow.keras.parallelism.pipeline.training_stage \
    import BaseTrainingStage
from tensorflow.keras.layers import Layer
from tensorflow.python.framework.ops import Tensor
import tensorflow as tf
import abc
import json


class PipelineLayer(Layer, metaclass=abc.ABCMeta):

    def __init__(
            self, trainable=True, name=None, dtype=None, dynamic=False,
            **kwargs):
        super().__init__(
            trainable=trainable, name=name, dtype=dtype, dynamic=dynamic,
            **kwargs
        )
        self.__compiled = False
        self.__previous_stage_rank = None
        self.__communicator = None

    @property
    def previous_stage_rank(self) -> int:
        """
        @return: 上一阶段的模型所属的rank, 如果是第一阶段的模型, 则返回-1
        """
        assert self.__compiled, 'access previous stage rank before' \
                                ' compile pipeline model'
        return self.__previous_stage_rank

    @property
    def communicator(self) -> Communicator:
        assert self.__compiled, 'access communicator before' \
                                ' compile pipeline model'
        return self.__communicator

    def compile_by_pipeline_model(self, pipeline_model) -> None:
        """
        由PipelineModel对象编译即将完成时对此对象执行的方法, 不要直接调用,
        参数设置成PipelineModel对象也正是此意
        @param pipeline_model: 即将编译完成的PipelineModel
        @return: None
        """
        from ddl.tensorflow.keras.parallelism.pipeline.model import \
            PipelineModel
        assert isinstance(pipeline_model, PipelineModel)
        self.__previous_stage_rank = \
            pipeline_model.pipeline_communicator.rank - 1
        self.__communicator = pipeline_model.pipeline_communicator
        self.__compiled = True


class PipelineInputLayer(PipelineLayer):
    def __init__(
            self, input_shape: tuple = None, name: str = None,
            index: int = 0, **kwargs):
        """
        @param input_shape: 输入形状
        @param name: 名字
        @param index: 输入的索引, 当模型有输入时, 此项才可能非0
        @param kwargs:
        """
        kwargs.pop('trainable', None)
        if input_shape is not None:
            kwargs['input_shape'] = input_shape
        super().__init__(trainable=True, name=name, **kwargs)

        self.__index = index

        self.__shape = input_shape

        @tf.custom_gradient
        def input_grad(x, _):
            """
            输入层的梯度, 因为一般情况下是不计算输入层的梯度的, 所以需要一个Variable去欺骗
            tensorflow这里有需要计算的梯度, 这个自定义梯度就是为了获取输入层的梯度
            @param x: 输入层的输入tensor
            @param _: 无用的变量输入, 不需要使用, 只是为了欺骗tensorflow这里有梯度要计算
            @return:
            """

            def grad(dy):
                fake_grad = CPPBackend.tf_lib().forward_and_send(
                    tf.zeros((1,)), dy,
                    receiver=self.previous_stage_rank,
                    msg=json.dumps({
                        'code': BaseTrainingStage.MessageCode.BPROP_GRAD.value,
                        'index': self.__index
                    }),
                    communicator_id=self.communicator.id,
                    name=f'pipeline-input-{self.__index}'
                )
                return None, fake_grad

            return tf.identity(x), grad

        self.__input_grad_fn = input_grad
        self.__fake_kernel = None

    def build(self, input_shape):
        self.__fake_kernel = self.add_weight(shape=(1,), trainable=True)
        if self.__shape is None:
            self.__shape = input_shape[1:]
        self.built = True

    def compute_output_shape(self, input_shape):
        if self.__shape is None:
            self.__shape = input_shape[1:]
        else:
            assert self.__shape == input_shape[1:]
        return input_shape

    def call(self, inputs, **kwargs):
        if self.__shape is None:
            self.__shape = inputs.shape[1:]
        return self.__input_grad_fn(inputs, self.__fake_kernel)
