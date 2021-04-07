from tensorflow.keras.layers import Layer
from ddl.tensorflow.communicator import Communicator
from ddl.tensorflow.cpp_backend import CPPBackend
from ddl.tensorflow.keras.parallelism.pipeline.training_stage \
    import BaseTrainingStage
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
        if not self.__compiled:
            raise Exception(
                'access last stage rank before compile pipeline model'
            )
        return self.__previous_stage_rank

    @property
    def communicator(self) -> Communicator:
        if not self.__compiled:
            raise Exception(
                'access last stage rank before compile pipeline model'
            )
        return self.__communicator

    def compile_by_pipeline_model(self, pipeline_model) -> None:
        """
        由PipelineModel对象编译即将完成时对此对象执行的方法, 不要直接调用,
        参数设置成PipelineModel对象也正是此意
        @param pipeline_model: 即将编译完成的PipelineModel
        @return: None
        """
        self.__previous_stage_rank = \
            pipeline_model.pipeline_communicator.rank - 1
        self.__communicator = pipeline_model.pipeline_communicator
        self.__compiled = True


class PipelineInputLayer(PipelineLayer):
    def __init__(self, input_shape: tuple, name: str = None, **kwargs):
        kwargs.pop('trainable', None)
        super().__init__(
            input_shape=input_shape, trainable=True, name=name, **kwargs
        )

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
                        'code': BaseTrainingStage.MessageCode.BPROP_GRAD.value
                    }),
                    communicator_id=self.communicator.id
                )
                return None, fake_grad

            return x, grad

        self.__input_grad_fn = input_grad
        self.__fake_kernel = None

    def build(self, input_shape):
        self.__fake_kernel = self.add_weight(shape=(1,), trainable=True)
        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        return (input_shape[0],) + self.__shape

    def call(self, inputs, **kwargs):
        return self.__input_grad_fn(inputs, self.__fake_kernel)
