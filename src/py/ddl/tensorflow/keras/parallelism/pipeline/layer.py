from ddl.tensorflow.cpp_backend import CPPBackend
from ddl.tensorflow.keras.parallelism.pipeline.training import TrainingExecutor
from ddl.tensorflow.keras.parallelism.pipeline.pipe import PipelinePipe
from tensorflow.keras.layers import Layer
import tensorflow as tf
import abc
import json


class PipelineInputLayer(Layer, metaclass=abc.ABCMeta):

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
        self.__stage = stage
        self.__communicator = stage.pipeline_model.pipeline_communicator
        self.__pipe = pipe
        self.__fake_kernel = None

    def build(self, input_shape):
        self.__fake_kernel = self.add_weight(shape=(1,), trainable=True)
        self.built = True

    def compute_output_shape(self, input_shape):
        return input_shape

    @tf.autograph.experimental.do_not_convert
    def call(self, inputs, **kwargs):
        assert self.__communicator is not None

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
                if self.__pipe.comes_from is None:
                    return None, tf.zeros((1,))
                fake_grad = CPPBackend.tf_lib().forward_and_send(
                    tf.zeros((1,)), dy,
                    receiver=self.__pipe.comes_from.stage_rank,
                    msg=json.dumps({
                        'code': TrainingExecutor.MessageCode.BPROP_GRAD.value,
                        'index': self.__pipe.index_of(self.__pipe.comes_from)
                    }),
                    communicator_id=self.__communicator.id,
                    name=f'gradients-of-stage-'
                         f'{self.__pipe.comes_from.stage_rank}-pipeline-input-'
                         f'{self.__pipe.index_of(self.__pipe.comes_from)}'
                )
                return None, fake_grad

            return tf.identity(x), grad

        return input_grad(inputs, self.__fake_kernel)
