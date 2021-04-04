from tensorflow.keras.layers import Dense, Layer
from tensorflow.python.framework.ops import Tensor, RegisterGradient
# noinspection PyProtectedMember
from tensorflow.python.ops.nn_grad import _BiasAddGrad
from ddl.tensorflow.communicator import Communicator
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
        self.__previous_stage_rank = pipeline_model.pipeline_communicator.rank - 1
        self.__communicator = pipeline_model.pipeline_communicator
        self.__compiled = True


class DensePipelineInputLayer(Dense, PipelineLayer):
    """
    分布式流水线并行中向上一层中进行误差后向传播的全连接层
    """

    def __init__(
            self, units, activation=None,
            kernel_initializer='glorot_uniform', bias_initializer='zeros',
            kernel_regularizer=None, bias_regularizer=None,
            activity_regularizer=None, kernel_constraint=None,
            bias_constraint=None,
            **kwargs):
        """
        @param 见Dense的文档
        """
        Dense.__init__(
            self, units, activation, True, kernel_initializer,
            bias_initializer, kernel_regularizer,
            bias_regularizer, activity_regularizer,
            kernel_constraint, bias_constraint, **kwargs)
        PipelineLayer.__init__(self, **kwargs)

    def call(self, inputs):
        """
        替换原有的偏置加法OP, 让其正常完成加添置的同时, 将批次内的所有样例的偏差传递到上一个节点
        @param inputs: 见父类文档
        @return: 见父类文档
        """

        # 以下代码要求单例, 因为不允许注册相同名字的梯度
        @RegisterGradient('DenseForwardAndSendGrad')
        def dense_forward_and_send_grad(op, received_grad: Tensor):
            """
            替换原有OP, 将其传给ForwardAndSendOp, 将上一阶段
            @param op: _BiasAddGrad的第一函数参数
            @param received_grad: 偏置应收到的每个样例产生的梯度
            @return: _BiasAddGrad returns
            """
            from ddl.tensorflow.cpp_backend import CPPBackend
            from ddl.tensorflow.keras.parallelism.pipeline.training_stage \
                import BaseTrainingStage
            # 需要进行一次矩阵乘法
            last_stage_errors = tf.transpose(tf.matmul(
                self.kernel,
                tf.transpose(received_grad)
            ))

            received_grad = CPPBackend.tf_lib().forward_and_send(
                received_grad, last_stage_errors,
                receiver=self.previous_stage_rank,
                msg=json.dumps({
                    'code': BaseTrainingStage.MessageCode.BPROP_GRAD.value
                }),
                communicator_id=self.communicator.id
            )
            return _BiasAddGrad(op, received_grad)

        # 替换OP
        with tf.compat.v1.get_default_graph().gradient_override_map({
            'BiasAdd': 'DenseForwardAndSendGrad'
        }):
            return super(DensePipelineInputLayer, self).call(inputs)
