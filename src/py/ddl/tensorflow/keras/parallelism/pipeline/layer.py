from ddl.tensorflow.cpp_backend import CPPBackend
from ddl.tensorflow.keras.parallelism.pipeline.pipe import PipelinePipe
from tensorflow.keras.layers import Layer
import tensorflow as tf
from tensorflow.python.ops.resource_variable_ops import ResourceVariable


class PipelineInputLayer(Layer):
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
        self.__pipeline_model_rank = stage.pipeline_model.pipeline_model_rank
        self.__convey_gradient = pipe.convey_gradient()

    def build(self, input_shape):
        if self.__pipe.comes_from is not None:
            self.__fake_kernel = self.add_weight(shape=(1,), trainable=True)
        self.built = True

    def compute_output_shape(self, input_shape):
        return input_shape

    @tf.autograph.experimental.do_not_convert
    def call(self, inputs, **kwargs):
        assert self.__communicator is not None

        if self.__pipe.comes_from is None:
            return inputs

        forward = CPPBackend.tf_lib().receive_tensor(
            inputs,
            sender=self.__pipe.comes_from.stage_rank,
            communicator_id=self.__communicator.id,
            tag=self.__pipe.index_of(self.__stage),
            name=f'pipeline-{self.__pipeline_model_rank}-'
                 f'stage-{self.__communicator.rank}-'
                 f'input-{self.__pipe.index_of(self.__stage)}-'
                 f'receive-forward-from-stage-'
                 f'{self.__pipe.comes_from.stage_rank}-output-'
                 f'{self.__pipe.index_of(self.__pipe.comes_from)}'
        )

        @tf.custom_gradient
        def pipeline_input(x, _):
            """
            输入层的梯度, 因为一般情况下是不计算输入层的梯度的, 所以需要一个Variable去欺骗
            tensorflow这里有需要计算的梯度, 这个自定义梯度就是为了获取输入层的梯度
            @param x: 输入层的输入tensor
            @param _: 无用的变量输入, 不需要使用, 只是为了欺骗tensorflow这里有梯度要计算
            @return:
            """

            def grad(dy):
                if self.__convey_gradient:
                    fake_grad = CPPBackend.tf_lib().forward_and_send(
                        tf.zeros((1,)), dy,
                        receiver=self.__pipe.comes_from.stage_rank,
                        tag=self.__pipe.index_of(self.__stage),
                        communicator_id=self.__communicator.id,
                        name=f'gradients-of-stage-'
                             f'{self.__pipe.comes_from.stage_rank}'
                             f'-pipeline-input-'
                             f'{self.__pipe.index_of(self.__pipe.comes_from)}'
                    )
                else:
                    fake_grad = tf.reshape(
                        tf.reduce_sum(dy, axis=None), shape=(1,)) * 0
                return None, fake_grad

            return x, grad

        return pipeline_input(forward, self.__fake_kernel)


class PipelineOutputLayer(Layer):
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
        self.__pipeline_model_rank = stage.pipeline_model.pipeline_model_rank
        self.__fake_kernel = None
        self.__convey_gradient = pipe.convey_gradient()

    def build(self, input_shape):
        if len(self.__pipe.send_to) > 0:
            self.__fake_kernel = self.add_weight(shape=(1,), trainable=True)
        self.built = True

    def compute_output_shape(self, input_shape):
        if input_shape[0] is not None:
            return (None, *input_shape)
        return input_shape

    @tf.autograph.experimental.do_not_convert
    def call(self, inputs, **kwargs):
        # 是中间输出, 就再前向传播中发送结果, 在后向传播中接收梯度
        assert self.__communicator is not None
        from ddl.tensorflow.keras.parallelism.pipeline.stage import \
            PipelineStage

        if len(self.__pipe.send_to) == 0:
            return inputs

        if isinstance(inputs, ResourceVariable):
            raise Exception('ResourceVariable '
                            'can not be direct output of a stage')

        @tf.custom_gradient
        def pipeline_output(x, _):
            """
            如果是中间输出, 那么就在前向传播中发送前向传播结果, 在后向传播中接收来自下一阶段的梯度
            @param x: 输入层的输入tensor
            @param _: 无用的变量输入, 不需要使用, 只是为了欺骗tensorflow这里有梯度要计算
            @return:
            """

            # 定义后向传播图
            def grad(dy):
                if self.__convey_gradient:
                    recv_grad_ops = []
                    dy = tf.zeros_like(dy)
                    for recv_from_stage in self.__pipe.send_to:
                        assert isinstance(recv_from_stage, PipelineStage)
                        recv_from_input_index = self.__pipe.index_of(
                            recv_from_stage)
                        recv_op = CPPBackend.tf_lib().receive_tensor(
                            dy,
                            sender=recv_from_stage.stage_rank,
                            tag=recv_from_input_index,
                            name=f'pipeline-'
                                 f'{self.__pipeline_model_rank}'
                                 f'-stage-'
                                 f'{self.__stage.stage_rank}-'
                                 f'backward-gradient-from-stage-'
                                 f'{recv_from_stage.stage_rank}-input-'
                                 f'{recv_from_input_index}',
                            # 这里要传入handle(整数值), 而不是一个python对象
                            communicator_id=self.__communicator.id
                        )

                        recv_grad_ops.append(recv_op)

                    recv_grad = tf.add_n(recv_grad_ops)
                else:
                    recv_grad = tf.zeros_like(dy)
                # 为了欺骗tensorflow这里有梯度需要计算, 如果不加这一句那么如果之前的层
                # 不需要recv_grad
                fake_grad = CPPBackend.tf_lib().do_but_pass_by(
                    tf.zeros((1,)), recv_grad
                )
                return recv_grad, fake_grad

            # 确保所有分支都会被计算到
            merged_send_ops = None

            output_index = self.__pipe.index_of(self.__stage)

            for i, recv_stage in enumerate(self.__pipe.send_to):
                sending_to_input_index = self.__pipe.index_of(recv_stage)
                send_op = CPPBackend.tf_lib().forward_and_send(
                    x, x,
                    receiver=recv_stage.stage_rank,
                    tag=sending_to_input_index,
                    communicator_id=self.__communicator.id,
                    name=f'pipeline-{self.__pipeline_model_rank}-stage-'
                         f'{self.__stage.stage_rank}-output-'
                         f'{output_index}-forward-to-stage-'
                         f'{recv_stage.stage_rank}-input-'
                         f'{sending_to_input_index}'
                )

                if i > 0:
                    merged_send_ops = CPPBackend.tf_lib().do_but_pass_by(
                        send_op, merged_send_ops)
                else:
                    merged_send_ops = send_op

            return CPPBackend.tf_lib().do_but_pass_by(x, merged_send_ops), grad

        return pipeline_output(inputs, self.__fake_kernel)
