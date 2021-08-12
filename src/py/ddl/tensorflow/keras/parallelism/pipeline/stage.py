from ddl.tensorflow.keras.parallelism.pipeline.model import PipelineModel
from ddl.tensorflow.keras.parallelism.pipeline.pipe import PipelinePipe, \
    PipelineInput
from ddl.tensorflow.cpp_backend import CPPBackend
from ddl.tensorflow.keras.parallelism.pipeline.layer import \
    EagerPipelineInputLayer, EagerPipelineOutputLayer
from ddl.tensorflow import util
from ddl.message import Message
from ddl.log import Log, TimeUnit, TimeLogCallback

from tensorflow.python.keras.engine.training import Model as ModelV2
from tensorflow.python.keras.mixed_precision.experimental \
    import loss_scale_optimizer as lso
from tensorflow.python.keras.engine import data_adapter
from tensorflow.python.eager import backprop
from tensorflow import control_dependencies
from tensorflow.keras.models import Model
from tensorflow.keras import Input
from abc import ABCMeta, abstractmethod
from typing import Tuple, List
import json
import tensorflow as tf

pmb = Log.new_log_type(0, True, 'pipeline micro batch')
pb = Log.new_log_type(0, True, 'pipeline batch')


class PipelineKerasImplementModel(ModelV2):
    def __init__(
            self, inputs: tuple or list, outputs: tuple or list,
            stage: 'PipelineStage', *args, **kwargs):
        if not isinstance(inputs, (tuple, list)):
            inputs = (inputs,)
        if not isinstance(outputs, (tuple, list)):
            outputs = (outputs,)
        super().__init__(inputs=inputs, outputs=outputs, *args, **kwargs)
        self._mb_size = 0
        self.__stage = stage

    @property
    def stage(self) -> 'PipelineStage':
        return self.__stage

    def compile(
            self,
            optimizer='rmsprop',
            loss=None,
            metrics=None,
            loss_weights=None,
            sample_weight_mode=None,
            weighted_metrics=None,
            **kwargs):
        run_eagerly = kwargs.pop('run_eagerly', True)
        if not run_eagerly:
            raise NotImplementedError(
                'pipeline model noly support run_eagerly=True')
        super().compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics,
            loss_weights=loss_weights,
            sample_weight_mode=sample_weight_mode,
            weighted_metrics=weighted_metrics,
            run_eagerly=True,
            **kwargs
        )

    def fit(
            self, x=None, y=None, batch_size=None, epochs=1, verbose=1,
            callbacks=None, validation_split=0., validation_data=None,
            shuffle=True, class_weight=None, sample_weight=None,
            initial_epoch=0, steps_per_epoch=None,
            validation_steps=None, validation_batch_size=None,
            validation_freq=1, max_queue_size=10,
            workers=1, use_multiprocessing=False,
            micro_batch_size=None):
        if micro_batch_size is None:
            self._mb_size = batch_size
        else:
            self._mb_size = min(batch_size, micro_batch_size)
        if Log.is_logging_type(pb):
            if callbacks is not None:
                callbacks.append(TimeLogCallback())
            else:
                callbacks = [TimeLogCallback()]
        super().fit(
            x=x,
            y=y,
            batch_size=batch_size,
            epochs=epochs,
            verbose=verbose,
            callbacks=callbacks,
            validation_split=validation_split,
            validation_data=validation_data,
            shuffle=shuffle,
            class_weight=class_weight,
            sample_weight=sample_weight,
            initial_epoch=initial_epoch,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            validation_batch_size=validation_batch_size,
            validation_freq=validation_freq,
            max_queue_size=max_queue_size,
            workers=workers,
            use_multiprocessing=use_multiprocessing
        )

    def predict(
            self,
            x,
            batch_size=None,
            verbose=0,
            steps=None,
            callbacks=None,
            max_queue_size=10,
            workers=1,
            use_multiprocessing=False,
            micro_batch_size: int = None
    ):
        if micro_batch_size is None:
            self._mb_size = batch_size
        else:
            self._mb_size = min(batch_size, micro_batch_size)
        res = super().predict(
            x,
            batch_size=batch_size,
            verbose=verbose,
            steps=steps,
            callbacks=callbacks,
            max_queue_size=max_queue_size,
            workers=workers,
            use_multiprocessing=use_multiprocessing
        )
        if isinstance(res, (list, tuple)) and len(res) == 1:
            return res[0]
        return res

    def train_step(self, data):
        data = data_adapter.expand_1d(data)
        x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)

        assert sample_weight is None, 'sample_weight is not supported'

        stage = self.stage

        input_num = len(stage.pipeline_input_layers)
        output_num = len(stage.pipeline_output_layers)

        batch_size = tf.shape(x[0])[0]

        mb_begin = 0
        mb_sizes = []

        is_output_stage = self.stage.is_output_stage

        while mb_begin < batch_size:
            mb_end = tf.minimum(mb_begin + self._mb_size, batch_size)
            mb_sizes.append((mb_begin, mb_end))
            mb_begin = mb_end

        predicts = [[] for _ in range(output_num)]
        mb_preds = []
        with backprop.GradientTape(persistent=True) as tape:
            for i, (mb_begin, mb_end) in enumerate(mb_sizes):
                mb_inputs = stage.get_micro_batch_inputs(
                    [x[j][mb_begin:mb_end] for j in range(input_num)])
                mb_pred = self(mb_inputs, training=True)
                # todo: 让模型都返回tuple而不是直接返回张量
                if not isinstance(mb_pred, (tuple, list)):
                    mb_pred = [mb_pred]
                stage.send_micro_batch_outputs(mb_pred, mb_begin, mb_end)
                Log.time_log(
                    f'micro batch forward propagation {i} '
                    f'[{mb_begin}, {mb_end}] finished',
                    TimeUnit.MS, log_type=pmb
                )
                for j in range(output_num):
                    predicts[j].append(mb_pred[j])
                mb_preds.append(mb_pred)

        accumulated_gradients = [
            tf.zeros_like(variable, dtype=variable.dtype)
            for variable in self.trainable_variables
        ]

        float_batch_size = tf.cast(batch_size, dtype=tf.float32)

        for i, (mb_begin, mb_end) in enumerate(mb_sizes):
            with tape:
                mb_target = stage.get_micro_batch_targets(
                    [y[j][mb_begin:mb_end] for j in range(output_num)])

                if is_output_stage:
                    target = self.compiled_loss(
                        mb_target, mb_preds[i],
                        regularization_losses=self.losses,
                        sample_weight=None,
                    )
                    if isinstance(self.optimizer, lso.LossScaleOptimizer):
                        target = self.optimizer.get_scaled_loss(target)
                else:
                    target = mb_preds[i]

            if is_output_stage:
                grads = tape.gradient(target, self.trainable_variables)
            else:
                grads = tape.gradient(
                    target, self.trainable_variables,
                    output_gradients=mb_target
                )
            stage.send_micro_batch_grads()
            Log.time_log(
                f'micro batch backward propagation {i} '
                f'[{mb_begin}, {mb_end}] finished',
                TimeUnit.MS,
                log_type=pmb
            )
            for j, gard in enumerate(grads):
                if gard is not None:
                    if isinstance(self.optimizer, lso.LossScaleOptimizer):
                        gard = self.optimizer.get_unscaled_gradients(
                            gard)
                    accumulated_gradients[j] = accumulated_gradients[j] + (
                            gard *
                            tf.cast(mb_end - mb_begin, dtype=tf.float32) /
                            float_batch_size
                    )

        y_pred = [tf.concat(predicts[i], axis=0) for i in range(output_num)]
        # noinspection PyProtectedMember
        accumulated_gradients = self.optimizer._clip_gradients(
            accumulated_gradients)  # pylint: disable=protected-access
        if self.trainable_variables:
            self.optimizer.apply_gradients(
                zip(accumulated_gradients, self.trainable_variables))
        self.compiled_metrics.update_state(y, y_pred, sample_weight)

        return {m.name: m.result() for m in self.metrics}

    def predict_step(self, data):
        data = data_adapter.expand_1d(data)
        x, _, _ = data_adapter.unpack_x_y_sample_weight(data)

        stage = self.stage

        input_num = len(stage.pipeline_input_layers)
        output_num = len(stage.pipeline_output_layers)

        batch_size = tf.shape(x[0])[0]

        mb_begin = 0
        mb_sizes = []

        while mb_begin < batch_size:
            mb_end = tf.minimum(mb_begin + self._mb_size, batch_size)
            mb_sizes.append((mb_begin, mb_end))
            mb_begin = mb_end

        predicts = [[] for _ in range(output_num)]

        for i, (mb_begin, mb_end) in enumerate(mb_sizes):
            mb_inputs = stage.get_micro_batch_inputs(
                [x[j][mb_begin:mb_end] for j in range(input_num)])
            mb_pred = self(mb_inputs, training=True)
            # todo: 让模型都返回tuple而不是直接返回张量
            if not isinstance(mb_pred, (tuple, list)):
                mb_pred = (mb_pred,)
            stage.send_micro_batch_outputs(mb_pred, mb_begin, mb_end)
            Log.time_log(
                f'micro batch forward propagation {i} '
                f'[{mb_begin}, {mb_end}] finished',
                TimeUnit.MS, log_type=pmb
            )
            for j in range(output_num):
                predicts[j].append(mb_pred[j])

        y_pred = [tf.concat(predicts[i], axis=0) for i in range(output_num)]
        return y_pred


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
        self.__pipeline_input_layers = []
        self.__pipeline_output_layers = []
        self.__is_output_stage = False

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

    @property
    def pipeline_input_layers(self) -> List[EagerPipelineInputLayer]:
        return self.__pipeline_input_layers

    @property
    def pipeline_output_layers(self) -> List[EagerPipelineOutputLayer]:
        return self.__pipeline_output_layers

    @property
    def is_output_stage(self) -> bool:
        return self.__is_output_stage

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

    def build(self, is_output_stage: bool) -> None:
        self.__is_output_stage = is_output_stage
        self.__input_shape = self.__prepare_input_shape()

        self.__pipeline_input_tensors = []
        self.__input_tensors = []
        for i in range(len(self.__input_shape)):
            shape = self.__input_shape[i]
            this_inputs = Input(shape=shape)
            self.__input_tensors.append(this_inputs)
            layer = EagerPipelineInputLayer(
                self, self.input_pipes[i],
                name=f'pipeline-input-{i}'
            )
            self.__pipeline_input_layers.append(layer)
            self.__pipeline_input_tensors.append(layer(this_inputs))
        outputs = self.call(*self.__pipeline_input_tensors)

        if not isinstance(outputs, (tuple, list)):
            outputs = (outputs,)

        self.__output_tensors = []
        for i in range(len(self.__output_pipes)):
            layer = EagerPipelineOutputLayer(
                self, self.__output_pipes[i],
                name=f'pipeline-output-{i}'
            )
            self.__pipeline_output_layers.append(layer)
            self.__output_tensors.append(layer(outputs[i]))
        self.__model = PipelineKerasImplementModel(
            inputs=self.__input_tensors,
            outputs=self.__output_tensors,
            stage=self
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

    def get_micro_batch_inputs(self, mb_inputs):
        for i, each in enumerate(self.pipeline_input_layers):
            each.receive_forward(mb_inputs[i])
        for i, each in enumerate(self.pipeline_input_layers):
            mb_inputs[i] = each.wait_and_get_receive_forward(mb_inputs[i])
        return mb_inputs

    def send_micro_batch_outputs(self, mb_outputs, mb_begin, mb_end):
        for i, each in enumerate(self.pipeline_output_layers):
            if len(mb_outputs[i].shape) == len(self.output_shape[i]):
                # 如果输出的没有批次这一维度，需要扩充维度使得维度相等
                mb_outputs[i] = tf.tile(
                    tf.expand_dims(mb_outputs[i], 0),
                    [mb_end - mb_begin, *[1 for each in self.output_shape[i]]]
                )

            each.send_forward(mb_outputs[i])
        for each in self.pipeline_output_layers:
            each.join_send_forward()

    def get_micro_batch_targets(self, mb_targets):
        for i, each in enumerate(self.pipeline_output_layers):
            each.receive_backward(mb_targets[i])
        for i, each in enumerate(self.pipeline_output_layers):
            mb_targets[i] = each.wait_and_get_backward(mb_targets[i])
        return mb_targets

    def send_micro_batch_grads(self):
        for each in self.pipeline_input_layers:
            each.send_backward()
        for each in self.pipeline_input_layers:
            each.join_send_backward()

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
                    send_op = CPPBackend.tf_lib().send_tensor(
                        output,
                        receiver=recv_stage.stage_rank,
                        tag=sending_to_input_index,
                        communicator_id=communicator.id,
                        key=f'pipeline-{pipeline_model_rank}-stage-'
                            f'{self.stage_rank}-output-'
                            f'{i}-forward-to-stage-'
                            f'{recv_stage.stage_rank}-input-'
                            f'{sending_to_input_index}'
                    )
                    send_ops[j] = send_op
                    current_stage_send_ops.append(send_op)
            last_stage_send_ops = current_stage_send_ops
        return output_layer_send_ops
