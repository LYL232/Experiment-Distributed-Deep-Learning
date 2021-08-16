from tensorflow.python.keras.engine.training import Model as ModelV2, _minimize
import tensorflow as tf
from tensorflow.python.keras.engine import data_adapter
from tensorflow.python.eager import backprop


class PipelineKerasImplementModel(ModelV2):
    def __init__(
            self, inputs: tuple or list, outputs: tuple or list,
            *args, **kwargs):
        if not isinstance(inputs, (tuple, list)):
            inputs = (inputs,)
        if not isinstance(outputs, (tuple, list)):
            outputs = (outputs,)
        super().__init__(inputs=inputs, outputs=outputs, *args, **kwargs)
        self._mb_size = 0

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

    # @tf.autograph.experimental.do_not_convert
    def train_step(self, data):
        data = data_adapter.expand_1d(data)
        x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)

        assert sample_weight is None, 'sample_weight is not supported'

        output_num = len(y)
        batch_size = tf.shape(x[0])[0]

        @tf.function
        def forward_fn():
            def cond(_mb_begin, _mb_size, _x, _outputs):
                return _mb_begin < tf.shape(_x[0])[0]

            def body(_mb_begin, _mb_size, _x, _outputs):
                _bs = tf.shape(_x[0])[0]
                _mb_end = tf.minimum(_mb_begin + _mb_size, _bs)

                ins = [_i[_mb_begin:_mb_end] for _i in _x]
                ous = self(ins, training=True)
                # todo: 让模型都返回tuple而不是直接返回张量
                if not isinstance(ous, (tuple, list)):
                    ous = (ous,)
                ous = [
                    tf.concat([_outputs[_i], ous[_i]], axis=0)
                    for _i in range(output_num)
                ]
                # if is_root:
                #     with tf.control_dependencies(ous):
                #         tf.print('fp[', _mb_begin, '][', _mb_end, ']')
                return _mb_end, _mb_size, _x, ous

            _, _, _, forward = tf.while_loop(
                cond,
                body,
                [
                    tf.zeros(shape=(), dtype=tf.int32),
                    self._mb_size,
                    x,
                    [
                        tf.zeros(shape=(0, *each.shape[1:]))
                        for each in self.outputs
                    ]
                ],
                shape_invariants=[
                    tf.TensorShape([]), tf.TensorShape([]),
                    tuple([each.shape for each in x]),
                    [
                        tf.TensorShape((None, *each.shape[1:]))
                        for each in self.outputs
                    ]
                ]
            )
            return forward

        @tf.function
        def compute_loss(_y_pred, _tape):
            def cond(_mb_begin, _mb_size, _pred, _target, _loss):
                return _mb_begin < tf.shape(_pred[0])[0]

            def body(_mb_begin, _mb_size, _pred, _target, _loss):
                _bs = tf.shape(_pred[0])[0]
                _mb_end = tf.minimum(_mb_begin + _mb_size, _bs)

                _mb_pred = [each[_mb_begin:_mb_end] for each in _pred]
                _mb_target = [
                    _target[_i][_mb_begin:_mb_end]
                    for _i in range(output_num)
                ]
                mb_loss = self.compiled_loss(
                    _mb_target,
                    _mb_pred,
                    sample_weight=None,
                    regularization_losses=self.losses
                )
                # with _tape.stop_recording():
                #     _g = _tape.gradient(mb_loss, self.trainable_variables)
                #     print('gard', _g)
                #     _new_grads = \
                #         [
                #             tf.zeros_like(
                #                 self.variables[i],
                #                 dtype=self.variables[i].dtype
                #             )
                #             if g is None else g for i, g in enumerate(_g)
                #         ]
                #     _grads = [
                #         _grads[_i] + _g
                #         for _i, _g in enumerate(_new_grads)
                #     ]
                # tf.print('rank', rank, '_mb_target shape:',
                #          tf.shape(_mb_target))
                # tf.print(
                #     'rank', rank, ': bp[', _mb_begin,
                #     '][', _mb_end, '], shape', tf.shape(_mb_target)
                # )
                _accumulate = (mb_loss * tf.cast(
                    _mb_end - _mb_begin, dtype=tf.float32
                ) / tf.cast(_bs, dtype=tf.float32)) + _loss

                return _mb_end, _mb_size, _pred, _target, _loss

            _, _, _, _, accumulated_grads = tf.while_loop(
                cond,
                body,
                [
                    tf.zeros(shape=(), dtype=tf.int32),
                    self._mb_size,
                    _y_pred, y,
                    tf.zeros(shape=(), dtype=tf.float32),
                    # [
                    #     tf.zeros_like(each, dtype=each.dtype)
                    #     for each in self.trainable_variables
                    # ]
                ],
            )

            return accumulated_grads

        mb_begin = 0
        mb_sizes = []

        while mb_begin < batch_size:
            mb_end = tf.minimum(mb_begin + self._mb_size, batch_size)
            mb_sizes.append((mb_begin, mb_end))
            mb_begin = mb_end

        with backprop.GradientTape() as tape:
            y_pred = forward_fn()
            losses = compute_loss(y_pred)
        _minimize(self.distribute_strategy, tape, self.optimizer, losses,
                  self.trainable_variables)
        self.compiled_metrics.update_state(y, y_pred, sample_weight)

        return {m.name: m.result() for m in self.metrics}
