from ddl.tensorflow.communicator import Communicator
from tensorflow.keras.callbacks import Callback
from tensorflow.keras import backend


class LearningRateScheduleCallback(Callback):
    def __init__(
            self,
            multiplier, start_epoch=0, end_epoch=None,
            staircase=True,
            momentum_correction=True, steps_per_epoch=None,
            initial_lr=None, *args):
        super().__init__()
        self.start_epoch = start_epoch
        self.end_epoch = end_epoch
        self.staircase = staircase
        self.momentum_correction = momentum_correction
        self.initial_lr = initial_lr
        self.restore_momentum = None
        self.steps_per_epoch = steps_per_epoch
        self.current_epoch = None

        if not callable(multiplier):
            self.staircase = True
            self.multiplier = lambda epoch: multiplier
        else:
            self.multiplier = multiplier

    def _autodetect_steps_per_epoch(self):
        if self.params.get('steps'):
            # The number of steps is provided in the parameters.
            return self.params['steps']
        elif self.params.get('samples') and self.params.get('batch_size'):
            # Compute the number of steps
            # per epoch using # of samples and a batch size.
            return self.params['samples'] // self.params['batch_size']
        else:
            raise ValueError(
                'Could not autodetect the number of steps per epoch. '
                'Please specify the steps_per_epoch parameter to the '
                '%s() or upgrade to the latest version of Keras.'
                % self.__class__.__name__
            )

    def _adjust_learning_rate(self, epoch):
        old_lr = backend.get_value(self.model.optimizer.lr)
        new_lr = self.initial_lr * self.multiplier(epoch)
        backend.set_value(self.model.optimizer.lr, new_lr)

        if hasattr(self.model.optimizer,
                   'momentum') and self.momentum_correction:
            # See the paper cited above
            # for more information about momentum correction.
            self.restore_momentum = backend.get_value(
                self.model.optimizer.momentum)
            backend.set_value(self.model.optimizer.momentum,
                              self.restore_momentum * new_lr / old_lr)

    def _restore_momentum_if_needed(self):
        if self.restore_momentum:
            backend.set_value(self.model.optimizer.momentum,
                              self.restore_momentum)
            self.restore_momentum = None

    def on_train_begin(self, logs=None):
        if self.initial_lr is None:
            self.initial_lr = backend.get_value(self.model.optimizer.lr)
        if not self.staircase and not self.steps_per_epoch:
            self.steps_per_epoch = self._autodetect_steps_per_epoch()

    def on_epoch_begin(self, epoch, logs=None):
        self.current_epoch = epoch

    def on_batch_begin(self, batch, logs=None):
        if (self.current_epoch < self.start_epoch or
                (self.end_epoch is not None and self.current_epoch >=
                 self.end_epoch)):
            return

        if self.staircase and batch == 0:
            # Do on first batch of every epoch.
            self._adjust_learning_rate(self.current_epoch)
        elif not self.staircase:
            epoch = self.current_epoch + float(batch) / self.steps_per_epoch
            self._adjust_learning_rate(epoch)

    def on_batch_end(self, batch, logs=None):
        self._restore_momentum_if_needed()

    def on_epoch_end(self, epoch, logs=None):
        if logs is not None:
            # Log current learning rate.
            logs['lr'] = backend.get_value(self.model.optimizer.lr)


class LearningRateWarmupCallback(LearningRateScheduleCallback):
    """
    学习率warmup回调类，直接参考horovod实现
    """

    def __init__(self,
                 warmup_epochs=5,
                 momentum_correction=True,
                 steps_per_epoch=None,
                 verbose=0, initial_lr=None,
                 communicator: Communicator = Communicator.world(),
                 *args):
        def multiplier(epoch):
            # Adjust epoch to produce round numbers
            # at the end of each epoch, so that TensorBoard
            # learning rate graphs look better.
            epoch += 1. / self.steps_per_epoch
            return 1. / communicator.size * (
                    epoch * (communicator.size - 1) / warmup_epochs + 1)

        super().__init__(
            multiplier=multiplier,
            start_epoch=0, end_epoch=warmup_epochs,
            staircase=False,
            momentum_correction=momentum_correction,
            steps_per_epoch=steps_per_epoch, initial_lr=initial_lr,
            *args
        )
        self.verbose = verbose if communicator.rank == 0 else 0

    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch, logs)

        if epoch == self.end_epoch - 1 and self.verbose > 0:
            new_lr = backend.get_value(self.model.optimizer.lr)
            print('\nEpoch %d: finished gradual learning rate warmup to %g.' %
                  (epoch + 1, new_lr))
