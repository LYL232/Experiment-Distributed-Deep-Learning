import tensorflow as tf
from tensorflow.python.keras import backend
from tensorflow.python.ops import control_flow_ops
from tensorflow.keras.optimizers import Optimizer


class MicroBatchController:
    def __init__(
            self, stage, optimizer: Optimizer, session: tf.compat.v1.Session):
        from ddl.tensorflow.keras.parallelism.pipeline.stage import \
            PipelineStage
        assert isinstance(stage, PipelineStage)
        self.__session = session

        self.__pipeline_model_rank = stage.pipeline_model.pipeline_model_rank

        with tf.name_scope(f'pipeline-{self.__pipeline_model_rank}-vars'):
            self.__micro_batch_counter = \
                tf.Variable(0, name='current_micro_batch', dtype=tf.int32)
            # 总批次数
            self.__micro_batches = tf.Variable(
                0, name='micro_batches', dtype=tf.int32
            )
            # 每个批次的大小: 第一批大小和最后一批大小
            self.__first_micro_batch_size = tf.Variable(
                0, name='first_micro_batch_size', dtype=tf.int32
            )
            self.__last_micro_batch_size = tf.Variable(
                0, name='last_micro_batch_size', dtype=tf.int32
            )
            self.__grads_accumulator = []
            for param in stage.model.weights:
                self.__grads_accumulator.append(
                    tf.Variable(
                        tf.zeros(shape=param.shape),
                        name='micro-batch-saved-grads-' + param.name.replace(
                            '/', '-').replace(':', '-'),
                        dtype=param.dtype
                    )
                )
        for v in [
            self.__micro_batch_counter,
            self.__first_micro_batch_size,
            self.__last_micro_batch_size,
            self.__micro_batches,
            *self.__grads_accumulator
        ]:
            backend.track_variable(v)

        self.__micro_batch_first_size_placeholder = \
            tf.compat.v1.placeholder(shape=(), dtype=tf.int32)

        self.__init_micro_first_batch_size_op = \
            self.__first_micro_batch_size.assign(
                self.__micro_batch_first_size_placeholder
            )

        self.__micro_batch_last_size_placeholder = \
            tf.compat.v1.placeholder(shape=(), dtype=tf.int32)
        self.__init_micro_last_batch_size_op = \
            self.__last_micro_batch_size.assign(
                self.__micro_batch_last_size_placeholder
            )

        self.__micro_batch_sizes_placeholder = \
            tf.compat.v1.placeholder(shape=(), dtype=tf.int32)
        self.__init_micro_batches_op = self.__micro_batches.assign(
            self.__micro_batch_sizes_placeholder
        )

        self.__init_current_micro_batch_op = \
            self.__micro_batch_counter.assign(0)
        self.__init_micro_batch_grads_ops = []

        for each in self.__grads_accumulator:
            self.__init_micro_batch_grads_ops.append(
                each.assign(tf.zeros_like(each))
            )

        opt_apply_gradients = optimizer.apply_gradients

        if hasattr(optimizer, 'is_distributed_optimizer'):
            # 由于数据并行分布式优化器每一批次都会进行一次allreduce,
            # 所以需要复写一下优化器的get_gradient方法和_aggregate_gradients方法,
            # 防止不必要的allreduce

            # get_gradients方法直接使用原优化器的方法即可
            optimizer.get_gradients = optimizer.original_get_gradients

            # 记录一下原分布式的_aggregate_gradients方法,
            # noinspection PyProtectedMember
            distributed_aggregate_gradients = optimizer._aggregate_gradients

            def micro_batch_aggregate_gradients(grads_and_vars):
                """
                根据微批次计数器进行allreduce的方法, 当该批次是最后一批次时, 进行
                allreduce
                @param grads_and_vars: [(梯度, 变量)]
                @return: [梯度], todo: 注意, tf2.4版本的返回值有所不同
                """
                grads, variables = list(zip(*grads_and_vars))
                return tf.cond(
                    # 这counter是未进行自加操作的值, 所以减一
                    self.__micro_batch_counter
                    >= self.__micro_batches - 1,
                    lambda: distributed_aggregate_gradients(grads_and_vars),
                    lambda: list(grads),
                    name=f'if-original-aggregate-gradients'
                )

            optimizer._aggregate_gradients = micro_batch_aggregate_gradients

        def micro_batch_apply_gradients(
                grads_and_vars,
                name=None,
                experimental_aggregate_gradients=True):
            """
            替换掉模型优化器的应用梯度方法, 其实就是一个装饰器, 使得只有在当前批次的所有微批次结束时
            才进行梯度的更新
            @param grads_and_vars: List of (gradient, variable) pairs.
            @param name:
            @param experimental_aggregate_gradients:
            @return:
            """
            assert len(grads_and_vars) == len(self.__grads_accumulator)

            # 计数器自加1op
            current_batch_added = self.__micro_batch_counter.assign_add(1)

            for i in range(len(grads_and_vars)):
                # 微批次累加变量
                cumsum_grad = self.__grads_accumulator[i]
                # 该批次计算出的梯度
                grad = grads_and_vars[i][0]
                # 梯度对应的变量
                variable = grads_and_vars[i][1]
                assert cumsum_grad.shape == variable.shape

                # 将该批次计算出来的梯度进行累加的op
                assign_added = cumsum_grad.assign_add(
                    grad * tf.cast(
                        tf.cond(
                            current_batch_added < self.__micro_batches,
                            # 由于一批次的微批次大小只有两种,
                            # 第一到倒数第二微批次的大小都是一致的,
                            # 最后一微批次可能会有所不同, 所以这里要加一个判断
                            lambda: self.__first_micro_batch_size,
                            lambda: self.__last_micro_batch_size,
                        ),
                        dtype=grad.dtype
                    )
                )

                applying_gradient = tf.cond(
                    # 小于的时候就返回0梯度
                    current_batch_added < self.__micro_batches,
                    lambda: tf.zeros_like(variable),
                    lambda: tf.divide(
                        assign_added,
                        tf.cast(
                            (self.__micro_batches - 1) *
                            self.__first_micro_batch_size +
                            self.__last_micro_batch_size,
                            dtype=variable.dtype
                        )
                    ),
                )
                grads_and_vars[i] = (applying_gradient, grads_and_vars[i][1])

            return tf.cond(
                current_batch_added >= self.__micro_batches,
                lambda: opt_apply_gradients(
                    grads_and_vars, name, experimental_aggregate_gradients),
                # 如果没有到最后一微批次, 就什么也不做, 相当于把值赋给了梯度累计变量
                lambda: control_flow_ops.no_op(),
                name=f'original-applying-grads'
            )

        optimizer.apply_gradients = micro_batch_apply_gradients

    @property
    def session(self) -> tf.compat.v1.Session:
        return self.__session

    def initialize_micro_batch_vars(self, micro_batch_inputs: list):
        """
        初始化微批次所需的变量
        @param: micro_batch_inputs, 这一批次的微批次输入列表
        @return: None
        """
        # 这里其实有一些浪费计算资源, 因为传输给下一个阶段计算了一次前向传播, 后面进行fit的时候
        # 也计算了一次前向传播, todo: 可优化一次前向传播的计算, 或许可以直接调用优化器的apply_gradient
        assert len(micro_batch_inputs) > 0

        self.session.run([
            self.__init_micro_first_batch_size_op,
            self.__init_micro_last_batch_size_op,
            self.__init_micro_batches_op,
            self.__init_current_micro_batch_op,
            *self.__init_micro_batch_grads_ops
        ], feed_dict={
            self.__micro_batch_first_size_placeholder:
                micro_batch_inputs[0][0].shape[0],
            self.__micro_batch_last_size_placeholder:
                micro_batch_inputs[-1][0].shape[0],
            self.__micro_batch_sizes_placeholder:
                len(micro_batch_inputs)
        })
