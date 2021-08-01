from ddl.tensorflow.util import executing_eagerly, make_tf_function
import tensorflow as tf
from tensorflow.python.keras import backend
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
            micro_batch_counter_var = \
                tf.Variable(0, name='current_micro_batch', dtype=tf.int32)
            # 总批次数
            micro_batches_var = tf.Variable(
                0, name='micro_batches', dtype=tf.int32
            )
            # 每个批次的大小: 第一批大小和最后一批大小
            first_micro_batch_size_var = tf.Variable(
                0, name='first_micro_batch_size', dtype=tf.int32
            )
            last_micro_batch_size_var = tf.Variable(
                0, name='last_micro_batch_size', dtype=tf.int32
            )
            grads_accumulator = []
            for param in stage.model.weights:
                grads_accumulator.append(
                    tf.Variable(
                        tf.zeros(shape=param.shape),
                        name='micro-batch-saved-grads-' + param.name.replace(
                            '/', '-').replace(':', '-'),
                        dtype=param.dtype
                    )
                )

        for v in [
            micro_batch_counter_var,
            first_micro_batch_size_var,
            last_micro_batch_size_var,
            micro_batches_var,
            *grads_accumulator
        ]:
            backend.track_variable(v)

        self.__execute_eagerly = executing_eagerly()

        if self.__execute_eagerly:

            def init_micro_batch_function(
                    first_micro_batch_size: int,
                    last_micro_batch_size: int,
                    micro_batches: int
            ):
                first_micro_batch_size_var.assign(
                    first_micro_batch_size)
                last_micro_batch_size_var.assign(last_micro_batch_size)
                micro_batches_var.assign(micro_batches)

            init_micro_batch_function = make_tf_function(
                init_micro_batch_function)

            self.__init_micro_batch_function = init_micro_batch_function

            micro_batch_first_size_placeholder = None
            init_micro_first_batch_size_op = None
            micro_batch_last_size_placeholder = None
            init_micro_last_batch_size_op = None
            micro_batch_sizes_placeholder = None
            init_micro_batches_op = None
        else:
            micro_batch_first_size_placeholder = \
                tf.compat.v1.placeholder(shape=(), dtype=tf.int32)

            init_micro_first_batch_size_op = \
                first_micro_batch_size_var.assign(
                    micro_batch_first_size_placeholder
                )

            micro_batch_last_size_placeholder = \
                tf.compat.v1.placeholder(shape=(), dtype=tf.int32)
            init_micro_last_batch_size_op = \
                last_micro_batch_size_var.assign(
                    micro_batch_last_size_placeholder
                )
            micro_batch_sizes_placeholder = \
                tf.compat.v1.placeholder(shape=(), dtype=tf.int32)
            init_micro_batches_op = micro_batches_var.assign(
                micro_batch_sizes_placeholder
            )

            def init_micro_batch_function(
                    first_micro_batch_size: int,
                    last_micro_batch_size: int,
                    micro_batches: int
            ):
                with self.session.graph.as_default():
                    with self.session.as_default():
                        self.session.run([
                            init_micro_first_batch_size_op,
                            init_micro_last_batch_size_op,
                            init_micro_batches_op
                        ], feed_dict={
                            micro_batch_first_size_placeholder:
                                first_micro_batch_size,
                            micro_batch_last_size_placeholder:
                                last_micro_batch_size,
                            micro_batch_sizes_placeholder: micro_batches
                        })

            self.__init_micro_batch_function = init_micro_batch_function

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

            # noinspection PyTypeChecker
            def micro_batch_aggregate_gradients(grads_and_vars):
                """
                根据微批次计数器进行allreduce的方法, 当该批次是最后一批次时, 进行
                allreduce
                @param grads_and_vars: [(梯度, 变量)]
                @return: [梯度], todo: 注意, tf2.4版本的返回值有所不同
                """
                if isinstance(grads_and_vars, zip):
                    grads_and_vars = list(grads_and_vars)

                return tf.cond(
                    # 这counter是未进行自加操作的值, 所以减一
                    micro_batch_counter_var
                    >= micro_batches_var - 1,
                    lambda: distributed_aggregate_gradients(grads_and_vars),
                    lambda: list(list(zip(*grads_and_vars))[0]),
                    name=f'if-original-aggregate-gradients'
                )

            optimizer._aggregate_gradients = micro_batch_aggregate_gradients

        # noinspection PyTypeChecker
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
            if isinstance(grads_and_vars, zip):
                grads_and_vars = list(grads_and_vars)
            assert len(grads_and_vars) == len(grads_accumulator)

            for i in range(len(grads_and_vars)):
                # 微批次累加变量
                cumsum_grad = grads_accumulator[i]
                # 该批次计算出的梯度
                grad = grads_and_vars[i][0]
                # 梯度对应的变量
                variable = grads_and_vars[i][1]

                # 将该批次计算出来的梯度进行累加的op
                assign_added = cumsum_grad.assign_add(
                    grad * tf.cast(
                        tf.cond(
                            micro_batch_counter_var < micro_batches_var - 1,
                            # 由于一批次的微批次大小只有两种,
                            # 第一到倒数第二微批次的大小都是一致的,
                            # 最后一微批次可能会有所不同, 所以这里要加一个判断
                            lambda: first_micro_batch_size_var,
                            lambda: last_micro_batch_size_var,
                        ),
                        dtype=grad.dtype
                    )
                )

                applying_gradient = tf.cond(
                    # 小于的时候就返回0梯度
                    micro_batch_counter_var < micro_batches_var - 1,
                    lambda: tf.zeros_like(variable),
                    lambda: tf.divide(
                        assign_added,
                        tf.cast(
                            (micro_batches_var - 1) *
                            first_micro_batch_size_var +
                            last_micro_batch_size_var,
                            dtype=variable.dtype
                        )
                    ),
                )
                grads_and_vars[i] = (
                    applying_gradient, grads_and_vars[i][1])

            def do_apply():
                with tf.control_dependencies([
                    opt_apply_gradients(
                        grads_and_vars, name,
                        experimental_aggregate_gradients
                    )]
                ):
                    return tf.group(
                        micro_batch_counter_var.assign(0),
                        *[each.assign(tf.zeros_like(each))
                          for each in grads_accumulator
                          ]
                    )

            def no_apply():
                # 如果没有到最后一微批次, 就只累加
                return tf.group(
                    micro_batch_counter_var.assign_add(1)
                )

            return tf.cond(
                micro_batch_counter_var >= micro_batches_var - 1,
                do_apply, no_apply,
                name=f'original-applying-grads'
            )

        optimizer.apply_gradients = micro_batch_apply_gradients

        self.__micro_batch_counter = micro_batch_counter_var

        self.__first_micro_batch_size = first_micro_batch_size_var
        self.__grads_accumulator = grads_accumulator
        self.__micro_batches = micro_batches_var
        self.__last_micro_batch_size = last_micro_batch_size_var
        self.__micro_batch_first_size_placeholder = \
            micro_batch_first_size_placeholder
        self.__init_micro_first_batch_size_op = init_micro_first_batch_size_op
        self.__micro_batch_last_size_placeholder = \
            micro_batch_last_size_placeholder
        self.__init_micro_last_batch_size_op = init_micro_last_batch_size_op
        self.__micro_batch_sizes_placeholder = micro_batch_sizes_placeholder
        self.__init_micro_batches_op = init_micro_batches_op

    @property
    def session(self) -> tf.compat.v1.Session:
        return self.__session

    def initialize_micro_batch_vars(self, micro_batch_inputs: list):
        """
        初始化微批次所需的变量
        @param: micro_batch_inputs, 这一批次的微批次输入列表
        @return: None
        """
        assert len(micro_batch_inputs) > 0
        self.__init_micro_batch_function(
            micro_batch_inputs[0][0].shape[0],
            micro_batch_inputs[-1][0].shape[0],
            len(micro_batch_inputs)
        )
