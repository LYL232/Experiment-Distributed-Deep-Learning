# 模仿horovod通过自定义Op实现分布式数据并行训练tensorflow2 模型

import tensorflow as tf


def main():
    (mnist_images, mnist_labels), _ = \
        tf.keras.datasets.mnist.load_data(
            path='mnist-%d.npz' % Global.process_rank()
        )

    dataset = tf.data.Dataset.from_tensor_slices(
        (tf.cast(mnist_images[..., tf.newaxis] / 255.0, tf.float32),
         tf.cast(mnist_labels, tf.int64))
    )

    dataset = dataset.repeat().shuffle(10000).batch(128)

    # 个人PC性能不佳, 降低网络复杂度
    mnist_model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(4, [3, 3], activation='relu',
                               input_shape=(28, 28, 1)),
        tf.keras.layers.Conv2D(8, [3, 3], activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    # 获取数据并行分布式优化器
    opt = data_parallelism_distributed_optimizer_wrapper(
        tf.optimizers.Adam(0.001)
    )

    # 设置`experimental_run_tf_function=False` 让TensorFlow
    # 使用opt计算梯度
    mnist_model.compile(loss=tf.losses.SparseCategoricalCrossentropy(),
                        optimizer=opt,
                        metrics=['accuracy'],
                        experimental_run_tf_function=False)

    callbacks = [
        # 将0号节点的模型参数广播到所有节点上
        InitialParametersBroadcastCallBack(0)
    ]

    if Global.process_rank() == 0:
        mnist_model.summary()
        callbacks.append(
            tf.keras.callbacks.ModelCheckpoint('./checkpoint-{epoch}.h5'))

    verbose = 1 if Global.process_rank() == 0 else 0

    mnist_model.fit(
        dataset, steps_per_epoch=500 // Global.processes(),
        callbacks=callbacks,
        epochs=5, verbose=verbose
    )


if __name__ == '__main__':
    import sys
    from os.path import abspath, join

    sys.path.append(abspath(join(__file__, '../../../')))

    from ddl.tensorflow import Global
    from ddl.tensorflow.keras import \
        data_parallelism_distributed_optimizer_wrapper
    from ddl.tensorflow.keras import InitialParametersBroadcastCallBack

    main()
