from tensorflow.keras.layers import Dense, Flatten, Conv2D, \
    MaxPooling2D, Dropout, Reshape
import tensorflow as tf
import numpy as np


def main():
    from ddl.tensorflow.communicator import Communicator
    from ddl.tensorflow.keras.parallelism.data import \
        data_parallelism_distributed_optimizer_wrapper
    from ddl.tensorflow.keras.parallelism.data import \
        InitialParametersBroadcastCallBack
    from ddl.tensorflow.keras.parallelism.data.lr_warm_up_callback import \
        LearningRateWarmupCallback

    base_lr = 0.001

    model = tf.keras.Sequential([
        Reshape(input_shape=(28, 28), target_shape=(28, 28, 1)),
        Conv2D(32, [3, 3], activation='relu'),
        Conv2D(64, [3, 3], activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(10, activation='softmax')
    ])

    (data, label), (test_data, test_label) = tf.keras.datasets.mnist.load_data(
        path='original-mnist.npz')
    world = Communicator.world()
    samples_per_rank = len(data) // world.size
    begin = world.rank * samples_per_rank
    end = begin + samples_per_rank
    if world.rank == world.size - 1 and end < len(data):
        end = len(data)

    data = data[begin:end, ...] / 255.0
    label = label[begin:end, ...]

    scaled_lr = world.size * base_lr

    optimizer = data_parallelism_distributed_optimizer_wrapper(
        tf.optimizers.Adam(scaled_lr)
    )

    model.compile(
        loss=tf.losses.SparseCategoricalCrossentropy(),
        optimizer=optimizer,
        metrics=['accuracy'],
    )

    if world.size > 1:
        # 统一初始权重, 由根节点广播
        callbacks = [InitialParametersBroadcastCallBack(0)]
    else:
        callbacks = []

    callbacks.append(
        LearningRateWarmupCallback(
            warmup_epochs=3, initial_lr=scaled_lr,
            verbose=1,
            communicator=world
        )
    )

    model.fit(
        x=data, y=label,
        batch_size=1000,
        epochs=5, verbose=1 if world.rank == 0 else 0,
        callbacks=callbacks
    )

    # 只需要0进程进行预测计算精确度即可
    if world.rank == 0:
        test_data = test_data / 255.0
        predict = np.argmax(
            model.predict(test_data, batch_size=1000, verbose=1), axis=-1)
        assert len(test_label) == len(predict)
        corrects = 0
        for i in range(len(test_label)):
            if test_label[i] == predict[i]:
                corrects += 1
        print(f'test accuracy: {corrects / float(len(test_label))}')


if __name__ == '__main__':
    import sys
    from os.path import abspath, join

    sys.path.append(abspath(join(__file__, '../../../')))

    main()
