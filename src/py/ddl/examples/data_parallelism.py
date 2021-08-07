from tensorflow.keras.layers import Dense, Flatten, Conv2D, \
    MaxPooling2D, Dropout, Reshape
import tensorflow as tf
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--dataset', default='mnist', type=str)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--epochs', default=20, type=int)
parser.add_argument('--warmup_epochs', default=5, type=int)
parser.add_argument('--lr', default=0.001, type=float)

args = parser.parse_args()

dataset = args.dataset

model_layers = []
if dataset == 'mnist':
    (train_inputs, train_label), (test_inputs, test_label) = \
        tf.keras.datasets.mnist.load_data(path='original-mnist.npz')
    input_shape = (28, 28)
    reshape = (28, 28, 1)
    model_layers.extend([
        Reshape(input_shape=(28, 28), target_shape=(28, 28, 1)),
        Conv2D(32, [3, 3], activation='relu')
    ])
elif dataset == 'cifar10':
    (train_inputs, train_label), (test_inputs, test_label) = \
        tf.keras.datasets.cifar10.load_data()
    model_layers.append(
        Conv2D(32, [3, 3], activation='relu', input_shape=(32, 32, 3)),
    )
else:
    raise ValueError('--dataset only support mnist and cifar10')

model_layers.extend([
    Conv2D(64, [3, 3], activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])


def get_processing_data(data, communicator):
    samples_per_rank = len(data) // communicator.size
    begin = communicator.rank * samples_per_rank
    end = begin + samples_per_rank
    if communicator.rank == communicator.size - 1 and end < len(data):
        end = len(data)
    return data[begin:end, ...]


def main():
    from ddl.tensorflow.communicator import Communicator
    from ddl.tensorflow.keras.parallelism.data import \
        data_parallelism_distributed_optimizer_wrapper
    from ddl.tensorflow.keras.parallelism.data import \
        InitialParametersBroadcastCallBack
    from ddl.tensorflow.keras.parallelism.data.lr_warm_up_callback import \
        LearningRateWarmupCallback
    from ddl.tensorflow.keras.parallelism.data.metric_average_callback import \
        MetricAverageCallback

    # 基础学习率
    base_lr = args.lr

    model = tf.keras.Sequential(model_layers)

    world = Communicator.world()

    x_train = get_processing_data(train_inputs, world) / 255.0
    y_train = get_processing_data(train_label, world)
    x_test = get_processing_data(test_inputs, world) / 255.0
    y_test = get_processing_data(test_label, world)

    # 将学习率乘上数据并行组数，然后在训练过程中学习率从略大于base_lr逐步warm up到这个值
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
        callbacks = [InitialParametersBroadcastCallBack(0, communicator=world)]
    else:
        callbacks = []

    callbacks.append(MetricAverageCallback())
    callbacks.append(
        LearningRateWarmupCallback(
            warmup_epochs=args.warmup_epochs, initial_lr=scaled_lr,
            verbose=1,
            communicator=world
        )
    )

    model.fit(
        x=x_train, y=y_train,
        batch_size=args.batch_size,
        epochs=args.epochs, verbose=1 if world.rank == 0 else 0,
        validation_data=(x_test, y_test),
        callbacks=callbacks
    )


if __name__ == '__main__':
    import sys
    from os.path import abspath, join

    sys.path.append(abspath(join(__file__, '../../../')))

    main()
