"""
data_and_pipeline_*.py的模型的单机版本, 作为对比
"""
from tensorflow.keras.layers import Dense, Flatten, Conv2D, \
    MaxPooling2D, Dropout, Reshape
import tensorflow as tf


def main():
    (mnist_images, mnist_labels), _ = \
        tf.keras.datasets.mnist.load_data(path='mnist.npz')

    batch_size = 1000
    samples = mnist_images.shape[0]

    def data_generator():
        begin = 0
        while True:
            end = min(begin + batch_size, samples)
            inputs = mnist_images[begin: end, ...] / 255.0
            labels = mnist_labels[begin: end, ...]
            begin = 0 if end >= samples else begin + batch_size
            yield inputs, labels

    mnist_model = tf.keras.Sequential([
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

    mnist_model.compile(loss=tf.losses.SparseCategoricalCrossentropy(),
                        optimizer=tf.optimizers.Adam(0.001),
                        metrics=['accuracy'])

    mnist_model.fit(
        data_generator(), steps_per_epoch=samples // batch_size,
        epochs=5, verbose=1
    )


if __name__ == '__main__':
    main()
