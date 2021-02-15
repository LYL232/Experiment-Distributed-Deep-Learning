# keras_pipeline的模型的单机版本, 作为对比

import tensorflow as tf


def main():
    (mnist_images, mnist_labels), _ = \
        tf.keras.datasets.mnist.load_data(path='mnist.npz')

    batch_size = 200
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
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(784, activation='relu'),
        tf.keras.layers.Dense(196, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    # 设置`experimental_run_tf_function=False` 让TensorFlow
    # 使用opt计算梯度
    mnist_model.compile(loss=tf.losses.SparseCategoricalCrossentropy(),
                        optimizer=tf.optimizers.Adam(0.001),
                        metrics=['accuracy'])

    mnist_model.fit(
        data_generator(), steps_per_epoch=samples // batch_size,
        epochs=5, verbose=1
    )


if __name__ == '__main__':
    main()
