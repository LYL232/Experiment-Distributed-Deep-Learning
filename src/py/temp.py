from tensorflow.keras.layers import Dense, Flatten, Conv2D, \
    MaxPooling2D, Dropout, Reshape, Input, concatenate
from tensorflow.keras.models import Model
import tensorflow as tf

(data, label), _ = tf.keras.datasets.mnist.load_data(
    path='original-mnist.npz')
data = data / 255.0

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

inputs = Input(shape=(28, 28))

branch_0 = Reshape((28, 28, 1))(inputs)
branch_0 = Conv2D(8, [3, 3], activation='relu')(branch_0)
branch_0 = Conv2D(16, [3, 3], activation='relu')(branch_0)
branch_0 = MaxPooling2D(pool_size=(2, 2))(branch_0)
branch_0 = Dropout(0.25)(branch_0)
branch_0 = Flatten()(branch_0)

branch_1 = Dense(32, activation='relu')(inputs)
branch_1 = Dense(64, activation='relu')(branch_1)
branch_1 = Dropout(0.5)(branch_1)
branch_1 = Flatten()(branch_1)

merged = concatenate([branch_0, branch_1])
outputs = Dense(10, activation='softmax')(merged)

model = Model(inputs=inputs, outputs=outputs)

model.compile(
    loss=tf.losses.SparseCategoricalCrossentropy(),
    optimizer=tf.optimizers.Adam(0.001),
    metrics=['accuracy'],
)

model.fit(
    x=data, y=label,
    batch_size=200,
    epochs=5, verbose=1
)
