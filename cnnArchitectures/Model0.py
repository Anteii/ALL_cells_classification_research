import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Activation


def CNN_model0(input_shape):
    seed = 32
    weight_initializer = tf.keras.initializers.GlorotNormal(seed=seed)
    model = tf.keras.Sequential()

    model.add(Conv2D(32, (3, 3), strides=(1, 1), padding="valid",
                     input_shape=input_shape, kernel_initializer=weight_initializer))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(32, (3, 3), strides=(1, 1), padding="valid",
                     kernel_initializer=weight_initializer))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(64, (3, 3), strides=(1, 1), padding="valid",
                     kernel_initializer=weight_initializer))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Flatten())

    model.add(Dense(64))
    model.add(Activation("relu"))
    model.add(Dense(64))

    model.add(Dropout(0.5))

    model.add(Dense(1))
    model.add(Activation("sigmoid"))

    return model
