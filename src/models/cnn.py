from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from keras.models import Sequential
from keras.optimizers import SGD


def cnn(is_hex=False):
    """
    :param is_hex: True if training or loading a model on hex character dataset; False if training
        or loading on full dataset.
    :return: A compiled Tensorflow CNN model.
    """
    model = Sequential()
    model.add(Conv2D(
        16,
        (5, 5),
        activation="relu",
        kernel_initializer="he_uniform",
        input_shape=(28, 28, 1)
    ))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(
        32,
        (3, 3),
        activation="relu",
        kernel_initializer="he_uniform"
    ))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(100, activation="relu", kernel_initializer="he_uniform"))
    if is_hex:
        model.add(Dense(22, activation="softmax"))
    else:
        model.add(Dense(62, activation="softmax"))
    # Compile model
    opt = SGD(learning_rate=0.01, momentum=0.9)
    model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])
    return model
