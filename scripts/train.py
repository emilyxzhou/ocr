import cv2
import git
import imgaug
import itertools
import keras_ocr
import math
import matplotlib.pyplot as plt
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import random
import sklearn.model_selection
import string
import tensorflow as tf

from tools import load_from_json, show_cropped, TRAIN_FOLDER


# assert tf.config.list_physical_devices('GPU'), 'No GPU is available.'

dataset = load_from_json()
train_labels = [(filepath, np.asarray(box), word.lower()) for filepath, box, word in dataset]

recognizer = keras_ocr.recognition.Recognizer()
recognizer.compile()

batch_size = 16

train_labels, validation_labels = sklearn.model_selection.train_test_split(train_labels, test_size=0.2, random_state=42)
(training_image_gen, training_steps), (validation_image_gen, validation_steps) = [
    (
        keras_ocr.datasets.get_recognizer_image_generator(
            labels=labels,
            height=recognizer.model.input_shape[1],
            width=recognizer.model.input_shape[2],
            alphabet=recognizer.alphabet
        ),
        len(labels) // batch_size
    ) for labels in [train_labels, validation_labels]
]
training_gen, validation_gen = [
    recognizer.get_batch_generator(
        image_generator=image_generator,
        batch_size=batch_size
    )
    for image_generator in [training_image_gen, validation_image_gen]
]

# for i in range(10):
#     show_cropped(os.path.join(TRAIN_FOLDER, train_labels[i][0]))
#     image, text = next(training_image_gen)
#     print(image.shape)
    # image = cv2.imread(os.path.join(TRAIN_FOLDER, train_labels[i][0]))
    # print(image.shape)
    # print('text:', text)
    # cv2.imshow("frame", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, restore_best_weights=False),
    tf.keras.callbacks.ModelCheckpoint('recognizer_pixel_operator.h5', monitor='val_loss', save_best_only=True),
    tf.keras.callbacks.CSVLogger('recognizer_pixel_operator.csv')
]
recognizer.training_model.fit_generator(
    generator=training_gen,
    steps_per_epoch=training_steps,
    validation_steps=validation_steps,
    validation_data=validation_gen,
    callbacks=callbacks,
    epochs=10,
)

image_filepath, _, actual = train_labels[1]
predicted = recognizer.recognize(image_filepath)
print(f'Predicted: {predicted}, Actual: {actual}')
_ = plt.imshow(keras_ocr.tools.read(image_filepath))
