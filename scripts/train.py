import git
import imgaug
import itertools
import keras_ocr
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import sklearn.model_selection
import string
import tensorflow as tf

from tools import load_training_data


# assert tf.config.list_physical_devices('GPU'), 'No GPU is available.'

cwd = os.getcwd()
git_repo = git.Repo(cwd, search_parent_directories=True)
git_root = git_repo.git.rev_parse("--show-toplevel")
train_folder = os.path.join(
    git_root, "data", "training"
)
train_folder = train_folder.replace("/", "\\")

dataset = load_training_data(train_folder, 60)
train_labels = [(filepath, box, word.lower()) for filepath, box, word in dataset]

recognizer = keras_ocr.recognition.Recognizer()
recognizer.compile()

batch_size = 8
augmenter = imgaug.augmenters.Sequential([
    imgaug.augmenters.GammaContrast(gamma=(0.25, 3.0)),
])

train_labels, validation_labels = sklearn.model_selection.train_test_split(train_labels, test_size=0.2, random_state=42)
(training_image_gen, training_steps), (validation_image_gen, validation_steps) = [
    (
        keras_ocr.datasets.get_recognizer_image_generator(
            labels=labels,
            height=recognizer.model.input_shape[1],
            width=recognizer.model.input_shape[2],
            alphabet=recognizer.alphabet,
            augmenter=augmenter
        ),
        len(labels) // batch_size
    ) for labels, augmenter in [(train_labels, augmenter), (validation_labels, None)]
]
training_gen, validation_gen = [
    recognizer.get_batch_generator(
        image_generator=image_generator,
        batch_size=batch_size
    )
    for image_generator in [training_image_gen, validation_image_gen]
]

for i in range(50):
    image, text = next(training_image_gen)
    print('text:', text)
    plt.imshow(image)
