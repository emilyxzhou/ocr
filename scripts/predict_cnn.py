import cv2
from keras.utils import to_categorical
import numpy as np
import os
import random

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from models import cnn
from tools import get_class_labels_from_prediction, load_training_data, scale_pixels, Constants

BATCH_SIZE = 16
EPOCHS = 50
HEX = list("0123456789ABCDEF")

model = cnn()
model_hex = cnn(is_hex=True)
model.load_weights(Constants.CNN_WEIGHTS_PATH_FULL_H5)
model_hex.load_weights(Constants.CNN_WEIGHTS_PATH_HEX_H5)

num_indices = sorted(random.sample(range(0, 350), 350))
char_indices = sorted(random.sample(range(0, 36), 36))
num_indices_hex = sorted(random.sample(range(0, 350), 350))
char_indices_hex = sorted(random.sample(range(0, 16), 16))
chars = list(Constants.CLASSES)
files = []
files_hex = []
for char in char_indices:
    for num in num_indices:
        files.append(f"{chars[char]}_{num}.jpg")
for char in char_indices_hex:
    for num in num_indices_hex:
        files_hex.append(f"{HEX[char]}_{num}.jpg")
paths = [os.path.join(Constants.TRAIN_FOLDER, file) for file in files]
paths_hex = [os.path.join(Constants.TRAIN_FOLDER, file) for file in files_hex]
images = [cv2.imread(path) for path in paths]
images = [scale_pixels(image) for image in images]
images_hex = [cv2.imread(path) for path in paths_hex]
images_hex = [scale_pixels(image) for image in images_hex]

correct = 0
count = 0
incorrect_paths_full = []
for i, image in enumerate(images):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    test_input = np.reshape(image, (1, 28, 28, 1))
    prediction = model.predict(test_input, verbose=0)
    prediction = get_class_labels_from_prediction(prediction)
    actual = files[i].split("_")[0]
    # print(f"Prediction: {prediction}; actual: {actual}")
    if prediction == actual:
        correct += 1
    else:
        incorrect_paths_full.append(files[i])
    count += 1

correct_hex = 0
count_hex = 0
incorrect_paths_hex = []
for i, image in enumerate(images_hex):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    test_input = np.reshape(image, (1, 28, 28, 1))
    prediction = model_hex.predict(test_input, verbose=0)
    prediction = get_class_labels_from_prediction(prediction)
    actual = files_hex[i].split("_")[0]
    # print(f"Prediction: {prediction}; actual: {actual}")
    if prediction == actual:
        correct_hex += 1
    else:
        incorrect_paths_hex.append(files_hex[i])
    count_hex += 1

correct_full_hex = 0
count_full_hex = 0
for i, image in enumerate(images_hex):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    test_input = np.reshape(image, (1, 28, 28, 1))
    prediction = model.predict(test_input, verbose=0)
    prediction = get_class_labels_from_prediction(prediction)
    actual = files_hex[i].split("_")[0]
    # print(f"Prediction: {prediction}; actual: {actual}")
    if prediction == actual:
        correct_full_hex += 1
    count_full_hex += 1

print(f"Test accuracy for full weights: {correct/count}")
print(f"Test accuracy for hex weights: {correct_hex/count_hex}")
print(f"Test accuracy for full weights on hex data: {correct_full_hex/count_full_hex}")

results_full_path = os.path.join(
    os.getcwd(), Constants.DATA_FOLDER, "test_results_cnn_full.txt"
)
results_hex_path = os.path.join(
    os.getcwd(), Constants.DATA_FOLDER, "test_results_cnn_hex.txt"
)

with open(results_full_path, "w") as f:
    for file_name in incorrect_paths_full:
        f.write(f"{file_name}\n")

with open(results_hex_path, "w") as f:
    for file_name in incorrect_paths_hex:
        f.write(f"{file_name}\n")