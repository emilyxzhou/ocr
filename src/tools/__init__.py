import cv2
import numpy as np
import os
import random

from os.path import abspath, dirname

cwd = os.getcwd()


class Constants:
    DATA_FOLDER = os.path.join(dirname(dirname(dirname(abspath(__file__)))), "data")
    TRAIN_FOLDER = os.path.join(DATA_FOLDER, "training")
    VALIDATION_FOLDER = os.path.join(DATA_FOLDER, "validation")
    METADATA_FOLDER = os.path.join(DATA_FOLDER, "metadata")
    CHECKPOINTS_FOLDER = os.path.join(DATA_FOLDER, "checkpoints")
    CLASSES = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    IMAGE_SIZE = 28
    CNN_WEIGHTS_PATH_FULL_H5 = os.path.join(DATA_FOLDER, "weights", "recognizer_pixel_operator_cnn_full.h5")
    CNN_WEIGHTS_PATH_HEX_H5 = os.path.join(DATA_FOLDER, "weights", "recognizer_pixel_operator_cnn_hex.h5")
    MLP_WEIGHTS_PATH_FULL = os.path.join(DATA_FOLDER, "weights", "recognizer_pixel_operator_mlp_full.pkl")
    MLP_WEIGHTS_PATH_HEX = os.path.join(DATA_FOLDER, "weights", "recognizer_pixel_operator_mlp_hex.pkl")
    CNN_WEIGHTS_PATH_FULL_CSV = os.path.join(DATA_FOLDER, "weights", "recognizer_pixel_operator_cnn_full.csv")
    CNN_WEIGHTS_PATH_HEX_CSV = os.path.join(DATA_FOLDER, "weights", "recognizer_pixel_operator_cnn_hex.csv")
    OCR_OUTPUT_FILE_CNN = os.path.join(DATA_FOLDER, "ocr_results_cnn.txt")
    OCR_OUTPUT_FILE_MLP = os.path.join(DATA_FOLDER, "ocr_results_mlp.txt")


def load_training_data(num_files=None, grayscale=True):
    image_paths = [
        f for f in os.listdir(Constants.TRAIN_FOLDER)
    ]

    if num_files is None:
        print("Loading all image files ...")
        num_files = len(image_paths)

    indices = random.sample(range(0, len(image_paths)), num_files)
    indices.sort()
    image_paths = [image_paths[i] for i in indices]
    labels = _read_labels(image_paths)
    full_image_paths = [
        os.path.join(Constants.TRAIN_FOLDER, f) for f in image_paths
    ]
    if grayscale:
        images = [cv2.imread(path, cv2.IMREAD_UNCHANGED) for path in full_image_paths]
    else:
        images = [cv2.imread(path) for path in full_image_paths]
    final_images = np.array([np.reshape(image, (28, 28, 1)) for image in images])
    return final_images, labels


def load_hex_training_data(grayscale=True):
    hex = list("0123456879ABCDEF")
    image_paths = [
        f for f in os.listdir(Constants.TRAIN_FOLDER) if f.split("_")[0] in hex
    ]
    labels = _read_labels(image_paths)
    full_image_paths = [
        os.path.join(Constants.TRAIN_FOLDER, f) for f in image_paths
    ]
    if grayscale:
        images = [cv2.imread(path, cv2.IMREAD_UNCHANGED) for path in full_image_paths]
    else:
        images = [cv2.imread(path) for path in full_image_paths]
    final_images = np.array([np.reshape(image, (28, 28, 1)) for image in images])
    return final_images, labels


def _read_labels(image_paths):
    print("Reading labels ...")
    count = 0
    symbols = "\"\"\\!@#$%^&*()-_=+,./<>?;:|~`[]{}"
    symbols = [sym for sym in symbols]
    labels = []

    for f in image_paths:
        file_info = f.split("_")
        if len(file_info[0]) > 1:  # text is a symbol instead of a character
            symbol_index = int(file_info[1])
            labels.append(symbols[symbol_index])
        else:
            char = file_info[0].split(".")[0]
            labels.append(_get_class_labels(char))
        # print(f"Image {count}/{total}")
        count += 1

    return np.array(labels)


def _get_class_labels(char):
    try:
        return Constants.CLASSES.index(char)
    except Exception:
        print("Invalid character")


def _get_box(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    white_rows = []
    white_cols = []
    for row in range(gray.shape[0]):
        for col in range(gray.shape[1]):
            if gray[row][col] > 10:
                white_rows.append(row)
                white_cols.append(col)

    min_row = min(white_rows) - 5
    min_col = min(white_cols) - 5
    max_row = max(white_rows) + 5
    max_col = max(white_cols) + 5

    return [
        [min_row, min_col],
        [min_row, max_col],
        [max_row, max_col],
        [max_row, min_col]
    ]


# scale pixels
def scale_pixels(train):
    # convert from integers to floats
    train_norm = train.astype("float32")
    # normalize to range 0-1
    train_norm = train_norm / 255.0
    # return normalized images
    return train_norm


def get_class_labels_from_prediction(pred):
    return Constants.CLASSES[np.argmax(pred)]


def get_cropped_size(image_path):
    boxes = _get_box(image_path)
    min_row = boxes[0][0]
    min_col = boxes[0][1]
    max_row = boxes[2][0]
    max_col = boxes[2][1]
    width = max_col - min_col
    height = max_row - min_row
    return width, height


def show_cropped(image_path):
    boxes = _get_box(image_path)
    min_row = boxes[0][0]
    min_col = boxes[0][1]
    max_row = boxes[2][0]
    max_col = boxes[2][1]
    image = cv2.imread(image_path)
    cv2.imshow("Frame", image[min_row:max_row, min_col:max_col])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
