import cv2
import math
import numpy as np
import os
import pickle

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from models.cnn import cnn
from ocr_engine import OCREngineBase
from tools import get_class_labels_from_prediction, scale_pixels, Constants


class OCREngine(OCREngineBase):

    def __init__(self, weights_path=None, model="cnn", is_hex=True):
        super().__init__()
        self._model_type = model
        if model == "cnn":
            self._model = cnn(is_hex=is_hex)
            if weights_path is None:
                weights_path = Constants.CNN_WEIGHTS_PATH_HEX_H5
            try:
                self._model.load_weights(weights_path)
            except Exception:
                print("Invalid path to pretrained model weights, loading CNN hex weights by default")
                self._model.load_weights(Constants.CNN_WEIGHTS_PATH_HEX_H5)
        elif model == "mlp":
            if weights_path is None:
                weights_path = Constants.MLP_WEIGHTS_PATH_HEX
            try:
                with open(weights_path, "rb") as f:
                    self._model = pickle.load(f)
            except Exception:
                print("Invalid path to pretrained model weights, loading MLP hex weights by default")
                with open(Constants.MLP_WEIGHTS_PATH_HEX, "rb") as f:
                    self._model = pickle.load(f)
        else:
            print(f"Model type {model} not supported, defaulting to CNN.")

    def get_text(self, image, verbose=False):
        lines = self._segment_characters(image)
        num_chars = sum([len(line) for line in lines])
        if verbose:
            print(f"{len(lines)} lines detected, {num_chars} chars total")
        predictions = []
        # Resize cropped characters to the shape expected by the model (28x28).
        # Keeps correct scaling and pads to 28x28 with black pixels if necessary.
        for line in lines:
            for char in line:
                # The following 5 lines are for testing purposes.
                # print(f"Shape before resizing: {char.shape}")
                # cv2.namedWindow("char", cv2.WINDOW_NORMAL)
                # cv2.resizeWindow("char", 200, 200)
                # cv2.imshow("char", char)
                # cv2.waitKey(0)

                scale = min(Constants.IMAGE_SIZE / max(char.shape), 1)
                # char = cv2.resize(char, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
                char = cv2.resize(char, None, fx=scale, fy=scale)
                l_r = (math.floor((Constants.IMAGE_SIZE - char.shape[1])/2), (math.ceil((Constants.IMAGE_SIZE - char.shape[1])/2)))
                t_b = (math.floor((Constants.IMAGE_SIZE - char.shape[0])/2), (math.ceil((Constants.IMAGE_SIZE - char.shape[0])/2)))
                char = np.pad(char, (t_b, l_r), mode="constant", constant_values=0)
                char = scale_pixels(char)
                # print(f"Shape after resizing: {char.shape}")
                # cv2.imshow("resized", char)
                # cv2.waitKey(0)

                if self._model_type == "cnn":
                    input_char = np.reshape(char, (1, 28, 28, 1))
                    pred = self._model.predict(input_char, verbose=0)
                    predictions.append(get_class_labels_from_prediction(pred))
                elif self._model_type == "mlp":
                    input_char = char.reshape(1, 784)
                    pred = self._model.predict(input_char)
                    predictions.append(Constants.CLASSES[pred[0]])
            predictions.append("\n")
        text = "".join(x for x in predictions)
        return text
