import math
import numpy as np
import os
import pickle

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from models.cnn import cnn
from ocr_engine import OCREngineBase
from tools import get_class_labels_from_prediction, scale_pixels, Constants


class OCREngine(OCREngineBase):

    def __init__(self, model="cnn", weights="full", weights_path=None, is_hex=True):
        if weights_path is None:
            weights_path = Constants.WEIGHTS_DICT[model][weights]
        super().__init__()
        self._model_type = model
        if model == "cnn":
            self._model = cnn(is_hex=is_hex)
            try:
                self._model.load_weights(weights_path)
            except Exception:
                print("Invalid path to pretrained model weights, loading CNN hex weights by default")
                self._model.load_weights(Constants.CNN_WEIGHTS_PATH_HEX_H5)
        elif model == "mlp":
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
        lines = self.segment_characters(image)
        num_chars = sum([len(line) for line in lines])
        if verbose:
            print(f"{len(lines)} lines detected, {num_chars} chars total")
        predictions = []
        # Resize cropped characters to the shape expected by the model (28x28).
        # Keeps correct scaling and pads to 28x28 with black pixels if necessary.
        for line in lines:
            for char in line:
                char = self._resize_model_input(char)
                char = scale_pixels(char)

                if self._model_type == "cnn":
                    input_char = np.reshape(char, (1, Constants.IMAGE_SIZE, Constants.IMAGE_SIZE, 1))
                    pred = self._model.predict(input_char, verbose=0)
                    predictions.append(get_class_labels_from_prediction(pred))
                elif self._model_type == "mlp":
                    input_char = char.reshape(1, int(math.pow(Constants.IMAGE_SIZE, 2)))
                    pred = self._model.predict(input_char)
                    predictions.append(Constants.CLASSES[pred[0]])
            predictions.append("\n")
        text = "".join(x for x in predictions)
        return text
