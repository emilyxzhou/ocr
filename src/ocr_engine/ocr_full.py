import cv2
import math
import numpy as np
import os
import pickle

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from models.cnn import cnn
from tools import get_class_labels_from_prediction, scale_pixels, Constants


class OCREngine:

    def __init__(self, weights_path=None, model="cnn", is_hex=True):
        self._model_type = model
        if model == "cnn":
            self._model = cnn(is_hex=is_hex)
            if weights_path is None:
                weights_path = Constants.CNN_WEIGHTS_PATH_HEX_H5
            try:
                self._model.load_weights(weights_path)
            except Exception:
                print("Invalid path to pretrained model weights, loading MLP hex weights by default")
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
                # print(F"Shape before resizing: {char.shape}")
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
                # print(F"Shape after resizing: {char.shape}")
                # cv2.imshow("resized", char)
                # cv2.waitKey(0)

                if self._model_type == "cnn":
                    input_char = np.reshape(char, (1, 28, 28, 1))
                    pred = self._model.predict(input_char)
                    predictions.append(get_class_labels_from_prediction(pred))
                elif self._model_type == "mlp":
                    input_char = char.reshape(1, 784)
                    pred = self._model.predict(input_char)
                    predictions.append(Constants.CLASSES[pred[0]])
            predictions.append("\n")
        text = "".join(x for x in predictions)
        return text

    def _segment_characters(self, image):
        """
        Segments the input image into individual characters.
        Returns a list of cropped images containing a single character.
        """
        if type(image) is not np.ndarray:
            raise TypeError(f"Input image is not a numpy array.")
        lines = []
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, th = cv2.threshold(image_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # temp = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        # th_dilated = cv2.morphologyEx(th, cv2.MORPH_DILATE, temp)
        contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for line in self._sort_bounding_boxes(contours):
            chars = []
            for b in line:
                (x, y, w, h) = b
                cv2.rectangle(image, (x, y), (x + w, y + h), color=(0, 255, 0), thickness=2)
                curr_num = image_gray[y:y + h, x:x + w]
                _, curr_num = cv2.threshold(curr_num, 220, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                chars.append(curr_num)
                # cv2.imshow("boxes", image)
                # cv2.waitKey(0)
            lines.append(chars)
        # cv2.imshow("boxes", image)
        # cv2.waitKey(0)
        return lines

    def _sort_bounding_boxes(self, contours):
        """
        Sorts bounding boxes of contours top to bottom, left to right.
        Returns a list of lists, where the outer lists represent lines in order from top to bottom.
        Bounding boxes within each list are ordered from left to right.
        """
        contour_boxes = [cv2.boundingRect(c) for c in contours]
        valid_contour_boxes = []
        sorted_y = sorted(contour_boxes, key=lambda b: b[1] + b[3])
        new_line_indices = [0]
        min_y = sorted_y[0][1]
        max_char_height = max([bounds[3] for bounds in sorted_y])
        for box in contour_boxes:
            if box[3] >= max_char_height / 3:
                valid_contour_boxes.append(box)
        # Separate bounding boxes into horizontal lines
        for i in range(len(sorted_y)):
            if sorted_y[i][1] > min_y + max_char_height:
                min_y = sorted_y[i][1]
                new_line_indices.append(i)
        # Sort boxes within lines from left to right
        lines = [sorted_y[
                 new_line_indices[i]:new_line_indices[i+1]
                 ] for i in range(len(new_line_indices)-1)]
        lines.append(sorted_y[new_line_indices[-1]:])
        sorted_lines = [sorted(line, key=lambda b: b[0] + b[2]) for line in lines]
        return sorted_lines
