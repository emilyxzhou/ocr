import cv2
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from models import cnn
from tools import get_class_labels, Constants


class OCREngine:

    def __init__(self, weights_path):
        self._model = cnn()
        self._model.load_weights(weights_path)

    def get_text(self, image):
        chars = self._segment_characters(image)
        text = []
        for char in chars:
            scale = 28 / max(char.shape)
            char = cv2.resize(char, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
            padding_lr = (char.shape[1])
            padding_tb = (char.shape[0])
            # text.append(self._model.predict(char))

    def _segment_characters(self, image):
        """
        Segments the input image into individual characters.
        Returns a list of cropped images containing a single character.
        """
        if type(image) is not np.ndarray:
            raise TypeError(f"Input image is not a numpy array.")
        chars = []
        digit_w = 28
        digit_h = 28
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, th = cv2.threshold(image_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        temp = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        th_dilated = cv2.morphologyEx(th, cv2.MORPH_DILATE, temp)
        contours, _ = cv2.findContours(th_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for line in self._sort_bounding_boxes(contours):
            for b in line:
                (x, y, w, h) = b
                cv2.rectangle(image, (x, y), (x + w, y + h), color=(0, 255, 0), thickness=2)
                curr_num = image_gray[y:y + h, x:x + w]
                curr_num = cv2.resize(curr_num, dsize=(digit_w, digit_h))
                _, curr_num = cv2.threshold(curr_num, 220, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                # cv2.imshow("char", curr_num)
                # cv2.waitKey(0)
                chars.append(curr_num)
            # cv2.imshow("boxes", image)
            # cv2.waitKey(0)
        return chars

    def _sort_bounding_boxes(self, contours):
        contour_boxes = [cv2.boundingRect(c) for c in contours]
        sorted_y = sorted(contour_boxes, key=lambda b: b[1] + b[3])
        new_line_indices = [0]
        min_y = sorted_y[0][1]
        max_char_height = max([bounds[3] for bounds in sorted_y])
        for i in range(len(sorted_y)):
            if sorted_y[i][1] > min_y + max_char_height:
                min_y = sorted_y[i][1]
                new_line_indices.append(i)
        lines = [sorted_y[
                 new_line_indices[i]:new_line_indices[i+1]
                 ] for i in range(len(new_line_indices)-1)]
        lines.append(sorted_y[new_line_indices[-1]:])
        sorted_lines = [sorted(line, key=lambda b: b[0] + b[2]) for line in lines]
        return sorted_lines


if __name__ == "__main__":
    import os

    from tools import load_training_data, scale_pixels
    from keras.utils import to_categorical

    test_image_path = os.path.join(
        os.getcwd(), "..", "..", "data", "train_0.jpg"
    )
    test_char_path = os.path.join(
        os.getcwd(), "..", "..", "data", "training", "A_0.jpg"
    )
    image = cv2.imread(test_image_path)
    # image = cv2.imread(test_char_path)
    # cv2.imshow("image", image)
    # cv2.waitKey(0)

    weights_path = os.path.join(os.getcwd(), "..", "..", "scripts", "recognizer_pixel_operator.h5")
    ocr_engine = OCREngine(weights_path)
    ocr_engine.get_text(image)

    # X_train, y_train = load_training_data()
    # X_train = scale_pixels(X_train)
    # X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
    # y_train = to_categorical(y_train)

    # img = X_train[0, :, :]
    # test_input = np.reshape(img, (1, 28, 28, 1))
    # prediction = model.predict(test_input)
    # prediction = Constants.CLASSES[np.argmax(prediction)]
    # actual = Constants.CLASSES[np.argmax(y_train[0])]
    # print(f"Prediction: {prediction}; actual: {actual}")
    # cv2.imshow("char", img)
    # cv2.waitKey(0)
