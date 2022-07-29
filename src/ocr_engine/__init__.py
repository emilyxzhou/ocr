import cv2
import math
import numpy as np

from tools import scale_pixels, Constants


class OCREngineBase:

    def __init__(self):
        self._model_type = None
        self._model = None

    def segment_characters(self, image):
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
        sorted_bounding_boxes = self._sort_bounding_boxes(contours)
        for line in sorted_bounding_boxes:
            chars = []
            i = 0
            while i < len(line):
                box1 = line[i]
                (x, y, w, h) = box1
                if 0 < i < len(line) - 1:
                    box0 = line[i-1]
                    box1 = line[i]
                    box2 = line[i+1]
                    if self._is_skip_box(box0, box1, box2):
                        i += 1
                        continue

                # Compare current box to next box and see if they should be merged into one
                if i < len(line) - 1:
                    box2 = line[i+1]
                    if self._is_merge_boxes(box1, box2):
                        (x, y, w, h) = self._merge_boxes(box1, box2)
                        i += 1
                curr_num = image_gray[y:y + h, x:x + w]
                _, curr_num = cv2.threshold(curr_num, 220, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                chars.append(curr_num)
                # The following three lines are used for debugging purposes
                # cv2.rectangle(image, (x, y), (x + w, y + h), color=(0, 255, 0), thickness=1)
                # cv2.imshow("boxes", image)
                # cv2.waitKey(0)
                i += 1
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
        for box in sorted_y:
            if box[3] >= max_char_height / 3:
                valid_contour_boxes.append(box)
        # Separate bounding boxes into horizontal lines
        for i in range(len(valid_contour_boxes)):
            if valid_contour_boxes[i][1] > min_y + max_char_height:
                min_y = valid_contour_boxes[i][1]
                new_line_indices.append(i)
        # Sort boxes within lines from left to right
        lines = [valid_contour_boxes[
                 new_line_indices[i]:new_line_indices[i+1]
                 ] for i in range(len(new_line_indices)-1)]
        lines.append(sorted_y[new_line_indices[-1]:])
        sorted_lines = [sorted(line, key=lambda b: b[0] + b[2]) for line in lines]
        return sorted_lines

    def _is_skip_box(self, box0, box1, box2, threshold=15):
        """
        :param box0: Leftmost box of three adjacent boxes in the same line.
        :param box1: Middle box of three adjacent boxes in the same line.
        :param box2: Rightmost box of three adjacent boxes in the same line.
        :param threshold: Maximum number of pixels the middle box can differ from the left and right boxes
        :return: True if the y-coordinate of the upper left point of the middle box (box1) differs from the
            left and right boxes by more than the threshold number of pixels.
        """
        return abs(box1[1] - box0[1]) > threshold and abs(box2[1] - box1[1]) > threshold

    def _is_merge_boxes(self, box1, box2, x_thresh=1, y_thresh=3):
        """
        Checks if two contour boxes should be merged into one.
        :param box1: Contour box in the format (x, y, w, h). (x, y) denotes the top left corner.
        :param box2: Contour box in the format (x, y, w, h).
        :return: True if boxes have very close x or y coordinates and are nearly overlapping.
            'Close' if x or y coordinates are within the threshold # pixels of one another.
            'Nearly overlapping' if edges are within the threshold # pixels of one another.
        """
        (x1, y1, w1, h1) = box1
        (x2, y2, w2, h2) = box2
        if abs(x1 - x2) < x_thresh and (abs(y1 + h1 - y2) < y_thresh or abs(y2 + h2 - y1) < y_thresh):
            return True
        elif abs(y1 - y2) < y_thresh and (abs(x1 + w1 - x2) < x_thresh or abs(x2 + w2 - x1) < x_thresh):
            return True
        else:
            return False

    def _merge_boxes(self, box1, box2, x_thresh=1, y_thresh=3):
        """
        Merges two bounding boxes into one.
        :param box1: Contour box in the format (x, y, w, h). (x, y) denotes the top left corner.
        :param box2: Contour box in the format (x, y, w, h).
        :return: Merged bounding box in the format (x, y, w, h).
        """
        (x1, y1, w1, h1) = box1
        (x2, y2, w2, h2) = box2
        # Merge boxes vertically
        if abs(x1 - x2) < x_thresh:
            (x, y, w, h) = (
                max(x1, x2),
                min(y1, y2),
                max(w1, w2),
                h1 + h2
            )
        # Merge boxes horizontally
        elif abs(y1 - y2) < y_thresh:
            (x, y, w, h) = (
                min(x1, x2),
                min(y1, y2),
                w1 + w2,
                max(h1, h2)
            )
        else:
            raise ValueError("Boxes did not meet criteria for merging")

        return x, y, w, h

    def _resize_model_input(self, char):
        """
        Resizes the input character image to the input size of the OCR model.
        :param char: Image to be resized; numpy array expected
        :return: Resized image. Scales original image down such that the maximum height or width is
            equivalent to the model input side length. Pads remaining space with black pixels.
        """
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
        # print(f"Shape after resizing: {char.shape}")
        # cv2.imshow("resized", char)
        # cv2.waitKey(0)

        return char
