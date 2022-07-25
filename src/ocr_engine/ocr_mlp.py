import math
import pickle

from ocr_engine import OCREngineBase
from tools import scale_pixels, Constants


class OCRMLP(OCREngineBase):

    def __init__(self, weights_path=None):
        super().__init__()
        try:
            with open(weights_path, "rb") as f:
                self._model = pickle.load(f)
        except Exception:
            print("Invalid path to pretrained model weights, loading MLP hex weights by default")
            with open(Constants.MLP_WEIGHTS_PATH_HEX, "rb") as f:
                self._model = pickle.load(f)

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

                input_char = char.reshape(1, int(math.pow(Constants.IMAGE_SIZE, 2)))
                pred = self._model.predict(input_char)
                predictions.append(Constants.CLASSES[pred[0]])
            predictions.append("\n")
        text = "".join(x for x in predictions)
        return text
