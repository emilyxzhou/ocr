import keras_ocr
import matplotlib.pyplot as plt
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tools import load_from_json


dataset = load_from_json(1000)
test_labels = [(filepath, np.asarray(box), word.lower()) for filepath, box, word in dataset]

recognizer = keras_ocr.recognition.Recognizer()
recognizer.compile()
recognizer.model.load_weights('recognizer_pixel_operator.h5')

image_filepath, _, actual = test_labels[1]
predicted = recognizer.recognize(image_filepath)
print(f'Predicted: {predicted}, Actual: {actual}')
_ = plt.imshow(keras_ocr.tools.read(image_filepath))
plt.show()
