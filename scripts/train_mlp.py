import cv2
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

from tools import load_training_data, scale_pixels, Constants

BATCH_SIZE = 32
EPOCHS = 50


X_train, y_train = load_training_data()
X_train = scale_pixels(X_train)
X_train = X_train.reshape(X_train.shape[0], 784)
X_train, X_test, y_train, y_test = train_test_split(
    X_train, y_train, test_size=0.20, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5)

# print(y_val[0])
# cv2.imshow("frame", X_val[0])
# cv2.waitKey(0)

# create model
model = MLPClassifier(
    hidden_layer_sizes=(150, 100, 50),
    max_iter=300,
    activation="relu",
    solver="adam",
    random_state=1
)
# set up checkpoints
model_path = os.path.join(Constants.CHECKPOINTS_FOLDER, "weights-{epoch:02d}-{val_accuracy:.2f}.hdf5")

model.fit(X_train, y_train)
train_accuracy = model.score(X_train, y_train)
print(f"Training accuracy: {train_accuracy}")
test_accuracy = model.score(X_test, y_test)
print(f"Test accuracy: {test_accuracy}")

# model.save_weights("recognizer_pixel_operator.h5")

accuracy_score(y_test, model.predict(X_test))

test_inputs = X_test[0:10]
prediction = model.predict(test_inputs)
prediction = [Constants.CLASSES[i] for i in prediction]
actual = [Constants.CLASSES[i] for i in y_test[0:10]]
print(f"Prediction: {prediction}\nActual: {actual}")
# img = img.reshape((28, 28, 1)) * 255
# cv2.imshow("char", img)
# cv2.waitKey(0)
