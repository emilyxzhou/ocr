import cv2
import os

from ocr_engine.ocr_full import OCREngine
from tools import Constants


def test_ocr_full():
    files = []
    for i in range(0, 20):
        files.append(f"train_{i}.jpg")
    paths = [os.path.join(
        os.getcwd(), Constants.DATA_FOLDER, file
    ) for file in files]
    images = [cv2.imread(path) for path in paths]

    mlp_full = OCREngine(model="mlp", weights_path=Constants.MLP_WEIGHTS_PATH_FULL, is_hex=False)
    mlp_hex = OCREngine(model="mlp", weights_path=Constants.MLP_WEIGHTS_PATH_HEX, is_hex=True)
    cnn_full = OCREngine(model="cnn", weights_path=Constants.CNN_WEIGHTS_PATH_FULL_H5)
    cnn_hex = OCREngine(model="cnn", weights_path=Constants.CNN_WEIGHTS_PATH_HEX_H5)
    mlp_full_output = ""
    mlp_hex_output = ""
    cnn_full_output = ""
    cnn_hex_output = ""
    for i, image in enumerate(images):
        prediction = mlp_full.get_text(image)
        mlp_full_output += f"Image {i}:\n{prediction}\n"
        prediction = mlp_hex.get_text(image)
        mlp_hex_output += f"Image {i}:\n{prediction}\n"
        prediction = cnn_full.get_text(image)
        cnn_full_output += f"Image {i}:\n{prediction}\n"
        prediction = cnn_hex.get_text(image)
        cnn_hex_output += f"Image {i}:\n{prediction}\n"

    with open(Constants.OCR_OUTPUT_FILE_FULL, "w") as f:
        f.write("EXPECTED TEXT:\n\n")
        f.write("NEWHAVEN DISPLAY\n4x20 CHARACTER OLEDS\nSLIM DESIGN ONLY 5MM\nOLED COLOR WHITE\n\n")
        f.write("ABCDEFGHIJKLM\nNOPQRSTUVWXYZ\n0123456789\n\n")
        f.write("ABCDEF\n01234\n56789\n\n")
        f.write("A B C D E F\n0 1 2 3 4\n5 6 7 8 9\n\n\n")
        f.write("TEST RESULTS:\n\n")
        f.write(f"MLP FULL WEIGHTS:\n{mlp_full_output}\n")
        f.write(f"MLP HEX WEIGHTS:\n{mlp_hex_output}\n")
        f.write(f"CNN FULL WEIGHTS:\n{cnn_full_output}\n")
        f.write(f"CNN HEX WEIGHTS:\n{cnn_hex_output}\n")

    # X_train, y_train = load_training_data()
    # X_train = scale_pixels(X_train)
    # X_train = X_train.reshape(X_train.shape[0], Constants.IMAGE_SIZE, Constants.IMAGE_SIZE, 1)
    # y_train = to_categorical(y_train)

    # img = X_train[0, :, :]
    # test_input = np.reshape(img, (1, Constants.IMAGE_SIZE, Constants.IMAGE_SIZE, 1))
    # prediction = model.predict(test_input)
    # prediction = Constants.CLASSES[np.argmax(prediction)]
    # actual = Constants.CLASSES[np.argmax(y_train[0])]
    # print(f"Prediction: {prediction}; actual: {actual}")
    # cv2.imshow("char", img)
    # cv2.waitKey(0)


if __name__ == "__main__":
    test_ocr_full()
