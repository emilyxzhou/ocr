import cv2
import os

from ocr_engine.ocr_mlp import OCRMLP
from tools import Constants


def test_ocr_mlp():
    files = []
    for i in range(0, 20):
        files.append(f"train_{i}.jpg")
    paths = [os.path.join(
        os.getcwd(), Constants.DATA_FOLDER, file
    ) for file in files]
    images = [cv2.imread(path) for path in paths]

    mlp_full = OCRMLP(weights_path=Constants.MLP_WEIGHTS_PATH_FULL)
    mlp_hex = OCRMLP(weights_path=Constants.MLP_WEIGHTS_PATH_HEX)
    mlp_full_output = ""
    mlp_hex_output = ""
    for i, image in enumerate(images):
        prediction = mlp_full.get_text(image)
        mlp_full_output += f"Image {i}:\n{prediction}\n"
        prediction = mlp_hex.get_text(image)
        mlp_hex_output += f"Image {i}:\n{prediction}\n"

    with open(Constants.OCR_OUTPUT_FILE, "w") as f:
        f.write("EXPECTED TEXT:\n\n")
        f.write("NEWHAVEN DISPLAY\n4x20 CHARACTER OLEDS\nSLIM DESIGN ONLY 5MM\nOLED COLOR WHITE\n\n")
        f.write("ABCDEFGHIJKLM\nNOPQRSTUVWXYZ\n0123456789\n\n")
        f.write("ABCDEF\n01234\n56789\n\n")
        f.write("A B C D E F\n0 1 2 3 4\n5 6 7 8 9\n\n\n")
        f.write("TEST RESULTS:\n\n")
        f.write(f"MLP FULL WEIGHTS:\n{mlp_full_output}\n")
        f.write(f"MLP HEX WEIGHTS:\n{mlp_hex_output}\n")


if __name__ == "__main__":
    test_ocr_mlp()
