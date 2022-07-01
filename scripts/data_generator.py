import cv2
import git
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import random

from PIL import Image, ImageDraw, ImageFont, ImageOps
from tools import get_cropped_size, load_training_data, Constants


def generate_ocr_data(font_file_path, image_path, save_folder):
    count = 0
    spacing = 10
    font_sizes = [28]
    texts = [
        "((NEWHAVEN DISPLAY))\n4x20 CHARACTER OLEDS\nSLIM DESIGN ONLY 5MM\nOLED COLOR: <WHITE>",
        "ABCDEFGHIJKLM\nNOPQRSTUVWXYZ\n0123456789"
    ]
    # fill: font
    fill = ["white"]
    for font_size in font_sizes:
        for color in fill:
            for text in texts:
                image = Image.open(image_path)
                draw = ImageDraw.Draw(image)
                x = 15
                y = 15
                font = ImageFont.truetype(font_file_path, font_size)
                draw.text(
                    (x, y), text,
                    fill=color, font=font, spacing=spacing
                )  # (x, y) is the top left corner of the text to be drawn
                image.show()

                file_path = os.path.join(save_folder, f"train_{count}.jpg")
                image.save(file_path, "JPEG")
                count += 1


def generate_training_data(image_path, save_folder, fonts):
    image = Image.open(image_path)
    width, height = image.size
    count = 0
    chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    chars = [char for char in chars]
    font_sizes = [18, 19, 20, 21, 22]
    angles = [-30, -20, -10, 0, 10, 20, 30]
    color = "white"
    for char in chars:
        for font_path in fonts:
            for font_size in font_sizes:
                for angle in angles:
                    font = ImageFont.truetype(font_path, font_size)
                    image = Image.open(image_path)
                    draw = ImageDraw.Draw(image)
                    file_path = os.path.join(save_folder, f"{char}_{count}.jpg")
                    with open(file_path, "w") as f:
                        draw.text(
                            (width // 2, height // 2), char,
                            fill=color, font=font, anchor="mm"
                        )
                        image = ImageOps.grayscale(image)
                        image = image.rotate(angle)
                        # image.show()

                        image.save(f, "JPEG")
                    count += 1
        count = 0


if __name__ == "__main__":
    cwd = os.getcwd()
    _git_repo = git.Repo(cwd, search_parent_directories=True)
    _git_root = _git_repo.git.rev_parse("--show-toplevel")
    _data_folder = os.path.join(
        _git_root, "data"
    )
    pixel_font_paths = [
        os.path.join(_data_folder, "fonts", "pixel_operator", "PixelOperator.ttf"),
        os.path.join(_data_folder, "fonts", "pixel_operator", "PixelOperator8.ttf"),
        os.path.join(_data_folder, "fonts", "pixel_operator", "PixelOperatorMono.ttf"),
        os.path.join(_data_folder, "fonts", "pixel_operator", "PixelOperatorMono8.ttf"),
        os.path.join(_data_folder, "fonts", "pixel_operator", "PixelOperatorSC.ttf"),
    ]
    base_image = os.path.join(_data_folder, "images", "base_image.jpg")
    # generate_ocr_data(pixel_font_path, base_image, _data_folder)
    generate_training_data(base_image, Constants.TRAIN_FOLDER, pixel_font_paths)
    # generate_validation_data(pixel_font_path, base_image, VALIDATION_FOLDER)
    # save_to_json()
    # dataset = load_from_json()
    # for data in dataset:
    #     print(get_cropped_size(data[0]))
