import cv2
import os
import numpy as np

from ocr_engine import OCREngineBase
from PIL import Image, ImageDraw, ImageFont, ImageOps
from tools import Constants


def generate_ocr_data(image_path, save_folder, fonts, texts=None):
    """
    Generates JPG test images for the OCR engine.
    :param image_path: Path to base image to overlay text.
    :param save_folder: Folder to save training images to.
    :param fonts: List of paths to font files in .ttf format.
    :param texts: List of strings to overlay on the base image.
    :return: None
    """
    count = 0
    spacing = 10
    font_sizes = [22, 24, 26, 28]
    if texts is None:
        texts = [
            "NEWHAVEN DISPLAY\n4x20 CHARACTER OLEDS\nSLIM DESIGN ONLY 5MM\nOLED COLOR WHITE",
            "Newhaven Display\n4x20 Character OLEDs\nSlim Design Only 5MM\nOLED Color White",
            "ABCDEFGHI\nJKLMNOPQR\nSTUVWXYZ\n0123456789",
            "abcdefghi\njklmnopqr\nstuvwxyz\n0123456789",
            "ABCDEF\n01234\n56789",
            "abcdef\n01234\n56789",
            "A B C D E F\n0 1 2 3 4\n5 6 7 8 9",
            "a b c d e f\n0 1 2 3 4\n5 6 7 8 9"
        ]
    fill = ["white"]
    for font_path in fonts:
        for font_size in font_sizes:
            for color in fill:
                for text in texts:
                    image = Image.open(image_path)
                    draw = ImageDraw.Draw(image)
                    x = 15
                    y = 15
                    font = ImageFont.truetype(font_path, font_size)
                    draw.text(
                        (x, y), text,
                        fill=color, font=font, spacing=spacing
                    )  # (x, y) is the top left corner of the text to be drawn

                    file_path = os.path.join(save_folder, f"train_{count}.jpg")
                    image.save(file_path, "JPEG")
                    count += 1


def generate_training_data(
        image_path,
        save_folder,
        fonts
):
    """
    Generates 28x28x3 JPG images of individual characters.
    Uses a variety of fonts and rotation angles.
    :param image_path: Path to base image to overlay text.
    :param save_folder: Folder to save training images to.
    :param fonts: List of paths to font files in .ttf format.
    :return: None
    """
    image = Image.open(image_path)
    width, height = image.size
    count = 0
    chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
    chars = [char for char in chars]
    font_sizes = list(range(16, 22))
    angles = [-10, -8, -5, 0, 5, 8, 10]
    color = "white"
    for char in chars:
        for font_path in fonts:
            for font_size in font_sizes:
                for angle in angles:
                    font = ImageFont.truetype(font_path, font_size)
                    image = Image.open(image_path)
                    draw = ImageDraw.Draw(image)
                    if not char.isdigit():
                        if char == char.lower():
                            prefix = f"{char.lower()}_lower"
                        else:
                            prefix = f"{char.lower()}_upper"
                    else:
                        prefix = char
                    file_path = os.path.join(save_folder, f"{prefix}_{count}.jpg")
                    with open(file_path, "w") as f:
                        draw.text(
                            (width // 2, height // 2), char,
                            fill=color, font=font, anchor="mm"
                        )
                        image = ImageOps.grayscale(image)
                        image = image.rotate(angle)

                        image.save(f, "JPEG")
                    count += 1

                    # Save dilated version of image
                    image = np.asarray(image)
                    file_path = os.path.join(save_folder, f"{prefix}_{count}.jpg")
                    kernel = np.ones((3, 3), np.uint8)
                    image = cv2.dilate(image, kernel, iterations=1)
                    cv2.imwrite(file_path, image)
                    count += 1
        count = 0


def generate_cropped_training_data(
        image_path,
        save_folder,
        fonts,
        texts=None
):
    """
    Generates 28x28x3 JPG images of individual characters by segmenting individual characters
        from a block of text.
    Uses the OCREngine's character segmentation algorithm to optimize the OCR model's performance.
    :param image_path: Path to base image to overlay text.
    :param save_folder: Folder to save training images to.
    :param fonts: List of paths to font files in .ttf format.
    :param texts: List of strings of text to draw on the base image.
    :return: None
    """
    ocr_engine = OCREngineBase()
    spacing = 10
    font_sizes = [24, 26, 28]
    if texts is None:
        texts = [
            "ABCDEFGHI\nJKLMNOPQR\nSTUVWXYZ\n0123456789",
            "abcdefghi\njklmnopqr\nstuvwxyz\n0123456789"
        ]
    fill = ["white"]
    index = 0
    for font_path in fonts:
        for font_size in font_sizes:
            for color in fill:
                for text in texts:
                    chars = list(t for t in text if t != "\n" and t != " ")
                    image = Image.open(image_path)
                    draw = ImageDraw.Draw(image)
                    x = 15
                    y = 15
                    font = ImageFont.truetype(font_path, font_size)
                    draw.text(
                        (x, y), text,
                        fill=color, font=font, spacing=spacing
                    )  # (x, y) is the top left corner of the text to be drawn

                    image = np.asarray(image)
                    lines = ocr_engine.segment_characters(image)

                    count = 0
                    # for line in lines:
                    #     for char in line:
                    #         cv2.imshow("char", char)
                    #         cv2.waitKey(0)

                    for line in lines:
                        for char in line:
                            if not chars[count].isdigit():
                                if chars[count] == chars[count].lower():
                                    prefix = f"{chars[count].lower()}_lower"
                                else:
                                    prefix = f"{chars[count].lower()}_upper"
                            else:
                                prefix = chars[count]
                            char = ocr_engine._resize_model_input(char)
                            file_path = os.path.join(save_folder, f"{prefix}_cropped_{index}.jpg")
                            final_image = Image.fromarray(char)
                            with open(file_path, "w") as f:
                                final_image.save(f, "JPEG")
                            index += 1

                            file_path = os.path.join(save_folder, f"{prefix}_cropped_{index}.jpg")
                            kernel = np.ones((3, 3), np.uint8)
                            char = cv2.dilate(char, kernel, iterations=1)
                            cv2.imshow("dilated", char)
                            cv2.waitKey(0)
                            cv2.imwrite(file_path, char)
                            count += 1

                    index += 1


if __name__ == "__main__":
    # Generate blocks of text to test full OCR system ---------------
    pixel_font_paths = [
        os.path.join(Constants.DATA_FOLDER, "fonts", "pixel_operator", "PixelOperator.ttf"),
        os.path.join(Constants.DATA_FOLDER, "fonts", "pixel_operator", "PixelOperatorMono.ttf"),
        os.path.join(Constants.DATA_FOLDER, "fonts", "pixel_operator", "PixelOperator8.ttf"),
        os.path.join(Constants.DATA_FOLDER, "fonts", "pixel_operator", "PixelOperatorMono8.ttf")
    ]
    # Base image for training data is a 28x28 black square
    base_image = os.path.join(Constants.DATA_FOLDER, "images", "base_image.jpg")
    black_rectangle = os.path.join(Constants.DATA_FOLDER, "images", "black_rectangle.jpg")

    # Generate test data for OCR engine
    generate_ocr_data(black_rectangle, Constants.OCR_TEST_FOLDER, pixel_font_paths)

    # Generate training data for OCR models
    generate_training_data(base_image, Constants.TRAIN_FOLDER, pixel_font_paths)
    generate_cropped_training_data(black_rectangle, Constants.TRAIN_FOLDER, pixel_font_paths[0:2])
