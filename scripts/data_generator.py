import os

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
    font_sizes = [28]
    if texts is None:
        texts = [
            "NEWHAVEN DISPLAY\n4x20 CHARACTER OLEDS\nSLIM DESIGN ONLY 5MM\nOLED COLOR WHITE",
            "ABCDEFGHIJKLM\nNOPQRSTUVWXYZ\n0123456789",
            "ABCDEF\n01234\n56789",
            "A B C D E F\n0 1 2 3 4\n5 6 7 8 9"
        ]
    fill = ["white"]
    for text in texts:
        for font_path in fonts:
            for font_size in font_sizes:
                for color in fill:
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
    chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    chars = [char for char in chars]
    font_sizes = list(range(16, 22))
    angles = [-15, -10, -5, 0, 5, 10, 15]
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

                        image.save(f, "JPEG")
                    count += 1
        count = 0


if __name__ == "__main__":
    # Generate blocks of text to test full OCR system ---------------
    pixel_font_paths = [
        os.path.join(Constants.DATA_FOLDER, "fonts", "pixel_operator", "PixelOperator.ttf"),
        os.path.join(Constants.DATA_FOLDER, "fonts", "pixel_operator", "PixelOperator8.ttf"),
        os.path.join(Constants.DATA_FOLDER, "fonts", "pixel_operator", "PixelOperatorMono.ttf"),
        os.path.join(Constants.DATA_FOLDER, "fonts", "pixel_operator", "PixelOperatorMono8.ttf"),
        os.path.join(Constants.DATA_FOLDER, "fonts", "pixel_operator", "PixelOperatorSC.ttf"),
    ]
    # Base image for training data is a 28x28 black square
    base_image = os.path.join(Constants.DATA_FOLDER, "images", "base_image.jpg")
    black_rectangle = os.path.join(Constants.DATA_FOLDER, "images", "black_rectangle.jpg")

    # Generate training data for OCR models
    generate_training_data(base_image, Constants.TRAIN_FOLDER, pixel_font_paths)
    # Generate test data for OCR engine
    generate_ocr_data(black_rectangle, Constants.OCR_TEST_FOLDER, pixel_font_paths)
