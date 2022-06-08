import cv2
import matplotlib.pyplot as plt
import os

from PIL import ImageFont, ImageDraw, Image


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


def generate_alphabet_data(font_file_path, image_path, save_folder):
    image = Image.open(image_path)
    count = 0
    chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    # chars = "\'\"\\!@#$%^&*()-_=+,./<>?;:|~`[]{}"
    chars = [char for char in chars]
    font_size = 30
    color = "white"
    spacing = 15
    font = ImageFont.truetype(font_file_path, font_size)
    x_values = list(range(5, image.size[0]-font_size, spacing))
    y_values = list(range(5, image.size[1]-font_size, spacing))
    for char in chars:
        for x in x_values:
            for y in y_values:
                image = Image.open(image_path)
                location = (x, y)
                draw = ImageDraw.Draw(image)
                file_path = os.path.join(save_folder, f"symbol_{count}.jpg")
                draw.text(
                    location, char,
                    fill=color, font=font, spacing=spacing
                )  # (x, y) is the top left corner of the text to be drawn
                # image.show()

                image.save(file_path, "JPEG")
                count += 1
        # count = 0


if __name__ == "__main__":
    cwd = os.getcwd()
    pixel_font_path = os.path.join(
        cwd, "..", "data", "fonts", "pixel_operator", "PixelOperator8.ttf"
    )
    image_folder = os.path.join(
        cwd, "..", "data", "images"
    )
    test_image = os.path.join(image_folder, "black_rectangle.jpg")
    save_folder = os.path.join(
        cwd, "..", "data", "training"
    )
    # generate_ocr_data(pixel_font_path, test_image, save_folder)
    generate_alphabet_data(pixel_font_path, test_image, save_folder)
