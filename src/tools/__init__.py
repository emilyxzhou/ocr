import cv2
import git
import os


def load_training_data(train_folder):
    image_paths = [
        f for f in os.listdir(train_folder)
    ]
    boxes, labels = _read_labels(train_folder, image_paths)
    return list(zip(image_paths, boxes, labels))


def _read_labels(train_folder, file_paths):
    # 231 images per character
    symbols = "\'\"\\!@#$%^&*()-_=+,./<>?;:|~`[]{}"
    labels = []
    boxes = []

    for f in file_paths:
        file_info = f.split("_")
        if len(file_info[0]) > 1:  # text is a symbol instead of a character
            symbol_index = int(file_info[1])
            labels.append(symbols[symbol_index])
        else:
            char_info = file_info[0].split(".")
            labels.append(char_info[0])
        boxes.append(
            _get_box(os.path.join(train_folder, f))
        )

    return boxes, labels


def _get_box(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    white_rows = []
    white_cols = []
    for row in range(gray.shape[0]):
        for col in range(gray.shape[1]):
            if gray[row][col] > 100:
                white_rows.append(row)
                white_cols.append(col)

    min_row = min(white_rows)
    min_col = min(white_cols)
    max_row = max(white_rows)
    max_col = max(white_cols)

    return [
        (min_row, min_col),
        (min_row, max_col),
        (max_row, max_col),
        (max_row, min_col)
    ]


def show_cropped(image_path):
    boxes = _get_box(image_path)
    min_row = boxes[0][0]
    min_col = boxes[0][1]
    max_row = boxes[2][0]
    max_col = boxes[2][1]
    image = cv2.imread(image_path)
    cv2.imshow("Frame", image[min_row:max_row, min_col:max_col])
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    cwd = os.getcwd()
    git_repo = git.Repo(cwd, search_parent_directories=True)
    git_root = git_repo.git.rev_parse("--show-toplevel")
    train_folder = os.path.join(
        git_root, "data", "training"
    )
    train_folder = train_folder.replace("/", "\\")

    dataset = load_training_data(train_folder)
    print(dataset[0])
