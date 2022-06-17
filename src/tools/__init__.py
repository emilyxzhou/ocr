import cv2
import git
import json
import os
import random

cwd = os.getcwd()
_git_repo = git.Repo(cwd, search_parent_directories=True)
_git_root = _git_repo.git.rev_parse("--show-toplevel")
_data_folder = os.path.join(
    _git_root, "data"
)
DATA_FOLDER = _data_folder.replace("/", "\\")
TRAIN_FOLDER = os.path.join(DATA_FOLDER, "training")
METADATA_FOLDER = os.path.join(DATA_FOLDER, "metadata")


class MultiDimensionalArrayEncoder(json.JSONEncoder):
    def encode(self, obj):
        def hint_tuples(item):
            if isinstance(item, tuple):
                return {'__tuple__': True, 'items': item}
            if isinstance(item, list):
                return [hint_tuples(e) for e in item]
            if isinstance(item, dict):
                return {key: hint_tuples(value) for key, value in item.items()}
            else:
                return item

        return super(MultiDimensionalArrayEncoder, self).encode(hint_tuples(obj))


def hinted_tuple_hook(obj):
    if '__tuple__' in obj:
        return tuple(obj['items'])
    else:
        return obj


def load_from_json(num_samples=None):
    file_name = "pixel_operator_dataset.json"
    file_path = os.path.join(METADATA_FOLDER, file_name)
    with open(file_path) as f:
        json_string = json.load(f)
    dataset = json.loads(json_string, object_hook=hinted_tuple_hook)
    if num_samples is None:
        num_samples = len(dataset)
    return random.sample(dataset, num_samples)


def load_training_data(num_files=None):
    # TODO: add train/test split option
    image_paths = [
        f for f in os.listdir(TRAIN_FOLDER)
    ]

    if num_files is None:
        print("Loading all image files ...")
        num_files = len(image_paths)

    indices = random.sample(range(0, len(image_paths)), num_files)
    image_paths = [image_paths[i] for i in indices]
    boxes, labels = _read_labels(TRAIN_FOLDER, image_paths)
    full_image_paths = [
        os.path.join(TRAIN_FOLDER, f) for f in image_paths
    ]
    return list(zip(full_image_paths, boxes, labels))


def _read_labels(train_folder, image_paths):
    print("Reading labels ...")
    count = 0
    total = len(image_paths)
    # 231 images per character
    symbols = "\'\"\\!@#$%^&*()-_=+,./<>?;:|~`[]{}"
    symbols = [sym for sym in symbols]
    labels = []
    boxes = []

    for f in image_paths:
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
        print(f"Image {count}/{total}")
        count += 1

    return boxes, labels


def _get_box(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    white_rows = []
    white_cols = []
    for row in range(gray.shape[0]):
        for col in range(gray.shape[1]):
            if gray[row][col] > 10:
                white_rows.append(row)
                white_cols.append(col)

    min_row = min(white_rows) - 5
    min_col = min(white_cols) - 5
    max_row = max(white_rows) + 5
    max_col = max(white_cols) + 5

    return [
        [min_row, min_col],
        [min_row, max_col],
        [max_row, max_col],
        [max_row, min_col]
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

