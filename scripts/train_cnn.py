from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from sklearn.model_selection import train_test_split

from models.cnn import cnn
from tools import load_training_data, load_hex_training_data, scale_pixels, Constants

BATCH_SIZE = 32
EPOCHS = 500


def train_hex():
    """
    Trains the CNN on the hex characters (A-F, 0-9) found in ocr/data/training/.
    """
    # Load training data
    X_train, y_train = load_hex_training_data()
    X_train = scale_pixels(X_train)
    X_train = X_train.reshape(X_train.shape[0], Constants.IMAGE_SIZE, Constants.IMAGE_SIZE, 1)
    y_train = to_categorical(y_train)
    X_train, X_test, y_train, y_test = train_test_split(
        X_train, y_train, test_size=0.20, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5)

    # Create model
    model = cnn(is_hex=True)

    # Set up checkpoints
    model_path = os.path.join(Constants.CHECKPOINTS_FOLDER, "weights-hex-{epoch:02d}-{val_accuracy:.2f}.hdf5")
    checkpoint = ModelCheckpoint(model_path, monitor="val_accuracy", verbose=1, save_best_only=True, mode="max")
    callbacks_list = [checkpoint]

    # Start training
    model.fit(
        x=X_train, y=y_train,
        steps_per_epoch=len(X_train) // BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=callbacks_list,
        validation_data=(X_val, y_val)
    )

    model.save_weights(Constants.CNN_WEIGHTS_PATH_HEX_H5)


def train_full():
    """
    Trains the CNN on the full dataset found in ocr/data/training/.
    """
    # Load training data
    X_train, y_train = load_training_data()
    X_train = scale_pixels(X_train)
    X_train = X_train.reshape(X_train.shape[0], Constants.IMAGE_SIZE, Constants.IMAGE_SIZE, 1)
    y_train = to_categorical(y_train)
    X_train, X_test, y_train, y_test = train_test_split(
        X_train, y_train, test_size=0.20, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5)

    # print(y_val[0])
    # cv2.imshow("frame", X_val[0])
    # cv2.waitKey(0)

    # create model
    model = cnn(is_hex=False)
    # set up checkpoints
    model_path = os.path.join(Constants.CHECKPOINTS_FOLDER, "weights-{epoch:02d}-{val_accuracy:.2f}.hdf5")
    checkpoint = ModelCheckpoint(model_path, monitor="val_accuracy", verbose=1, save_best_only=True, mode="max")
    callbacks_list = [checkpoint]

    model.fit(
        x=X_train, y=y_train,
        steps_per_epoch=len(X_train) // BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=callbacks_list,
        validation_data=(X_val, y_val)
    )

    model.save_weights(Constants.CNN_WEIGHTS_PATH_FULL_H5)


if __name__ == "__main__":
    print("Training on full dataset ...")
    train_full()
    print("Training on hex dataset ...")
    train_hex()
