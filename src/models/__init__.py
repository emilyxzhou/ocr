from models.cnn import cnn
from sklearn.neural_network import MLPClassifier


def mlp(hidden_layer_sizes=(150, 100, 50)):
    return MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        max_iter=300,
        activation="relu",
        solver="adam",
        random_state=1
    )
