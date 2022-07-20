from sklearn.neural_network import MLPClassifier


def mlp(hidden_layer_sizes=(150, 100, 50)):
    """
    :param hidden_layer_sizes: A tuple of hidden layer sizes to pass to the MLPClassifier.
    :return: A sklearn MLPClassifier with default parameters.
    """
    return MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        max_iter=300,
        activation="relu",
        solver="adam",
        random_state=1
    )
