from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPool2D, Input


def cnnpred_2d_mine(seq_len=60, n_features=82, n_filters=(8,8,8), droprate=0.1):
    """
    2D-CNNpred My own model architecture
    This function creates a 2D-CNN model with 3 convolutional layers, 2 max pooling layers, a dropout
    layer, and a dense layer with a sigmoid activation function."
    
    The first layer is a convolutional layer with a kernel size of (1, 82) and 8 filters. The kernel
    size is (1, 82) because we want to convolve over the sequence length (1) and the number of features
    (82). The second layer is a convolutional layer with a kernel size of (3, 1) and 8 filters. The
    kernel size is (3, 1) because we want to convolve over the sequence length (3) and the number of
    features (1). The third layer is a max pooling layer with a pool size of (2, 1). The fourth layer is
    a convolutional layer with a kernel size of (3, 1) and 8 filters. The fifth layer is a max pool
    
    :param seq_len: the length of the sequence, defaults to 60 (optional)
    :param n_features: number of features in the input data, defaults to 82 (optional)
    :param n_filters: number of filters for each convolutional layer
    :param droprate: the dropout rate
    :return: A model object
    """
    model = Sequential([
        Input(shape=(seq_len, n_features, 1)),
        Conv2D(n_filters[0], kernel_size=(1, n_features), activation="relu"),
        Conv2D(n_filters[1], kernel_size=(3,1), activation="relu"),
        MaxPool2D(pool_size=(2,1)),
        Conv2D(n_filters[2], kernel_size=(3,1), activation="relu"),
        MaxPool2D(pool_size=(2,1)),
        Flatten(),
        Dropout(droprate),
        Dense(1, activation="sigmoid")
    ])
    
    return model