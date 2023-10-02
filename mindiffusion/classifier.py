def LeNet5v1(input_shape=(32, 32, 1), classes=10):
    """
    Implementation of a modified LeNet-5.
    Modified Architecture -- ConvNet --> ConvNet --> Pool --> (Dropout) --> ConvNet --> Pool --> (Dropout) --> (Flatten) --> FullyConnected --> FullyConnected --> Softmax

    Arguments:
    input_shape -- shape of the images of the dataset
    classes -- integer, number of classes

    Returns:
    model -- a Model() instance in Keras
    """

    model = Sequential(
        [
            # Layer 1
            Conv2D(
                filters=6,
                kernel_size=5,
                strides=1,
                activation="relu",
                input_shape=(32, 32, 1),
                name="convolution_1",
            ),
            # Layer 2
            Conv2D(
                filters=6,
                kernel_size=5,
                strides=1,
                activation="relu",
                name="convolution_2",
            ),
            # -------------------------------- #
            MaxPooling2D(pool_size=2, strides=2, name="max_pool_1"),
            Dropout(0.25, name="dropout_1"),
            # -------------------------------- #
            # Layer 3
            Conv2D(
                filters=16,
                kernel_size=5,
                strides=1,
                activation="relu",
                kernel_regularizer=l2(0.01),
                name="convolution_3",
            ),
            # -------------------------------- #
            MaxPooling2D(pool_size=2, strides=2, name="max_pool_2"),
            Dropout(0.25, name="dropout_2"),
            Flatten(name="flatten"),
            # -------------------------------- #
            # Layer 4
            Dense(units=120, activation="relu", name="fully_connected_1"),
            # Layer 5
            Dense(units=84, activation="relu", name="fully_connected_2"),
            # Output
            Dense(units=10, activation="softmax", name="output"),
        ]
    )

    model._name = "LeNet5v1"

    return model
