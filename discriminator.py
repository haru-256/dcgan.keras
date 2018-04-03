from keras.models import Model
from keras.layers import Dense, Activation, Input, Flatten, Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.advanced_activations import LeakyReLU


def discriminator_model():
    """
    build discriminator model

    Returns
    -------------------
    model: class
        discriminator model
    """
    inputs = Input(shape=(28, 28, 1))
    # Eliminate pooling by strides=(2, 2)
    outputs = Conv2D(filters=64,
                     kernel_size=(5, 5),
                     padding="same",
                     strides=(2, 2))(inputs)
    outputs = LeakyReLU(alpha=0.2)(outputs)
    outputs = Flatten()(outputs)
    outputs = Dense(units=256)(outputs)
    outputs = LeakyReLU(alpha=0.2)(outputs)
    outputs = Dropout(rate=0.5)(outputs)
    outputs = Dense(1)(outputs)
    outputs = Activation("sigmoid")(outputs)

    model = Model(inputs=inputs, outputs=outputs)

    return model


if __name__ == "__main__":
    model = discriminator_model()
    print(model.summary())
