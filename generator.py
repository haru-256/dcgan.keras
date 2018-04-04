from keras.models import Model
from keras.layers import Dense, Activation, Reshape, Input
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D, Conv2DTranspose
from keras.layers.advanced_activations import LeakyReLU

def generator_model():
    """
    build Generator model

    Returns
    -------------
    model: class
        generator model

    """

    inputs = Input(shape=(100,))
    outputs = Dense(units=1024)(inputs)
    outputs = BatchNormalization()(outputs)
    outputs = LeakyReLU(alpha=0.2)(outputs)
    outputs = Dense(units=128*7*7)(outputs)
    outputs = BatchNormalization()(outputs)
    outputs = LeakyReLU(alpha=0.2)(outputs)
    outputs = Reshape(target_shape=(7, 7, 128))(outputs)
    # outputs = UpSampling2D(size=(2, 2))(outputs)
    outputs = Conv2DTranspose(filters=64,
                              kernel_size=(5, 5),
                              strides=(2, 2),
                              padding="same")(outputs)
    outputs = BatchNormalization()(outputs)
    outputs = LeakyReLU(alpha=0.2)(outputs)
    # outputs = UpSampling2D(size=(2, 2))(outputs)
    outputs = Conv2DTranspose(filters=1,
                              kernel_size=(5, 5),
                              strides=(2,2),
                              padding="same")(outputs)
    outputs = Activation("tanh")(outputs)

    model = Model(inputs=inputs, outputs=outputs)

    return model


if __name__ == "__main__":
    model = generator_model()
    print(model.summary())
