import pathlib
from keras.datasets import mnist
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
from discriminator import discriminator_model
from generator import generator_model
from combine_images import combine_images
from keras.models import Sequential

BATCH_SIZE = 128 
NUM_EPOCH = 20 
GENERATED_IMAGE_PATH = 'generated_images/'  # 生成画像の保存先
path = pathlib.Path(GENERATED_IMAGE_PATH)


def set_trainable(model, trainable):
    """
    Set the trainable

    Parameters
    -------------------
    trainable: bool

    model: class model
        keras model
    """
    model.trainable = trainable
    for layer in model.layers:
        layer.trainable = trainable


def train():
    """
    train dcgan
    """
    # X_train.shape=(60000, 28, 28)
    (X_train, _), (_, _) = mnist.load_data()
    X_train = (X_train.astype(np.float32) - 127.5)/127.5  # -1~1の範囲にする
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1],
                              X_train.shape[2], 1)  # X_train's dataformat=NHWC

    # 後に重みは固定させるためにdiscriminator単体を先に作る
    discriminator = discriminator_model()
    d_opt = Adam(lr=2e-4, beta_1=0.5) 
    # d_opt = Adam(lr=1e-5, beta_1=0.1)  # 論文ではDiscriminato, Generatorともにlr=2e-4, beta_1 = 0.5であったが，コード例より変更した
    discriminator.compile(loss='binary_crossentropy',
                          optimizer=d_opt, metrics=["accuracy"])

    # generator+discriminator （discriminator部分の重みは固定）
    set_trainable(discriminator, False)
    generator = generator_model()
    dcgan = Sequential([generator, discriminator])
    g_opt = Adam(lr=2e-4, beta_1=0.5)
    dcgan.compile(loss='binary_crossentropy',
                  optimizer=g_opt, metrics=["accuracy"])

    num_batches = X_train.shape[0] // BATCH_SIZE
    print('Number of batches:', num_batches)

    for epoch in range(NUM_EPOCH):

        for index in range(num_batches):
            noise = np.array([np.random.uniform(-1, 1, 100)
                              for _ in range(BATCH_SIZE)])  # noiseデータを作成
            # train_batshを作成
            image_batch = X_train[index*BATCH_SIZE:(index+1)*BATCH_SIZE]
            # generatorから偽の画像を作成
            generated_images = generator.predict(noise, verbose=0)

            # discriminatorを更新
            X = np.concatenate((image_batch, generated_images), axis=0)
            y = [1]*BATCH_SIZE + [0]*BATCH_SIZE
            d_loss, d_acc = discriminator.train_on_batch(X, y)

            # generatorを更新
            noise = np.array([np.random.uniform(-1, 1, 100)
                              for _ in range(BATCH_SIZE)])
            g_loss, g_acc = dcgan.train_on_batch(noise, [1]*BATCH_SIZE)
            print("epoch: {0:4d} index: {5:4d} g_loss: {1:.4f} d_loss:{2:.4f} " \
                  "g_acc: {3:.4f} d_acc: {4:.4f}".format(epoch + 1, g_loss,
                                                         d_loss, g_acc, d_acc, index))

        # epoch ごとに生成画像を出力
        noise = np.array([np.random.uniform(-1, 1, 100)
                              for _ in range(7*7)])  # noiseデータを作成
        generated_images = generator.predict(noise, verbose=0)
        image = combine_images(generated_images)
        image = image*127.5 + 127.5
        if not path.exists():
            path.mkdir()
        plt.imshow(image[:, :], cmap=plt.cm.gray)
        plt.savefig(path / "epoch{0}.jpg".format(epoch + 1))

        generator.save_weights('generator.h5')
        discriminator.save_weights('discriminator.h5')


if __name__ == "__main__":
    train()
