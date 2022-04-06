from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Reshape, Flatten, BatchNormalization, Dense, Dropout

class Model:
    def __init__(self):
        self.channel = 1
        self.input_dim = 100
        self.input_shape = (28,28,1)

    def generator(self):
        generator_model = Sequential()
        generator_model.add(Dense(7*7*256,activation="relu",input_dim=self.input_dim))
        generator_model.add(BatchNormalization())
        generator_model.add(Reshape((7,7,256)))
        generator_model.add(Conv2DTranspose(256,(5,5),strides=(2,2),padding="same",activation="relu"))
        generator_model.add(BatchNormalization())
        generator_model.add(Conv2DTranspose(128,(5,5),strides=(1,1),padding="same",activation="relu"))
        generator_model.add(BatchNormalization())
        generator_model.add(Conv2DTranspose(64,(5,5),strides=(1,1),padding="same",activation="relu"))
        generator_model.add(BatchNormalization())
        generator_model.add(Conv2DTranspose(self.channel,(5,5),strides=(2,2),padding="same",activation="tanh"))

        return generator_model

    def discriminator(self):
        discriminator_model = Sequential()
        discriminator_model.add(Conv2D(64,(5,5),strides=(2,2),activation="relu",padding="same",input_shape=self.input_shape))
        discriminator_model.add(Dropout(0.2))
        discriminator_model.add(Conv2D(128,(5,5),strides=(2,2),activation="relu",padding="same"))
        discriminator_model.add(Dropout(0.2))
        discriminator_model.add(Conv2D(256,(5,5),strides=(2,2),activation="relu",padding="same"))
        discriminator_model.add(Dropout(0.2))
        discriminator_model.add(Flatten())
        discriminator_model.add(Dense(1,activation="sigmoid"))

        return discriminator_model
