import tensorflow.keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import LSTM,Add,Input,BatchNormalization
from tensorflow.keras.layers import LeakyReLU,Flatten
from tensorflow.keras import models, layers
from tensorflow.keras import optimizers
from tensorflow.compat.v1.keras.layers import CuDNNLSTM, Bidirectional


def create_network():
    input_ = Input(shape=(100))
    x = input_
    #cnn = Conv1D(64 ,8,activation = 'relu', padding="same")(input_)
    #norm = BatchNormalization()(cnn)
    #cnn2 = Conv1D(32 ,8,activation = 'relu', padding="same")(norm1)
    #norm2 = BatchNormalization()(cnn2)
    #cnn3 = Conv1D(16 ,8,activation = 'relu', padding="same")(norm2)
    #norm3 = BatchNormalization()(cnn3)
    #f = Flatten()(norm3)
    x = Bidirectional(CuDNNLSTM(units=100,return_sequences=True))(x)
    x = Bidirectional(CuDNNLSTM(units=100,return_sequences=False))(x)
    result = Dense(1, activation = "sigmoid")(x)
    model = Model(inputs=input_,outputs=result)
    model.compile(optimizer=optimizers.Adam(lr=1e-4, clipnorm=1),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model