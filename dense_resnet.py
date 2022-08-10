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

def res_net_block(input_data, units):
    x = layers.BatchNormalization()(input_data)
    x = layers.Activation('relu')(x)
    x = Dense(units)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = Dense(units)(x)
    x = layers.Add()([x, input_data])
    return x

def create_network():
    #if not os.path.isdir(path):
    #    os.mkdir(path)
    input_ = Input(shape=(64))
    x = input_
    num_res_net_blocks = 10
    for i in range(num_res_net_blocks):
        x = res_net_block(x, 64)
    x = Dense(128, activation = "relu")(x)
    result = Dense(1, activation = "sigmoid")(x)
    model = Model(inputs=input_,outputs=result)

    model.compile(optimizer=optimizers.Adam(lr=1e-4, clipnorm=1),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])     
    return model
