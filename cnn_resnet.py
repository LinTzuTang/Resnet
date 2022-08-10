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


'''
def res_net_block(input_data, filters, conv_size):
    x = layers.BatchNormalization()(input_data)
    x = LeakyReLU(alpha=0.1)(x)
    x = Conv1D(filters, conv_size, padding="same")(x) 
    x = layers.BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(input_data)
    x = Conv1D(filters, conv_size, padding="same")(x)
    x = layers.Add()([x, input_data])
    return x
'''
def res_net_block(input_data, filters, conv_size):
    x = Conv1D(filters, conv_size, activation = 'relu', padding="same")(input_data)
    x = layers.BatchNormalization()(x)
    x = Conv1D(filters, conv_size, activation = 'relu', padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([x, input_data])
    x = layers.Activation('relu')(x)
    return x


def create_network():
    #if not os.path.isdir(path):
    #    os.mkdir(path)
    input_ = Input(shape=(100,1))
    x = input_
    num_res_net_blocks = 10
    for i in range(num_res_net_blocks):
        x = res_net_block(x, 64, 16)
    f = Flatten()(x)
    result = Dense(1, activation = "sigmoid")(f)
    model = Model(inputs=input_,outputs=result)
    model.compile(optimizer=optimizers.Adam(lr=1e-4, clipnorm=1),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model