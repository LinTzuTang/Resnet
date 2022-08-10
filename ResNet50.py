from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import LSTM,Add,Input,BatchNormalization
from tensorflow.keras.layers import LeakyReLU,Flatten
from tensorflow.keras import models, layers
from tensorflow.keras import optimizers
from tensorflow.compat.v1.keras.layers import CuDNNLSTM, Bidirectional

def create_network(preModel=ResNet50):
    pred_model = preModel(include_top=False, weights='imagenet',
                          input_shape=(48, 48, 3),
                          pooling='max', classifier_activation='softmax')
    # input_shape: It should have exactly 3 inputs channels, and width and height should be no smaller than 32. E.g. (200, 200, 3) would be one valid value.
    result = Dense(1, activation="softmax", name="output_layer")

    model = Model(pred_model.inputs, output_layer(result.output))

    model.compile(optimizer=optimizers.Adam(lr=1e-4, clipnorm=1),
                  loss='binary_crossentropy', metrics=['accuracy'])

    return model
