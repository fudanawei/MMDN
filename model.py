from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Dropout, Flatten, Dense
from tensorflow.keras.callbacks import EarlyStopping, CSVLogger, TensorBoard

def createCNN(outsize, input_shape):
    model_input = Input(shape=input_shape)

    x = Conv2D(8, kernel_size=(3, 3), activation='relu')(model_input)
    x = BatchNormalization(axis=-1)(x)
    x = Dropout(0.4)(x) #

    x = Conv2D(8, kernel_size=(3, 3), strides=2, activation='relu')(x)
    x = BatchNormalization(axis=-1)(x)
    x = Dropout(0.4)(x)

    x = Flatten()(x)
    x = Dense(outsize, activation='sigmoid')(x)

    cnn = Model(inputs=model_input, outputs=x)

    return cnn


def get_callbacks():
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=5e-6, patience=10, verbose=2, mode='min') # min_delta=5e-6
    csv_logger = CSVLogger('./log.csv', append=True)
    tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=False)
    return [tensorboard, csv_logger, early_stopping]
    