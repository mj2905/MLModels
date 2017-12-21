"""
This file contains the method used to build the Keras models, see configuration.py for parameters
"""
from keras import layers, models, callbacks


def model_cnn1(we_layer, config):
    """
    First model:
        - one embeddings layer
        - multiple (parallel and different kernels size) convolutionnal layers
        - One fully connected layer
    :param we_layer: the embeddings layer to use
    :param config: the parameters of the layer
    :return: a compiled Keras model
    """
    model_conf = config.keras_model
    in_layer = layers.Input(shape=(config.tw_length,),
                            dtype='int32',
                            name='input_layer')
    we = we_layer(in_layer)
    submodels = []
    for ks in model_conf.cnn_kernels:
        conv = layers.Conv1D(model_conf.cnn_filters, ks, padding='valid')(we)
        conv = layers.LeakyReLU()(conv)
        submodels.append(layers.GlobalMaxPooling1D()(conv))
    x = layers.concatenate(submodels)
    x = layers.Dropout(model_conf.dropout)(x)
    x = layers.Dense(model_conf.dense_size)(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(model_conf.dropout)(x)
    out_layer = layers.Dense(units=2, activation='softmax')(x)

    model = models.Model(in_layer, out_layer)
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['acc'])
    model.summary()
    return model


def model_cnn2(we_layer, config):
    """
    Second model:
        - one embeddings layer
        - multiple (parallel and different kernels size) convolutionnal layers
        - One fully connected layer
    :param we_layer: the embeddings layer to use
    :param config: the parameters of the layer
    :return: a compiled Keras model
    """
    model_conf = config.keras_model
    # create the input layer
    in_layer = layers.Input(shape=(config.tw_length,),
                            dtype='int32',
                            name='input_layer')
    we = we_layer(in_layer)
    we = layers.Dropout(model_conf.dropout)(we)

    # add the convolutional layers
    submodels = []
    for ks in model_conf.cnn_kernels:
        conv = layers.Conv1D(model_conf.cnn_filters, ks, padding='valid')(we)
        conv = layers.LeakyReLU()(conv)
        submodels.append(layers.GlobalMaxPooling1D()(conv)) # max pooling

    # concatenate all the max pooling output in one layer
    x = layers.concatenate(submodels)
    x = layers.Dropout(model_conf.dropout)(x)

    # add the (multiple) fully connected layers
    for dense_size in model_conf.dense_sizes:
        x = layers.Dense(dense_size)(x)
        x = layers.LeakyReLU()(x)
        x = layers.Dropout(model_conf.dropout)(x)

    # create the output layer
    out_layer = layers.Dense(units=2, activation='softmax')(x)

    # compile the model
    model = models.Model(in_layer, out_layer)
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['acc'])

    # print the description of the model
    model.summary()
    return model

def model_cnn3(we_layer, config):
    """
    Keras model that reproduce the dying relu phenomena
    """
    model_conf = config.keras_model
    in_layer = layers.Input(shape=(config.tw_length,),
                            dtype='int32',
                            name='input_layer')
    we = we_layer(in_layer)
    submodels = []
    for ks in model_conf.cnn_kernels:
        conv = layers.Conv1D(model_conf.cnn_filters, ks, padding='valid', activation='relu')(we)
        submodels.append(layers.GlobalMaxPooling1D()(conv))
    x = layers.concatenate(submodels)
    x = layers.Dropout(model_conf.dropout)(x)
    x = layers.Dense(model_conf.dense_size, activation='relu')(x)
    x = layers.Dropout(model_conf.dropout)(x)
    out_layer = layers.Dense(units=2, activation='softmax')(x)

    model = models.Model(in_layer, out_layer)
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['acc'])
    model.summary()
    return model
