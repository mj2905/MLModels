"""
Script that produce the same prediction file that gave us our best score on Kaggle.
Type: "python run.py" in terminal to create the submission file.
Type "python run.py model_xyz" to run an another model

See "configuration.py" for models parameters details.
"""
import os
import sys
import numpy as np
np.random.seed(1)

from keras import models, callbacks

from utils.utils import save_pkl, plot_keras_accuracy_graph, create_submission_file_from_keras_model
from utils.tweets_embeddings_builder import TweetsEmbeddingsBuilder

from configuration import config as _c  # load the configuration.py file


def main():
    # configuration to use, see file configuration.py
    arguments = sys.argv[1:]
    config = _c[arguments[0]] if len(arguments) > 0 else _c[_c.default_model]  # get config. parameters

    print('Script START')
    print(config.welcome_message)

    model = None

    if config.training:
        """
        To enable the training phase, set the flag 'training': True in the config file
        Be aware that this step take several hours without a decent GPU
        """
        if not os.path.exists(config.base_folder):
            # create the directory where to write results
            os.makedirs(config.base_folder)

        print('Training enabled! Go take a coffee or, better, come back tomorrow ' +
              '(unless you have a good CPU and GPU)!')

        print('Preparing data...')
        fb = TweetsEmbeddingsBuilder(preprocess_func=config.preprocess_func)
        fb.add_tweet_file(config.datasets_folder, config.tw_pos_file, 1)
        fb.add_tweet_file(config.datasets_folder, config.tw_neg_file, 0)
        fb.add_tweet_file(config.datasets_folder, config.tw_test_file, None)

        fb.build_word_index(max_seq_length=config.tw_length,
                            dict_max_words=config.dictionary_size,
                            post_padding=config.post_padding)

        _x_train, _y_train = fb.get_training_data(shuffle=True, seed=config.seed)
        x_test, ids_test = fb.get_test_data(shuffle=False)
        save_pkl((x_test, ids_test), config.test_data_pkl)

        sub_sample = config.datasets_subsample
        num_samples = int(len(_y_train)*sub_sample)
        num_validation_samples = int(config.val_data_ratio * num_samples)

        x_train = _x_train[:num_samples][:-num_validation_samples]
        y_train = _y_train[:num_samples][:-num_validation_samples]
        x_val = _x_train[:num_samples][-num_validation_samples:]
        y_val = _y_train[:num_samples][-num_validation_samples:]

        # ***********************************
        # create keras model
        we = fb.create_embeddings_layer(config.we_ver, config.we_file,
                                        lower_case=True,
                                        trainable=config.we_trainable,
                                        save_to=None)

        """
        The keras models are defined in file keras_models.py and their models in configuration.py file
        """
        model = config.keras_model.builder(we, config)

        # **********************************
        # train model
        print("Training phase started ...")

        # create Keras callbacks
        cb = []
        metric = 'val_acc' if config.val_data_ratio < 1.0 else 'acc'
        cb += [callbacks.ReduceLROnPlateau(monitor=metric, factor=0.2, patience=2, min_lr=0.001, epsilon=1e-4)]
        cb += [callbacks.EarlyStopping(monitor=metric, min_delta=1e-6, patience=4, mode="max")]
        cb += [callbacks.ModelCheckpoint(config.models_checkpoints, monitor=metric,
                                         verbose=0, save_best_only=True, save_weights_only=False, mode='auto')]

        # Fit the model
        np.random.seed(config.seed)
        history = model.fit(x_train, y_train,
                            batch_size=config.batch_size,
                            epochs=config.epochs if config.unfreeze_we_at_epoch is None else config.unfreeze_we_at_epoch,
                            validation_data=(x_val, y_val) if config.val_data_ratio < 1.0 else None,
                            callbacks=cb)
        # plot the training and validation accuracy graph
        plot_keras_accuracy_graph(history, config.base_folder + 'train_graph.png')

        if config.unfreeze_we_at_epoch is not None:
            # Second phase of training with embeddings set as trainable
            we.trainable = True
            model.compile(loss='binary_crossentropy',
                          optimizer='rmsprop',
                          metrics=['acc'])
            history = model.fit(x_train, y_train,
                                batch_size=config.batch_size,
                                initial_epoch=config.unfreeze_we_at_epoch,
                                epochs=config.epochs,
                                validation_data=(x_val, y_val) if config.val_data_ratio < 1.0 else None,
                                callbacks=cb)
            plot_keras_accuracy_graph(history, config.base_folder + 'train_graph_2.png')

        print('Training phase finished ...')

    else:
        print('Training phase skipped, to enable it set training=True in configuration.py')

    if model is None:
        # if the training was skipped, load the model from file
        print('Load model from: ', config.submission_model)
        model = models.load_model(config.submission_model)

    # build the submission file
    create_submission_file_from_keras_model(model,
                                            config.test_data_pkl,
                                            config.submission_filename)

    print('Script END.')


if __name__ == "__main__":
    main()
