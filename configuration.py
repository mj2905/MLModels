"""
Configuration file for all the different algorithms we have implemented
"""
from utils.tweets_cleanup import tweet_preprocess_1
from keras_models import *

_config = {

    'default_model': 'submitted_model',

    'submitted_model': {
        'welcome_message': 'CNN with one layer of convolution (handle multiples kernels size) and one dense layer',
        'training': False,  # set to True to train, otherwise use saved model to make predictions
        'seed': 1,
        'epochs': 15,
        'batch_size': 128,
        'datasets_folder': 'twitter-datasets',
        'tw_pos_file': 'train_pos_full.txt',
        'tw_neg_file': 'train_neg_full.txt',
        'tw_test_file': 'test_data.txt',
        'datasets_subsample': 1.0,  # used to reduce training data size
        'base_folder': 'data/submitted_model/',  # where to write file
        'dictionary_size': 20000,  # keep n words in dictionary --> embeddings matrix size
        'tw_length': 40,  # crop tweets longer than n words
        'preprocess_func': tweet_preprocess_1,  # no preprocessing: lambda t, ht=False: t
        'post_padding': True,  # if set to true, tweets shorter than tw_length are post-padded with 0s, otherwize they are pre-padded
        'we_file': 'data/glove.twitter.27B.200d.txt',  # use empty embeddings file
        'we_ver': 1,
        'we_trainable': True,  # if false --> embeddings are static, otherwize they are trained
        'test_data_pkl': 'data/submitted_model/test_data.pkl',  # where to write cleaned tweets (cached for prediction)
        'val_data_ratio': 0.03,  # set to 0.0 to train in blind mode (without validation data)
        'keras_model': {
            'builder': model_cnn1,  # see keras_models.py
            'cnn_filters': 100,  # numbers of filters of the convolution layer
            'cnn_kernels': (3, 4, 5),  # kernels length
            'dropout': 0.4,  # dropout rate
            'dense_size': 64,  # size of the dense (fully connected) layer
        },
        'models_checkpoints': 'data/submitted_model/model_weights_{epoch:02d}.hdf5',  # where to write model weights during training
        'submission_model': 'data/submitted_model/model_submission.hdf5',  # location of model weights used for the submission
        'submission_filename': 'submission.csv'  # submission file name
    },

    'model_two_phases_training': {
        'welcome_message': 'CNN with one layer of convolution (handle multiples kernels size) and one dense layer and 2 phases training',
        'training': True,
        'seed': 1,
        'epochs': 15,
        'unfreeze_we_at_epoch': 5,
        'batch_size': 128,
        'datasets_folder': 'twitter-datasets',
        'tw_pos_file': 'train_pos_full.txt',
        'tw_neg_file': 'train_neg_full.txt',
        'tw_test_file': 'test_data.txt',
        'datasets_subsample': 1.0,  # used to reduce training data size
        'base_folder': 'data/submitted_model_two_phases_training/',  # where to write file
        'dictionary_size': 20000,  # keep n words in dictionary --> embeddings matrix size
        'tw_length': 40,  # crop tweets longer than n words
        'preprocess_func': tweet_preprocess_1,  # tweet preprocess function to use
        'post_padding': True,  # if set to true, tweets shorter than tw_length are post-padded with 0s, otherwize they are pre-padded
        'we_file': 'data/glove.twitter.27B.200d.txt',  # use empty embeddings file
        'we_ver': 1,
        'we_trainable': False,  # if false --> embeddings are static, otherwize they are trained
        'test_data_pkl': 'data/submitted_model_two_phases_training/test_data.pkl',  # where to write cleaned tweets (cached for prediction)
        'val_data_ratio': 0.03,  # set to 0.0 to train in blind mode (without validation data)
        'keras_model': {
            'builder': model_cnn1,  # see keras_models.py
            'cnn_filters': 100,  # numbers of filters of the convolution layer
            'cnn_kernels': (3, 4, 5),  # kernels length
            'dropout': 0.4,  # dropout rate
            'dense_size': 64,  # size of the dense (fully connected) layer
        },
        'models_checkpoints': 'data/submitted_model_two_phases_training/model_weights_{epoch:02d}.hdf5',  # where to write model weights during training
        'submission_model': 'data/submitted_model_two_phases_training/saved_model.hdf5',  # location of model weights used for the submission
        'submission_filename': 'data/submitted_model_two_phases_training/submission.csv'  # submission file name
    },

    'cnn_tests_with_2_dense_layers': {
        'welcome_message': 'CNN with one layer of convolution (handle multiples kernels size) and multiple dense layers',
        'training': True,  # set to train, otherwise use saved model to make predictions
        'seed': 1,
        'epochs': 15,
        'batch_size': 128,
        'datasets_folder': 'twitter-datasets',
        'tw_pos_file': 'train_pos_full.txt',
        'tw_neg_file': 'train_neg_full.txt',
        'tw_test_file': 'test_data.txt',
        'datasets_subsample': 1.0,  # used to reduce training data size
        'base_folder': 'data/cnn_tests_with_2_dense_layers/',  # where to write file
        'dictionary_size': 40000,  # keep n words in dictionary --> embeddings matrix size
        'tw_length': 40,  # crop tweets longer than n words
        'preprocess_func': tweet_preprocess_1,  # tweet preprocess function to use
        'post_padding': True,  # if set to true, tweets shorter than tw_length are post-padded with 0s, otherwize they are pre-padded
        'we_file': 'data/glove.twitter.27B.200d.txt',  # embeddings tweets files (in this case Standford ones, 27B tokens)
        'we_trainable': True,  # if false --> embeddings are static, otherwize they are trained
        'test_data_pkl': 'data/cnn_tests_with_2_dense_layers/test_data.pkl',  # where to write cleaned tweets (cached for prediction)
        'val_data_ratio': 0.03,  # set to 0.0 to train in blind mode (without validation data)
        'keras_model': {
            'builder': model_cnn2, # see keras_models.py
            'cnn_filters': 100,  # numbers of filters of the convolution layer
            'cnn_kernels': (3, 4, 5),  # kernels length
            'dropout': 0.4,  # dropout rate
            'dense_sizes': [128, 64],  # size of the dense (fully connected) layer
        },
        'models_checkpoints': 'data/cnn_tests_with_2_dense_layers/model_weights_{epoch:02d}.hdf5',  # where to write model weights during training
        'submission_model': 'data/cnn_tests_with_2_dense_layers/model_submission.hdf5',  # location of model weights used for the submission
        'submission_filename': 'submission_2denses_layers.csv'  # submission file name
    },

    'glove27b_relu_activation': {
        'welcome_message': 'CNN with one layer of convolution (handle multiples kernels size) and one dense layer',
        'training': True, 
        'seed': 1,
        'epochs': 10,
        'batch_size': 128,
        'datasets_folder': 'twitter-datasets',
        'tw_pos_file': 'train_pos_full.txt',
        'tw_neg_file': 'train_neg_full.txt',
        'tw_test_file': 'test_data.txt',
        'datasets_subsample': 1.0,  # used to reduce training data size
        'base_folder': 'data/submitted_model/',  # where to write file
        'dictionary_size': 20000,  # keep n words in dictionary --> embeddings matrix size
        'tw_length': 40,  # crop tweets longer than n words
        'preprocess_func': tweet_preprocess_1,  # tweet preprocess function to use
        'post_padding': True,  # if set to true, tweets shorter than tw_length are post-padded with 0s, otherwize they are pre-padded
        'we_file': 'data/glove.twitter.27B.200d.txt',  # use empty embeddings file
        'we_ver': 1,
        'we_trainable': True,  # if false --> embeddings are static, otherwize they are trained
        'test_data_pkl': 'data/submitted_model/test_data.pkl',  # where to write cleaned tweets (cached for prediction)
        'val_data_ratio': 0.03,  # set to 0.0 to train in blind mode (without validation data)
        'keras_model': {
            'builder': model_cnn3,  # see keras_models.py
            'cnn_filters': 100,  # numbers of filters of the convolution layer
            'cnn_kernels': (3, 4, 5),  # kernels length
            'dropout': 0.4,  # dropout rate
            'dense_size': 64,  # size of the dense (fully connected) layer
        },
        'models_checkpoints': 'data/submitted_model/model_weights_{epoch:02d}.hdf5',  # where to write model weights during training
        'submission_model': 'data/submitted_model/model_submission.hdf5',  # location of model weights used for the submission
        'submission_filename': 'submission.csv'  # submission file name
    },
}


# credits: https://stackoverflow.com/questions/13520421/recursive-dotdict
class DotDict(dict):
    """
    a dictionary that supports dot notation
    as well as dictionary access notation
    usage: d = DotDict() or d = DotDict({'val1':'first'})
    set attributes: d.val2 = 'second' or d['val2'] = 'second'
    get attributes: d.val2 or d['val2']
    """
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __init__(self, dct):
        for key, value in dct.items():
            if hasattr(value, 'keys'):
                value = DotDict(value)
            self[key] = value


# convert the dict to a DotDict for 'dot' access
config = DotDict(_config)
