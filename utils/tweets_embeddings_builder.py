import os
import numpy as np
from keras.layers import Embedding
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from utils.utils import save_pkl, load_pkl


class TweetsEmbeddingsBuilder(object):
    """
    Utility class used to preprocess tweets, tranform them to sequences of integer
    and finally build the embedding layer for the Keras model

    Methods call order:
    1. add_tweet_file: preprocess tweets files
    2. build_word_index: tokenizing tweets, compute word frequency, convert tweets to integer vector
    3. create_embeddings_layer: used in the Keras model
    4. get_training/test_data: get data used for training and prediction
    """

    def __init__(self, preprocess_func):
        """
        Constructor
        :param preprocess_func: the tweets preprocess function to use
        """
        self.preprocess_func = preprocess_func
        self.tweets_array = []
        self.labels = []
        self.tweets_ids = []
        self.int_mapped_tweets = None
        self.tokenizer = None
        self.max_seq_length = None
        self.dict_max_words = None

    def save(self, filename):
        """
        Save the build to a pkl file
        """
        save_pkl(self.__dict__, filename)

    def load(self, filename):
        """
        Initialize the builder from a pkl file
        """
        self.__dict__.update(load_pkl(filename))

    def add_tweet_file(self, path, file, y, preprocess_func=None):
        """
        Add and preprocess a tweet file that will be added to the words index
        :param path: the folder containing the file
        :param file: the filename
        :param y: the class of all tweets in this file (positive=1 / negative=0)
        :param preprocess_func: to override the function passed in constructor
        :return: None
        """
        p_func = self.preprocess_func if preprocess_func is None else preprocess_func

        print("Process tweets file:", file, '...')
        with open(os.path.join(path, file), encoding='utf8') as f:
            seen = set([])
            for tw in f:
                t = p_func(tw, y is None)
                if y is None:
                    self.tweets_array.append(t)
                    self.labels.append(None)
                    self.tweets_ids.append(int(tw.split(',', 1)[0]))
                elif t not in seen:
                    self.tweets_array.append(t)
                    self.labels.append(y)
                    self.tweets_ids.append(None)
                    seen.add(t)
        n = len(seen)
        n = np.count_nonzero(self.tweets_ids) if n == 0 else n
        print('Found %s unique tweets.' % n)
        
    def build_word_index(self, max_seq_length, dict_max_words, post_padding=True):
        """
        Build the word index, use the keras Tokenizer class
        :param max_seq_length: max number of words in a tweet
        :param dict_max_words: dictionnary size
        :param post_padding: how to align tweets shorter than max_seq_length
        :return: None
        """
        print('Build words index...')
        assert(len(self.tweets_array) > 0)
        self.dict_max_words = dict_max_words
        self.max_seq_length = max_seq_length
        tokenizer = Tokenizer(num_words=self.dict_max_words, filters='')
        self.tokenizer = tokenizer
        tokenizer.fit_on_texts(self.tweets_array)
        sequences = tokenizer.texts_to_sequences(self.tweets_array)

        seq_len = [len(s) for s in sequences]
        l, b = np.histogram(seq_len, np.concatenate(([0], range(0, 101, 10))))
        print('Tweets sequences length histogram bins:')
        print(np.concatenate(([b[1:]], [l])))

        self.int_mapped_tweets = pad_sequences(sequences, maxlen=max_seq_length, padding='post' if post_padding else 'pre')
        print('Found {} unique tokens, index shape: {}'.format(len(tokenizer.word_index), str(self.int_mapped_tweets.shape)))

    def get_training_data(self, shuffle=True, seed=1):
        """
        Get the training dataset: (X, y)
        X: matrix of integers with each lines representing a tweets and each cell a word index
        y: the corresponding class for each tweet
        :param shuffle: if True the tweets are shuffled
        :param seed: the seed to use
        :return: (X, y)
        """
        assert(self.tokenizer is not None)
        y = np.asarray(self.labels)
        x_train = np.asarray(self.int_mapped_tweets)[y != np.array([None])]
        y_train = to_categorical(y[y != np.array([None])])
        if shuffle:
            np.random.seed(seed)
            indices = np.arange(len(x_train))
            np.random.seed(1)
            np.random.shuffle(indices)
            x_train = x_train[indices]
            y_train = y_train[indices]
        return x_train, y_train

    def get_test_data(self, shuffle=True, seed=1):
        """
        Get the test dataset: (X, ids)
        X: matrix of integers with each lines representing a tweets and each cell a word index
        ids: the corresponding ids for each tweet
        :param shuffle: if True the tweets are shuffled
        :param seed: the seed to use
        :return: (X, ids)
        """
        assert(self.tokenizer is not None)
        ids = np.asarray(self.tweets_ids)
        x_test = np.asarray(self.int_mapped_tweets)[ids != np.array([None])]
        ids_test = ids[ids != np.array([None])]
        if shuffle:
            np.random.seed(seed)
            indices = np.arange(x_test.shape[0])
            np.random.seed(1)
            np.random.shuffle(indices)
            x_test = x_test[indices]
            ids_test = ids_test[indices]
        return x_test, ids_test

    def create_embeddings_layer(self, ver, words_embeddings_file, lower_case=True, layer_name=None, trainable=False, save_to=None):
        """
        Create the Keras embeddings layer
        :param words_embeddings_file: pretrained words embeddings file (eg. glove_tw_27b_200.txt)
        :param lower_case: if the words in the embeddings file need to be lower cased
        :param layer_name: the name of the layer
        :param trainable: True --> the embeddings are trainable
        :param save_to: None or filename if you want to save the layer
        :return: the layer (keras.layers.Embedding)
        """
        if ver == 1:
            return self.create_embeddings_layer_v1(words_embeddings_file, lower_case, layer_name, trainable, save_to)
        else:
            return self.create_embeddings_layer_v2(words_embeddings_file, lower_case, layer_name, trainable, save_to)

    def create_embeddings_layer_v1(self, words_embeddings_file, lower_case=True, layer_name=None, trainable=False, save_to=None):
        assert(self.tokenizer is not None)
        embeddings_index = {}
        we_dim = 0
        wd = sorted(self.tokenizer.word_docs, key=self.tokenizer.word_index.get, reverse=False)
        wd = set(wd[:self.dict_max_words])
        with open(words_embeddings_file, encoding='utf8') as f:
            for line in f:
                if len(line) < 100:
                    continue
                values = line.split()
                word = values[0]
                if lower_case:
                    word = word.lower()
                if word in embeddings_index or word not in wd:
                    continue
                coefs = np.asarray(values[1:], dtype='float32')
                if we_dim == 0:  # detect number of features in WE file
                    we_dim = len(coefs)
                embeddings_index[word] = coefs

        # prepare embedding matrix
        num_words = min(self.dict_max_words, len(self.tokenizer.word_index))+1
        embedding_matrix = np.zeros((num_words, we_dim))
        for i, word in enumerate(wd):
            if i > num_words:
                continue
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i+1] = embedding_vector

        # load pre-trained word embeddings into an Embedding layer
        # note that we set trainable = False so as to keep the embeddings fixed
        embedding_layer = Embedding(num_words,
                                    output_dim=we_dim,
                                    weights=[embedding_matrix],
                                    input_length=self.max_seq_length,
                                    trainable=trainable,
                                    name=words_embeddings_file if layer_name is None else layer_name)

        return embedding_layer

    def create_embeddings_layer_v2(self, words_embeddings_file, lower_case=True, layer_name=None, trainable=False, save_to=None):
        assert(self.tokenizer is not None)
        print('Create embeddings layer from file: ', words_embeddings_file)
        embeddings_index = {}
        we_dim = 0

        # read words embeddings file
        with open(words_embeddings_file, encoding='utf8') as f:
            wd = sorted(self.tokenizer.word_index, key=self.tokenizer.word_index.get, reverse=False)[:self.dict_max_words]
            wd_set = set(wd)
            for line in f:
                if len(line) < 100:
                    continue
                values = line.split()
                word = values[0]
                if lower_case:
                    word = word.lower()
                if word in embeddings_index or word not in wd_set:
                    continue
                coefs = np.asarray(values[1:], dtype='float32')
                if we_dim == 0:  # detect number of features in WE file
                    we_dim = len(coefs)
                embeddings_index[word] = coefs

        we_dim = 200 if we_dim == 0 else we_dim  # empty embeddings file, set 200 dimensional vectors

        # prepare embedding matrix
        num_words = min(self.dict_max_words+1, len(self.tokenizer.word_index))
        embedding_matrix = np.zeros((num_words, we_dim))
        i = r = 0
        for word, idx in self.tokenizer.word_index.items():
            if idx >= num_words:
                continue
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                i += 1
                embedding_matrix[idx] = embedding_vector
            else:
                r += 1
                # words not found in embedding index will be initialized randomly.
                embedding_matrix[idx] = np.random.normal(0, 1, we_dim)

        print('{0:.2f}% words found in embeddings file!'.format(100.0 * i / (i + r)))

        # load pre-trained word embeddings into an Embedding layer
        # note that we set trainable = False so as to keep the embeddings fixed
        embedding_layer = Embedding(num_words,
                                    output_dim=we_dim,
                                    weights=[embedding_matrix],
                                    input_length=self.max_seq_length,
                                    trainable=trainable,
                                    name=words_embeddings_file if layer_name is None else layer_name)
        print("Embeddings created from ", words_embeddings_file)
        if save_to is not None:
            save_pkl(embedding_layer, save_to)
            print('Embeddings layer saved to: ', save_to)

        return embedding_layer
