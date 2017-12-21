import pandas as pd
import numpy as np
import sklearn as sk
import gensim
from sklearn.model_selection import train_test_split
import re
from sklearn.metrics import accuracy_score
from nltk.tokenize import TweetTokenizer
from sklearn.preprocessing import normalize
from random import shuffle

PATH_POSITIVE = 'twitter-datasets/train_pos.txt'
PATH_NEGATIVE = 'twitter-datasets/train_neg.txt'
PATH_FULL_POSITIVE = 'twitter-datasets/train_pos_full.txt'
PATH_FULL_NEGATIVE = 'twitter-datasets/train_neg_full.txt'

def getDataFrame(path):
    with open(path, 'r') as tweets:
        lines = tweets.readlines()
        df = pd.DataFrame(lines)
        df = df.rename(columns={0: 'Tweets'})
    return df

def load_data(PATH_FULL_POSITIVE, PATH_FULL_NEGATIVE):
    full_positive = getDataFrame(PATH_FULL_POSITIVE)
    full_negative = getDataFrame(PATH_FULL_NEGATIVE)
    full_positive['Category'] = 1
    full_negative['Category'] = 0

    total = pd.concat([full_positive, full_negative]).reset_index(drop=True)

    total_shuffled = total.sample(frac=1, random_state=1)
    total_shuffled = total_shuffled.drop_duplicates().reset_index(drop=True)
    
    total_shuffled.Tweets = total_shuffled.replace(r"[\\|\:|\.|\*|\-|\'|\`|\@|\;|\,|\_|\^|\+|\[|\]|\(|\)|\=|\&|\%|\"|\~]|[0-9]", '', regex=True)
    total_shuffled.Tweets = total_shuffled.replace(r'(.)\1+', r'\1\1', regex=True)
    total_shuffled.Tweets = total_shuffled.replace(r'(. )\1+', r'\1\1', regex=True)
	
    return total_shuffled

def tokenize(tweets):
    tknzr = TweetTokenizer()
    tweets_tokens = []
    for i,tweet in enumerate(total_shuffled.Tweets):
        tok = gensim.models.doc2vec.LabeledSentence(tknzr.tokenize(tweet), ['X$'+str(i)])
        tweets_tokens += [tok]
    return tweets_tokens

def get_doc2vec_model(tweets, size):
    model = gensim.models.Doc2Vec(size=size, window=5, min_count=1, workers=4, sample=1e-4, iter=1)
    model.build_vocab(tweets)
    for epoch in range(1):
        shuffle(tweets)
        model.train(tweets, epochs=epoch, total_examples=len(tweets))
    return model

def obtain_tweet_vector(tweets, size, model):
    tweets_score = np.empty((len(tweets), size))
    for i, tweet in enumerate(tweets):
        tweets_score[i] = model.docvecs['X$'+str(i)]
    return tweets_score

def create_keras_model(tweets_entry_dimension):
    model_k = Sequential()
    model_k.add(Dense(128, activation='relu', input_dim= tweets_entry_dimension))
    model_k.add(Dropout(0.5))
    model_k.add(Dense(1, activation='sigmoid'))
    model_k.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
    return model_k


total_shuffled = load_data(PATH_FULL_POSITIVE, PATH_FULL_NEGATIVE)
print('--- data loaded')
tweets_tokens = tokenize(total_shuffled.Tweets)
print('--- data tokenized')
SIZE_doc2vec = 100
np.random.seed(1337)
                                                   
model = get_doc2vec_model(tweets_tokens, SIZE_doc2vec)
print('--- doc2vec obtained')
tweets_score = obtain_tweet_vector(tweets_tokens, SIZE_doc2vec, model)
print('--- tweet vector obtained')
normalize(tweets_score, axis=0, norm='max', copy=False)
print('--- normalization done')

np.random.seed(42)
from keras.models import Sequential
from keras.layers import Dense, Dropout

X_train, X_test, y_train, y_test = train_test_split(tweets_score, total_shuffled.Category, test_size=0.1, random_state=42)
print('--- data split')

del total_shuffled
del tweets_tokens
del tweets_score

model_k = create_keras_model(X_train.shape[1])
print('--- model created')
model_k.fit(X_train, y_train, epochs=9, batch_size=64, verbose=2, validation_data = [X_test, y_test])
print('--- end of algorithm')

