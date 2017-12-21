import pandas as pd
import numpy as np
import sklearn as sk
import gensim
from sklearn.model_selection import train_test_split
import re
from sklearn.metrics import accuracy_score
from nltk.tokenize import TweetTokenizer
from sklearn.preprocessing import normalize

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

    total_shuffled = total.sample(frac=1, random_state=0)
    total_shuffled = total_shuffled.drop_duplicates().reset_index(drop=True)
    
    total_shuffled.Tweets = total_shuffled.replace(r"[\\|\:|\.|\*|\-|\'|\`|\@|\;|\,|\_|\^|\+|\[|\]|\(|\)|\=|\&|\%|\"|\~]|[0-9]", '', regex=True)
    total_shuffled.Tweets = total_shuffled.replace(r'(.)\1+', r'\1\1', regex=True)
    total_shuffled.Tweets = total_shuffled.replace(r'(. )\1+', r'\1\1', regex=True)
	
    return total_shuffled

def tokenize(tweets):

    tknzr = TweetTokenizer()
    tweets_tokens = []

    for tweet in tweets:
    	tok = tknzr.tokenize(tweet)
    	tweets_tokens += [tok]
    return tweets_tokens

def get_word2vec(tweets, size):
    model = gensim.models.Word2Vec(tweets, size=size, min_count=5, workers=4)
    return dict(zip(model.wv.index2word, model.wv.syn0))

def obtain_tweet_vector(tweets, size, w2v):
    tweets_score = np.empty((len(tweets), size))
    for i, tweet in enumerate(tweets):
        tweet_vectors = []
        
        for j, word in enumerate(tweet):
            
            vector = w2v.get(word)
            if not vector is None:
                tweet_vectors += [vector]

        if len(tweet_vectors) == 0:
            tweet_vectors = [np.zeros(size)]
        tweets_score[i] = np.mean(tweet_vectors, axis=0)
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
SIZE_word2vec = 200
w2v = get_word2vec(tweets_tokens, SIZE_word2vec)
print('--- word2vec obtained')
tweets_score = obtain_tweet_vector(tweets_tokens, SIZE_word2vec, w2v)
print('--- tweet vector obtained')
normalize(tweets_score, axis=1, norm='l2', copy=False)
print('--- normalization done')

np.random.seed(42)
from keras.models import Sequential
from keras.layers import Dense, Dropout

X_train, X_test, y_train, y_test = train_test_split(tweets_score, total_shuffled.Category, test_size=0.1, random_state=42)
print('--- data split')

del total_shuffled
del tweets_tokens
del w2v
del tweets_score

model_k = create_keras_model(X_train.shape[1])
print('--- model created')
model_k.fit(X_train, y_train, epochs=9, batch_size=64, verbose=2, validation_data = [X_test, y_test])
                                                   
f1_score()
print('--- end of algorithm')

