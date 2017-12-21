import pandas as pd
import numpy as np
import sklearn as sk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import re
from sklearn.metrics import accuracy_score
from nltk.tokenize import TweetTokenizer

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
	
    return total_shuffled

def tokenize(tweets):

    tknzr = TweetTokenizer()
    tweets_tokens = []

    for tweet in tweets:
    	tok = tknzr.tokenize(tweet)
    	tweets_tokens += [tok]
    return tweets_tokens

def get_tfidf_matrix(tweets, min_df = 0.0007, max_df=0.8):
    vectorizer = TfidfVectorizer(stop_words='english',min_df = min_df, max_df = max_df)
    return vectorizer.fit_transform(tweets)

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
tfidf_result = get_tfidf_matrix(total_shuffled.Tweets)
print('--- tf-idf obtained')

np.random.seed(42)
from keras.models import Sequential
from keras.layers import Dense, Dropout

X_train, X_test, y_train, y_test = train_test_split(tfidf_result, total_shuffled.Category, test_size=0.1, random_state=42)
print('--- data split')

del tweets_tokens

model_k = create_keras_model(X_train.shape[1])
print('--- model created')
model_k.fit(X_train.toarray(), y_train, epochs=9, batch_size=64, verbose=2, validation_data = [X_test.toarray(), y_test])
print('--- end of algorithm')

