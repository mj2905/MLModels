"""
Contains several methods used to do text preprocessing of tweets
"""
import re

from math import log

words = open("data/words-by-frequency.txt").read().split()
wordcost = dict((k, log((i+1)*log(len(words)))) for i, k in enumerate(words))
maxword = max(len(x) for x in words)

def infer_spaces(s):
    """Uses dynamic programming to infer the location of spaces in a string
    without spaces.
    credits: https://stackoverflow.com/questions/8870261/how-to-split-text-without-spaces-into-list-of-words"""
    s = re.sub(r'[^0-9a-z]', '', s)

    # Find the best match for the i first characters, assuming cost has
    # been built for the i-1 first characters.
    # Returns a pair (match_cost, match_length).
    def best_match(i):
        candidates = enumerate(reversed(cost[max(0, i-maxword):i]))
        return min((c + wordcost.get(s[i-k-1:i], 9e999), k+1) for k,c in candidates)

    # Build the cost array.
    cost = [0]
    for i in range(1,len(s)+1):
        c,k = best_match(i)
        cost.append(c)

    # Backtrack to recover the minimal-cost string.
    out = []
    i = len(s)
    while i>0:
        c,k = best_match(i)
        assert c == cost[i]
        out.append(s[i-k:i])
        i -= k

    return " ".join(reversed(out))


def tweet_clean_dup_chars(word):
    """
    Delete chars that are duplicate more than two times in a word
    :param word: the word to clean
    :return: the cleaned word
    """
    c = 0
    last = 0
    res = ''
    for m in re.finditer(r'(.)\1{2,}', word):
        if len(m.group(1)) > 0:
            c += m.end()-m.start()
            res += word[last:m.start()]+m.group(1)+m.group(1)
            last = m.end()

    res += word[last: len(word)]
    if c > 0:
        res += ' <rep_chars' + ('>' if c < 5 else '+>')
    return res.replace('\n', '')


def tweet_split_hashtags(word, append_hashtag):
    """
    Split a #hashtag in multiple words
    eg. '#darthgradient' --> '<hashtag> darth gradient'
    :param word: the word to process
    :param append_hashtag: if True --> '<hashtag> darth gradient #darthgradient'
    :return: the splitted hashtag
    """
    if word.startswith('#') and len(word) > 1:
        res = ''
        res += '<hashtag> '
        res += infer_spaces(word[1:])
        if append_hashtag:
            res += ' '
            res += word
        return res
    else:
        return word


def tweet_clean_numbers(word):
    """
    Delete words composed of "numbers" (can surely be improved)
    """
    if not re.search(r'[0-9]+', word):
        return word
    if len(word)==4 and re.search(r'[0-9]{4}', word) and 1900 < int(word) < 2019:
        return word
    word = re.sub(r'^([0-9]|[\+\-%/\*\.:])+[0-9%/\+\*\.x:]*$', '<number>', word)
    return word


def tweet_grammar_rules(word):
    """
    "Grammar" replacement rules:
    eg. don't --> do not
        it's --> it 's
    :param word:
    :return:
    """
    w = re.sub(r"'s$", " 's", word)
    w = re.sub(r"n't$", " not", w)
    w = re.sub(r"'ll$|'l$", " will", w)
    w = re.sub(r"'d$", " 'd", w)
    w = re.sub(r"'m$", " am", w)
    return w


def tweet_preprocess(tweet, append_hashtag, test_file=False):
    """
    Apply all the precedents methods on all word of a tweet + do some other cleanup
    :param tweet: the tweet
    :param append_hashtag: see tweet_split_hashtag()
    :param test_file: True: handle the ids in test_data.txt file
    :return: the processed tweet
    """
    if test_file:
        tweet = tweet.split(',', 1)[1]
    tweet = re.sub(r'<3|< 3', '<heart>', tweet)
    res = []
    for word in tweet.split(' '):
        w = re.sub(r'[\.\*,%/\\"\-\_]+', ' ', word)
        w = tweet_grammar_rules(w)
        w = tweet_clean_numbers(w)
        w = tweet_clean_dup_chars(w)
        w = tweet_split_hashtags(w, append_hashtag)
        res.append(w)
    tweet = ' '.join(res).strip()
    tweet = re.sub(r'[ ]+', ' ', tweet)
    return tweet


def tweet_preprocess_1(tweet, test_file=False):
    return tweet_preprocess(tweet, append_hashtag=False, test_file=test_file)


def tweet_preprocess_2(tweet, test_file=False):
    return tweet_preprocess(tweet, append_hashtag=True, test_file=test_file)
