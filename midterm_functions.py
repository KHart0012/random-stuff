# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 08:05:04 2019

@author: KHart0012 | s526939
"""

# BIG WARNING!!!! 
#
# 1. DO NOT RUN create_dictionary() UNLESS YOU HAVE ABOVE 6GB OF RAM!
#    - UNLESS YOU HAVE AROUND/ABOVE 16GB OF RAM DO NOT 
#      USE THE VARIABLE EXPLORER ON THE DICTIONARY
#
# 2. I AM PREPROCESSING ALL OF THE DATA SO YOU DON'T HAVE TO RUN IT
#
# 3. IF YOU PLAN ON RUNNING SOME FUNCTIONS, LOWER MAX_TREND TO BELOW 30
#    OTHERWISE FUNCTIONS COULD TAKE MINUTES TO COMPLETE

import tweepy
import pickle
import matplotlib.pyplot as plt
import numpy as np
#import nltk
#nltk.download('vader_lexicon')
#nltk.download('stopwords')
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from prettytable import PrettyTable

analyzer = SentimentIntensityAnalyzer()
pt = PrettyTable()
tweet_tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
trend_dictionary = {}
day_night_trends = {}
stop_words = set(stopwords.words('english'))
for punct in ",.'?/:;~*&%!()$+=-_<>":
    stop_words.add(punct)
stop_words.add('...')
stop_words.add('rt')
trend_names = []

# 65 is the largest full set of tweets
MAX_TREND = 65

# Importing data to do analysis on
def create_dictionary():
    # Initialize keys
    for i in range(10, MAX_TREND):
        with open('pickles/trend' + str(i), 'rb') as f:
            trend_names.extend(pickle.load(f))
	
    # initialize dict
    for trend in set(trend_names):
        trend_dictionary[trend] = []
        day_night_trends[trend] = []
		
    # Loading lists of tweets.
    for i in range(10, MAX_TREND):
        for j in range(10, 14):
            try:
                with open('pickles/trend' + str(i) + 'tweets' + str(j), 'rb') as f:
                    t = pickle.load(f)
                for key in t:
                    trend_dictionary[key].extend(t[key])
                    if t[key][0].created_at.hour <= 19 and t[key][0].created_at.hour >= 5:
                        day_night_trends[key].append('night')
                    else:
                        day_night_trends[key].append('day')
            except:
                pass
    
    
    # Gets rid of duplicate day/night in lists
    for trend in set(trend_names):
        day_night_trends[trend] = list(set(day_night_trends[trend]))


# Lexcial Diversity Functions
def lexical_diversity(tokens):
    if len(tokens) > 3:
        return len(set(tokens)) / len(tokens)
    else:
        return 0

def avg_lexical_diversity(trend):
    lex_divs = []
    tweets = trend_dictionary[trend]
    for tweet in tweets:
        lex_divs.append(lexical_diversity(remove_punctuation(tokenize_tweet(tweet))))
    return sum(lex_divs) / len(lex_divs)

def avg_lex_div_day():
    lex_divs = []
    day_trends = list_day_trends()
    for trend in day_trends:
        lex_divs.append(avg_lexical_diversity(trend))
    return sum(lex_divs) / len(lex_divs)

def avg_lex_div_night():
    lex_divs = []
    night_trends = list_night_trends()
    for trend in night_trends:
        lex_divs.append(avg_lexical_diversity(trend))
    return sum(lex_divs) / len(lex_divs)
  
def plot_lex_divs():
    objects = ('Day', 'Night')
    y_pos = np.arange(len(objects))
    perf = grab_data('avg_lex_div_data')
    
    plt.bar(y_pos, perf, align='center', alpha=0.5, color=['red', 'blue'])
    plt.xticks(y_pos, objects)
    plt.ylabel('Lexical Diversity')
    plt.title('Day/Night Lexical Diversity Difference')
    
    plt.show()


# Most Popular functions
def most_popular_trends(num):
    num_tweets = dict()
    for trend in set(trend_names):
        num_tweets[trend] = len(trend_dictionary[trend])
    sorted_dict = sorted(num_tweets.items(), key=lambda x: x[1], reverse=True)[:num]
    pop_trends = list(sorted_dict)
    return pop_trends  

def popularity_table():
    pt.field_names = ['Popularity', 'Trend', 'Lifetime', 'Lexical Diversity', 'Pos/Neg', 'Day/Night']
    data = grab_data('popularity_data')
    j = 0
    for i in range(15):
        pt.add_row([str(i + 1), data[j + i], data[(j + i) + 1], data[(j + i) + 2], data[(j + i) + 3], data[(j + i) + 4]])
        j += 4
    print(pt)
    pt.clear()

def grab_daynight(trend):
    if 'night' in day_night_trends[trend] and 'day' in day_night_trends[trend]:
        return 'both'
    elif 'night' in day_night_trends[trend]:
        return 'night'
    elif 'day' in day_night_trends[trend]:
        return 'day'
    return 'none'

# Sentiment Analysis functions
def count_day_sentiments():
    pos = []
    neg = []
    day_trends = list_day_trends()
    for trend in day_trends:
        texts = list_tweets(trend)
        avg_pos, avg_neg = compute_avg_sentiments(texts)
        if avg_pos >= avg_neg:
            pos.append(trend)
        else:
            neg.append(trend)
        
    return len(pos), len(neg)

def count_night_sentiments():
    pos = []
    neg = []
    night_trends = list_night_trends()
    for trend in night_trends:
        texts = list_tweets(trend)
        avg_pos, avg_neg = compute_avg_sentiments(texts)
        if avg_pos >= avg_neg:
            pos.append(trend)
        else:
            neg.append(trend)
        
    return len(pos), len(neg)

def compute_avg_sentiments(texts):
    scores = []
    for text in texts:
        scores.append(analyzer.polarity_scores(text))
    pos = []
    neg = []
    for sent in scores:
        pos.append(sent['pos'])
        neg.append(sent['neg'])
    return sum(pos)/len(pos), sum(neg)/len(neg)

def pos_or_neg(trend):
    texts = list_tweets(trend)
    avg_pos, avg_neg = compute_avg_sentiments(texts)
    if avg_pos >= avg_neg:
        return 'pos'
    else:
        return 'neg'
    

def plot_average_sentiments():
    objects = ('Pos/Day', 'Neg/Day', 'Pos/Night', 'Neg/Night')
    y_pos = np.arange(len(objects))
    perf = grab_data('day_night_sentiment_data')
    
    plt.bar(y_pos, perf, align='center', alpha=0.5, color=['red', 'red', 'blue', 'blue'])
    plt.xticks(y_pos, objects)
    plt.ylabel('# of Positive/Negative trends')
    plt.title('Day/Night Sentiment Difference')
    
    plt.show()


# Average lifetime of a trend -- Need one more plot of length of time for time of day
def lifetime(trend):
    return (num_tweets(trend) / 400) * 60

def avg_lifetime_of_trends():
    lifetimes = []
    for trend in set(trend_names):
        lifetimes.append(lifetime(trend))
    return sum(lifetimes) / len(lifetimes)

def avg_lifetime_day():
    lifetimes = []
    for trend in set(trend_names):
        if 'day' in day_night_trends[trend] and not 'night' in day_night_trends[trend]:
            lifetimes.append(lifetime(trend))
    return sum(lifetimes) / len(lifetimes)

def avg_lifetime_night():
    lifetimes = []
    for trend in set(trend_names):
        if 'night' in day_night_trends[trend] and not 'day' in day_night_trends[trend]:
            lifetimes.append(lifetime(trend))
    return sum(lifetimes) / len(lifetimes)

def avg_lifetime_daynight():
    lifetimes = []
    for trend in set(trend_names):
        if 'night' in day_night_trends[trend] and 'day' in day_night_trends[trend]:
            lifetimes.append(lifetime(trend))
    return sum(lifetimes) / len(lifetimes)

def plot_length_trends():
    objects = ('Only Day', 'Only Night', 'Day/Night', 'Overall')
    y_pos = np.arange(len(objects))
    perf = grab_data('lifetime_of_trends')
    
    plt.barh(y_pos, perf, align='center', alpha=0.5, color=['red', 'blue', 'purple', 'green'])
    plt.yticks(y_pos, objects)
    plt.xlabel('Time (Minutes)')
    plt.title('Average Lifetime of a Trend')
    
    plt.show()

def plot_day_night_trends():
    labels = ('Day', 'Night', 'Both')
    sizes = grab_data('num_day_night_trends')
    explode = (0, 0, 0.1)
    
    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', shadow=True, startangle=90)
    ax1.axis('equal')
    plt.title('% of Trends throughout day')
    
    plt.show()

# Tweet Editiing Functions
def list_tweets(trend):
    tweets = trend_dictionary[trend]
    return [t.full_text for t in tweets]

def list_day_trends():
    day_trends = []
    for trend in set(trend_names):
        if 'day' in day_night_trends[trend]: 
            day_trends.append(trend)
    return day_trends

def num_only_day_trends():
    day_trends = []
    for trend in set(trend_names):
        if 'day' in day_night_trends[trend] and not 'night' in day_night_trends[trend]: 
            day_trends.append(trend)
    return len(day_trends)

def list_night_trends():
    night_trends = []
    for trend in set(trend_names):
        if 'night' in day_night_trends[trend]:
            night_trends.append(trend)
    return night_trends

def num_only_night_trends():
    night_trends = []
    for trend in set(trend_names):
        if 'night' in day_night_trends[trend] and not 'day' in day_night_trends[trend]: 
            night_trends.append(trend)
    return len(night_trends)

def tokenize_trend(trend):
    tweets = trend_dictionary[trend]
    tokens = []
    for tweet in tweets:
        tokens.extend(tweet_tokenizer.tokenize(tweet.full_text))
    return remove_hypers(tokens)

def tokenize_tweet(tweet):
    return remove_hypers(tweet_tokenizer.tokenize(tweet.full_text))

def remove_hypers(tokens):
    return list(filter(lambda x: not x.startswith('http'), tokens))

def remove_stopwords(tokens):
    return list(filter(lambda x: not x in stop_words, tokens))

def remove_punctuation(tokens):
    return list(filter(lambda x: not x in ",.'?/:;~*&%!()$+=-_<>", tokens))

def num_tweets(trend):
    return len(trend_dictionary[trend])

def total_num_tweets():
    return sum([len(trend_dictionary[trend]) for trend in set(trend_names)])


# Data Saving/Retrieval Functions
def save_data(thing, name):
    with open('data/' + name + '.pkl', 'wb') as f:
        pickle.dump(thing, f)

def grab_data(name):
    with open('data/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

# Main
#create_dictionary()
#plot_length_trends()
#popularity_table()
#plot_lex_divs()
#plot_day_night_trends()
#plot_average_sentiments()