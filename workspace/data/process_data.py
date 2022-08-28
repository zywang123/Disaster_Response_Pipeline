import sys
import pandas as pd
import numpy as np
import re
import pickle
from time import strptime
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
import nltk
from nltk import word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
import operator
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from sqlalchemy import create_engine

nltk.download(['stopwords', 'wordnet', 'punkt'])


def load_data(messages_filepath, categories_filepath):
    """
    Function: load dataset
    Input: 
        messages_filepath: message file path
        categories_filepath: category file path
    Return:
        df: merge file of message and category
    """
    
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, left_on = 'id', right_on = 'id', how = 'inner')
    
    return df


def clean_data(df):
    """
    Function: clean the dataset
    Input: 
        df: original dataset
    Return:
        df: clean dataset
    """
    
    categories = df['categories'].str.split(';', expand = True)
    
    # select the first row of the categories dataframe
    row = categories.head(1)

    # use this row to extract a list of new column names for categories.
    category_colnames = row.applymap(lambda x: x[:-2]).iloc[0,:].tolist()
    
    # rename the columns of `categories`
    categories.columns = category_colnames
    
    # Convert category values to just numbers 0 or 1.
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).str[-1]
    
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    
    # remove the data which related is equal to 2
    categories = categories[categories['related'] != 2]
    # drop the original categories column from `df`
    df = df.drop(['categories'], axis = 1)
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis = 1, join = 'inner')
    
    # drop duplicates
    df.drop_duplicates(inplace = True)
    
    return df

def save_data(df, database_filename):
    """
    Function: save  dataset
    Input: 
        df: dataset need to saved
        database_filename: file name
    """
    
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('disaster_message_tbl', engine, index=False, if_exists = 'replace')


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()