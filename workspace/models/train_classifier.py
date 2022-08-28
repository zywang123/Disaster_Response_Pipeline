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

def load_data(database_filepath):
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('disaster_message_tbl', engine)
    X = df['message']
    Y = df.iloc[:, 4:]
    return X, Y


def tokenize(text):
    """
    Function: convert sentense into words and transform words to its original (root) form
    Input: 
        text: the message
    Return:
        lemm: the original form of the words
    """
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # Tokenize text
    words = word_tokenize(text) 
    
    # Remove stop words
    stop = stopwords.words("english")
    words = [x for x in words if x not in stop]
    
    # Lemmatization
    lemm = [WordNetLemmatizer().lemmatize(x) for x in words]
    return lemm


def build_model():
    """
    Function: build model for random forest classification of diasaster message
    Return:
        cv: classification model
    """
    # Random Forest
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer = tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    parameters = {
        'tfidf__use_idf': (True, False), 
        'clf__estimator__n_estimators': [10, 20]
    }
    
    cv = GridSearchCV(pipeline, param_grid = parameters)
    
    return cv


def evaluate_model(model, X_test, Y_test):
    """
    Function: Test the model and report the f1 score, precision and recall for each output category of the dataset.
    Input: 
        model: Random Forest Classification model
        X_test: test message
        Y_test: message to fit the model
    Return:
        accurarcy: model accurarcy
    """
    y_pred = model.predict(X_test)
    i = 0
    for col in Y_test:
        print('Feature {}:{}'.format(i+1, col))
        print(classification_report(Y_test[col], y_pred[:, i]))
        i = i + 1
    accuracy = (y_pred == Y_test.values).mean()
    print('The model accuracy is {:.3f}'.format(accuracy))

def save_model(model, model_filepath):
    """
    Function: Export your model as a pickle file
    Input: 
        model: Random Forest Classification model
        model_filepath: pickle file path
    """
    
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()