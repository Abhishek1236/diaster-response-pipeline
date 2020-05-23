"""
Classifier Trainer
Project: Disaster Response Pipeline (Udacity - Data Science Nanodegree)
Sample Script Syntax:
> python train_classifier.py <path to sqllite  destination db> <path to the pickle file>
Sample Script Execution:
> python train_classifier.py ../data/disaster_response_db.db classifier.pkl
Arguments:
    1) Path to SQLite destination database (e.g. disaster_response_db.db)
    2) Path to pickle file name where ML model needs to be saved (e.g. classifier.pkl)
"""
#importing the libraries
import sys
import pandas as pd
import numpy as np
import os
import pickle
from sqlalchemy import create_engine
import re
import nltk
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score, f1_score, fbeta_score, classification_report
from scipy.stats import hmean
from scipy.stats.mstats import gmean
import nltk
nltk.download(['punkt', 'wordnet'])
nltk.download('stopwords')

def load_data(database_filepath):
    '''
    Args : location of database file
    Output :
    X : Features dataframe
    Y : Target Dataframe
    category names = Target Labels
    '''
    engine = create_engine('sqlite:///' + database_filepath)
    table_name = os.path.basename(database_filepath).replace(".db","") + "_table"
    df = pd.read_sql_table(table_name,engine)
    
    #Remove child alone as it has all zeros only
    df = df.drop(['child_alone'],axis=1)
    
    # Given value 2 in the related field are neglible so it could be error. Replacing 2 with 1 to consider it a valid response.
    # Alternatively, we could have assumed it to be 0 also. In the absence of information I have gone with majority class.
    df['related']=df['related'].map(lambda x: 1 if x == 2 else x)
    
    X = df['message']
    y = df.iloc[:,4:]
    
    #print(X)
    #print(y.columns)
    category_names = y.columns # This will be used for visualization purpose
    return X, y, category_names



def tokenize(text):
    '''
    Args : Text
    Output : Text converted into tokens which is used in our further steps
    '''
    token = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    tokens = []
    for tok in token:
        token_ = lemmatizer.lemmatize(tok).lower().strip()
        tokens.append(token_)
    return tokens


def build_model():
    '''
    Args : None
    Output : A model by using pipeline and gridsearch on the pipeline
    '''
    pipeline = Pipeline([
                ('vect',CountVectorizer(tokenizer=tokenize)),
                ('tfidf',TfidfTransformer()),
                ('clf',MultiOutputClassifier(RandomForestClassifier()))])
     # hyper-parameter grid
    parameters =  {'tfidf__use_idf': (True, False), 
              'clf__estimator__n_estimators': [50, 100], 
              'clf__estimator__min_samples_split': [2, 4]} 

    # create model
    model = GridSearchCV(estimator=pipeline,
            param_grid=parameters)
    return model
    

def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Args :
    model : Model defined in the build_model function
    X_test : Test Features
    Y_test : Test Target
    category_names : Test Labels
    Output : 
    print classification report 
    print accuracy of model
    '''
    y_pred = model.predict(X_test)

    # print classification report
    print(classification_report(Y_test.values, y_pred, target_names=category_names))

    # print accuracy score
    print('Accuracy: {}'.format(np.mean(Y_test.values == y_pred)))


def save_model(model, model_filepath):
    '''
    Args :
    model = Trained model
    model_filepath = path to save the pickle file
    '''
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Read the documentation about running the models')


if __name__ == '__main__':
    main()
