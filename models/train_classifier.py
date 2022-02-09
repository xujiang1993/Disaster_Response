# import libraries
from sqlite3 import connect
import pandas as pd
from sqlalchemy import create_engine
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import nltk
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import sys
import pickle

from utils import tokenize

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')



def load_data(database_filepath):
    '''
    Load sql database into dataframe and seperate the features and labels
    
    Args:
      database_filepath (str): name of database containing data
      
    Returns:
      X (dataframe): message data
      Y (dataframe): categories (labels)
      category_names: headers of categories
    '''
    
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql('SELECT * FROM massage_and_categories', engine)
    X = df['message']
    Y = df[df.columns[4:-1]]
    category_names = Y.columns
    return X, Y, category_names




def build_model():
    '''build a model pipeline
    Function builds a pipeline by combining CountVectorize, TfidfTransformers and a KNN's Multioutputclassifier.
    Optimize model using cross valicdation and measure the performance with f1score.
    
    Args:
      None
      
    Returns:
      scikit learn pipeline model
    '''   
    pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(KNeighborsClassifier()))
    ]

    )

    parameters = {
        'clf__estimator__n_neighbors': [5, 8]
    }

    cv = GridSearchCV(pipeline, param_grid=parameters)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    evaluate the trained model with test dataset
    
    Function prints the model performance on test dataset,
    and prints multiple metrics (precision, recall, f1score)
    with various averaging
    
    Args:
      model (scikit learn pipeline): name of model
      X_test (dataframe): test dataset features
      Y_test (dataframe): test dataset labels
      category_names(string): output class
      
    Returns:
      None
    '''
        
    def column(matrix, i): # Extract the column from input matrix
        return [row[i] for row in matrix]
    y_pred = model.predict(X_test)
    for i in range(Y_test.shape[1]):
        print('The performance of the ' + category_names[i] +' prediction is:')
        print(classification_report(Y_test.iloc[:,i], column(y_pred, i)))


def save_model(model, model_filepath):
    '''
    save the trained model as .pkl file
    
    Args:
      model (scikit learn pipeline): name of model
      model_filepath (str): path and name of the .pkl file
      
    Returns:
      None
    ''' 
    filename = model_filepath
    pickle.dump(model, open(filename, 'wb'))


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
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
