import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
import pickle
from sqlalchemy import create_engine
from DRapp import app


# app = Flask(__name__)
    
def tokenize(text):
    '''
    Tokenize the text data into vector features
    
    Args:
      text (str): a message in text form
      
    Returns:
      clean_tokens (array): array of words after processing
    '''
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data

engine = create_engine('sqlite:///DisasterResponse.db')
print('Loading data from database')
df = pd.read_sql_table('messages', engine)
print('Finished loading from database')
# load model
with open("models/classifier.pkl", 'rb') as f:
    model = pickle.load(f)
print('Finished loading model')


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    '''
    Build the home page of the web
    
    Args:
      None
      
    Returns:
      None
    '''
        
    # extract data needed for visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    related_counts = df[df.columns[4:-1]].sum()
    related_names = list(df.columns[4:-1])
    
    # create visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "<b>Count</b>"
                },
                'xaxis': {
                    'title': "<b>Genre</b>"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=related_names,
                    y=related_counts
                )
            ],

            'layout': {
                'title': 'Distribution of categories',
                'yaxis': {
                    'title': "<b>Count</b>"
                },
                'xaxis': {
                    'title': "<b>categories</b>",
                    'tickangle': 30
                    
                }
                
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    '''save user input in query
    
    Arge:
      None
      
    Returns:
      None
    '''
    
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


#def main():
#    app.run(host='0.0.0.0', port=3000, debug=True)


#if __name__ == '__main__':
#    main()
