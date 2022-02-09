# import libraries
import pandas as pd
from sqlalchemy import create_engine
import sys


def load_data(messages_filepath, categories_filepath):
    '''
    Read the file paths for two csv files and load them
    into two pandas dataframe, and return merged them to dataframe
    
    Args:
      messages_filepath (str): name of csv datafile
      categories_filepath (str): name of csv datafile
      
    Returns:
      df (dataframe): merged dataframe
    '''
        
    # load messages dataset
    messages = pd.read_csv(messages_filepath) 
    # load categories dataset
    categories = pd.read_csv(categories_filepath)
    # merge datasets
    df = messages.merge(categories, on='id')
    return df


    
    
    
def clean_data(df):
    ''' 
    Cleaning the raw data and convert them into usable data. 
    
    Args:
      df (dataframe): name of dataframe
      
    Returns:
      df (dataframe): name of dataframe
    '''
    
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(pat=';',  expand=True)
    # select the first row of the categories dataframe
    row = categories.loc[0]
    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = row.apply(lambda x: x[0:-2])

    # rename the columns of `categories`
    categories.columns = category_colnames

    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: x[-1])

        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df.drop(columns='categories'), categories],axis=1)

    # drop duplicates
    df = df.drop_duplicates(subset=['id'])
    return df

def save_data(df, table_name, database_filename):
    ''' 
    Saving data in SQL database
    
    Args:
      df (dataframe): name of dataframe
      table_name (str): the name of the table
      database_filename: the name of the sql database
      
    Returns:
      None
    '''
    
    # Save the clean dataset into an sqlite database
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql(table_name, engine, index=False, if_exists='replace', chunksize=100 )


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, 'messages',database_filepath)
        
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
