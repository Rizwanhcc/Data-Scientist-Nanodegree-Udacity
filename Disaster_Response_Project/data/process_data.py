import sys

import pandas as pd
import numpy as np
import sqlite3
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    This function loads two files as input and return 
    merged files
    """
    #load messages file
    messages = pd.read_csv(messages_filepath)
    #load categories file
    categories = pd.read_csv(categories_filepath)
    #merge these two datasets
    df = messages.merge(categories, how='inner', on= 'id')
    
    return df



def clean_data(df):
    """
    INPUT: 
        This function takes merged dataframe as input 
        Process the data by spliting categroies into individual columns
        converts categories to binary values
        replaces old categories columns with new ones
        and drops duplicates
    
    RETURNS: 
        Returns dataset after cleaning
    """
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';', expand=True)
    # select the first row of the categories dataframe
    row = categories.iloc[0]
    category_colnames =  row.apply(lambda x: x.split('-')[0])
    print(category_colnames)
    # rename the columns of `categories`
    categories.columns = category_colnames
    #convert to numeric
    for column in categories:
    # set each value to be the last character of the string
        categories[column]=categories[column].str[-1]
    # convert column from string to numeric
        categories[column]=categories[column].apply(pd.to_numeric)
    
    categories['related'] = categories['related'].replace(2, 1)
    # drop the original categories column from `df`
    df.drop('categories', axis = 1, inplace = True)
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)
    # drop duplicates
    df = df.drop_duplicates(keep='first')
    
    return df

def save_data(df, database_filename):
    """
    Function:
        This function takes input the cleaned data file
        and saves into sql database.
     """
    engine =  create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('Disaster_cleaned', engine, index=False, if_exists='replace')


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