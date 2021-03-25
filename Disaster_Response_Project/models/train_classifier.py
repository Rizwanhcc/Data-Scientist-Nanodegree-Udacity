import sys
import pandas as pd
import numpy as np
import pickle
from sqlalchemy import create_engine
import re
import nltk 
nltk.download('punkt')
nltk.download('stopwords')
from nltk.stem import PorterStemmer
stemmer=PorterStemmer()
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import classification_report


def load_data(database_filepath):
    """
    FUNCTION:
        This function loads data from database
    INPUT: 
        It takes input as SQL database filepath
    OUTPUT: 
        It returns X (features) and Y (target) for ML
    """
    engine = create_engine('sqlite:///data/DisasterResponse.db')
    df = pd.read_sql_table('Disaster_cleaned', engine)
    X = df['message']
    #Y = df.drop(['id', 'message', 'original', 'genre'], axis=1)
    #X = df['message']
    Y = df.iloc[:, 4:]
    category_names = list(df.columns[4:])

    return X, Y, category_names
    


def tokenize(text):
    """
    FUNCTION:
        This is a function to create tokenized text:
    INPUT:
        It takes raw text as input
    OUTPUT:
        This function returns tokenized text after performing nlp steps.
    """
    #Step 1: normalize text, remove meta-characters
    text = re.sub(r'[^a-zA-Z0-9]', ' ', text.lower())

    #Step 2: Tokenize text
    tokens = word_tokenize (text)
                  
    #Step 3: Lemmatize/Stemming and stopword elimination
    #stemmed = [stemmer.stem(w) for w in tokens]
    lemmet_text = [WordNetLemmatizer().lemmatize(w) for w in tokens]
    #lemmet_text = [WordNetLemmatizer().lemmatize(w) for w in stop_words if w not in stop_words]
    stop_words = stopwords.words("english")
    cleaned_tokens = [token for token in lemmet_text if token not in stop_words]
    
    return cleaned_tokens



def build_model():
    """
    FUNCTION:
        This function builds an NLP ML classification pipeline
        It count words, tf-idf, multiple output classifier, 
        and uses a grid search to find the best parameters.
        
    RETURNS:
        It returns ML model described above.
    """
    model = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))])
    
    parameters = {
        'vect__ngram_range': ((1, 1), (1, 2)), # unigrams or bigrams
        'vect__max_df': (0.5, 0.75, 1.0),
        'tfidf__use_idf': (True, False)}

    cv = GridSearchCV(model, param_grid=parameters)
    #cv.fit(X_train, y_train)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
    This function takes input pipeline, xtest and ytest
    and returns a dataframe with three columns i.e.,
    f1 score, precision and recall
    all these columns evaluate the performance of the model.
    """
    y_pred = model.predict(X_test)
    # an empty list and looping over columns.. 
    evaluations = []
    for i in range(len(Y_test.columns)):
        evaluations.append([f1_score(Y_test.iloc[:, i].values, y_pred[:, i], average='micro'),
                             precision_score(Y_test.iloc[:, i].values, y_pred[:, i], average='micro'),
                             recall_score(Y_test.iloc[:, i].values, y_pred[:, i], average='micro')])
    #converting to dataframe
    evaluations = pd.DataFrame(evaluations, columns = ['F1_Score', 'Precision', 'Recall'],
                                index = Y_test.columns)   
    print(classification_report(Y_test, y_pred, target_names=category_names))



def save_model(model, model_filepath):
    """
    FUNCTION:
        This function saves the classification model to a pickle file
        It takes validated classification model as input, a filepath 
        and saves the model in picle file.
    """
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
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()


