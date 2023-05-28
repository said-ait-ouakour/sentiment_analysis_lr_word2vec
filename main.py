import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
import joblib
from sklearn.preprocessing import StandardScaler
from gensim.models import Word2Vec
import warnings
import argparse

from logistic_regression import LogisticRegression

warnings.filterwarnings("ignore", category=UserWarning)

# nltk.download('wordnet')
# nltk.download('stopwords')
# nltk.download('punkt')

# set of stopwords
stop_words = set(stopwords.words('english'))

from nltk.corpus import stopwords
import matplotlib.pyplot as plt

def tweet_cleaner_without_stopwords(text):

    new_text = re.sub(r"'s\b"," is ", text)
    new_text = re.sub("#", "", new_text)
    new_text = re.sub("@[A-Za-z0-9]+", "", new_text)
    new_text = re.sub(r"http\S+", "", new_text)
    new_text = re.sub(r"[^a-zA-Z]", " ", new_text)    
    new_text = new_text.lower().strip()
    
    cleaned_text = ''
    for token in new_text.split(): 
      
          lemmatizer = WordNetLemmatizer()
          cleaned_text = cleaned_text + lemmatizer.lemmatize(token) + ' '
    
    return cleaned_text

def document_vector(doc):
    """Create document vectors by averaging word vectors. Remove out-of-vocabulary words."""
  
    cbow_model = Word2Vec.load("cbow_model.model")
    
    # doc1 contains those words of the document which are included in the vocab
    doc1 = [word for word in doc.split() if word in cbow_model.wv.index_to_key]
    
    wv1 = []  
    for word in doc1:
        wv1.append(cbow_model.wv.get_vector(word))
    wv1_ = np.array(wv1)
    wv1_mean = wv1_.mean(axis=0)
    return wv1_mean

def preprocess_text(text):
    cleaned_text = tweet_cleaner_without_stopwords(text)  # Apply the same preprocessing steps as during training
    doc_vector = document_vector(cleaned_text)  # Convert the preprocessed text to a document vector
    return doc_vector

def main():
    
    parser = argparse.ArgumentParser(
                        prog='Document Similarity Checker',
                        description='This program checks for similarity on two documents takes documents as input and returns similarity results',
                        epilog='Similarity Checker')
        # document file
    parser.add_argument('-s', '--sentence', help='phrase to analyze', type=str)
       
    lr= joblib.load("joblib_RL_Model.pkl")
    
    # Example usage
    new_text = "i'll kill you "
    
    args = parser.parse_args()
    
    if args.sentence:
        
        # Preprocess the new text
        cleaned_text = preprocess_text(args.sentence)
        
        preprocessed_text = cleaned_text.reshape(1, -1)

        # Make predictions using the logistic regression model
        prediction = lr.predict(preprocessed_text)

        # Print the predicted sentiment
        if prediction == 0:
            print("Positive sentiment 😊")
        else:
            print("Negative sentiment 🥺")

    else : 
        parser.error("sentence cannot be empty")
    
if __name__ == "__main__":
    main()