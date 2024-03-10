import streamlit as st
import pickle
import sklearn
import re
import joblib
import nltk
import requests
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

model_url = 'https://github.com/Prashanth-1/spam_classfication/blob/main/Model.joblib'
tfidf_url = 'https://github.com/Prashanth-1/spam_classfication/blob/main/TFIDF.joblib'

response_model = requests.get(model_url)
response_tfidf = requests.get(tfidf_url)

with open('Model.joblib', 'wb') as file:
    file.write(response_model.content)

with open('TFIDF.joblib', 'wb') as file:
    file.write(response_tfidf.content)

# Load the model
model = joblib.load('Model.joblib')

# Load TF-IDF vectorizer
tfidf = joblib.load('TFIDF.joblib')


# Function to classify emails
def prediction(MSG: str):
    vectorized_email = tfidf.transform([MSG]).toarray()
    prediction = model.predict(vectorized_email)
    probabilities = model.predict_proba(vectorized_email)
    spam_probability = probabilities[0][1] * 100
    ham_probability = probabilities[0][0] * 100
    return prediction, spam_probability, ham_probability



# Page title and description
st.title('Message Classification App')
st.write('This app classifies Messages into spam or not spam.')


message = st.text_input('Enter your Message here')

if st.button("Predict"):
    result, spam_prob, ham_prob = prediction(message)
    if result == 1:
        st.write('This message is a Spam with a probability of {:.2f}%'.format(spam_prob))
    else:
        st.write('This message is not Spam with a probability of {:.2f}%'.format(ham_prob))

