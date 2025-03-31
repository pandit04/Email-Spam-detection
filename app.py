import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
import sklearn

tfidf = pickle.load(open("vectorizer.pkl","rb"))
model = pickle.load(open("model.pkl","rb"))

st.title("Email Spam Detector")

input_mail = st.text_area("Enter Email Below!")

def converted(text):
    ## lower case
    text = text.lower()
    
    ## Word Tokenize
    text = nltk.word_tokenize(text)
    
    ## Remove special characters    
    z = []
    for i in text:
        if i.isalnum() and i not in stopwords.words('english') and i not in string.punctuation:
            z.append(i)
            
    return " ".join(z)


if st.button("Predict", type = "primary"):
    #1. Preprocess 
    converted_mail = converted(input_mail)

    #2. Vectorize
    vector_input = tfidf.transform([converted_mail])

    #3. Predict 
    result = model.predict(vector_input)[0]

    #4. Display
    if result == 0:
        st.header("It is a _Ham_ Mail.")
    else:
        st.header("It is a _Spam_ Mail.")
