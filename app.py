import streamlit as st
import pickle
import string
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

import re
from nltk.stem.porter import PorterStemmer



ps = PorterStemmer()
def text_transform(text):
    text = text.lower()

    # Define the regular expression pattern for special characters
    pattern = re.compile(r'[^a-z0-9\s]')

    # Replace special characters with an empty string
    text = re.sub(pattern, '', text)

    list = []

    for i in text.split():
        if (i not in stopwords.words("english")):
            list.append(i)
    text = list[:]
    list.clear()

    for i in text:
        list.append(ps.stem(i))

    text = list[:]
    list.clear()
    return " ".join(text)

cv = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):

    # 1. preprocess
    transformed_sms = text_transform(input_sms)
    # 2. vectorize
    vector_input = cv.transform([transformed_sms]).toarray()
    # 3. predict
    result = model.predict(vector_input)[0]
    # 4. Display
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")

