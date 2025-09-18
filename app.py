import streamlit as st
import pickle
import joblib
import string
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

# Download required nltk data
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
# Initialize stemmer
port_stemmer = PorterStemmer()

# Load models and vectorizers
# --- Spam Classifier ---
tfidf_spam = pickle.load(open('vectorizer.pkl', 'rb'))
spam_model = pickle.load(open('model.pkl', 'rb'))

# --- Fake News Detector ---
vectorizer_news = joblib.load("vectorizer.jb")
news_model = joblib.load("lr_model.jb")

# ------------------ Preprocessing function ------------------
def clean_text(text):
    text = word_tokenize(text)  # Tokenize
    text = " ".join(text)
    text = [char for char in text if char not in string.punctuation]  # Remove punctuation
    text = ''.join(text)
    text = [char for char in text if char not in re.findall(r"[0-9]", text)]  # Remove numbers
    text = ''.join(text)
    text = [word.lower() for word in text.split() if word.lower() not in set(stopwords.words('english'))]
    text = ' '.join(text)
    text = list(map(lambda x: port_stemmer.stem(x), text.split()))
    return " ".join(text)

# ------------------ Streamlit UI ------------------
st.title("Text Classification App")

# Option to choose between tasks
app_mode = st.radio("Choose a Task:", ["📩 SMS Spam Classifier", "📰 Fake News Detector"])

if app_mode == "📩 SMS Spam Classifier":
    st.subheader("SMS Spam Classifier")
    input_sms = st.text_input("Enter the Message")

    if st.button("Predict Spam"):
        if input_sms.strip() == "":
            st.warning("⚠️ Please Enter Your Message !!!")
        else:
            # Preprocess
            transform_text = clean_text(input_sms)
            # Vectorize
            vector_input = tfidf_spam.transform([transform_text])
            # Prediction
            result = spam_model.predict(vector_input)
            # Output
            if result == 1:
                st.error("🚨 Spam Message")
            else:
                st.success("✅ Not Spam")

elif app_mode == "📰 Fake News Detector":
    st.subheader("Fake News Detector")
    inputn = st.text_area("Enter News Article:")

    if st.button("Check News"):
        if inputn.strip():
            transform_input = vectorizer_news.transform([inputn])
            prediction = news_model.predict(transform_input)
            if prediction[0] == 1:
                st.success("🟢 The News is Real!")
            else:
                st.error("🔴 The News is Fake!")
        else:
            st.warning("⚠️ Please enter some text to Analyze.")


