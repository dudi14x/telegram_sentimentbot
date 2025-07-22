import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords

# Download stopwords only once
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load model components safely
@st.cache_resource
def load_components():
    model = joblib.load("sentiment_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
    return model, vectorizer, label_encoder

# ✅ Try/except to reveal any errors in model loading
try:
    model, vectorizer, label_encoder = load_components()
except Exception as e:
    st.error(f"❌ Failed to load model files: {e}")
    st.stop()

# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

# Streamlit UI
st.set_page_config(page_title="Sentiment Analyzer", layout="centered")
st.title("🧠 Sentiment Analysis App")
st.write("Enter a review or sentence and get the predicted sentiment.")

user_input = st.text_area("✍️ Your Text", height=150)

if st.button("🔍 Analyze Sentiment"):
    if not user_input.strip():
        st.warning("⚠️ Please enter some text.")
    else:
        cleaned = clean_text(user_input)
        vect_input = vectorizer.transform([cleaned])
        prediction = model.predict(vect_input)
        sentiment = label_encoder.inverse_transform(prediction)[0]

        st.markdown(f"### 🎯 Predicted Sentiment: **{sentiment.capitalize()}**")
