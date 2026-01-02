import streamlit as st
import joblib

# Load trained model
model = joblib.load("sentiment_model.pkl")

st.title("Sentiment Analysis App")
st.write("Enter a sentence to analyze sentiment")

user_input = st.text_area("Your Text")

if st.button("Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter some text")
    else:
        prediction = model.predict([user_input])[0]

    sentiment_map = {
        "positive": "😊 Positive Sentiment",
        "negative": "😡 Negative Sentiment",
        "neutral": "😐 Neutral Sentiment"
    }
    st.success(sentiment_map[prediction])
