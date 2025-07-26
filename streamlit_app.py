import streamlit as st
import pandas as pd
import numpy as np
import re
import string
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns

st.title("Tweet Sentiment Analyzer")
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    words = text.split()
    return ' '.join(words)

def get_sentiment(text):
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0.1:
        return 'positive'
    elif polarity < -0.1:
        return 'negative'
    else:
        return 'neutral'


st.header("Single Tweet/Test Input")
user_input = st.text_area("Paste a tweet or some text here:", "")
if st.button("Analyze Sentiment"):
    if user_input.strip():
        cleaned = clean_text(user_input)
        sentiment = get_sentiment(cleaned)
        st.markdown(f"**Predicted Sentiment:** `{sentiment}`")
    else:
        st.warning("Please enter some text.")


st.header("Tweet Analysis (CSV Upload)")
st.write("Upload a CSV with a column named `text` for sentiment tweet analysis.")

uploaded_file = st.file_uploader("Choose CSV file", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    if 'text' in df.columns:
        df['clean_text'] = df['text'].apply(clean_text)
        df['predicted_sentiment'] = df['clean_text'].apply(get_sentiment)
        sentiment_counts = df['predicted_sentiment'].value_counts().reindex(['positive','negative','neutral'], fill_value=0)
        st.write("Sentiment Counts:")
        st.bar_chart(sentiment_counts)
        st.write(df[['text', 'predicted_sentiment']].head())
        if 'sentiment' in df.columns:
            from sklearn.metrics import confusion_matrix
            import matplotlib.pyplot as plt
            import seaborn as sns
            y_true = df['sentiment']
            y_pred = df['predicted_sentiment']
            labels = ['positive','negative','neutral']
            cm = confusion_matrix(y_true, y_pred, labels=labels)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, xticklabels=labels, yticklabels=labels, ax=ax)
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.title('Confusion Matrix')
            st.pyplot(fig)
    else:
        st.warning("CSV must contain a `text` column!")

