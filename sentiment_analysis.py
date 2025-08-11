import streamlit as st
import pandas as pd
import os
# import seaborn as sns
# import matplotlib.pyplot as plt
# import numpy as np
# from sklearn.metrics import confusion_matrix
import plotly.express as px
from openai import OpenAI

# CONFIGURATION
st.set_page_config(page_title="Sentiment Analysis Dashboard", layout="wide")
API_KEY = os.getenv("OPENAI_API_BOOK_KEY")
fine_tuned_model = "ft:gpt-3.5-turbo-0125:personal::Bvd5KAKa"

# Initialize OpenAI client
client = OpenAI(api_key=API_KEY)

# LOAD DATA
@st.cache_data
def load_data():
    df = pd.read_csv("train_df.csv")
    return df

df = load_data()

# APP UI
st.title("📊 Sentiment Analysis Dashboard - Fine-Tuned GPT-3.5 Turbo")
st.markdown("""
This dashboard explores the sentiment dataset and interact with a fine-tuned GPT model 
for sentiment prediction.
""")

tab_overview, tab_graphs, tab_predict = st.tabs(["📈 Data Overview", "📊 Visualizations", "🤖 Model Prediction"])


# TAB 1: Data Overview

with tab_overview:
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.markdown("### Sentiment Class Mapping")
    st.table({
        "Label Code": [0, 1, 2],
        "Sentiment": ["Negative", "Neutral", "Positive"]
    })

    st.write(f"**Total Records:** {len(df)}")
    st.write(df['sentiment'].value_counts())

# TAB 2: Visualizations

with tab_graphs:
    st.subheader("Exploratory Data Analysis")

    # 1. Sentiment Distribution
    st.markdown("#### 1. Sentiment Distribution")
    sentiment_counts = df['sentiment'].value_counts().reset_index()
    sentiment_counts.columns = ['Sentiment', 'Count']
    fig1 = px.bar(sentiment_counts, x='Sentiment', y='Count', color='Sentiment',
                  title="Sentiment Distribution in Dataset")
    st.plotly_chart(fig1, use_container_width=True)

    # 2. Sentiment Trend Over Index
    st.markdown("#### 2. Sentiment Trend Over Dataset Index")
    fig2 = px.line(df.reset_index(), x='index', y='label', title='Sentiment Trend by Sample Order')
    st.plotly_chart(fig2, use_container_width=True)

    # 3. Text Length vs Sentiment
    st.markdown("#### 3. Text Length vs Sentiment")
    df['text_length'] = df['text'].apply(len)
    fig3 = px.box(df, x='sentiment', y='text_length', color='sentiment',
                  title="Text Length Distribution by Sentiment")
    st.plotly_chart(fig3, use_container_width=True)

    # 4. Confusion Matrix (Simulated Example for Demo)
    # st.markdown("#### 4. Confusion Matrix (Example - Random Predictions for Demo)")
    # np.random.seed(42)
    # fake_preds = np.random.choice(df['sentiment'].unique(), size=len(df))
    # cm = confusion_matrix(df['sentiment'], fake_preds, labels=['negative','neutral','positive'])
    # fig_cm, ax = plt.subplots()
    # sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Neg','Neu','Pos'], yticklabels=['Neg','Neu','Pos'], ax=ax)
    # ax.set_xlabel("Predicted Label")
    # ax.set_ylabel("True Label")
    # st.pyplot(fig_cm)

# TAB 3: Model Prediction

with tab_predict:
    st.subheader("Real-Time Sentiment Prediction")
    user_input = st.text_area("Enter text for sentiment analysis:", height=100)

    if st.button("Analyze Sentiment"):
        if user_input.strip():
            try:
                response = client.chat.completions.create(
                    model=fine_tuned_model,
                    messages=[
                        {"role": "system", "content": "You are a sentiment analysis assistant."},
                        {"role": "user", "content": user_input}
                    ]
                )
                sentiment = response.choices[0].message.content
                st.success(f"Predicted Sentiment: {sentiment}")
            except Exception as e:
                st.error(f"Error: {e}")
        else:
            st.warning("Please enter some text.")

st.markdown("---")
st.caption("Sentiment Analysis Dashboard using Fine-Tuned GPT-3.5 Turbo")
