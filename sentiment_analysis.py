import streamlit as st
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import plotly.express as px
from openai import OpenAI

# CONFIGURATION
# Set Streamlit page configuration with a title and wide layout for better UI
st.set_page_config(page_title="Sentiment Analysis Dashboard", layout="wide")

# Retrieve OpenAI API key from environment variables for secure authentication
API_KEY = os.getenv("OPENAI_API_BOOK_KEY")

# Specify the fine-tuned GPT-3.5 Turbo model ID to be used for sentiment predictions
fine_tuned_model = "ft:gpt-3.5-turbo-0125:personal::Bvd5KAKa"

# Initialize OpenAI client with the API key to interact with OpenAI services
client = OpenAI(api_key=API_KEY)

# LOAD DATA
@st.cache_data
def load_data():
    # Cache this function to avoid reloading data on every app interaction for performance
    # Load sentiment dataset from a CSV file into a pandas DataFrame
    df = pd.read_csv("train_df.csv")
    return df

# Load dataset once into the DataFrame 'df'
df = load_data()

# APP UI
# Set the main title of the dashboard
st.title("📊 Sentiment Analysis Dashboard - Fine-Tuned GPT-3.5 Turbo")

# Provide a brief description of the dashboard purpose
st.markdown("""
This dashboard explores the sentiment dataset and interact with a fine-tuned GPT model 
for sentiment prediction.
""")

# Create three tabs for organized UI: data overview, visualizations, and model interaction
tab_overview, tab_graphs, tab_predict = st.tabs(["📈 Data Overview", "📊 Visualizations", "🤖 Model Prediction"])

# TAB 1: Data Overview
with tab_overview:
    st.subheader("Dataset Preview")
    # Display the first few records of the dataset for user to inspect
    st.dataframe(df.head())

    # Show a table mapping label codes to sentiment categories for clarity
    st.markdown("### Sentiment Class Mapping")
    st.table({
        "Label Code": [0, 1, 2],
        "Sentiment": ["Negative", "Neutral", "Positive"]
    })

    # Display summary statistics: total records and counts per sentiment
    st.write(f"**Total Records:** {len(df)}")
    st.write(df['sentiment'].value_counts())

# TAB 2: Visualizations
with tab_graphs:
    st.subheader("Exploratory Data Analysis")

    # 1. Sentiment Distribution
    st.markdown("#### 1. Sentiment Distribution")
    # Count how many samples per sentiment to understand dataset balance
    sentiment_counts = df['sentiment'].value_counts().reset_index()
    sentiment_counts.columns = ['Sentiment', 'Count']
    # Create an interactive bar chart using Plotly to visualize sentiment counts
    fig1 = px.bar(sentiment_counts, x='Sentiment', y='Count', color='Sentiment',
                  title="Sentiment Distribution in Dataset")
    st.plotly_chart(fig1, use_container_width=True)

    # 2. Sentiment Trend Over Index
    st.markdown("#### 2. Sentiment Trend Over Dataset Index")
    # Plot sentiment labels over dataset index to observe any ordering effects or trends
    fig2 = px.line(df.reset_index(), x='index', y='label', title='Sentiment Trend by Sample Order')
    st.plotly_chart(fig2, use_container_width=True)

    # 3. Text Length vs Sentiment
    st.markdown("#### 3. Text Length vs Sentiment")
    # Calculate text length for each record for further analysis of textual features
    df['text_length'] = df['text'].apply(len)
    # Box plot comparing text length distributions across sentiment classes to examine differences
    fig3 = px.box(df, x='sentiment', y='text_length', color='sentiment',
                  title="Text Length Distribution by Sentiment")
    st.plotly_chart(fig3, use_container_width=True)

    # 4. Confusion Matrix (Simulated Example for Demo)
    st.markdown("#### 4. Confusion Matrix (Example - Random Predictions for Demo)")
    # For demonstration: generate random sentiment predictions as a baseline example
    np.random.seed(42)
    fake_preds = np.random.choice(df['sentiment'].unique(), size=len(df))
    # Create confusion matrix to compare true vs predicted labels (here random)
    cm = confusion_matrix(df['sentiment'], fake_preds, labels=['negative','neutral','positive'])
    # Plot confusion matrix heatmap using seaborn for easy visualization
    fig_cm, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Neg','Neu','Pos'], yticklabels=['Neg','Neu','Pos'], ax=ax)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    # Display the confusion matrix plot in the app
    st.pyplot(fig_cm)

# TAB 3: Model Prediction
with tab_predict:
    st.subheader("Real-Time Sentiment Prediction")
    # Text area for user to input text for sentiment analysis
    user_input = st.text_area("Enter text for sentiment analysis:", height=100)

    if st.button("Analyze Sentiment"):
        if user_input.strip():
            try:
                # Call OpenAI fine-tuned model for real-time sentiment prediction
                response = client.chat.completions.create(
                    model=fine_tuned_model,
                    messages=[
                        {"role": "system", "content": "You are a sentiment analysis assistant."},
                        {"role": "user", "content": user_input}
                    ]
                )
                # Extract predicted sentiment from the model's response
                sentiment = response.choices[0].message.content
                # Show the predicted sentiment as success message
                st.success(f"Predicted Sentiment: {sentiment}")
            except Exception as e:
                # Show error if model call fails for troubleshooting
                st.error(f"Error: {e}")
        else:
            # Warn user to enter input text before prediction
            st.warning("Please enter some text.")

# Footer separator and caption for dashboard
st.markdown("---")
st.caption("Sentiment Analysis Dashboard using Fine-Tuned GPT-3.5 Turbo")