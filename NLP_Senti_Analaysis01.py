# ---------------------------------------------------------------
# AI Echo: Streamlit Dashboard for Sentiment Analysis
# ---------------------------------------------------------------

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import joblib
import re

# ---------------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------------
st.set_page_config(page_title="AI Echo - Sentiment Dashboard", layout="wide")

st.title("🤖 AI Echo: Your Smartest Conversational Partner")
st.markdown("### Sentiment Analysis Dashboard for ChatGPT User Reviews")

# ---------------------------------------------------------------
# LOAD MODEL + EMBEDDER
# ---------------------------------------------------------------
try:
    model = joblib.load("sentiment_logreg_embeddings.pkl")
    embedder = joblib.load("sentence_transformer_model.pkl")
    st.success("✅ Model and Embedding loaded successfully!")
except:
    st.warning("⚠️ Model or embedder not found. Prediction section will be disabled.")
    model = None
    embedder = None

# ---------------------------------------------------------------
# DATA UPLOAD SECTION
# ---------------------------------------------------------------
uploaded_file = st.file_uploader("📂 Upload your review dataset (.xlsx or .csv):", type=["xlsx", "csv"])
if uploaded_file:
    if uploaded_file.name.endswith(".xlsx"):
        df = pd.read_excel(uploaded_file)
    else:
        df = pd.read_csv(uploaded_file)
else:
    df = pd.read_excel("chatgpt_style_reviews_dataset.xlsx")

st.success(f"✅ Dataset Loaded Successfully! Shape: {df.shape}")
st.dataframe(df.head())

# ---------------------------------------------------------------
# BASIC CLEANING
# ---------------------------------------------------------------
df = df.dropna(subset=["review"])
df["rating"] = df["rating"].astype(int)

def rating_to_sentiment(r):
    if r <= 2:
        return "negative"
    elif r == 3:
        return "neutral"
    else:
        return "positive"

df["sentiment"] = df["rating"].apply(rating_to_sentiment)

# ---------------------------------------------------------------
# SECTION 1: SENTIMENT DISTRIBUTION
# ---------------------------------------------------------------
st.subheader("📊 Sentiment Distribution")
sent_counts = df["sentiment"].value_counts()
col1, col2 = st.columns(2)
with col1:
    st.bar_chart(sent_counts)
with col2:
    plt.figure(figsize=(4, 4))
    plt.pie(sent_counts, labels=sent_counts.index, autopct="%1.1f%%", colors=["red", "gold", "green"])
    st.pyplot(plt.gcf())

# ---------------------------------------------------------------
# SECTION 2: PLATFORM-WISE AVERAGE RATING
# ---------------------------------------------------------------
if "platform" in df.columns:
    st.subheader("💻 Platform-wise Average Rating")
    platform_avg = df.groupby("platform")["rating"].mean().reset_index()
    plt.figure(figsize=(6, 4))
    sns.barplot(data=platform_avg, x="platform", y="rating", palette="coolwarm")
    plt.title("Average Rating by Platform")
    st.pyplot(plt.gcf())

# ---------------------------------------------------------------
# SECTION 3: WORDCLOUDS
# ---------------------------------------------------------------
st.subheader("☁️ Word Clouds by Sentiment")
pos_text = " ".join(df[df["sentiment"] == "positive"]["review"].astype(str))
neg_text = " ".join(df[df["sentiment"] == "negative"]["review"].astype(str))

col3, col4 = st.columns(2)
with col3:
    st.markdown("#### 👍 Positive Reviews")
    wc = WordCloud(width=600, height=400, background_color="white", colormap="Greens").generate(pos_text)
    st.image(wc.to_array())
with col4:
    st.markdown("#### 👎 Negative Reviews")
    wc = WordCloud(width=600, height=400, background_color="white", colormap="Reds").generate(neg_text)
    st.image(wc.to_array())

# ---------------------------------------------------------------
# SECTION 4: AVERAGE RATING OVER TIME (FIXED)
# ---------------------------------------------------------------
if "date" in df.columns:
    st.subheader("📆 Average Rating Over Time")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    date_avg = df.groupby(df["date"].dt.to_period("M"))["rating"].mean()
    date_avg.index = date_avg.index.to_timestamp()  # ✅ FIXED for Matplotlib

    plt.figure(figsize=(8, 4))
    plt.plot(date_avg.index, date_avg.values, marker='o', linestyle='-', color='teal')
    plt.title("Average Rating Over Time")
    plt.xlabel("Month")
    plt.ylabel("Average Rating")
    plt.grid(True)
    st.pyplot(plt.gcf())

# ---------------------------------------------------------------
# SECTION 5: LIVE SENTIMENT PREDICTOR (FIXED LOGIC)
# ---------------------------------------------------------------
st.subheader("💬 Try the Live Sentiment Predictor")

user_review = st.text_area("Enter a review to analyze sentiment:")

def clean_text(txt):
    txt = re.sub(r"[^a-zA-Z0-9\s]", "", txt.lower())
    return txt

# Map for numeric predictions (if model returns integers)
label_map = {0: "negative", 1: "neutral", 2: "positive"}

if st.button("Predict Sentiment"):
    if not model or not embedder:
        st.error("⚠️ Model files not found. Please check your folder and reload.")
    elif user_review.strip() == "":
        st.warning("Please enter a review text first!")
    else:
        cleaned = clean_text(user_review)
        vec = embedder.encode([cleaned])
        pred = model.predict(vec)[0]

        # ✅ Handle both numeric and string outputs
        if isinstance(pred, (int, float)):
            pred_label = label_map.get(int(pred), "neutral")
        else:
            pred_label = str(pred).lower().strip()

        if pred_label == "positive":
            st.success(f"✅ Predicted Sentiment: **{pred_label.upper()}**")
        elif pred_label == "negative":
            st.error(f"❌ Predicted Sentiment: **{pred_label.upper()}**")
        else:
            st.warning(f"⚪ Predicted Sentiment: **{pred_label.upper()}**")

# ---------------------------------------------------------------
# FOOTER
# ---------------------------------------------------------------
st.markdown("---")
