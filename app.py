import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import base64

from PIL import Image
from io import BytesIO

# Load model, vectorizer, and label encoder
model = joblib.load("model.pk1")
vectorizer = joblib.load("vectorizer.pk1")
label_encoder = joblib.load("label_encoder.pk1")
st.set_page_config(page_title="Mental Health Emotion Detector", layout="centered")

# ----- Set Background -----
def set_background(image_file):
    with open(image_file, "rb") as image:
        encoded = base64.b64encode(image.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: linear-gradient(rgba(255,255,255,0.65), rgba(255,255,255,0.65)), 
                        url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_background("background.jpg")

# ----- Emotion Metadata -----
emotion_details = {
    "happy": {"emoji": "😊", "quote": "Happiness is not by chance, but by choice."},
    "sad": {"emoji": "😢", "quote": "Every storm runs out of rain."},
    "angry": {"emoji": "😠", "quote": "For every minute you remain angry, you give up sixty seconds of peace."},
    "anxious": {"emoji": "😰", "quote": "You don’t have to control your thoughts. You just have to stop letting them control you."},
    "depressed": {"emoji": "😔", "quote": "This too shall pass. You are stronger than you think."},
    "love": {"emoji": "❤️", "quote": "Where there is love, there is life."},
    "neutral": {"emoji": "😐", "quote": "Even a neutral state is a step toward clarity."}
}

# ----- Session Initialization -----
if "history" not in st.session_state:
    st.session_state.history = []

if "submitted" not in st.session_state:
    st.session_state.submitted = False

# ----- Title -----
st.markdown(f"""
    <div style='text-align:center;'>
        <h1 style='color:#222;'>🧠 Mental Health Emotion Detector</h1>
        <p style='color:#444; font-size:18px;'>Let your feelings flow. We’ll help you understand them.</p>
    </div>
""", unsafe_allow_html=True)

# ----- Input Box -----
st.markdown("### ✏️ How are you feeling right now?")
user_input = st.text_area("Type your thoughts here...", placeholder="Type your thoughts...", height=100)

col1, col2 = st.columns([1, 1])
with col1:
    analyze = st.button("🔍 Analyze")
with col2:
    reset = st.button("🔁 Reset")

# ----- Reset Functionality -----
if reset:
    st.session_state.history.clear()
    st.session_state.submitted = False
    st.rerun()

# ----- Prediction Flow -----
if analyze and user_input.strip():
    X = vectorizer.transform([user_input])
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)
    emotion = label_encoder.inverse_transform(y_pred)[0]
    confidence = np.max(y_proba)

    emoji = emotion_details[emotion]["emoji"]
    quote = emotion_details[emotion]["quote"]

    # Save in session history
    st.session_state.history.insert(0, {
        "Text": user_input,
        "Emotion": emotion,
        "Confidence": round(confidence * 100, 2)
    })

    st.session_state.submitted = True

# ----- Result Display -----
if st.session_state.submitted and st.session_state.history:
    last_result = st.session_state.history[0]
    st.markdown(f"""
        <div style="background-color: rgba(255, 255, 255, 0.85); padding: 20px; border-radius: 12px;">
            <h2 style="text-align:center; color:#111;">{emotion_details[last_result['Emotion']]['emoji']} Emotion Detected: 
            <span style="color:#d6336c;">{last_result['Emotion'].upper()}</span></h2>
            <p style="text-align:center; font-size:16px; color:#333;">
            Confidence Score: <b>{last_result['Confidence']:.2f}%</b></p>
            <p style="text-align:center; font-style: italic; font-size:18px; color:#111;">
            “{emotion_details[last_result['Emotion']]['quote']}”</p>
        </div>
    """, unsafe_allow_html=True)

# ----- History Table and Chart -----
if st.session_state.history:
    st.markdown("### 🕒 Last 5 Predictions")
    df = pd.DataFrame(st.session_state.history[:5])
    st.dataframe(df, use_container_width=True)

    st.markdown("### 📊 Emotion Distribution")
    emotion_counts = df["Emotion"].value_counts()
    fig, ax = plt.subplots()
    ax.pie(emotion_counts, labels=emotion_counts.index, autopct='%1.1f%%', startangle=90)
    ax.axis("equal")
    st.pyplot(fig)
