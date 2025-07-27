# 🧠 Mental Health Emotion Detector

A smart and empathetic emotion detection web app that analyzes user text input to predict mental health-related emotions like **happy**, **sad**, **anxious**, **angry**, **depressed**, **love**, and **neutral**.

Built with `Streamlit`, this tool helps users reflect on their emotions while offering motivational feedback and a clean, mobile-friendly interface.

---

## 🚀 Demo

Try the live demo: [Streamlit Cloud Link Here](https://mental-health-predictor-7cnaov2v2p9q7qz3upeckx.streamlit.app/)

---

## 🌟 Features

- ✅ Emotion detection using ML (Logistic Regression + TF-IDF)
- 🎨 Beautiful and responsive UI
- 📊 Pie chart of emotion distribution
- 📜 Shows history of last 5 predictions
- ❤️ Emojis + motivational quotes per emotion
- 🖼️ Custom background and logo support
- 📱 Mobile-friendly design

---

## 🧠 How It Works

1. User enters a thought or feeling in text.
2. Model vectorizes the input using `TF-IDF`.
3. A trained classifier predicts the emotion.
4. The app displays:
   - Emotion with emoji
   - Confidence %
   - Motivational quote
   - History + pie chart

---

## 🗂️ Project Structure

Mental health predictor/
├── app.py
├── model.pk1
├── vectorizer.pk1
├── label_encoder.pk1
├── background.jpg
├── requirements.txt
└── README.md

---

## 📦 requirements.txt

streamlit
pandas
scikit-learn
joblib
matplotlib
Pillow

---

## 🌐 Deployment (Streamlit Cloud)

Push all files to a public GitHub repo.
Go to streamlit.io/cloud and click "New app".
Connect your GitHub repo.
Set app.py as the entry file.
Deploy!
✅ Make sure model.pk1, vectorizer.pk1, and label_encoder.pk1 are all in the root of the repo.

---

## 📬 Contact

Want to connect or give feedback?

📧 Gmail – satwikabehara99@gmail.com  
🔗 [LinkedIn](https://www.linkedin.com/in/satwika-behara-457578353/)  
🐙 [GitHub](https://github.com/SatwikaBehara99)  

---

# ❤️ Acknowledgements
Streamlit Team
Scikit-learn
Mental Health Dataset

---

#📜 License
MIT License. Use responsibly and ethically.
