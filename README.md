# 🍿 AI Movie Sentiment Analysis

An NLP project that uses Deep Learning (LSTM) to understand human emotion in text.

## 🚀 Overview
This application takes a written movie review and predicts whether the sentiment is **Positive** or **Negative** using a Recurrent Neural Network.

## 🧠 Model Architecture
- **Embedding Layer:** Converts words into dense vectors to capture semantic meaning.
- **LSTM Layer:** A Recurrent Neural Network (RNN) that understands the sequence of words.
- **Dense Layer:** Outputs a probability (Positive vs Negative).

## 🛠️ Tech Stack
- **Deep Learning:** TensorFlow / Keras
- **Data Processing:** Pandas, NumPy, Scikit-Learn
- **Interface:** Streamlit
- **Data Source:** IMDB Dataset (50k Movie Reviews)

## 📊 Performance
- Trained on 50,000 reviews.
- Achieved **~85%+ Accuracy** on unseen test data.

## 📸 How to Run
1. Clone the repository.
2. Install dependencies: `pip install tensorflow streamlit pandas`
3. Run the app: `streamlit run app.py`