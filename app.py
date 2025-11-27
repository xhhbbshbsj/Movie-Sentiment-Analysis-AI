import streamlit as st
import tensorflow as tf
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# --- 1. SETUP & LOAD ---
st.set_page_config(page_title="AI Movie Critic", page_icon="🍿")

@st.cache_resource
def load_resources():
    # Load the Model
    model = tf.keras.models.load_model('sentiment_model.keras')
    
    # Load the Tokenizer (Dictionary)
    # We re-fit it quickly so the app knows which numbers belong to which words
    dataset = pd.read_csv('IMDB Dataset.csv')
    vocab_size = 10000
    tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
    tokenizer.fit_on_texts(dataset['review'])
    
    return model, tokenizer

st.write("Loading AI Brain... (This takes 5 seconds)")
model, tokenizer = load_resources()

# --- 2. THE UI ---
st.title("🍿 AI Movie Sentiment Analyzer")
st.markdown("Type a movie review below, and the AI will detect your emotion.")

# Text Input
user_review = st.text_area("Write your review here:", height=150, placeholder="Example: The movie was terrible but the popcorn was good.")

# --- 3. PREDICTION LOGIC ---
if st.button("Analyze Sentiment"):
    if user_review:
        # 1. Tokenize (Convert words to numbers)
        # We put it in a list [] because the model expects a batch of data
        sequences = tokenizer.texts_to_sequences([user_review])
        
        # 2. Pad (Ensure it is 200 length)
        padded = pad_sequences(sequences, maxlen=200, padding='post', truncating='post')
        
        # 3. Predict
        prediction = model.predict(padded)
        score = prediction[0][0] # The probability (0.0 to 1.0)
        
        st.write("---")
        
        # 4. Display Result
        if score > 0.5:
            # Positive
            confidence = score * 100
            st.success(f"😊 **POSITIVE Review**")
            st.write(f"Confidence: **{confidence:.2f}%**")
            st.balloons()
        else:
            # Negative
            confidence = (1 - score) * 100
            st.error(f"😡 **NEGATIVE Review**")
            st.write(f"Confidence: **{confidence:.2f}%**")
    else:
        st.warning("Please type a review first!")