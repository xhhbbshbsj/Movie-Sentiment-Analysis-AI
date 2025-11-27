import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

# ==========================================
# --- PHASE 1: LOAD & PREPARE DATA ---
# ==========================================
print("Step 1: Loading data...")
dataset = pd.read_csv('IMDB Dataset.csv')

# Map 'positive' to 1 and 'negative' to 0
dataset['sentiment'] = dataset['sentiment'].map({'positive': 1, 'negative': 0})

# ==========================================
# --- PHASE 2: TOKENIZATION & PADDING ---
# ==========================================
print("Step 2: Preprocessing text...")
vocab_size = 10000
tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(dataset['review'])

sequences = tokenizer.texts_to_sequences(dataset['review'])

max_length = 200
padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post', truncating='post')

# Split Data
X_train, X_test, y_train, y_test = train_test_split(padded_sequences, dataset['sentiment'], test_size=0.2)
print(f"Data Ready! Training shape: {X_train.shape}")

# ==========================================
# --- PHASE 3: BUILD LSTM MODEL ---
# ==========================================
print("Step 3: Building the Brain...")
model = tf.keras.Sequential([
    # Embedding: Understands the meaning of words
    tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=128, input_length=max_length),
    
    # LSTM: Reads the sentence sequence (Memory)
    tf.keras.layers.LSTM(128),
    
    # Dense: Decides Positive (1) or Negative (0)
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# ==========================================
# --- PHASE 4: TRAIN & SAVE ---
# ==========================================
print("Step 4: Training (This will take a few minutes)...")
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train for 5 epochs
history = model.fit(X_train, y_train, epochs=5, batch_size=64, validation_data=(X_test, y_test))

# Save
model.save('sentiment_model.keras')
print("✅ Success! Model saved as 'sentiment_model.keras'")