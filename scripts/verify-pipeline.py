# ... (same imports as train_sentiment.py)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
import torch
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from transformers import (
    DistilBertTokenizerFast, 
    DistilBertForSequenceClassification, 
    Trainer, 
    TrainingArguments
)
from datasets import Dataset

# --- 1. Configuration & Setup ---
DATA_PATH = "../data/internship_feedback_4000_unique.csv"
DEVICE = "cpu" # Force CPU for verification
print(f"🚀 [VERIFICATION MODE] Using device: {DEVICE}")

# --- 2. Load Dataset (Subset) ---
print("📊 Loading subset of dataset...")
df = pd.read_csv(DATA_PATH).sample(200, random_state=42) # Only 200 samples

# --- 3. Preprocessing ---
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text.strip()

df['clean_feedback'] = df['feedback_text'].apply(clean_text)
label2id = {"negative": 0, "positive": 1}
id2label = {0: "negative", 1: "positive"}
df['label'] = df['sentiment_label'].map(label2id)

X_train, X_test, y_train, y_test = train_test_split(df['clean_feedback'], df['label'], test_size=0.2, random_state=42)

# --- 4. Logistic Regression ---
print("🤖 Training Logistic Regression...")
tfidf = TfidfVectorizer(max_features=1000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)
lr_model = LogisticRegression().fit(X_train_tfidf, y_train)
print(f"✅ LR Accuracy: {accuracy_score(y_test, lr_model.predict(X_test_tfidf)):.4f}")

# --- 5. DistilBERT (1 Epoch, No save) ---
print("🔥 Verifying DistilBERT Pipeline...")
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
train_ds = Dataset.from_dict({'text': X_train.tolist(), 'label': y_train.tolist()})
test_ds  = Dataset.from_dict({'text': X_test.tolist(), 'label': y_test.tolist()})

def tokenize_function(examples):
    return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=64) # Shorter for speed

tokenized_train = train_ds.map(tokenize_function, batched=True)
tokenized_test  = test_ds.map(tokenize_function, batched=True)

model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2).to(DEVICE)

training_args = TrainingArguments(
    output_dir="./temp_results",
    num_train_epochs=1,
    per_device_train_batch_size=8,
    logging_steps=10,
    report_to="none"
)

trainer = Trainer(model=model, args=training_args, train_dataset=tokenized_train)
print("⏳ Running 1 epoch on subset...")
trainer.train()
print("✨ Pipeline verification complete!")
