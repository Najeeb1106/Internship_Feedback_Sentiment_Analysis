import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
import torch
import shutil
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from transformers import (
    DistilBertTokenizerFast, 
    DistilBertForSequenceClassification, 
    Trainer, 
    TrainingArguments,
    EarlyStoppingCallback
)
from datasets import Dataset

# --- 1. Configuration & Setup ---
DATA_PATH = "../data/internship_feedback_4000_unique.csv"
SAVE_PATH = "../models/finetuned_distilbert"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Using device: {DEVICE}")

# Set visual style
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# --- 2. Load & Explore Dataset ---
print("📊 Loading dataset...")
df = pd.read_csv(DATA_PATH)

# Check for duplicates or missing values
print(f"Total entries: {len(df)}")
print(f"Missing values: {df.isnull().sum().sum()}")
print(f"Duplicate texts: {df['feedback_text'].duplicated().sum()}")

# Sentiment Distribution
plt.figure(figsize=(8, 5))
sns.countplot(x='sentiment_label', data=df, palette='viridis')
plt.title('Sentiment Distribution (Balanced 1:1)')
plt.savefig('01_sentiment_distribution.png')
print("✅ Saved: 01_sentiment_distribution.png")

# --- 3. Preprocessing ---
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text) # Remove URLs
    text = re.sub(r'<.*?>', '', text)               # Remove HTML tags
    text = re.sub(r'[^a-z\s]', '', text)            # Remove non-alphabetic
    text = re.sub(r'\s+', ' ', text).strip()        # Remove extra spaces
    return text

print("🧹 Preprocessing text...")
df['clean_feedback'] = df['feedback_text'].apply(clean_text)

# Map labels to integers
label2id = {"negative": 0, "positive": 1}
id2label = {0: "negative", 1: "positive"}
df['label'] = df['sentiment_label'].map(label2id)

# Train-Val-Test Split (70/15/15)
X_train_full, X_test, y_train_full, y_test = train_test_split(
    df['clean_feedback'], df['label'], test_size=0.15, random_state=42, stratify=df['label']
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.1765, random_state=42, stratify=y_train_full
)

print(f"Train size: {len(X_train)} | Val size: {len(X_val)} | Test size: {len(X_test)}")

# --- 4. Logistic Regression Baseline ---
print("🤖 Training Logistic Regression baseline...")
tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train_tfidf, y_train)

lr_preds = lr_model.predict(X_test_tfidf)
lr_acc = accuracy_score(y_test, lr_preds)
print(f"✅ Logistic Regression Accuracy: {lr_acc:.4f}")

# Confusion Matrix for LR
cm_lr = confusion_matrix(y_test, lr_preds)
plt.figure(figsize=(6, 5))
sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Greens', 
            xticklabels=id2label.values(), yticklabels=id2label.values())
plt.title('Confusion Matrix: Logistic Regression')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.savefig('07_cm_logistic.png')

# --- 5. DistilBERT Fine-Tuning ---
print("🔥 Preparing DistilBERT fine-tuning...")
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

def tokenize_function(examples):
    return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=128)

# Convert to HuggingFace Dataset format
train_ds = Dataset.from_dict({'text': X_train.tolist(), 'label': y_train.tolist()})
val_ds   = Dataset.from_dict({'text': X_val.tolist(), 'label': y_val.tolist()})
test_ds  = Dataset.from_dict({'text': X_test.tolist(), 'label': y_test.tolist()})

tokenized_train = train_ds.map(tokenize_function, batched=True)
tokenized_val   = val_ds.map(tokenize_function, batched=True)
tokenized_test  = test_ds.map(tokenize_function, batched=True)

# Load Model
model = DistilBertForSequenceClassification.from_pretrained(
    'distilbert-base-uncased', 
    num_labels=2,
    id2label=id2label,
    label2id=label2id
).to(DEVICE)

# Define Metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    f1_macro = f1_score(labels, predictions, average='macro')
    return {"accuracy": acc, "f1_macro": f1_macro}

# Training Arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=100,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=50,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
    fp16=torch.cuda.is_available(), # Use FP16 if GPU is available
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

print("⏳ Training model (this may take a while)...")
trainer.train()

# --- 6. Evaluation & Comparison ---
print("📈 Evaluating models...")
eval_results = trainer.evaluate(tokenized_test)
bert_ft_acc = eval_results['eval_accuracy']
print(f"✅ Fine-Tuned DistilBERT Accuracy: {bert_ft_acc:.4f}")

# Get predictions for confusion matrix
raw_preds = trainer.predict(tokenized_test)
bert_ft_preds = np.argmax(raw_preds.predictions, axis=-1)

cm_bert = confusion_matrix(y_test, bert_ft_preds)
plt.figure(figsize=(6, 5))
sns.heatmap(cm_bert, annot=True, fmt='d', cmap='Purples', 
            xticklabels=id2label.values(), yticklabels=id2label.values())
plt.title('Confusion Matrix: Fine-Tuned DistilBERT')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.savefig('10_cm_bert_finetuned.png')

# Model Comparison Plot
models     = ['Logistic Regression\n(TF-IDF)', 'Fine-Tuned DistilBERT']
accuracies = [lr_acc, bert_ft_acc]
plt.figure(figsize=(9, 6))
sns.barplot(x=models, y=accuracies, palette=['#4CAF50', '#9C27B0'])
plt.ylim(0.5, 1.0)
plt.title('Model Comparison: Accuracy')
plt.ylabel('Accuracy Score')
for i, acc in enumerate(accuracies):
    plt.text(i, acc + 0.01, f'{acc:.2%}', ha='center', fontweight='bold')
plt.savefig('11_model_comparison.png')

# Save the model
print(f"💾 Saving model to {SAVE_PATH}...")
trainer.save_model(SAVE_PATH)
tokenizer.save_pretrained(SAVE_PATH)

# --- 7. Department Insights ---
print("🔍 Generating department insights...")
dept_neg = df.groupby('department').apply(
    lambda x: (x['sentiment_label'] == 'negative').mean() * 100
).sort_values(ascending=False)

plt.figure(figsize=(12, 6))
dept_neg.plot(kind='barh', color='salmon')
plt.title('Negative Feedback % by Department')
plt.xlabel('Percentage (%)')
plt.tight_layout()
plt.savefig('13_department_insights.png')

print("✨ All tasks completed successfully!")
