# bert_sentiment_analysis.py
import pandas as pd
import torch
import transformers
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
print(f"Transformers version: {transformers.__version__}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
# ===== STEP 1: Data Loading =====
print("\n=== STEP 1: Loading Data ===")
import os
file_path = 'IMDB_Dataset.csv'
if not os.path.exists(file_path):
    print(f"Error: File not found at {os.path.abspath(file_path)}")
    print("Please ensure the CSV file is in the same directory as your script")
    exit()
df = pd.read_csv('IMDB_Dataset.csv')
# Synthetic mini dataset (20 samples)
# data = {
#     'review': [
#         "This movie was fantastic! The acting was superb.",
#         "Terrible plot and bad acting throughout.",
#         "I loved every minute of this film.",
#         "Worst movie I've ever seen.",
#         "The cinematography was beautiful.",
#         "Boring and predictable storyline.",
#         "A masterpiece of modern cinema.",
#         "Complete waste of time.",
#         "The performances were outstanding.",
#         "I couldn't stand this film.",
#         "Highly recommended for all audiences.",
#         "Painfully bad dialogue.",
#         "The director did an amazing job.",
#         "I fell asleep halfway through.",
#         "Perfect from start to finish.",
#         "Unbearably long and dull.",
#         "The soundtrack was incredible.",
#         "Not a single redeeming quality.",
#         "One of the best films this year.",
#         "I want my money back."
#     ],
#     'sentiment': [
#         "positive", "negative", "positive", "negative", "positive",
#         "negative", "positive", "negative", "positive", "negative",
#         "positive", "negative", "positive", "negative", "positive",
#         "negative", "positive", "negative", "positive", "negative"
#     ]
# }
# df = pd.DataFrame(data)
df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['review'].values,
    df['sentiment'].values,
    test_size=0.2,
    random_state=42
)

print(f"\nData loaded:")
print(f"- Training samples: {len(train_texts)}")
print(f"- Validation samples: {len(val_texts)}")
print(f"- Sample review: {train_texts[0][:50]}...")
print(f"- Corresponding label: {'Positive' if train_labels[0] == 1 else 'Negative'}")

# ===== STEP 2: Tokenization =====
print("\n=== STEP 2: Tokenizing Text ===")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize a sample first to see output
sample_text = "This movie was great!"
sample_tokens = tokenizer(sample_text, truncation=True, padding='max_length', max_length=16)
print(f"\nSample tokenization for: '{sample_text}'")
print(f"- Input IDs: {sample_tokens['input_ids']}")
print(f"- Attention Mask: {sample_tokens['attention_mask']}")

# Tokenize all data
train_encodings = tokenizer(list(train_texts), truncation=True, padding=True, max_length=128)
val_encodings = tokenizer(list(val_texts), truncation=True, padding=True, max_length=128)

# ===== STEP 3: Dataset Preparation =====
print("\n=== STEP 3: Creating PyTorch Dataset ===")
class IMDbDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = IMDbDataset(train_encodings, train_labels)
val_dataset = IMDbDataset(val_encodings, val_labels)

print("\nDataset created:")
print(f"- First training sample keys: {list(train_dataset[0].keys())}")
print(f"- Input IDs shape: {train_dataset[0]['input_ids'].shape}")

# ===== STEP 4: Model Initialization =====
print("\n=== STEP 4: Loading BERT Model ===")
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Test with a single sample
sample = train_dataset[0]
with torch.no_grad():
    outputs = model(input_ids=sample['input_ids'].unsqueeze(0),
                   attention_mask=sample['attention_mask'].unsqueeze(0))

print("\nModel initialized:")
print(f"- Sample output logits: {outputs.logits}")
print(f"- Predicted class: {'Positive' if outputs.logits.argmax().item() == 1 else 'Negative'}")
print(f"- Actual class: {'Positive' if sample['labels'].item() == 1 else 'Negative'}")
#
# ===== STEP 5: Configuring Training =====
print("\n=== STEP 5: Configuring Training ===")
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=1,  # Start with 1 epoch for testing
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    eval_strategy="steps",
    eval_steps=50,
    logging_dir='./logs',
    logging_steps=10,
    report_to="none",
    save_strategy="no",
    use_cpu=not torch.cuda.is_available()  # Updated parameter
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=lambda p: {'accuracy': accuracy_score(p.label_ids, p.predictions.argmax(axis=1))}
)

# ===== STEP 6: Execute Training =====
print("\n=== STEP 6: Starting Training ===")
trainer.train()

# ===== STEP 7: Quick Prediction Test =====
print("\n=== STEP 7: Making Predictions ===")
test_reviews = [
    "This film was absolutely wonderful!",
    "Terrible acting and boring plot.",
    "The movie was okay, not great but not bad either."
]

for review in test_reviews:
    inputs = tokenizer(review, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    prediction = outputs.logits.argmax().item()
    print(f"\nReview: {review}")
    print(f"Predicted sentiment: {'Positive' if prediction == 1 else 'Negative'}")
    print(f"Confidence scores: {torch.softmax(outputs.logits, dim=1).tolist()[0]}")

# ===== Enhanced Evaluation =====
print("\n=== Enhanced Evaluation ===")
from sklearn.metrics import classification_report

# Get all predictions
predictions = trainer.predict(val_dataset)
preds = predictions.predictions.argmax(-1)

# Detailed classification report
print("\nClassification Report:")
print(classification_report(val_labels, preds, target_names=['Negative', 'Positive']))

# Confusion Matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(val_labels, preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Negative', 'Positive'])
disp.plot()
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')
plt.show()

# ===== Model Saving =====
print("\n=== Saving Model ===")
model.save_pretrained("./sentiment_model")
tokenizer.save_pretrained("./sentiment_model")
print("Model and tokenizer saved to 'sentiment_model' directory")

# ===== Model Loading =====
print("\n=== Loading Model ===")
loaded_model = BertForSequenceClassification.from_pretrained("./sentiment_model")
loaded_tokenizer = BertTokenizer.from_pretrained("./sentiment_model")
print("Model successfully loaded!")


# ===== Interactive Prediction =====
def predict_interactive():
    while True:
        text = input("\nEnter a review (or 'quit' to exit): ")
        if text.lower() == 'quit':
            break

        inputs = loaded_tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        with torch.no_grad():
            outputs = loaded_model(**inputs)

        probs = torch.softmax(outputs.logits, dim=1)[0]
        pred = outputs.logits.argmax().item()

        print(f"\nPredicted: {'Positive' if pred == 1 else 'Negative'}")
        print(f"Confidence: {probs[pred]:.2%}")
        print(f"Negative: {probs[0]:.2%} | Positive: {probs[1]:.2%}")


print("\n=== Interactive Mode ===")
predict_interactive()