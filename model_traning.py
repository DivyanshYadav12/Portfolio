import pandas as pd
from datasets import Dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import torch
from sklearn.model_selection import train_test_split as sk_train_test_split
import numpy as np

# Step 1: Load data
df = pd.read_csv(r'D:\portfolio_project\intents.csv')  # Ensure it has 'text' and 'label' columns

# Step 2: Encode labels
label_encoder = LabelEncoder()
df['label_id'] = label_encoder.fit_transform(df['label'])

# Step 3: Convert to Hugging Face dataset
dataset = Dataset.from_pandas(df[['text', 'label_id']])  # Only keep necessary columns

# Step 4: Split into train/test
dataset_split = dataset.train_test_split(test_size=0.2, seed=42)
train_dataset = dataset_split['train']
test_dataset = dataset_split['test']

# Step 5: Tokenization
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def tokenize_function(example):
    return tokenizer(example['text'], padding="max_length", truncation=True, max_length=128)

train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

# ✅ Rename "label_id" to "labels" for the Trainer
train_dataset = train_dataset.rename_column("label_id", "labels")
test_dataset = test_dataset.rename_column("label_id", "labels")

# Set format for PyTorch
train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

# Step 6: Load model
num_labels = len(label_encoder.classes_)
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_labels)

# Step 7: Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy='epoch',
    save_strategy='epoch',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    warmup_steps=100,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model='accuracy',
    greater_is_better=True,
)

# Step 8: Evaluation metrics
def compute_metrics(p):
    preds = torch.tensor(p.predictions).argmax(dim=-1)
    labels = torch.tensor(p.label_ids)
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc}

# Step 9: Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
)

# Step 10: Train
trainer.train()

# Step 11: Evaluate
results = trainer.evaluate()
print("Evaluation Results:", results)

# Step 12: Save model/tokenizer
model.save_pretrained('./my_bert_model')
tokenizer.save_pretrained('./my_bert_model')

# Step 13: Inference function
def predict(text):
    model.eval()
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    predicted_class_id = outputs.logits.argmax(dim=-1).item()
    return label_encoder.inverse_transform([predicted_class_id])[0]

# Example prediction
sample_text = "Developed a Python-based data pipeline to parse system logs"
print("Prediction:", predict(sample_text))

# Step 14: Classification report on test set
all_preds = []
all_labels = []

for i in range(len(test_dataset)):
    item = test_dataset[i]
    inputs = {k: item[k].unsqueeze(0) for k in ['input_ids', 'attention_mask']}
    with torch.no_grad():
        outputs = model(**inputs)
    pred = outputs.logits.argmax(dim=-1).item()
    all_preds.append(pred)
    all_labels.append(item['labels'])  # it's 'labels' now

# Decode labels
all_preds_decoded = label_encoder.inverse_transform(all_preds)
all_labels_decoded = label_encoder.inverse_transform(all_labels)

# Report
print("\nClassification Report:")
print(classification_report(all_labels_decoded, all_preds_decoded))

print("Confusion Matrix:")
print(confusion_matrix(all_labels_decoded, all_preds_decoded))
