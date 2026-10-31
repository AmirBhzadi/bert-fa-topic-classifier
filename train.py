import tensorflow as tf
from transformers import BertTokenizer, TFBertModel
import numpy as np
import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from model import BertClassifier

# ------------------- Paths -------------------
DATA_PATH   = "D:/DeepLearning/Learning/BERT/topic_classifer/train.csv"
MODEL_DIR   = "saved_model"
WEIGHTS_PATH = os.path.join(MODEL_DIR, "weights.h5")
CONFIG_PATH  = os.path.join(MODEL_DIR, "config.json")
TOKENIZER_DIR = os.path.join(MODEL_DIR, "tokenizer")

os.makedirs(MODEL_DIR, exist_ok=True)

# ------------------- Load data -------------------
df = pd.read_csv(DATA_PATH, sep='\t', index_col=0)
data = df["content"].astype(str).tolist()
labels = df["label_id"].astype(int).tolist()

# Save class names if the dataset has a 'label_name' column
if "label_name" in df.columns:
    label_map = df.drop_duplicates("label").set_index("label")["label_name"].to_dict()
    label_names = [label_map[i] for i in sorted(label_map)]
else:
    # If no names, use the numbers as strings
    unique_labels = sorted(set(labels))
    label_names = [str(i) for i in unique_labels]

NUM_LABELS = len(sorted(set(labels)))
print("Number of classes:", NUM_LABELS)
print("Class names:", label_names)

# ------------------- Split -------------------
x_train, x_temp, y_train, y_temp = train_test_split(
    data, labels, test_size=0.3, random_state=42, stratify=labels
)
x_val, x_test, y_val, y_test = train_test_split(
    x_temp, y_temp, test_size=0.15, random_state=42, stratify=y_temp
)

# ------------------- Tokenizer -------------------
print("Loading ParsBERT tokenizer...")
tokenizer = BertTokenizer.from_pretrained("HooshvareLab/bert-fa-base-uncased")

MAX_LEN = 128
x_train_enc = tokenizer(x_train, padding=True, truncation=True, max_length=MAX_LEN, return_tensors="tf")
x_val_enc   = tokenizer(x_val,   padding=True, truncation=True, max_length=MAX_LEN, return_tensors="tf")
x_test_enc  = tokenizer(x_test,  padding=True, truncation=True, max_length=MAX_LEN, return_tensors="tf")

# ------------------- TF Datasets -------------------
BATCH_SIZE = 16
train_dataset = tf.data.Dataset.from_tensor_slices((dict(x_train_enc), y_train)).shuffle(1000).batch(BATCH_SIZE)
val_dataset   = tf.data.Dataset.from_tensor_slices((dict(x_val_enc),   y_val)).batch(BATCH_SIZE)
test_dataset  = tf.data.Dataset.from_tensor_slices((dict(x_test_enc),  y_test)).batch(BATCH_SIZE)

# ------------------- Model -------------------
print("Loading ParsBERT model...")
bert = TFBertModel.from_pretrained("HooshvareLab/bert-fa-base-uncased")
model = BertClassifier(bert, NUM_LABELS)

model.compile(
    optimizer=tf.keras.optimizers.Adam(2e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

# ------------------- Train -------------------
print("Training started...")
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=5,
    callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, mode="min")],
)

# ------------------- Evaluate -------------------
test_loss, test_acc = model.evaluate(test_dataset)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_acc)

# ------------------- Save everything -------------------
print("Saving model...")

# 1) Weights
model.save_weights(WEIGHTS_PATH)
print("Saved weights ->", WEIGHTS_PATH)

# 2) Tokenizer
tokenizer.save_pretrained(TOKENIZER_DIR)
print("Saved tokenizer ->", TOKENIZER_DIR)

# 3) Config (num_labels, max_len, class names)
config = {
    "num_labels": NUM_LABELS,
    "max_len": MAX_LEN,
    "label_names": label_names,
    "base_model": "HooshvareLab/bert-fa-base-uncased",
}
with open(CONFIG_PATH, "w", encoding="utf-8") as f:
    json.dump(config, f, ensure_ascii=False, indent=2)
print("Saved config ->", CONFIG_PATH)

print("Done. Model is ready for predict.py")