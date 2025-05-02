import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

import tensorflow as tf
from tensorflow.python.keras import layers, models
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

from transformers import TFAutoModel, AutoTokenizer

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Parameters
MAX_LENGTH = 128
BATCH_SIZE = 32
EPOCHS = 10
MODEL_NAME = "Rostlab/prot_bert"
OUTPUT_DIR = "./"

# Create output directory if not exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. Load dataset - auto-detect header
df = pd.read_csv('data/final_protein_dataset.csv')  # removed header=None

# Ensure column names are correct
expected_columns = ['id', 'protein', 'sequence', 'mutation', 'ptm', 'confidence', 'pmid', 'label']
if list(df.columns) != expected_columns:
    df.columns = expected_columns  # force rename if needed

# Sanity check and drop any non-numeric labels
df = df[df['label'].apply(lambda x: str(x).isdigit())]

labels = df['label'].astype(int).values
num_classes = len(np.unique(labels))
labels_cat = tf.keras.utils.to_categorical(labels, num_classes=num_classes)
sequences = df['sequence'].astype(str).values

# 2. Split data
X_train, X_test, y_train, y_test = train_test_split(
    sequences, labels_cat, test_size=0.2, stratify=labels, random_state=42
)

# 3. Load tokenizer and model with from_pt=True to load PyTorch weights
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, do_lower_case=False)
transformer_model = TFAutoModel.from_pretrained(MODEL_NAME, from_pt=True)

# 4. Tokenization helper function
def tokenize_sequences(seqs):
    """
    Tokenize sequences and ensure token IDs are within the vocabulary size.
    """
    seqs_spaced = [" ".join(list(seq)) for seq in seqs]  # Add spaces between characters for tokenization
    encodings = tokenizer(
        seqs_spaced,
        padding='max_length',  # Pad sequences to MAX_LENGTH
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors='tf'
    )
    
    # Debugging: Check for out-of-bound token IDs
    max_token_id = tf.reduce_max(encodings['input_ids']).numpy()
    if max_token_id >= tokenizer.vocab_size:
        print(f"Error: Token ID {max_token_id} exceeds vocabulary size {tokenizer.vocab_size}.")
        raise ValueError("Token ID exceeds vocabulary size. Check your input sequences.")
    
    return encodings

# Tokenize training and test sequences
train_encodings = tokenize_sequences(X_train)
test_encodings = tokenize_sequences(X_test)

# 5. Build model using functional API (updated version)
def build_transformer_model():
    input_ids = layers.Input(shape=(MAX_LENGTH,), dtype=tf.int32, name='input_ids')
    attention_mask = layers.Input(shape=(MAX_LENGTH,), dtype=tf.int32, name='attention_mask')

    outputs = transformer_model(input_ids=input_ids, attention_mask=attention_mask)
    sequence_output = outputs.last_hidden_state  # shape: (batch_size, seq_len, hidden_size)

    x = layers.GlobalAveragePooling1D()(sequence_output)
    x = layers.Dense(256, activation='gelu')(x)
    output = layers.Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=[input_ids, attention_mask], outputs=output)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=3e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

model = build_transformer_model()
model.summary()

# 6. Train model
history = model.fit(
    x={'input_ids': train_encodings['input_ids'], 'attention_mask': train_encodings['attention_mask']},
    y=y_train,
    validation_data=(
        {'input_ids': test_encodings['input_ids'], 'attention_mask': test_encodings['attention_mask']},
        y_test
    ),
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    verbose=1
)

# 7. Save weights and history
model.save_weights(os.path.join(OUTPUT_DIR, 'transformer_weights.h5'))
with open(os.path.join(OUTPUT_DIR, 'history_transformer.pkl'), 'wb') as f:
    pickle.dump(history.history, f)

# 8. Evaluate model on test set
y_pred_probs = model.predict(
    {'input_ids': test_encodings['input_ids'], 'attention_mask': test_encodings['attention_mask']},
    batch_size=BATCH_SIZE
)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_test, axis=1)

# Classification report
class_names = ['Alzheimers', 'Parkinsons', 'Other Neurodegenerative', 'Healthy']
report_dict = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
report_df = pd.DataFrame(report_dict).transpose()
report_df.to_csv(os.path.join(OUTPUT_DIR, 'results_transformer.csv'), index=True)

# 9. Plot training curves
def plot_training_curves(history, save_path):
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(history['accuracy'], label='Train Accuracy')
    plt.plot(history['val_accuracy'], label='Val Accuracy')
    plt.title('Transformer Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.subplot(1,2,2)
    plt.plot(history['loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Transformer Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

plot_training_curves(history.history, os.path.join(OUTPUT_DIR, 'training_curve_transformer.png'))

# 10. Plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, classes, save_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(7,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Transformer Model Confusion Matrix')
    plt.savefig(save_path)
    plt.close()

plot_confusion_matrix(y_true, y_pred, class_names, os.path.join(OUTPUT_DIR, 'confusion_matrix_transformer.png'))

print("Training and evaluation complete. All files saved in:", OUTPUT_DIR)