"""
Experiment 10: Word Sense Disambiguation (WSD) using LSTM/GRU

Implements a deep learning sequence model using PyTorch:
1. Labelled sentence dataset loading (wsd_dataset.csv)
2. Tokenization & Vocabulary construction
3. Word-to-index integer sequence encoding & padding
4. Embedding Layer -> Bidirectional LSTM/GRU Layer -> Dropout -> Dense Classification Layer
5. Training loop with epoch loss & accuracy tracking
6. Model evaluation (Train/Val/Test Accuracy)
7. Contextual Word Sense Disambiguation inference for words like "course", "bill", "order"
"""

import os
import re
import pickle
from typing import List, Dict, Any, Tuple
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize


DEFAULT_WSD_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "wsd_dataset.csv")
DEFAULT_WSD_MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "wsd_model.pt")
DEFAULT_WSD_META_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "wsd_meta.pkl")


class WSDDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = torch.tensor(sequences, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


class NeuralWSDModel(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, num_classes: int,
                 rnn_type: str = "LSTM", bidirectional: bool = True):
        super(NeuralWSDModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn_type = rnn_type.upper()
        self.bidirectional = bidirectional
        num_directions = 2 if bidirectional else 1

        if self.rnn_type == "GRU":
            self.rnn = nn.GRU(embed_dim, hidden_dim, batch_first=True,
                              bidirectional=bidirectional)
        else:
            self.rnn = nn.LSTM(embed_dim, hidden_dim, batch_first=True,
                               bidirectional=bidirectional)

        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim * num_directions, num_classes)

    def forward(self, x):
        embedded = self.embedding(x)  # [batch_size, seq_len, embed_dim]
        out, _ = self.rnn(embedded)   # [batch_size, seq_len, hidden_dim * num_dir]
        pooled = torch.mean(out, dim=1)  # Mean pooling over sequence
        dropped = self.dropout(pooled)
        logits = self.fc(dropped)
        return logits


class WSDDisambiguator:
    def __init__(self, data_path: str = DEFAULT_WSD_PATH,
                 model_path: str = DEFAULT_WSD_MODEL_PATH,
                 meta_path: str = DEFAULT_WSD_META_PATH):
        self.data_path = data_path
        self.model_path = model_path
        self.meta_path = meta_path

        self.word2idx = {"<PAD>": 0, "<UNK>": 1}
        self.label2idx = {}
        self.idx2label = {}
        self.max_len = 22
        self.embed_dim = 48
        self.hidden_dim = 64
        self.model = None
        self.rnn_type = "LSTM"
        self.metrics = {}

    def build_vocab(self, sentences: List[str]):
        """Constructs word-to-index mapping from corpus."""
        for s in sentences:
            words = [w.lower() for w in word_tokenize(s) if w.isalnum()]
            for w in words:
                if w not in self.word2idx:
                    self.word2idx[w] = len(self.word2idx)

    def text_to_sequence(self, sentence: str) -> List[int]:
        """Encodes sentence words into padded integer sequence."""
        words = [w.lower() for w in word_tokenize(sentence) if w.isalnum()]
        seq = [self.word2idx.get(w, 1) for w in words]
        if len(seq) < self.max_len:
            seq = seq + [0] * (self.max_len - len(seq))
        else:
            seq = seq[:self.max_len]
        return seq

    def train_model(self, rnn_type: str = "LSTM", epochs: int = 40, lr: float = 0.005) -> Dict[str, Any]:
        """
        Trains the neural LSTM or GRU WSD model.
        Tracks epoch loss and calculates training, validation, and test accuracy.
        """
        self.rnn_type = rnn_type.upper()
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"WSD dataset not found at {self.data_path}")

        df = pd.read_csv(self.data_path)
        sentences = df['sentence'].astype(str).tolist()
        senses = df['sense'].astype(str).tolist()

        # Build vocabulary & label encoding
        self.word2idx = {"<PAD>": 0, "<UNK>": 1}
        self.build_vocab(sentences)

        unique_labels = sorted(list(set(senses)))
        self.label2idx = {label: i for i, label in enumerate(unique_labels)}
        self.idx2label = {i: label for label, i in self.label2idx.items()}

        sequences = [self.text_to_sequence(s) for s in sentences]
        labels = [self.label2idx[s] for s in senses]

        # Train / Val / Test split (70% train, 15% val, 15% test)
        try:
            X_train, X_temp, y_train, y_temp = train_test_split(
                sequences, labels, test_size=0.30, random_state=42, stratify=labels
            )
            X_val, X_test, y_val, y_test = train_test_split(
                X_temp, y_temp, test_size=0.50, random_state=42
            )
        except Exception:
            X_train, X_test, y_train, y_test = train_test_split(
                sequences, labels, test_size=0.20, random_state=42
            )
            X_val, y_val = X_test, y_test

        train_dataset = WSDDataset(X_train, y_train)
        val_dataset = WSDDataset(X_val, y_val)
        test_dataset = WSDDataset(X_test, y_test)

        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

        vocab_size = len(self.word2idx)
        num_classes = len(self.label2idx)

        self.model = NeuralWSDModel(
            vocab_size=vocab_size,
            embed_dim=self.embed_dim,
            hidden_dim=self.hidden_dim,
            num_classes=num_classes,
            rnn_type=self.rnn_type,
            bidirectional=True
        )

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)

        epoch_losses = []
        self.model.train()

        for epoch in range(epochs):
            total_loss = 0.0
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            epoch_losses.append(round(total_loss / len(train_loader), 4))

        # Evaluate accuracies
        self.model.eval()
        with torch.no_grad():
            # Train Acc
            train_out = self.model(torch.tensor(X_train, dtype=torch.long))
            train_preds = torch.argmax(train_out, dim=1).numpy()
            train_acc = float(np.mean(train_preds == np.array(y_train)))

            # Val Acc
            val_out = self.model(torch.tensor(X_val, dtype=torch.long))
            val_preds = torch.argmax(val_out, dim=1).numpy()
            val_acc = float(np.mean(val_preds == np.array(y_val)))

            # Test Acc
            test_out = self.model(torch.tensor(X_test, dtype=torch.long))
            test_preds = torch.argmax(test_out, dim=1).numpy()
            test_acc = float(np.mean(test_preds == np.array(y_test)))

        self.metrics = {
            "model_architecture": f"Bidirectional {self.rnn_type}",
            "vocab_size": vocab_size,
            "training_samples": len(X_train),
            "validation_samples": len(X_val),
            "test_samples": len(X_test),
            "epochs": epochs,
            "final_epoch_loss": epoch_losses[-1],
            "loss_history": epoch_losses,
            "training_accuracy": round(train_acc * 100, 2),
            "validation_accuracy": round(val_acc * 100, 2),
            "test_accuracy": round(test_acc * 100, 2),
            "classes": unique_labels
        }

        # Save model and metadata
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        torch.save(self.model.state_dict(), self.model_path)
        with open(self.meta_path, 'wb') as f:
            pickle.dump({
                "word2idx": self.word2idx,
                "label2idx": self.label2idx,
                "idx2label": self.idx2label,
                "metrics": self.metrics,
                "rnn_type": self.rnn_type,
                "vocab_size": vocab_size,
                "num_classes": num_classes
            }, f)

        return self.metrics

    def load_or_train(self):
        """Loads saved weights or trains on startup."""
        if os.path.exists(self.model_path) and os.path.exists(self.meta_path):
            try:
                with open(self.meta_path, 'rb') as f:
                    meta = pickle.load(f)
                    self.word2idx = meta["word2idx"]
                    self.label2idx = meta["label2idx"]
                    self.idx2label = meta["idx2label"]
                    self.metrics = meta.get("metrics", {})
                    self.rnn_type = meta.get("rnn_type", "LSTM")
                    vocab_size = meta["vocab_size"]
                    num_classes = meta["num_classes"]

                self.model = NeuralWSDModel(
                    vocab_size=vocab_size,
                    embed_dim=self.embed_dim,
                    hidden_dim=self.hidden_dim,
                    num_classes=num_classes,
                    rnn_type=self.rnn_type,
                    bidirectional=True
                )
                self.model.load_state_dict(torch.load(self.model_path, weights_only=True))
                self.model.eval()
                return
            except Exception:
                pass
        self.train_model(rnn_type="LSTM", epochs=35)

    def disambiguate(self, sentence: str, target_word: str = "course") -> Dict[str, Any]:
        """
        Predicts the disambiguated word sense given contextual sentence.
        """
        if self.model is None:
            self.load_or_train()

        self.model.eval()
        seq = self.text_to_sequence(sentence)
        tensor_in = torch.tensor([seq], dtype=torch.long)

        with torch.no_grad():
            logits = self.model(tensor_in)
            probs = torch.softmax(logits, dim=1).numpy()[0]
            pred_idx = int(np.argmax(probs))
            predicted_sense = self.idx2label[pred_idx]
            confidence = float(probs[pred_idx])

        # Meaning description
        sense_descriptions = {
            "FOOD": "Culinary Dish / Meal Course served in a restaurant or banquet",
            "EDUCATION": "Academic / Vocational study course or degree curriculum",
            "PAYMENT": "Financial restaurant dining invoice or bill to pay",
            "LAW": "Government legislative act or parliamentary bill",
            "FOOD_REQUEST": "Placing food or beverage order at a restaurant",
            "SEQUENCE": "Arrangement, numerical, or chronological order"
        }

        all_probs = {self.idx2label[i]: round(float(p), 4) for i, p in enumerate(probs)}

        return {
            "sentence": sentence,
            "target_word": target_word,
            "predicted_sense": predicted_sense,
            "confidence": round(confidence * 100, 2),
            "sense_meaning": sense_descriptions.get(predicted_sense, "Domain context sense"),
            "probabilities": all_probs,
            "architecture": self.metrics.get("model_architecture", "Bidirectional LSTM"),
            "training_metrics": self.metrics
        }


# Singleton instance
_wsd_instance = None

def get_wsd_disambiguator() -> WSDDisambiguator:
    global _wsd_instance
    if _wsd_instance is None:
        _wsd_instance = WSDDisambiguator()
        _wsd_instance.load_or_train()
    return _wsd_instance


def run_experiment_10(sentence: str, target_word: str = "course") -> Dict[str, Any]:
    """Runs Experiment 10 on a sentence."""
    disambiguator = get_wsd_disambiguator()
    return disambiguator.disambiguate(sentence, target_word=target_word)
