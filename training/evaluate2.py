import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, average_precision_score, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import spacy
from spacy.tokens import DocBin
import numpy as np
import os

# 1. Device setup
DEVICE = torch.device("mps") if torch.backends.mps.is_available() else (
         torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
print(f"Using device: {DEVICE}")

# 2. Paths and constants
MODEL_PATH = "./best_model/ner_model.pth"
TOKENIZER_PATH = "./best_model"
DEV_DATA_PATH = "./labeled_dev.spacy"
BACKBONE_NAME = "roberta-base"

LABELS = [
    "COURT", "PETITIONER", "RESPONDENT", "JUDGE", "LAWYER",
    "DATE", "ORG", "GPE", "STATUTE", "PROVISION",
    "PRECEDENT", "CASE_NUMBER", "WITNESS", "OTHER_PERSON"
]
LABEL2ID = {label: idx for idx, label in enumerate(LABELS)}
ID2LABEL = {idx: label for label, idx in LABEL2ID.items()}
MAX_LENGTH = 256
BATCH_SIZE = 8

# 3. Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)

# 4. Define model
class NERModel(nn.Module):
    def __init__(self, backbone_name, num_labels):
        super(NERModel, self).__init__()
        self.backbone = AutoModel.from_pretrained(backbone_name)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.backbone.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        x = self.dropout(outputs.last_hidden_state)
        logits = self.classifier(x)
        return logits

# 5. Load model
model = NERModel(BACKBONE_NAME, len(LABELS))
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

# 6. Load dev data
def load_spacy_data(filepath):
    doc_bin = DocBin().from_disk(filepath)
    return list(doc_bin.get_docs(spacy.blank("en").vocab))

dev_docs = load_spacy_data(DEV_DATA_PATH)

class NERDataset(torch.utils.data.Dataset):
    def __init__(self, docs):
        self.docs = docs

    def __len__(self):
        return len(self.docs)

    def __getitem__(self, idx):
        doc = self.docs[idx]
        encoding = tokenizer(
            doc.text,
            truncation=True,
            max_length=MAX_LENGTH,
            padding="max_length",
            return_offsets_mapping=True,
            return_tensors="pt"
        )
        offsets = encoding.pop("offset_mapping").squeeze(0)
        labels = torch.full((MAX_LENGTH,), fill_value=-100, dtype=torch.long)

        for ent in doc.ents:
            for idx_tok, (start, end) in enumerate(offsets):
                if start == end:
                    continue
                char_start = start.item()
                char_end = end.item()
                if char_start >= ent.start_char and char_end <= ent.end_char:
                    labels[idx_tok] = LABEL2ID.get(ent.label_, 0)
                    break

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": labels
        }

dev_dataset = NERDataset(dev_docs)
dev_loader = DataLoader(dev_dataset, batch_size=BATCH_SIZE)

# 7. Evaluation loop
all_preds = []
all_labels = []

with torch.no_grad():
    for batch in dev_loader:
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        logits = model(input_ids, attention_mask)
        preds = torch.argmax(logits, dim=-1)

        active = labels.view(-1) != -100
        all_preds.extend(preds.view(-1)[active].cpu().numpy())
        all_labels.extend(labels.view(-1)[active].cpu().numpy())

# 8. Classification Report (text, PNG, CSV)
report_dict = classification_report(all_labels, all_preds, target_names=LABELS, digits=4, output_dict=True)
report_df = pd.DataFrame(report_dict).transpose()
report_df.to_csv("classification_report.csv")
print("✅ Saved classification_report.csv")

plt.figure(figsize=(12, len(report_df) * 0.5 + 2))
sns.heatmap(report_df.iloc[:-3, :-1], annot=True, cmap="YlGnBu", fmt=".4f")
plt.title("Classification Report")
plt.tight_layout()
plt.savefig("classification_report.png")
plt.close()
print("✅ Saved classification_report.png")

# 9. Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=LABELS, yticklabels=LABELS, cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.close()
print("✅ Saved confusion_matrix.png")

# 10. Accuracy
accuracy = accuracy_score(all_labels, all_preds)
print(f"✅ Overall Accuracy: {accuracy:.4f}")

# 11. Most confused labels (just names)
confusions = (cm - np.eye(len(LABELS)) * cm.diagonal()).sum(axis=1)
top_confused = np.argsort(confusions)[-5:]
top_confused_labels = [LABELS[i] for i in reversed(top_confused)]
print("\nTop 5 Most Confused Labels:")
for label in top_confused_labels:
    print(f"- {label}")

# 12. Precision-Recall Curve
y_true_bin = np.eye(len(LABELS))[all_labels]
y_pred_bin = np.eye(len(LABELS))[all_preds]

plt.figure(figsize=(12, 8))
for i in range(len(LABELS)):
    precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_pred_bin[:, i])
    ap = average_precision_score(y_true_bin[:, i], y_pred_bin[:, i])
    plt.plot(recall, precision, label=f"{LABELS[i]} (AP={ap:.2f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve (per class)")
plt.legend(loc="lower left")
plt.tight_layout()
plt.savefig("precision_recall_curve.png")
plt.close()
print("✅ Saved precision_recall_curve.png")
