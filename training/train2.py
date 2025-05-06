import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
import spacy
from spacy.tokens import DocBin
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from tqdm import tqdm
import os

# 1. Device
DEVICE = torch.device("mps") if torch.backends.mps.is_available() else (
         torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
print(f"Using device: {DEVICE}")

# 2. Settings
MODEL_NAME = "roberta-base"
LABELS = [
    "COURT", "PETITIONER", "RESPONDENT", "JUDGE", "LAWYER",
    "DATE", "ORG", "GPE", "STATUTE", "PROVISION",
    "PRECEDENT", "CASE_NUMBER", "WITNESS", "OTHER_PERSON"
]
LABEL2ID = {label: idx for idx, label in enumerate(LABELS)}
ID2LABEL = {idx: label for label, idx in LABEL2ID.items()}
MAX_LENGTH = 256
BATCH_SIZE = 8
NUM_EPOCHS = 5
LEARNING_RATE = 3e-5
TRAIN_DATA_PATH = "./new_train.spacy"
DEV_DATA_PATH = "./labeled_dev.spacy"

# 3. Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# 4. Load datasets
def load_spacy_data(filepath):
    doc_bin = DocBin().from_disk(filepath)
    return list(doc_bin.get_docs(spacy.blank("en").vocab))

train_docs = load_spacy_data(TRAIN_DATA_PATH)
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

        labels = torch.full((MAX_LENGTH,), fill_value=-100, dtype=torch.long)  # padding label = -100

        ents = [(ent.start_char, ent.end_char, ent.label_) for ent in doc.ents]

        for idx_tok, (start, end) in enumerate(offsets):
            if start == end:
                continue
            char_start = start.item()
            char_end = end.item()
            for ent_start, ent_end, label in ents:
                if char_start >= ent_start and char_end <= ent_end:
                    labels[idx_tok] = LABEL2ID.get(label, 0)
                    break

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": labels
        }

train_dataset = NERDataset(train_docs)
dev_dataset = NERDataset(dev_docs)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
dev_loader = DataLoader(dev_dataset, batch_size=BATCH_SIZE)

# 5. Model
class NERModel(nn.Module):
    def __init__(self, model_name, num_labels):
        super(NERModel, self).__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.backbone.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        x = self.dropout(outputs.last_hidden_state)
        logits = self.classifier(x)
        return logits

model = NERModel(MODEL_NAME, len(LABELS)).to(DEVICE)

for param in model.backbone.parameters():
    param.requires_grad = True

# 6. Optimizer, loss
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()

# 7. Training + Validation
best_val_loss = float('inf')

for epoch in range(NUM_EPOCHS):
    print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
    # --- Training ---
    model.train()
    running_loss = 0.0

    for batch in tqdm(train_loader, desc="Training"):
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        loss = criterion(logits.view(-1, len(LABELS)), labels.view(-1))
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_train_loss = running_loss / len(train_loader)
    print(f"Average Training Loss: {avg_train_loss:.4f}")

    # --- Validation ---
    model.eval()
    val_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dev_loader, desc="Validating"):
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            logits = model(input_ids, attention_mask)
            loss = criterion(logits.view(-1, len(LABELS)), labels.view(-1))
            val_loss += loss.item()

            preds = torch.argmax(logits, dim=-1)

            active_labels = labels.view(-1) != -100
            preds = preds.view(-1)[active_labels].cpu().numpy()
            labels = labels.view(-1)[active_labels].cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels)

    avg_val_loss = val_loss / len(dev_loader)
    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted', zero_division=0)

    print(f"Validation Loss: {avg_val_loss:.4f} | Accuracy: {acc:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1 Score: {f1:.4f}")

    # Save best model
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        os.makedirs("./best_model", exist_ok=True)
        torch.save(model.state_dict(), "./best_model/ner_model.pth")
        tokenizer.save_pretrained("./best_model")
        print("✅ Best model updated and saved!")

print("\n🏁 Training Complete!")
