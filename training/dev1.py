import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
import spacy
from spacy.tokens import DocBin
from sklearn.metrics import classification_report
from tqdm import tqdm

# 1. Settings
MODEL_PATH = "./output_model"
DEV_DATA_PATH = "./labeled_dev.spacy"
LABELS = [
    "COURT", "PETITIONER", "RESPONDENT", "JUDGE", "LAWYER",
    "DATE", "ORG", "GPE", "STATUTE", "PROVISION",
    "PRECEDENT", "CASE_NUMBER", "WITNESS", "OTHER_PERSON"
]
LABEL2ID = {label: idx for idx, label in enumerate(LABELS)}
ID2LABEL = {idx: label for label, idx in LABEL2ID.items()}
MAX_LENGTH = 256
BATCH_SIZE = 8

# 2. Detect device (prefer MPS for Mac)
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print("Using Apple MPS GPU")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print("Using CUDA GPU")
else:
    DEVICE = torch.device("cpu")
    print("Using CPU")

# 3. Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

# 4. Define model
class NERModel(torch.nn.Module):
    def __init__(self, model_name, num_labels):
        super(NERModel, self).__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        self.dropout = torch.nn.Dropout(0.3)
        self.classifier = torch.nn.Linear(self.backbone.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        x = self.dropout(outputs.last_hidden_state)
        logits = self.classifier(x)
        return logits

# 5. Load trained weights
model = NERModel(MODEL_PATH, len(LABELS))
model.load_state_dict(torch.load("./output_model/ner_model.pth", map_location=DEVICE))
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
        labels = torch.zeros(MAX_LENGTH, dtype=torch.long)

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

dev_dataset = NERDataset(dev_docs)
dev_loader = DataLoader(dev_dataset, batch_size=BATCH_SIZE)

# 7. Evaluation
all_preds = []
all_labels = []

with torch.no_grad():
    for batch in tqdm(dev_loader):
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        logits = model(input_ids, attention_mask)
        preds = torch.argmax(logits, dim=-1)

        preds = preds.view(-1).cpu().numpy()
        labels = labels.view(-1).cpu().numpy()

        # Only evaluate real labels (non-zero)
        mask = labels != 0
        preds = preds[mask]
        labels = labels[mask]

        all_preds.extend(preds)
        all_labels.extend(labels)

# 8. Classification report
print("\nClassification Report:")
print(classification_report(
    all_labels,
    all_preds,
    labels=list(LABEL2ID.values()),  # <--- THIS FIXES YOUR ERROR
    target_names=LABELS,
    digits=4,
    zero_division=0   # <--- no division by zero warning
))
