import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import os

# 1. Device
DEVICE = torch.device("mps") if torch.backends.mps.is_available() else (
    torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
print(f"Using device: {DEVICE}")

# 2. Settings
MODEL_WEIGHTS_PATH = "./best_model/ner_model.pth"  # saved weights
TOKENIZER_PATH = "./best_model"  # saved tokenizer
BACKBONE_NAME = "roberta-base"  # backbone model
LABELS = [
    "COURT", "PETITIONER", "RESPONDENT", "JUDGE", "LAWYER",
    "DATE", "ORG", "GPE", "STATUTE", "PROVISION",
    "PRECEDENT", "CASE_NUMBER", "WITNESS", "OTHER_PERSON"
]
LABEL2ID = {label: idx for idx, label in enumerate(LABELS)}
ID2LABEL = {idx: label for label, idx in LABEL2ID.items()}
MAX_LENGTH = 256

# 3. Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)


# 4. Define model again
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
model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()


# 6. Prediction function
def predict(text):
    encoding = tokenizer(
        text,
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length",
        return_offsets_mapping=True,
        return_tensors="pt"
    )
    input_ids = encoding["input_ids"].to(DEVICE)
    attention_mask = encoding["attention_mask"].to(DEVICE)
    offsets = encoding["offset_mapping"].squeeze(0)

    with torch.no_grad():
        logits = model(input_ids, attention_mask)
        preds = torch.argmax(logits, dim=-1).squeeze(0).cpu().numpy()

    tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze(0).cpu().numpy())
    entities = []

    for idx, pred_label_id in enumerate(preds):
        if pred_label_id == 0:  # not an entity
            continue
        start, end = offsets[idx].tolist()
        if start == end:
            continue
        entity_text = text[start:end]
        label = ID2LABEL[pred_label_id]
        entities.append((entity_text, label))

    return entities


# 7. Example usage
if __name__ == "__main__":
    sample_text = """The Supreme Court of India ruled in favor of the petitioner, Mr. Sameer Joshi, on 3rd June 2022 under Section 302 of the Indian Penal Code."""

    results = predict(sample_text)
    print("\nEntities Found:")
    for ent_text, ent_label in results:
        print(f"[{ent_label}] {ent_text}")
