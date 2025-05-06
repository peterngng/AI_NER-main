import spacy
from spacy.tokens import DocBin

# Path to your dev set
input_path = "./labeled_dev.spacy"

# Load
nlp = spacy.blank("en")
doc_bin = DocBin().from_disk(input_path)
docs = list(doc_bin.get_docs(nlp.vocab))

# Analyze entity labels
label_counter = {}

for doc in docs:
    for ent in doc.ents:
        label_counter[ent.label_] = label_counter.get(ent.label_, 0) + 1

# Print results
print("Entities found in labeled_dev.spacy:")
for label, count in label_counter.items():
    print(f"{label}: {count}")
