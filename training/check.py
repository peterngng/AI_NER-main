import spacy
from spacy.tokens import DocBin

# Load a blank English model just for vocab
nlp = spacy.blank("en")

# Read your DocBin
db = DocBin().from_disk("dev.spacy")

# Materialize all docs
docs = list(db.get_docs(nlp.vocab))

# Report counts
print("Total docs in train.spacy:   ", len(docs))
print("Docs with ≥1 entity span:    ", sum(1 for doc in docs if len(doc.ents) > 0))

# Peek at the first doc’s entities
if docs:
    print("\nExample spans in first doc:")
    for ent in docs[0].ents:
        print(f"  {ent.label_}: '{docs[0].text[ent.start_char:ent.end_char]}'")
