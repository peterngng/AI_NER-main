# Import necessary libraries
import spacy
from legal_ner import extract_entities_from_judgment_text
from legal_ner import get_csv
from spacy import displacy
import urllib

# Step 1: Load the pre-trained Legal-NER model
print("Loading the Legal-NER model...")
legal_nlp = spacy.load('en_legal_ner_trf')

# Step 2: Load SpaCy's small pre-trained model (for splitting preamble and judgment)
print("Loading SpaCy's pre-trained model for preamble splitting...")
preamble_splitting_nlp = spacy.load('en_core_web_sm')

# Step 3: Download a sample judgment text
print("Downloading sample judgment text...")
judgment_url = 'https://raw.githubusercontent.com/OpenNyAI/Opennyai/master/samples/sample_judgment1.txt'
judgment_text = urllib.request.urlopen(judgment_url).read().decode()

# Step 4: Extract entities from the judgment text
print("Extracting entities from the judgment text...")
run_type = 'sent'  # 'sent' for sentence-by-sentence processing (more accurate, slower)
do_postprocess = True  # Enable post-processing for better results

combined_doc = extract_entities_from_judgment_text(
    judgment_text, legal_nlp, preamble_splitting_nlp, run_type, do_postprocess
)

# Step 5: Print extracted entities
print("\nExtracted Entities:")
for ent in combined_doc.ents:
    print(f"{ent.text} ({ent.label_})")

# Step 6: Post-processing: Display clusters of precedents, statutes, and provision-statute pairs
print("\nPost-processed data:")

# Precedent clusters
precedent_clusters = combined_doc.user_data.get("precedent_clusters", {})
print("\nPrecedent Clusters:")
for head, cluster in precedent_clusters.items():
    print(f"Head: {head}, Variations: {cluster}")

# Statute clusters
statute_clusters = combined_doc.user_data.get("statute_clusters", {})
print("\nStatute Clusters:")
for head, cluster in statute_clusters.items():
    print(f"Head: {head}, Variations: {cluster}")

# Provision-Statute pairs
provision_statute_pairs = combined_doc.user_data.get("provision_statute_pairs", [])
print("\nProvision-Statute Pairs:")
for pair in provision_statute_pairs:
    print(pair)

# Step 7: Visualize the extracted entities
print("\nVisualizing extracted entities...")
extracted_ent_labels = list(set([ent.label_ for ent in combined_doc.ents]))
colors = {
    'COURT': "#bbabf2", 'PETITIONER': "#f570ea", "RESPONDENT": "#cdee81",
    'JUDGE': "#fdd8a5", "LAWYER": "#f9d380", 'WITNESS': "violet",
    "STATUTE": "#faea99", "PROVISION": "yellow",
    'CASE_NUMBER': "#fbb1cf", "PRECEDENT": "#fad6d6", 'DATE': "#b1ecf7",
    'OTHER_PERSON': "#b0f6a2", 'ORG': '#a57db5', 'GPE': '#7fdbd4'
}
options = {"ents": extracted_ent_labels, "colors": colors}

# Start a local server to visualize entities
displacy.serve(combined_doc, style='ent', port=8080, options=options)
print("Visit http://localhost:8080 in your web browser to view the visualization.")

# Step 8: Save the extracted entities to a CSV file
print("\nSaving extracted entities to a CSV file...")
file_name = "sample_judgment"
save_path = "./results"
get_csv(combined_doc, file_name, save_path)
print(f"Extracted entities saved to: {save_path}/{file_name}.csv")

print("\nProcess completed successfully!")