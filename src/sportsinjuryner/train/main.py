from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# Load model
print("Loading dslim/bert-large-NER...")
tokenizer = AutoTokenizer.from_pretrained("dslim/bert-large-NER")
model = AutoModelForTokenClassification.from_pretrained("dslim/bert-large-NER")

nlp = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

# Example text
example_text = """
Texans placed wide receiver Nico Collins on injured reserve with a hamstring injury.
Quarterback C.J. Stroud is expected to start against the Titans.
"""

print(f"\nAnalyzing text:\n{example_text}")

# Run NER
results = nlp(example_text)

# Print results
print("\nResults:")
for entity in results:
    print(f"{entity['entity_group']}: {entity['word']} (Score: {entity['score']:.4f})")
