# classifier.py

from transformers import pipeline

# Load the model
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

def classify(text, labels):
    result = classifier(text, labels)
    print("Input Text:", text)
    print("\n--- Results ---")
    for label, score in zip(result["labels"], result["scores"]):
        print(f"{label}: {score:.4f}")

# Example
if __name__ == "__main__":
    complaint = "There is no electricity in our area since morning."
    candidate_labels = ["Water", "Electricity", "Garbage", "Road", "Others"]
    classify(complaint, candidate_labels)
