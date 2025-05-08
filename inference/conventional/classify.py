import json
import requests

# Replace this with your Hugging Face API token
HF_API_TOKEN = "INJECT IT"
HF_API_URL = "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta"
HEADERS = {"Authorization": f"Bearer {HF_API_TOKEN}"}

BIN_CHOICES = [
    "form",
    "semantic meaning",
    "associative relations",
    "encyclopedic",
    "form_meaning"
]

def classify_category(name, words):
    """Call Hugging Face API to classify reasoning type."""
    prompt = (
        f"Classify the following NYT Connections category into one of these types: "
        f"form, semantic meaning, associative relations, encyclopedic, or form_meaning.\n\n"
        f"Category Name: {name}\n"
        f"Words: {', '.join(words)}\n\n"
        "Respond with just the classification label."
    )
    payload = {"inputs": prompt, "parameters": {"max_new_tokens": 10}}

    try:
        response = requests.post(HF_API_URL, headers=HEADERS, json=payload)
        response.raise_for_status()
        result = response.json()
        if isinstance(result, list) and "generated_text" in result[0]:
            label = result[0]["generated_text"].strip().lower()
        else:
            label = result[0].get("generated_text", "").strip().lower()
        return label if label in BIN_CHOICES else "unknown"
    except Exception as e:
        print(f"Error during classification: {e}")
        return "error"

def main():
    with open("results_output.json") as f:
        data = json.load(f)

    updated = []

    for row in data:
        annotations = row.get("reasoning_annotation")
        if not isinstance(annotations, list):
            row["incorrect_category_types"] = []
            updated.append(row)
            continue

        incorrect_groups = row.get("incorrect_inference_results", [])
        pred_sets = [set(group) for group in incorrect_groups]

        # Identify unmatched gold categories
        unmatched_annotations = []
        for annotation in annotations:
            gold_set = set(annotation["Words in Category"])
            if all(gold_set != pred for pred in pred_sets):
                unmatched_annotations.append(annotation)

        # Classify unmatched categories
        incorrect_types = []
        for ann in unmatched_annotations:
            cat_type = classify_category(ann["Categories"], ann["Words in Category"])
            incorrect_types.append(cat_type)

        row["incorrect_category_types"] = incorrect_types
        updated.append(row)

    with open("results_labeled.json", "w") as f:
        json.dump(updated, f, indent=2)

    print("✅ Done. Labels saved to results_labeled.json")

if __name__ == "__main__":
    main()
