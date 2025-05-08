# import json
# import re
# from pathlib import Path

# def extract_predicted_groups(text):
#     """Extract predicted word groups from raw prediction."""
#     matches = re.findall(r'Category\d+:\s*\[([^\]]+)\]', text)
#     return [sorted(re.findall(r'[A-Z][A-Z\s]*[A-Z]', group)) for group in matches if group]

# def find_matching_gold_entry(pred_words, gold_data):
#     """Match puzzle by full set of words."""
#     pred_set = set(pred_words)
#     for entry in gold_data:
#         if set(entry["allwords"]) == pred_set:
#             return entry
#     return None

# def match_groups(pred_groups, gold_groups):
#     """Match predicted groups to any gold group (one-to-one matching)."""
#     unmatched_gold = [set(g) for g in gold_groups]
#     correct_matches = []
#     incorrect_matches = []

#     for pred in pred_groups:
#         pred_set = set(pred)
#         match_found = False
#         for gold in unmatched_gold:
#             if pred_set == gold:
#                 correct_matches.append(pred)
#                 unmatched_gold.remove(gold)
#                 match_found = True
#                 break
#         if not match_found:
#             incorrect_matches.append(pred)

#     return correct_matches, incorrect_matches

# def main():
#     with open("data/inferences.json") as f:
#         inferences = json.load(f)
#     with open("data/gold_data.json") as f:
#         gold_data = json.load(f)

#     output = []

#     for entry in inferences:
#         new_pred = entry.get("new token prediction", "")
#         pred_groups = extract_predicted_groups(new_pred)
#         flat_pred_words = {w for g in pred_groups for w in g}

#         gold_entry = find_matching_gold_entry(flat_pred_words, gold_data)

#         if not gold_entry or len(pred_groups) != 4:
#             output.append({
#                 "new_token_prediction": new_pred,
#                 "real_results": "MATCHING GOLD PUZZLE NOT FOUND OR INVALID PREDICTION FORMAT",
#                 "predicted_groupings": pred_groups,
#                 "incorrect_inference_results": "N/A",
#                 "count_incorrect_inference_results": 4
#             })
#             continue

#         gold_annotations = gold_entry["reasoning_annotation"]
#         gold_group_tuples = [(g["Categories"], sorted(g["Words in Category"])) for g in gold_annotations]
#         gold_groups_only = [g for _, g in gold_group_tuples]

#         correct, incorrect = match_groups(pred_groups, gold_groups_only)

#         output.append({
#             "new_token_prediction": new_pred,
#             "real_results": [
#                 {"category_name": name, "words": words}
#                 for name, words in gold_group_tuples
#             ],
#             "predicted_groupings": pred_groups,
#             "incorrect_inference_results": incorrect,
#             "count_incorrect_inference_results": len(incorrect)
#         })

#     with open("results_output.json", "w") as f:
#         json.dump(output, f, indent=2)

#     print("✅ Flexible-matching results saved to results_output.json")

# if __name__ == "__main__":
#     main()

import json
import re
from pathlib import Path

def extract_predicted_groups(text):
    """Extract predicted word groups from raw model output."""
    matches = re.findall(r'Category\d+:\s*\[([^\]]+)\]', text)
    return [sorted(re.findall(r'[A-Z][A-Z\s]*[A-Z]', group)) for group in matches if group]

def find_matching_gold_entry(pred_words, gold_data):
    """Find gold puzzle whose allwords match the prediction words."""
    pred_set = set(pred_words)
    for entry in gold_data:
        if set(entry["allwords"]) == pred_set:
            return entry
    return None

def match_groups(pred_groups, gold_groups):
    """Match predicted groups to any gold group (no positional assumption)."""
    unmatched_gold = [set(g) for g in gold_groups]
    incorrect_matches = []

    for pred in pred_groups:
        pred_set = set(pred)
        match_found = False
        for gold in unmatched_gold:
            if pred_set == gold:
                unmatched_gold.remove(gold)
                match_found = True
                break
        if not match_found:
            incorrect_matches.append(pred)

    return incorrect_matches

def main():
    with open("data/inferences.json") as f:
        inferences = json.load(f)
    with open("data/gold_data.json") as f:
        gold_data = json.load(f)

    output = []

    for entry in inferences:
        new_pred = entry.get("new token prediction", "")
        pred_groups = extract_predicted_groups(new_pred)
        flat_pred_words = {w for g in pred_groups for w in g}

        gold_entry = find_matching_gold_entry(flat_pred_words, gold_data)

        if not gold_entry or len(pred_groups) != 4:
            output.append({
                "new_token_prediction": new_pred,
                "reasoning_annotation": "MATCHING GOLD PUZZLE NOT FOUND OR INVALID PREDICTION FORMAT",
                "incorrect_inference_results": pred_groups,
                "count_incorrect_inference_results": 4
            })
            continue

        gold_annotations = gold_entry["reasoning_annotation"]
        gold_groups_only = [sorted(g["Words in Category"]) for g in gold_annotations]

        incorrect = match_groups(pred_groups, gold_groups_only)

        output.append({
            "new_token_prediction": new_pred,
            "reasoning_annotation": gold_annotations,
            "incorrect_inference_results": incorrect,
            "count_incorrect_inference_results": len(incorrect)
        })

    with open("results_output.json", "w") as f:
        json.dump(output, f, indent=2)

    print("✅ Simplified results saved to results_output.json")

if __name__ == "__main__":
    main()
