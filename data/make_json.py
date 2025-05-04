import json
import re
import random

def parse_file(input_file):
    with open(input_file, 'r') as f:
        lines = f.readlines()
    
    puzzles = []

    for i in range(0, len(lines), 5):
        header = lines[i].strip().replace('“', '"').replace('”', '"')
        parts = header.split(" - ")
        match = re.search(r'\d+', parts[0])
        index = -1
        if match:
            index = int(match.group())
        date = parts[1].strip()

        puzzle = {
            "date": date,
            "puzzle_index": index,
            "allwords": [],
            "reasoning_annotation": []
        }
        
        complexity = 0
        for line in lines[i+1 : i+5]:
            line = line.strip().replace('“', '"').replace('”', '"')
            category, words_str = line.split(" - ")
            words = [word.strip() for word in words_str.split(',')]
            complexity += 1
            puzzle["allwords"].extend(words)
            puzzle["reasoning_annotation"].append({
                "Categories": category,
                "Words in Category": words,
                "Complexity": complexity
            })
        random.shuffle(puzzle["allwords"])
        puzzles.append(puzzle)
    
    return puzzles

def main():
    random.seed(123)
    in_file = "raw.txt"
    out_file = "primary_dataset.json"

    puzzles = parse_file(in_file)
    
    with open(out_file, 'w') as out:
        json.dump(puzzles, out, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    main()
