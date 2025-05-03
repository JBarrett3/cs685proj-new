import json
from datasets import Dataset

input_template = (
    "Solve today's NYT Connections game.\n"
    "Here are the instructions for how to play this game:\n"
    "Find groups of four items that share something in common.\n"
    "Category Examples:\n"
    "FISH: Bass, Flounder, Salmon, Trout\n"
    "FIRE ___: Ant, Drill, Island, Opal\n"
    "Categories will always be more specific than '5-LETTER-WORDS', 'NAMES' or 'VERBS.'\n\n"
    "Example 1:\n"
    "Words: ['DART', 'HEM', 'PLEAT', 'SEAM', 'CAN', 'CURE', 'DRY', 'FREEZE', 'BITE', 'EDGE', 'PUNCH', 'SPICE', 'CONDO', 'HAW', 'HERO', 'LOO']\n"
    "Groupings:\n"
    "Things to sew: ['DART', 'HEM', 'PLEAT', 'SEAM']\n"
    "Ways to preserve food: ['CAN', 'CURE', 'DRY', 'FREEZE']\n"
    "Sharp quality: ['BITE', 'EDGE', 'PUNCH', 'SPICE']\n"
    "Birds minus last letter: ['CONDO', 'HAW', 'HERO', 'LOO']\n\n"
    "Example 2:\n"
    "Words: ['COLLECTIVE', 'COMMON', 'JOINT', 'MUTUAL', 'CLEAR', 'DRAIN', 'EMPTY', 'FLUSH', 'CIGARETTE', 'PENCIL', 'TICKET', 'TOE', 'AMERICAN', 'FEVER', 'LUCID', 'PIPE']\n"
    "Groupings:\n"
    "Shared: ['COLLECTIVE', 'COMMON', 'JOINT', 'MUTUAL']\n"
    "Rid of contents: ['CLEAR', 'DRAIN', 'EMPTY', 'FLUSH']\n"
    "Associated with \"stub\": ['CIGARETTE', 'PENCIL', 'TICKET', 'TOE']\n"
    "____ Dream: ['AMERICAN', 'FEVER', 'LUCID', 'PIPE']\n\n"
    "Example 3:\n"
    "Words: ['HANGAR', 'RUNWAY', 'TARMAC', 'TERMINAL', 'ACTION', 'CLAIM', 'COMPLAINT', 'LAWSUIT', 'BEANBAG', 'CLUB', 'RING', 'TORCH', 'FOXGLOVE', 'GUMSHOE', 'TURNCOAT', 'WINDSOCK']\n"
    "Groupings:\n"
    "Parts of an airport: ['HANGAR', 'RUNWAY', 'TARMAC', 'TERMINAL']\n"
    "Legal terms: ['ACTION', 'CLAIM', 'COMPLAINT', 'LAWSUIT']\n"
    "Things a juggler juggles: ['BEANBAG', 'CLUB', 'RING', 'TORCH']\n"
    "Words ending in clothing: ['FOXGLOVE', 'GUMSHOE', 'TURNCOAT', 'WINDSOCK']\n\n"
    "Categories share commonalities:\n"
    "- There are 4 categories of 4 words each\n"
    "- Every word will be in only 1 category\n"
    "- One word will never be in two categories\n"
    "- There will never be a miscellaneous category\n"
    "- As the category number increases, the connections between the words and their category become more obscure. Category 1 is the most easy and intuitive and Category 4 is the hardest\n"
    "- There may be red herrings (words that seem to belong together but actually are in separate categories)\n"
    "- Category 4 often contains compound words with a common prefix or suffix word\n"
    "- A few other common categories include word and letter patterns, pop culture clues (such as music and movie titles) and fill-in-the-blank phrases\n\n"
    "You will be given a new example (Example 4) with today's list of words.\n"
    "First explain your reason for each category and then give your final answer following the structure below (Replace Category1, Category2, Category3, Category4 with their names instead):\n"
    "Groupings:\n"
    "Category1: [word1, word2, word3, word4]\n"
    "Category2: [word5, word6, word7, word8]\n"
    "Category3: [word9, word10, word11, word12]\n"
    "Category4: [word13, word14, word15, word16]\n\n"
    "Remember that the same word cannot be repeated across multiple categories, and you need to output 4 categories with 4 distinct words each. Also do not make up words not in the list. This is the most important rule. Please obey\n\n"
    "Example 4:\n"
    "Words: {words_list}\n"
    "Groupings:"
)

def format_target(groupings):
    lines = []
    for group in groupings:
        category = group["Categories"]
        words = group["Words in Category"]
        lines.append(f"{category.upper()}: {words}")
    return "\n".join(lines)

def make_ds(file, save_to_disk=True):
    with open(file, "r") as f:
        puzzles = json.load(f) 
    data_examples = []
    for puzzle in puzzles:
        words = str(puzzle["allwords"])
        input = input_template.format(words_list=words)
        target = format_target(puzzle["reasoning_annotation"])
        data_examples.append({
            "input": input,
            "target": target
        })

    ds = Dataset.from_list(data_examples)
    if save_to_disk:
        ds.save_to_disk("connections_ds")
    return ds

def main(file="output.json"):
    ds = make_ds(file)

    for i in range(5):
        sample = ds[i]
        print(f"Sample {i + 1}:")
        print("input:")
        print(sample["input"])
        print("target:")
        print(sample["target"])
        print("-" * 50)


if __name__ == "__main__":
    main()

