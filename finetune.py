import os
import json
import random
import torch
from unsloth import FastLanguageModel, Trainer  # Unsloth's fast fine-tuning API
from transformers import AutoTokenizer
import numpy as np

# -------------------------------------------------
# 1. Data Loading and Preprocessing
# -------------------------------------------------
def load_puzzle_data(json_file):
    """Load puzzles from a JSON file that contains an array of puzzle objects."""
    with open(json_file, "r") as f:
        data = json.load(f)
    return data

def create_grpo_example(puzzle):
    """
    Convert a puzzle JSON into an input-output pair for GRPO.
    
    The input includes:
      - A prompt with puzzle metadata (date, index, words)
      - A directive to provide an answer
    
    The target (reward signal) is indirectly derived from the gold-standard grouping.
    Here, we format the reward text (or score) as a string that the model must reproduce.
    """
    date = puzzle.get("date", "")
    index = puzzle.get("puzzle_index", "unknown")
    words = puzzle.get("allwords", [])
    annotations = puzzle.get("reasoning_annotation", [])

    prompt = f"Puzzle (Date: {date}, Index: {index}):\n" + ", ".join(words) + "\nAnswer:"
    
    # For GRPO, we want to compute a reward signal.
    # In this simplified version, we'll convert the annotations into a target string.
    reward_lines = []
    for ann in annotations:
        category = ann.get("Categories", "Unknown Category")
        complexity = ann.get("Complexity", "?")
        cat_words = ann.get("Words in Category", [])
        line = f"{category} (Complexity {complexity}): " + ", ".join(cat_words)
        reward_lines.append(line)
    target_reward = " ".join(reward_lines)
    
    # For GRPO we may also include a scalar reward.
    # Here, we compute a dummy reward: lower total complexity means a higher reward.
    total_complexity = sum(ann.get("Complexity", 1) for ann in annotations)
    # We'll define reward = 10 / (total_complexity) as a simple example.
    reward_value = 10.0 / (total_complexity if total_complexity > 0 else 1)
    
    return {"input": prompt, "target_text": target_reward, "reward": reward_value}

def build_grpo_dataset(json_file, output_filename="grpo_training_data.jsonl"):
    puzzles = load_puzzle_data(json_file)
    examples = [create_grpo_example(puzzle) for puzzle in puzzles]
    with open(output_filename, "w") as f:
        for ex in examples:
            json.dump(ex, f)
            f.write("\n")
    print(f"Saved {len(examples)} GRPO training examples to {output_filename}")
    return examples

# -------------------------------------------------
# 2. GRPO Training Setup with Unsloth
# -------------------------------------------------
# File paths
DATA_FILE = "data/puzzles.json"  # your JSON file with puzzles
TRAINING_DATA_FILE = "grpo_training_data.jsonl"
OUTPUT_DIR = "results/grpo_fine_tuned_model"

# Build the GRPO training dataset
_ = build_grpo_dataset(DATA_FILE, TRAINING_DATA_FILE)

# Set your base model ID
MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"

# Initialize the base model using Unsloth with quantization
model = FastLanguageModel.from_pretrained(
    MODEL_ID,
    load_in_4bit=True,   # Use 4-bit quantization for efficiency
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# Load GRPO training examples from JSONL
def load_jsonl(file_path):
    examples = []
    with open(file_path, "r") as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))
    return examples

raw_train_data = load_jsonl(TRAINING_DATA_FILE)

# Preprocess the data: combine prompt and target text.
# For GRPO we use the combined string as the training text.
def preprocess_grpo(example):
    # We include both the prompt and the expected target text.
    training_text = f"{example['input']}\nExpected: {example['target_text']}"
    # We also attach the reward value as metadata for later use in policy optimization.
    return {"text": training_text, "reward": example["reward"]}

train_dataset = [preprocess_grpo(ex) for ex in raw_train_data]

# Define a simple reward function.
# In a real GRPO setup, you would compute a reward based on how close the model's output is to target.
# Here, we simply use the provided reward value.
def reward_function(generated_text, target_text):
    # For demonstration, reward is negative absolute difference in length between generated and target.
    # More sophisticated methods (e.g. semantic similarity) can be used.
    return -abs(len(generated_text) - len(target_text))

# Define GRPO training hyperparameters (modify as needed)
training_args = {
    "num_train_epochs": 3,
    "per_device_train_batch_size": 4,
    "learning_rate": 2e-5,
    "logging_steps": 10,
    "output_dir": OUTPUT_DIR,
    # Additional GRPO-specific settings could be added here, such as reward scaling factors.
}

# Create a custom Trainer that can integrate the reward function.
# This is a simplified version that uses the provided training data.
class GRPOTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        # Standard cross-entropy loss from base Trainer.
        outputs = model(**inputs)
        loss = outputs.loss

        # Here, you would normally modify the loss by incorporating the reward signal.
        # For example, if the generated output (from `model.generate`) is available,
        # you could compute a policy gradient loss. This example adds a dummy reward term.
        # (In practice, you would perform a rollout, compute reward, and add a reinforcement learning loss.)
        dummy_reward = torch.tensor(0.0, device=loss.device)
        if "reward" in inputs:
            dummy_reward = inputs["reward"].float().mean() * 0.01  # scale reward term
        total_loss = loss - dummy_reward  # subtract reward to maximize it
        if return_outputs:
            return total_loss, outputs
        return total_loss

# Initialize our GRPO trainer.
grpo_trainer = GRPOTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    args=training_args
)

print("Starting GRPO fine-tuning...")
grpo_trainer.train()
grpo_trainer.save_model(OUTPUT_DIR)
print(f"GRPO fine-tuned model saved in {OUTPUT_DIR}")
