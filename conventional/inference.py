# args
MODEL_PATH = f"/home/jamesbarrett_umass_edu/cs685proj-new/conventional/results/best_model/checkpoints/checkpoint-48"
OUT_PATH = f'/home/jamesbarrett_umass_edu/cs685proj-new/conventional/results/best_model/inferences.json'

# imports
import json
from unsloth import is_bfloat16_supported, FastLanguageModel
from datasets import load_from_disk
from trl import SFTTrainer
from transformers import TrainingArguments, TrainerCallback
import torch
from datetime import datetime
from tqdm import tqdm
import os
import gc

# model loading
max_seq_length = 1256 # prompts are ~1000, so leaving 256 for response
dtype = None
load_in_4bit = True
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name =  "unsloth/Llama-3.1-8B-Instruct", 
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit
)
print("INFO: Model loaded")

# data prep
dataset = load_from_disk("/home/jamesbarrett_umass_edu/cs685proj-new/data/connections_ds") # Note that you will need to run make_ds.py ahead of time to generate this dataset
dataset = dataset.map(lambda example: {"text": example["input"]}, remove_columns=["input"])
dataset = dataset.map(lambda example: {"label": example["target"]}, remove_columns=["target"])
train_test_split = dataset.train_test_split(test_size=0.1) # splits 10% off to test
train_dataset = train_test_split['train']
test_dataset = train_test_split['test']
print(f"INFO: Loaded dataset of {len(dataset)} samples")

# inference
FastLanguageModel.for_inference(model) # Enable native 2x faster inference
predictions = []
for input in tqdm(test_dataset['text']):
    tok_input = tokenizer([input], truncation=True, padding=True, return_tensors="pt").to('cuda')
    output = model.generate(input_ids=tok_input['input_ids'][0].unsqueeze(0), attention_mask=tok_input['attention_mask'][0].unsqueeze(0), max_new_tokens = 128, use_cache = True)
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    predictions.append({
        'input_sentence': input,
        'whole prediction': generated_text,
        'new token prediction': generated_text.split('Groupings:')[-1]
    })
with open(OUT_PATH, 'w') as json_file:
    json.dump(predictions, json_file, indent=4)
print("INFO: Inference complete")