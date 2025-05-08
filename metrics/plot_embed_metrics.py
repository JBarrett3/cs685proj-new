# args
CKPT_DIR_PATH = f"/home/jamesbarrett_umass_edu/cs685proj-new/conventional/checkpoints/tuned"
OUT_PATH = f"/home/jamesbarrett_umass_edu/cs685proj-new/conventional/embed_metrics/tuned"

# imports
from unsloth import FastLanguageModel
import os
import sys
import json
import math
import matplotlib.pyplot as plt
from helpers.script_embed_metrics import compute_group_metrics
import gc
import numpy as np
import torch
from datasets import load_from_disk
import random

# Data prep
def extract_words(input_text):
    start = input_text.find("Words: [") + len("Words: [")
    end = input_text.find("]", start)
    words_str = input_text[start:end].strip()
    words_list = [word.strip().strip("'") for word in words_str.split(",")]
    return words_list
dataset = load_from_disk("/home/jamesbarrett_umass_edu/cs685proj-new/data/connections_ds") # Note that you will need to run make_ds.py ahead of time to generate this dataset
dataset = dataset.map(lambda example: {"text": example["input"]}, remove_columns=["input"])
dataset = dataset.map(lambda example: {"label": example["target"]}, remove_columns=["target"])
train_test_split = dataset.train_test_split(test_size=0.1, seed=42) # splits 10% off to test
test_dataset = train_test_split['test']
testWordLists = np.array(list(map(extract_words, test_dataset['text']))).reshape(len(test_dataset), 4, 4)

# get values for grp sim
avgIntraGrpSims = []
avgExtraGrpSims = []
epochs = []
epochNo = 1
for ckpt_end_path in os.listdir(CKPT_DIR_PATH):
    def run_model(pth):
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = os.path.join(CKPT_DIR_PATH, pth),
            max_seq_length = 1256,
            dtype = None,
            load_in_4bit = True, 
        )
        full_pth_single = os.path.join(OUT_PATH, 'singleEx', pth)
        os.makedirs(full_pth_single, exist_ok=True)
        compute_group_metrics(model, tokenizer, testWordLists[0], full_pth_single, plot=True)
        full_pth_average = os.path.join(OUT_PATH, 'average', pth)
        os.makedirs(full_pth_average, exist_ok=True)
        results = compute_group_metrics(model, tokenizer, testWordLists, full_pth_average, plot=True)
        return results['avg_intra_group_similarity'], results['avg_out_group_similarity']
    avgIntraGrpSim, avgExtraGrpSim = run_model(ckpt_end_path)
    avgIntraGrpSims.append(avgIntraGrpSim)
    avgExtraGrpSims.append(avgExtraGrpSim)
    epochs.append(epochNo)
    print(f"calculated values for epoch {epochNo}")
    epochNo += 1
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
print("embed metrics calculated")

# plot values
plt.figure(figsize=(10, 6))
plt.plot(epochs, avgIntraGrpSims, label="Average Intraclass Group Similarity", color="red", marker='o')
plt.plot(epochs, avgExtraGrpSims, label="Average Extraclass Group Similarity", color="green", marker='o')
plt.xlabel("Epochs")
plt.ylabel("Group similarity")
plt.title("Group similarity over time")
plt.legend()
plt.grid(True)
plt.savefig(OUT_PATH)
