# args
DATETIME = "20250505-000259"

# imports
from unsloth import FastLanguageModel
import os
import sys
import json
import math
import matplotlib.pyplot as plt
from embed_metrics import getAvgEmbedMetric
import gc
import torch

# set paths for inputs and outputs
ckpt_dir_path = f"/home/jamesbarrett_umass_edu/cs685proj-new/conventional/results/best_model/checkpoints"
unshuffled_dataset_path = "/home/jamesbarrett_umass_edu/cs685proj-new/data/output.json"
embed_plot_path = f"/home/jamesbarrett_umass_edu/cs685proj-new/conventional/results/best_model/embed_plot.png"

# Data prep
with open(unshuffled_dataset_path, 'r') as file:
    data = json.load(file)
rawWordLists = [item['allwords'] for item in data]
testWordLists = rawWordLists[math.ceil(len(rawWordLists)*0.9):]
print("dataset loaded")

# get values for grp sim
avgIntraGrpSims = []
avgExtraGrpSims = []
epochs = []
epochNo = 1
for ckpt_end_path in os.listdir(ckpt_dir_path):
    def run_model(pth):
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = os.path.join(ckpt_dir_path, pth),
            max_seq_length = 1256,
            dtype = None,
            load_in_4bit = True, 
        )
        return getAvgEmbedMetric(testWordLists[:3], model, tokenizer, pth)
    avgIntraGrpSim, avgExtraGrpSim = run_model(ckpt_end_path)
    avgIntraGrpSims.append(avgIntraGrpSim)
    avgExtraGrpSims.append(avgExtraGrpSim)
    epochs.append(epochNo)
    print(f"calculated values for checkpoint {epochNo}")
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
plt.savefig(embed_plot_path)
