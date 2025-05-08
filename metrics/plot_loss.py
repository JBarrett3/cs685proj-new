# args
LOGPATH = "/home/jamesbarrett_umass_edu/cs685proj-new/conventional/checkpoints/tuned/checkpoint-50/trainer_state.json"
OUT_PATH = f"/home/jamesbarrett_umass_edu/cs685proj-new/conventional/losses/tuned"

# imports
import matplotlib.pyplot as plt
import json
import numpy as np
import os 

# read logs
os.makedirs(OUT_PATH, exist_ok=True)
with open(LOGPATH, 'r') as file:
    logs = json.load(file)['log_history']

# Initialize lists for plotting
train_losses = []
eval_losses = [1.8093947172164917] # starts without an eval value on 0 epoch, this is copied in manually from the log of test loss

# Process the logs
for log in logs:
    if "loss" in log:
        train_losses.append(log["loss"])
    if "eval_loss" in log:
        eval_losses.append(log["eval_loss"])
# NOTE that there is very little error handling here
#   and we go on the assumption that there will always
#   be a train and eval loss every epoch

# Plotting the losses
plt.figure(figsize=(10, 6))
plt.plot(np.arange(len(train_losses)), train_losses, label="Training Loss", color="red", marker='o')
plt.plot(np.arange(len(eval_losses)), eval_losses, label="Validation Loss", color="green", marker='o')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.grid(True)
plt.savefig(f"{OUT_PATH}/loss_plot.png")

# Plotting the losses
plt.figure(figsize=(10, 6))
plt.plot(np.arange(len(train_losses)), train_losses, label="Training Loss", color="red", marker='o')
plt.plot(np.arange(len(eval_losses)), eval_losses, label="Validation Loss", color="green", marker='o')
plt.xlabel("Epochs")
plt.ylabel("Log Loss")
plt.title("Log Training and Validation Loss")
plt.yscale("log")
plt.legend()
plt.grid(True)
plt.savefig(f"{OUT_PATH}/log_loss_plot.png")