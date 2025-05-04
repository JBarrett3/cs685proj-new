# args
DATETIME = "20250504-133812"

# imports
import matplotlib.pyplot as plt
import json
import numpy as np

# read logs
with open(f"/home/jamesbarrett_umass_edu/cs685proj-new/conventional/losses/{DATETIME}.json", 'r') as file:
    logs = json.load(file)

# Initialize lists for plotting
train_losses = []
eval_losses = []
steps = []

# Process the logs
for log in logs:
    if "loss" in log:
        train_losses.append(log["loss"])
        steps.append(log["step"])
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
plt.savefig(f"/home/jamesbarrett_umass_edu/cs685proj-new/conventional/loss_plots/{DATETIME}.png")