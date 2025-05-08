# NYT Crossword Solver

## Structure

This repository contains four primary folders: `conventional`, `data`, `grpo`, `inference`, and `metrics`.

### `conventional`
This folder contains all the code relevant to training a standard-SFT (Supervised Fine-Tuning) model. 

- **finetune.ipynb** was used to test and converge on the final script, **finetune.py**, which trains the model. 
- **finetune.sh** is a batch script used to run the training process.

Additionally, there are folders for results, which include:
- checkpoints
- embed_metrics,
- inferences,
- losses,
- slurm_logs

We believe that these folders are all quite self-explanatory and note that the checkpoints are not committed to the repository due to their large size. However, these checkpoitns can be regenerated easily by running the `finetune` notebook, Python script, or batch script.

### `data`
This folder contains the raw NYT games file (`raw.txt`), which is converted into a JSON format (`originalJSON.json`) by **make_json.py**. The data from `raw.txt` are also shuffled and saved as `shuffledJSON` by **make_shuffled_json.py**. The shuffled data is further processed into a `connections_ds` format that can be used for the model by **make_ds.py**.

### `grpo`
\todo{Michael on GRPO}

### `inference`
This is a folder with utiltiies to help us annotate the results of the model.

### `metrics`
This folder contains the visualization code for evaluating model performance:

- **plot_embed_metrics_untrained.py** generates visualizations for the baseline untrained model.
- **plot_embed_metrics.py** iterates over checkpoints of a model to generate visualizations for each checkpoint.
- **plot_loss.py** generates standard and log scaled loss plots for training and validation loss over epochs.

Note that there is also a subdirectory, `helpers`, that contains earlier versions of the code for posterity and helper functions for the primary functions mentioned previously:

- **notebook_embed_metrics.ipynb** is used for initial engineering and visualizing metrics.
- **script_embed_metrics.py** is used to implement the same metrics in a script format.
