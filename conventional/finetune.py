# imports
import argparse
import json
from unsloth import is_bfloat16_supported, FastLanguageModel
from datasets import load_from_disk
from trl import SFTTrainer
from transformers import TrainingArguments, TrainerCallback
import torch
from datetime import datetime
from tqdm import tqdm
import subprocess
import os
import gc

# GPU check
result = subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
output = result.stdout.decode("utf-8")
print(output)
print("INFO: PyTorch Version:", torch.__version__)
print("INFO: CUDA Version:", torch.version.cuda)
print("INFO: cuDNN Version:", torch.backends.cudnn.version())
print("INFO: CUDA Available:", torch.cuda.is_available())
print("INFO: GPU Count:", torch.cuda.device_count())

# arg parse
parser = argparse.ArgumentParser(description="Model training script")
parser.add_argument("--training", action="store_true", help="Flag to train the model (default: just evaluation)")
parser.add_argument("--limit", type=int, default=1000, help="Limit on number of samples")
parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
parser.add_argument("--batch_size", type=int, default=10, help="Batch size for training")
parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
args = parser.parse_args()
TRAINING = args.training
LIMIT = args.limit
EPOCHS = args.epochs
BATCH_SIZE = args.batch_size
LR = args.lr
print(f"INFO: Training: {TRAINING}, Limit: {LIMIT}, Epochs: {EPOCHS}, Batch Size: {BATCH_SIZE}, Learning Rate: {LR}")

# model loading
max_seq_length = 1256 # prompts are ~1000, so leaving 256 for response
dtype = None
load_in_4bit = True
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Llama-3.1-8B-Instruct", # "unsloth/Llama-3.1-8B-Instruct",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit
)
print("INFO: Model loaded")


# layer freezing
total_layers = len(model.model.layers)
freeze_percentage = 80
num_freeze_layers = int(total_layers * (freeze_percentage / 100))
for i, layer in enumerate(model.model.layers):
    if i < num_freeze_layers:
        for param in layer.parameters():
            if param.dtype in [torch.float16, torch.bfloat16, torch.float32]:
                param.requires_grad = False
    else:
        for param in layer.parameters():
            if param.dtype in [torch.float16, torch.bfloat16, torch.float32]:
                param.requires_grad = True
print(f"INFO: First {num_freeze_layers} of {total_layers} layers frozen. Rest are trainable.")

# qlora
lora_rank = 32
model = FastLanguageModel.get_peft_model(
    model,
    r = lora_rank,
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_alpha = lora_rank,
    use_gradient_checkpointing = "unsloth",  # Enable long context finetuning
    random_state = 3407,
)
print("INFO: QLora applied")

# Data prep
dataset = load_from_disk("/home/jamesbarrett_umass_edu/cs685proj-new/data/connections_ds") # Note that you will need to run make_ds.py ahead of time to generate this dataset
dataset = dataset.map(lambda example: {"text": example["input"]}, remove_columns=["input"])
dataset = dataset.map(lambda example: {"label": example["target"]}, remove_columns=["target"])
if LIMIT != -1:
    dataset = dataset.select(range(0, LIMIT))
train_test_split = dataset.train_test_split(test_size=0.1) # splits 10% off to test
train_dataset = train_test_split['train']
test_dataset = train_test_split['test']
print(f"INFO: Loaded dataset of {len(dataset)} samples")

# Note prompt lengths
prompt_lengths = [tokenizer(example['text'], return_tensors="pt")["input_ids"].shape[1] for example in test_dataset]
print(f"INFO: Max prompted length:{max(prompt_lengths)}") # maxing out at most at 1000 tokens in prompt

# configuration
date = datetime.now().strftime('%Y%m%d-%H%M%S')
class ClearCacheCallback(TrainerCallback):
    def on_train_end(self, args, state, control, **kwargs):
        torch.cuda.empty_cache()
        print("cleaned cache post train\n")
        return control
    def on_evaluate(self, args, state, control, **kwargs):
        torch.cuda.empty_cache()
        print("cleaned cache post eval\n")
        return control
    def on_epoch_end(self, args, state, control, **kwargs):
        gc.collect()
        torch.cuda.empty_cache()
        print("Cache cleared after epoch\n")
        return control
training_args = TrainingArguments(
    per_device_train_batch_size = BATCH_SIZE,
    per_device_eval_batch_size = 1, # really fast even with small batches, so might as well not risk it
    gradient_accumulation_steps = 4,
    warmup_steps = 5,
    num_train_epochs = EPOCHS,
    learning_rate = LR,
    fp16 = not is_bfloat16_supported(),
    bf16 = is_bfloat16_supported(),
    optim = "adamw_8bit",
    weight_decay = 0.01,
    lr_scheduler_type = "linear",
    seed = 3407,
    output_dir = f"/home/jamesbarrett_umass_edu/cs685proj-new/conventional/checkpoints/{date}",
    logging_dir = f"/home/jamesbarrett_umass_edu/cs685proj-new/conventional/model_logs/{date}",
    logging_first_step = True,
    logging_strategy = "epoch",
    eval_strategy = "epoch",
    save_strategy = "epoch",
    report_to = "none",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    prediction_loss_only=True
)
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = train_dataset,
    eval_dataset = test_dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False,
    args = training_args,
    callbacks=[ClearCacheCallback()]
)
print("INFO: Configured")

# fine tune (if TRAINING)
if TRAINING:
    trainer_stats = trainer.train()
    print("INFO: Trained model")
    # output metrics
    with open(f"/home/jamesbarrett_umass_edu/cs685proj-new/conventional/losses/{date}.json", "w") as f:
        json.dump(trainer.state.log_history, f, indent=2)
else:
    print("INFO: Bypassing training")

# clear cache after training
gc.collect()
torch.cuda.empty_cache()
torch.cuda.ipc_collect()

# evaluate on test
eval_result = trainer.evaluate()
print("Test Loss:", eval_result["eval_loss"])

# clear cache after evaluation
gc.collect()
torch.cuda.empty_cache()
torch.cuda.ipc_collect()

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
os.makedirs(f"/home/jamesbarrett_umass_edu/cs685proj-new/conventional/inferences/{date}/")
with open(f'/home/jamesbarrett_umass_edu/cs685proj-new/conventional/inferences/{date}/trained={not TRAINING}.json', 'w') as json_file:
    json.dump(predictions, json_file, indent=4)
print("INFO: Inference complete")