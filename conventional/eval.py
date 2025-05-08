# imports
from unsloth import is_bfloat16_supported, FastLanguageModel
from datasets import load_from_disk
from trl import SFTTrainer
import torch

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
train_test_split = dataset.train_test_split(test_size=0.1) # splits 10% off to test
train_dataset = train_test_split['train']
test_dataset = train_test_split['test']
print(f"INFO: Loaded dataset of {len(dataset)} samples")

# Note prompt lengths
prompt_lengths = [tokenizer(example['text'], return_tensors="pt")["input_ids"].shape[1] for example in test_dataset]
print(f"INFO: Max prompted length:{max(prompt_lengths)}") # maxing out at most at 1000 tokens in prompt

# configuration
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = train_dataset,
    eval_dataset = test_dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False,
)
print("INFO: Configured")

# evaluate on test
eval_result = trainer.evaluate()
print("Test Loss:", eval_result["eval_loss"])