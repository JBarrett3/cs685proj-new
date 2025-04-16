from unsloth import FastLanguageModel
import torch
from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer
import json, random, re
from datasets import Dataset


max_seq_length = 1024     # Maximum sequence length for reasoning traces
lora_rank = 32            # Chosen LoRA rank

model, tokenizer = FastLanguageModel.from_pretrained(
    #model_name = "meta-llama/meta-Llama-3.1-8B-Instruct",
    model_name = "/datasets/ai/llama3/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659",
    max_seq_length = max_seq_length,
    load_in_4bit = True,
    fast_inference = True,
    max_lora_rank = lora_rank,
    gpu_memory_utilization = 0.6,
)

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

# Data prep
def convert_puzzle_to_example(puzzle):
    date = puzzle.get("date", "")
    index = puzzle.get("puzzle_index", "unknown")
    allwords = puzzle.get("allwords", [])
    annotations = puzzle.get("reasoning_annotation", [])
    
    prompt = f"Puzzle (Date: {date}, Index: {index}):\n" + ", ".join(allwords) + "\nAnswer:"
    
    reasoning_lines = []
    for ann in annotations:
        category = ann.get("Categories", "Unknown Category")
        complexity = ann.get("Complexity", 1)
        words_in_cat = ann.get("Words in Category", [])
        line = f"{category} (Complexity {complexity}): " + ", ".join(words_in_cat)
        reasoning_lines.append(line)
    reasoning_str = "\n".join(reasoning_lines)
    
    answer = f"<reasoning>\n{reasoning_str}\n</reasoning>\n<answer>\n{reasoning_str}\n</answer>\n"
    return {"prompt": prompt, "answer": answer}

DATA_FILE = "data/puzzles.json"
with open(DATA_FILE, "r") as f:
    puzzle_data = json.load(f)

training_examples = [convert_puzzle_to_example(p) for p in puzzle_data]
print(f"Converted {len(training_examples)} puzzles to training examples.")
train_dataset = Dataset.from_list(training_examples)

# Reward
def extract_xml_answer(text: str) -> str:
    """Extract text between <answer> and </answer> tags."""
    try:
        answer = text.split("<answer>")[-1].split("</answer>")[0]
        return answer.strip()
    except Exception:
        return ""

def get_completion_content(completion):
    """
    Helper to extract the content string from a completion.
    If completion is a list of dicts, returns the content of the first item.
    If it's a dict, returns the value for 'content'.
    If it's already a string, returns it.
    """
    if isinstance(completion, list):
        if len(completion) > 0 and isinstance(completion[0], dict):
            return completion[0].get("content", "")
        else:
            return str(completion)
    elif isinstance(completion, dict):
        return completion.get("content", "")
    elif isinstance(completion, str):
        return completion
    else:
        return str(completion)

def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    """Reward based on count of expected XML tags."""
    def count_xml(text: str) -> float:
        count = 0.0
        if text.count("<reasoning>\n") == 1:
            count += 0.125
        if text.count("\n</reasoning>\n") == 1:
            count += 0.125
        if text.count("\n<answer>\n") == 1:
            count += 0.125
        if text.count("\n</answer>") == 1:
            count += 0.125
        return count
    contents = [get_completion_content(comp) for comp in completions]
    return [count_xml(c) for c in contents]

def strict_format_reward_func(completions, **kwargs) -> list[float]:
    """Checks if completion exactly matches the expected XML format."""
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
    responses = [get_completion_content(comp) for comp in completions]
    return [0.5 if re.match(pattern, r) else 0.0 for r in responses]

def soft_format_reward_func(completions, **kwargs) -> list[float]:
    """Checks if completion contains the expected XML structure."""
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = [get_completion_content(comp) for comp in completions]
    return [0.5 if re.search(pattern, r) else 0.0 for r in responses]

def int_reward_func(completions, **kwargs) -> list[float]:
    """Dummy reward if the extracted answer is a number."""
    responses = [get_completion_content(comp) for comp in completions]
    extracted = [extract_xml_answer(r) for r in responses]
    return [0.5 if r.isdigit() else 0.0 for r in extracted]

def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    """Dummy correctness reward comparing the extracted answer with the target answer."""
    responses = [get_completion_content(comp) for comp in completions]
    extracted = [extract_xml_answer(r) for r in responses]
    return [2.0 if r.strip() == a.strip() else 0.0 for r, a in zip(extracted, answer)]

# GRPO
max_prompt_length = 256

training_args = GRPOConfig(
    learning_rate = 5e-6,
    adam_beta1 = 0.9,
    adam_beta2 = 0.99,
    weight_decay = 0.1,
    warmup_ratio = 0.1,
    lr_scheduler_type = "cosine",
    optim = "paged_adamw_8bit",
    logging_steps = 1,
    per_device_train_batch_size = 1,
    gradient_accumulation_steps = 1,  # Increase to smooth gradients if needed
    num_generations = 6,             # Number of generations per rollout
    max_prompt_length = max_prompt_length,
    max_completion_length = max_seq_length - max_prompt_length,
    max_steps = 250,
    save_steps = 250,
    max_grad_norm = 0.1,
    report_to = "none",              # Change to "wandb" for W&B reporting
    output_dir = "outputs",
)

# Training
trainer = GRPOTrainer(
    model = model,
    processing_class = tokenizer,
    reward_funcs = [
        xmlcount_reward_func,
        soft_format_reward_func,
        strict_format_reward_func,
        int_reward_func,
        correctness_reward_func,
    ],
    args = training_args,
    train_dataset = train_dataset,
)

print("Starting GRPO fine-tuning...")
trainer.train()
trainer.save_model(training_args.output_dir)
print(f"GRPO fine-tuned model saved in {training_args.output_dir}")

# Inference
from vllm import SamplingParams

sample = random.choice(puzzle_data)
inference_prompt = f"Puzzle (Date: {sample['date']}, Index: {sample['puzzle_index']}):\n" + ", ".join(sample["allwords"]) + "\nAnswer:"

sampling_params = SamplingParams(
    temperature = 0.8,
    top_p = 0.95,
    max_tokens = 1024,
)

generated_output = model.fast_generate(
    [inference_prompt],
    sampling_params = sampling_params,
    lora_request = None,
)[0].outputs[0].text

print("Inference prompt:")
print(inference_prompt)
print("\nGenerated Completion:")
print(generated_output)
