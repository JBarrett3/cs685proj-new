import multiprocessing as mp
import os

def main():
    from unsloth import FastLanguageModel
    import torch
    import sys
    from datasets import Dataset, load_from_disk


    max_seq_length = 2048
    lora_rank = 32

    MODEL = "/datasets/ai/llama3/hub/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/5206a32e0bd3067aef1ce90f5528ade7d866253f"

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = MODEL,
        max_seq_length = max_seq_length,
        load_in_4bit = False, # False for LoRA 16bit
        fast_inference = True, # Enable vLLM fast inference
        max_lora_rank = lora_rank,
        gpu_memory_utilization = 0.8, # Reduce if out of memory
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r = lora_rank, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ], # Remove QKVO if out of memory
        lora_alpha = lora_rank,
        use_gradient_checkpointing = "unsloth", # Enable long context finetuning
        random_state = 3407,
    )


    import re
    from datasets import load_dataset, Dataset

    # Load and prep dataset
    SYSTEM_PROMPT = """
    Respond in the following format:
    <reasoning>
    ...
    </reasoning>
    <answer>
    ...
    </answer>
    """

    XML_COT_FORMAT = """\
    <reasoning>
    {reasoning}
    </reasoning>
    <answer>
    {answer}
    </answer>
    """

    def extract_xml_answer(text: str) -> str:
        answer = text.split("<answer>")[-1]
        answer = answer.split("</answer>")[0]
        return answer.strip()



    data = load_from_disk("connections_ds")
    split = data.train_test_split(test_size=0.1, seed=3407)
    train_dataset = split["train"]
    eval_dataset  = split["test"]

    def convert_ds_to_prompt(data) -> Dataset:
        data = data.map(lambda x: { 
            'prompt': [
                {'role': 'system', 'content': SYSTEM_PROMPT},
                {'role': 'user', 'content': x['input']}
            ],
            'answer': x['target']
        }) 
        return data

    dataset = convert_ds_to_prompt(train_dataset)






    def parse_completion(text: str, to_print=False) -> dict[str, list[str]]:
        out = {}
        for line in text.strip().splitlines():
            i1, i2 = line.find('['), line.rfind(']')
            if i1 == -1 or i2 == -1 or i2 < i1:
                continue
            key = line[:i1].rstrip(':').strip()
            items_str = line[i1+1:i2]
            items = []
            for itm in items_str.split(','):
                itm = itm.strip()
                if len(itm) >= 2 and itm[0] in "\"'" and itm[-1] == itm[0]:
                    itm = itm[1:-1]
                if itm:
                    items.append(itm)
            out[key] = items
        if(to_print):
            print("------")
            print("ORIGINAL")
            print(text)
            print("*******")
            print("OUTPUT")
            print(out)
        return out

    def soft_category_reward(
        prompts, completions, target, **kwargs
    ) -> list[float]:
        rewards = []
        for comp, gt_text in zip(completions, target):
            pred_dict = parse_completion(comp[0]["content"] if isinstance(comp, list) else comp, to_print=True)
            gt_dict   = parse_completion(gt_text)
            total = sum(len(v) for v in gt_dict.values())
            if total == 0:
                rewards.append(0.0)
                continue
            correct = sum(
                len(set(pred_dict.get(cat, [])) & set(words))
                for cat, words in gt_dict.items()
            )
            rewards.append(correct / total)
        return rewards

    def strict_category_reward(
        prompts, completions, target, **kwargs
    ) -> list[float]:
        rewards = []
        for comp, gt_text in zip(completions, target):
            pred_dict = parse_completion(comp[0]["content"] if isinstance(comp, list) else comp)
            gt_dict   = parse_completion(gt_text)
            ok = all(
                set(pred_dict.get(cat, [])) == set(words)
                for cat, words in gt_dict.items()
            )
            rewards.append(1.0 if ok else 0.0)
        return rewards
    
    def format_category_reward(
        prompts, completions, target, **kwargs
    ) -> list[float]:

        rewards = []
        for comp, gt_text in zip(completions, target):
            # extract dicts via your parse_completion
            pred_dict = parse_completion(
                comp[0]["content"] if isinstance(comp, list) else comp
            )
            gt_dict = parse_completion(gt_text)
            # compare counts
            if len(pred_dict) == len(gt_dict):
                rewards.append(0.5)
            else:
                rewards.append(0.0)
        return rewards

    def strict_format_reward_func(completions, **kwargs) -> list[float]:
        """Reward function that checks if the completion has a specific format."""
        pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
        responses = [completion[0]["content"] for completion in completions]
        matches = [re.match(pattern, r) for r in responses]
        return [0.5 if match else 0.0 for match in matches]

    def soft_format_reward_func(completions, **kwargs) -> list[float]:
        """Reward function that checks if the completion has a specific format."""
        pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
        responses = [completion[0]["content"] for completion in completions]
        matches = [re.match(pattern, r) for r in responses]
        return [0.5 if match else 0.0 for match in matches]

    def count_xml(text) -> float:
        count = 0.0
        if text.count("<reasoning>\n") == 1:
            count += 0.125
        if text.count("\n</reasoning>\n") == 1:
            count += 0.125
        if text.count("\n<answer>\n") == 1:
            count += 0.125
            count -= len(text.split("\n</answer>\n")[-1])*0.001
        if text.count("\n</answer>") == 1:
            count += 0.125
            count -= (len(text.split("\n</answer>")[-1]) - 1)*0.001
        return count

    def xmlcount_reward_func(completions, **kwargs) -> list[float]:
        contents = [completion[0]["content"] for completion in completions]
        return [count_xml(c) for c in contents]


    max_prompt_length = 256

    from trl import GRPOConfig, GRPOTrainer
    training_args = GRPOConfig(
        learning_rate = 5e-6,
        adam_beta1 = 0.9,
        adam_beta2 = 0.99,
        weight_decay = 0.1,
        warmup_ratio = 0.1,
        lr_scheduler_type = "cosine",
        optim = "paged_adamw_8bit",
        logging_steps = 1,
        per_device_train_batch_size = 36,
        gradient_accumulation_steps = 1, # Increase to 4 for smoother training
        num_generations = 6, # Decrease if out of memory
        max_prompt_length = max_prompt_length,
        max_completion_length = max_seq_length - max_prompt_length,
        num_train_epochs = 3, # Set to 1 for a full training run
        #max_steps = 1,
        save_steps = 100,
        max_grad_norm = 0.1,
        report_to = "none", # Can use Weights & Biases
        output_dir = "epoch3v3/outputs",
    )


    trainer = GRPOTrainer(
        model = model,
        processing_class = tokenizer,
        reward_funcs = [
            xmlcount_reward_func,
            soft_format_reward_func,
            strict_format_reward_func,
            soft_category_reward,
            strict_category_reward,
            format_category_reward
        ],
        args = training_args,
        train_dataset = dataset,
    )
    trainer.train()


    model.save_lora("epoch3v3/grpo_saved_lora")



if __name__ == "__main__":
    # Ensure 'spawn' is set before any multiprocessing is used
    mp.set_start_method("spawn", force=True)
    main()