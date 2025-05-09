import json
import math
import re
import openai
from datasets import load_from_disk, Dataset
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
import torch



def main():
    data = load_from_disk("connections_ds")
    split = data.train_test_split(test_size=0.1, seed=3407)
    train_dataset = split["train"]
    eval_dataset  = split["test"]

    prompts = eval_dataset["input"]


    base_model = "/datasets/ai/llama3/hub/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/5206a32e0bd3067aef1ce90f5528ade7d866253f"

    t = ""
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        t= "bfloat16"
    else:
         t= "float16"

    llm = LLM(
        model=base_model,
        dtype=t,
        enable_lora=True,
        max_model_len = 16384,
        max_lora_rank=32
    )

    
    lora_adapter_path = "epoch3v2/grpo_saved_lora/"
    lora_req = LoRARequest("my_lora", 1, lora_adapter_path)

    
    sampling_params = SamplingParams(
        temperature=0.5,
        top_p=0.9,
        max_tokens=512
    )

    outputs = llm.generate(
        prompts,
        sampling_params,
        lora_request=lora_req   
    ) 

    targets = eval_dataset["target"]  
    results = []
    for i, out in enumerate(outputs):
        print("Prompt:", out.prompt)
        print("Completion:", out.outputs[0].text)
        print("-" * 60)

        results.append({
            "target": targets[i],
            "completion": out.outputs[0].text
        })

    with open("completions3v2.json", "w") as f:
        json.dump(results, f, indent=2)


    import csv
    with open("completions3v2.csv", "w", newline="", encoding="utf-8") as f_csv:
        writer = csv.DictWriter(f_csv, fieldnames=["target", "completion"])
        writer.writeheader()
        for row in results:
            writer.writerow(row)

if __name__ == "__main__":
    main()