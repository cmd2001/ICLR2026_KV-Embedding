import torch 
from transformers import AutoTokenizer, AutoModelForCausalLM
import tqdm
import argparse
import json
import datasets


CACHE_DIR = "./models_storage"
# model_name = "Qwen/Qwen3-8B"
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"


def parse_args():
    parser = argparse.ArgumentParser(description="Run dual thinking trail on a math problem.")
    parser.add_argument('--dataset_path', type=str, required=False, help='path to dataset', default='data/gsm8k/test.jsonl')
    return parser.parse_args()

def run_prompt(prompt, model, tokenizer):
    
    if 'Qwen' in model_name:
        prompt_fast_think = """<|im_start|>user
You are a math problem solver. For the following question, solve it using your direct answering ability.
Here is the question:
{}
<|im_end|>
<|im_start|>assistant\n<think>\n</think>\n\n""".format(prompt)
    elif 'DeepSeek' in model_name:
        prompt_fast_think = """<｜User｜> You are a math problem solver. For the following question, solve it using your direct answering ability.
Here is the question:
{}
<｜Assistant｜>\n<think>\n</think>\n\n""".format(prompt)
    else:
        raise ValueError("Unknown model name")
    
    if 'Qwen' in model_name:
        prompt_slow_think = """<|im_start|>user
You are a math problem solver. For the following question, solve it using your thinking ability.
Here is the question:
{}
<|im_end|>
<|im_start|>assistant\n<think>\n""".format(prompt)
    elif 'DeepSeek' in model_name:
        prompt_slow_think = """<｜User｜> You are a math problem solver. For the following question, solve it using your thinking ability.
Here is the question:
{}
<｜Assistant｜>\n<think>\n""".format(prompt)
    else:
        raise ValueError("Unknown model name")
    
    inputs_fast = tokenizer(prompt_fast_think, return_tensors="pt").input_ids.cuda()
    inputs_slow = tokenizer(prompt_slow_think, return_tensors="pt").input_ids.cuda()
    
    # generate the actual outputs
    outputs_fast = model.generate(inputs_fast, max_new_tokens=2048)
    outputs_slow = model.generate(inputs_slow, max_new_tokens=8192)
    
    # no batch decode, only one prompt
    output_fast_think = tokenizer.decode(outputs_fast[0].tolist()[inputs_fast.shape[1]:], skip_special_tokens=True)
    output_slow_think = tokenizer.decode(outputs_slow[0].tolist()[inputs_slow.shape[1]:], skip_special_tokens=True)
    
    
    return {
        'prompt': prompt,
        'prompt_fast_think': prompt_fast_think,
        'prompt_slow_think': prompt_slow_think,
        'output_fast_think': output_fast_think,
        'output_slow_think': output_slow_think,
        'output_fast_think_tokens': outputs_fast.shape[1] - inputs_fast.shape[1],
        'output_slow_think_tokens': outputs_slow.shape[1] - inputs_slow.shape[1],
    }


if __name__ == "__main__":
    model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=CACHE_DIR, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=CACHE_DIR, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    args = parse_args()
    
    fout = args.dataset_path.replace('.jsonl', '_dpsk_14b_trail.jsonl')
    
    dataset = datasets.load_dataset('json', data_files=args.dataset_path, split='train')
    # dataset = dataset['question']
    if 'question' in dataset.column_names:
        dataset = dataset['question']
    elif 'problem' in dataset.column_names:
        dataset = dataset['problem']
    
    
    with open(fout, 'w') as f:
        for prompt in tqdm.tqdm(dataset):
            result = run_prompt(prompt, model, tokenizer)
            f.write(json.dumps(result) + '\n')
    print(f"Results saved to {fout}")
