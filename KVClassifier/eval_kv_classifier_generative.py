import torch 
from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache, BitsAndBytesConfig
import tqdm
import argparse
import json
import gc
import datasets
from kv_classfier import KVClassifier


CACHE_DIR = "./models_storage"

torch.manual_seed(42)
torch.cuda.manual_seed(42)

# model_name = "Qwen/Qwen3-8B"
# tot_layers = 36 # model specific, e.g., Qwen3-8B has 36 layers
# n_heads = 8 # model specific, e.g., Qwen3-8B has 8 heads each layer
# n_dim = 128 # model specific, e.g., Qwen3-8B has 128 dimensions per head

# model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
# tot_layers = 48 # model specific, e.g., Qwen3-8B has 36 layers
# n_heads = 8 # model specific, e.g., Qwen3-8B has 8 heads each layer
# n_dim = 128 # model specific, e.g., Qwen3-8B has 128 dimensions per head

bnb_config = BitsAndBytesConfig(
    load_in_8bit_fp8=True  # <-- FP8 quantization
)
model_name = "Qwen/Qwen3-32B"
tot_layers = 64      # number of transformer blocks
n_heads = 8         # attention heads per layer
n_dim = 128          # dimension per head

seq_len = 64 # ust last 64 tokens for KV Classifier
n_layers = 2 # use KV cache from the last n layers
selected_layers = -1 # -1 means last n_layers, otherwise use the list
threshold_delta = 20


# kv_classifier_device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
kv_classifier_device = 'cpu'

def parse_args():
    parser = argparse.ArgumentParser(description="Run dual thinking trail on a math problem.")
    # parser.add_argument('--dataset_path', type=str, required=False, help='path to dataset', default='data/gsm8k/train_trail_prep_1000.jsonl,data/math/train_trail_prep_1000.jsonl')
    parser.add_argument('--dataset_path', type=str, required=False, help='path to dataset', default='data/math/test.jsonl')
    parser.add_argument('--checkpoint_path', type=str, required=False, help='checkpoint to eval', default="checkpoints_re_32b/kv_classifier_iter_179_seq_len_32_n_layers_8_selected_layers_56,57,58,59,60,61,62,63_batch_size_1024_lr_1e-05.pt")
    parser.add_argument('--allow_switch', action='store_true', help='allow switching between fast and slow thinking', default=True)
    args = parser.parse_args()
    global seq_len, n_layers, selected_layers
    ckpt_parts = args.checkpoint_path.split('/')[-1].replace('.pt', '').split('_')
    seq_len = int(ckpt_parts[6])
    n_layers = int(ckpt_parts[9])
    selected_layers = [int(x) for x in ckpt_parts[12].split(',')]
    print(f"Using sequence length: {seq_len}, number of layers: {n_layers}, selected layers: {selected_layers}")
    datasets_path = args.dataset_path.split(',')
    dataset = datasets.load_dataset('json', data_files=datasets_path, split='train')
    print('fout = ', args.dataset_path.replace('.jsonl', '_results_multi_{}_{}.jsonl'.format(seq_len, n_layers)))
    return args, dataset

def select_kv_cache(kv_cache, ending_pos = -1):
    k_cache_selected, v_cache_selected = [], []
    if ending_pos == -1:
        ending_pos = kv_cache.key_cache[0].shape[-2]
    for idx in selected_layers:
        k_cache_selected.append(kv_cache.key_cache[idx][:,:,ending_pos-seq_len:ending_pos])
        v_cache_selected.append(kv_cache.value_cache[idx][:,:,ending_pos-seq_len:ending_pos])
    k_cache_selected = torch.stack(k_cache_selected, dim=1)
    v_cache_selected = torch.stack(v_cache_selected, dim=1)
    k_cache_selected = k_cache_selected.to(kv_classifier_device)
    v_cache_selected = v_cache_selected.to(kv_classifier_device)
    return k_cache_selected, v_cache_selected

def select_kv_cache_multi_pos(kv_cache, starting_pos, ending_pos=-1):
    k_cache_selected, v_cache_selected = [], []
    if ending_pos == -1:
        ending_pos = kv_cache.key_cache[0].shape[-2]
    for i in range(starting_pos, ending_pos):
        # print(i, select_kv_cache(kv_cache, i).shape)
        k_cache_selected_i, v_cache_selected_i = select_kv_cache(kv_cache, i)
        k_cache_selected.append(k_cache_selected_i[0])
        v_cache_selected.append(v_cache_selected_i[0])
    k_cache_selected = torch.stack(k_cache_selected, dim=0) # (num_pos, n_layers, n_heads, seq_len, n_dim)
    v_cache_selected = torch.stack(v_cache_selected, dim=0)
    k_cache_selected = k_cache_selected.to(kv_classifier_device)
    v_cache_selected = v_cache_selected.to(kv_classifier_device)
    return k_cache_selected, v_cache_selected

def run_prompt(prompt, model, tokenizer, kv_classifier, truthfulqa=False):
    prompt_think = """<|im_start|>user
You are a math problem solver. For the following question, solve it using your thinking ability.
Here is the question:
{}
<|im_end|>
<|im_start|>assistant\n""".format(prompt)
    if 'DeepSeek' in model_name:
        prompt_think = """<｜User｜> You are a math problem solver. For the following question, solve it using your thinking ability.
Here is the question:
{}
<｜Assistant｜>\n""".format(prompt)
    if truthfulqa:
        prompt_think = f"""<|im_start|>user
You are a factual QA system. Answer the following question with detailed thinking.
Question:
{prompt}
<|im_end|>
<|im_start|>assistant
<think>
"""


    inputs = tokenizer(prompt_think, return_tensors="pt", padding='max_length', max_length=seq_len).input_ids.cuda()
    past_key_values = DynamicCache()
    model.generate(inputs, past_key_values=past_key_values, use_cache=True, max_new_tokens=1)
    
    # add first token according to kv_classifier
    k_cache, v_cache = select_kv_cache(past_key_values)
    predictions = kv_classifier(k_cache.bfloat16(), v_cache.bfloat16())
    difficulty = predictions[0].item()
    print(f"Prompt: {prompt}, Difficulty: {difficulty}")
    
    # add think tag according to difficulty
    first_think_mode = 'fast' if difficulty < 50 else 'slow'
    prompt_think += "<think>\n</think>\n\n" if first_think_mode == 'fast' else "<think>\n"
    
    inputs = tokenizer(prompt_think, return_tensors="pt", padding='max_length', max_length=seq_len).input_ids.cuda()
    past_key_values = DynamicCache()
    output_first = model.generate(inputs, past_key_values=past_key_values, use_cache=True, max_new_tokens=2048 if first_think_mode == 'fast' else 8192)[0].tolist()
    output_first_str = tokenizer.decode(output_first[inputs.shape[1]:], skip_special_tokens=True)
    
    switched = False
    if args.allow_switch:
        # allow one time switch between fast and slow thinking
        # print(past_key_values.key_cache[0].shape, past_key_values.value_cache[0].shape)
        k_cache, v_cache = select_kv_cache_multi_pos(past_key_values, inputs.shape[1] + 1) # for each generated token, use it as an possible ending position
        # print(f"KV cache shape: {k_cache.shape}, {v_cache.shape}, inputs shape: {inputs.shape}, output_first shape: {len(output_first)}")
        predictions = kv_classifier(k_cache.bfloat16(), v_cache.bfloat16())
        # convert to float and move to cpu for further processing
        difficulty = predictions.float().cpu().numpy()
        # find the first position where reqired difficulty switching
        for i in range(0, difficulty.shape[0]):
            if difficulty[i] < 50 - threshold_delta and first_think_mode == 'slow':
                # switch to fast thinking frrom this position
                second_think_mode, switched = 'fast', True
                switched_pos, switched_difficulty = inputs.shape[1] + i + 1, difficulty[i].item()
                print(f"Switching to fast thinking at position {switched_pos}, difficulty: {switched_difficulty}")
                output_first_selected = output_first[:inputs.shape[1] + i + 1]
                prompt_second = tokenizer.decode(output_first_selected, skip_special_tokens=True) + "</think>\n\n"
                inputs_second = tokenizer(prompt_second, return_tensors="pt").input_ids.cuda()
                output_second = model.generate(inputs_second, max_new_tokens=2048)[0].tolist()
                output_second_str = tokenizer.decode(output_second[inputs_second.shape[1]:], skip_special_tokens=True)
                break
            if difficulty[i] > 50 + threshold_delta and first_think_mode == 'fast':
                # switch to slow thinking from this position
                second_think_mode, switched = 'slow', True
                switched_pos, switched_difficulty = inputs.shape[1] + i + 1, difficulty[i].item()
                print(f"Switching to slow thinking at position {switched_pos}, difficulty: {switched_difficulty}")
                output_first_selected = output_first[:inputs.shape[1] + i + 1]
                prompt_second = tokenizer.decode(output_first_selected, skip_special_tokens=True) + "<think>\n"
                inputs_second = tokenizer(prompt_second, return_tensors="pt").input_ids.cuda()
                output_second = model.generate(inputs_second, max_new_tokens=8192)[0].tolist()
                output_second_str = tokenizer.decode(output_second[inputs_second.shape[1]:], skip_special_tokens=True)
                break
    
    if not switched:
        return {
            'prompt': prompt,
            'prompt_think': prompt_think,
            'output_first_str': output_first_str,
            'output_first_tokens': len(output_first) - inputs.shape[1],
            'first_think_mode': first_think_mode,
            'switched': False,
        }
    else:
        return {
            'prompt': prompt,
            'prompt_think': prompt_think,
            'output_first_str': output_first_str,
            'output_first_tokens': len(output_first) - inputs.shape[1],
            'first_think_mode': first_think_mode,
            'switched': True,
            'switched_pos': switched_pos,
            'switched_difficulty': switched_difficulty,
            'prompt_second': prompt_second,
            'output_second_str': output_second_str,
            'output_second_tokens': len(output_second) - inputs_second.shape[1],
            'second_think_mode': second_think_mode,
        }

if __name__ == "__main__":
    args, dataset = parse_args()
    # take 2 samples from the dataset
    # dataset = dataset.select(range(1, 10))
    
    model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=CACHE_DIR, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True) # for 14B, we have to use multi-gpu
    # model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=CACHE_DIR, torch_dtype=torch.bfloat16, device_map="cuda:0", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=CACHE_DIR, device_map="auto", quantization_config=bnb_config, trust_remote_code=True)


    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=CACHE_DIR, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    kv_classifier = KVClassifier(n_layers=n_layers, n_heads=n_heads, seq_len=seq_len, n_dim=n_dim).to(kv_classifier_device)
    kv_classifier.load_state_dict(torch.load(args.checkpoint_path, map_location=kv_classifier_device))
    kv_classifier.eval()
    
    print("LLM and KV classifier loaded.")
    if 'truthfulqa' in args.dataset_path.lower():
        print("Evaluating on TruthfulQA dataset...")
    with torch.no_grad():
        # Evaluate the KV classifier
        fout = args.dataset_path.replace('.jsonl', '_results_multi_{}_{}.jsonl'.format(seq_len, n_layers))
        with open(fout, 'w') as f_out:
            for i in tqdm.tqdm(range(0, len(dataset))):
                sample = dataset[i]
                prompt = sample['question'] if 'question' in sample else sample['problem'] if 'problem' in sample else sample['prompt']
                result = run_prompt(prompt, model, tokenizer, kv_classifier, truthfulqa=('truthfulqa' in args.dataset_path.lower()))
                f_out.write(json.dumps(result) + '\n')
                f_out.flush()