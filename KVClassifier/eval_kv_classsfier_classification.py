import torch 
from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache, BitsAndBytesConfig
import tqdm
import argparse
import json
import gc
import datasets
from kv_classfier import KVClassifier


CACHE_DIR = "./models_storage"
# model_name = "Qwen/Qwen3-8B"

bnb_config = BitsAndBytesConfig(
    load_in_8bit_fp8=True  # <-- FP8 quantization
)
model_name = "Qwen/Qwen3-32B"
tot_layers = 64      # number of transformer blocks
n_heads = 8         # attention heads per layer
n_dim = 128          # dimension per head

torch.manual_seed(42)
torch.cuda.manual_seed(42)

# tot_layers = 36 # model specific, e.g., Qwen3-8B has 36 layers
# n_heads = 8 # model specific, e.g., Qwen3-8B has 8 heads each layer
# n_dim = 128 # model specific, e.g., Qwen3-8B has 128 dimensions per head

# model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
# tot_layers = 48 # model specific, e.g., Qwen3-8B has 36 layers
# n_heads = 8 # model specific, e.g., Qwen3-8B has 8 heads each layer
# n_dim = 128 # model specific, e.g., Qwen3-8B has 128 dimensions per head

seq_len = 64 # ust last 64 tokens for KV Classifier
n_layers = 2 # use KV cache from the last n layers
selected_layers = -1 # -1 means last n_layers, otherwise use the list


kv_classifier_device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def parse_args():
    parser = argparse.ArgumentParser(description="Run dual thinking trail on a math problem.")
    # parser.add_argument('--dataset_path', type=str, required=False, help='path to dataset', default='data/gsm8k/test.jsonl')
    parser.add_argument('--dataset_path', type=str, required=False, help='path to dataset', default='../math500/math500_test.jsonl')
    # parser.add_argument('--dataset_path', type=str, required=False, help='path to dataset', default='../data/qwen3_32b-q8_0_math500_train_first_200.jsonl')
    # parser.add_argument('--dataset_path', type=str, required=False, help='path to dataset', default=',data/math/train_trail_prep_1000.jsonl')
    # parser.add_argument('--dataset_path', type=str, required=False, help='path to dataset', default='data/math/test.jsonl')
    # parser.add_argument('--seq_len', type=int, required=False, help='sequence length for KV cache', default=32)
    # parser.add_argument('--n_layers', type=int, required=False, help='number of layers to use for KV cache', default=8)
    # parser.add_argument('--selected_layers', type=str, required=False, help='selected layers for KV cache, e.g., -1 for last n_layers, or a list of layer indices', default='-1')
    parser.add_argument('--data_prep_batch_size', type=int, required=False, help='batch size for preprocessing', default=128)
    parser.add_argument('--eval_batch_size', type=int, required=False, help='batch size for training', default=128)
    parser.add_argument('--checkpoint_path', type=str, required=False, help='checkpoint to eval', default="checkpoints_re_32b/kv_classifier_iter_179_seq_len_32_n_layers_8_selected_layers_56,57,58,59,60,61,62,63_batch_size_1024_lr_1e-05.pt")
    args = parser.parse_args()
    global seq_len, n_layers, selected_layers
    # seq_len = args.seq_len
    # n_layers = args.n_layers
    # if args.selected_layers == '-1':
    #     selected_layers = list(range(tot_layers - n_layers, tot_layers))
    # else:
    #     selected_layers = [int(x) for x in args.selected_layers.split(',')]
    # parse from the checkpoint name
    ckpt_parts = args.checkpoint_path.split('/')[-1].replace('.pt', '').split('_')
    seq_len = int(ckpt_parts[6])
    n_layers = int(ckpt_parts[9])
    selected_layers = [int(x) for x in ckpt_parts[12].split(',')]
    print(f"Using sequence length: {seq_len}, number of layers: {n_layers}, selected layers: {selected_layers}")
    datasets_path = args.dataset_path.split(',')
    dataset = datasets.load_dataset('json', data_files=datasets_path, split='train')
    return args, dataset


def get_kv_cache(prompts, model, tokenizer):
    inputs = tokenizer(prompts, return_tensors="pt", truncation=True, padding='max_length', max_length=seq_len).input_ids.cuda()
    
    
    past_key_values = DynamicCache()
    model.generate(inputs, past_key_values=past_key_values, use_cache=True, max_new_tokens=1)
    return past_key_values


def prep_kv_cache(dataset, batch_size):
    print("Preparing KV cache...")
    ret_k_cache = []
    ret_v_cache = []
    for i in tqdm.tqdm(range(0, len(dataset), batch_size)):
        batch = dataset[i:i+batch_size]
        
        if 'question' in batch:
            kv_cache = get_kv_cache(batch['question'], model, tokenizer)
        elif 'problem' in batch:
            kv_cache = get_kv_cache(batch['problem'], model, tokenizer)
        else:
            kv_cache = get_kv_cache(batch['prompt'], model, tokenizer)

        k_cache_selected = []
        v_cache_selected = []
        for idx in selected_layers:
            k_cache_selected.append(kv_cache.key_cache[idx])
            v_cache_selected.append(kv_cache.value_cache[idx])
        k_cache_selected = torch.stack(k_cache_selected, dim=1)
        k_cache_selected = k_cache_selected.to('cpu') # only cpu can store such large tensors
        v_cache_selected = torch.stack(v_cache_selected, dim=1)
        v_cache_selected = v_cache_selected.to('cpu')
        
        ret_k_cache.append(k_cache_selected)
        ret_v_cache.append(v_cache_selected)
    
    # show cuda memory usage
    print(f"CUDA memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"CUDA memory reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
    
    return torch.cat(ret_k_cache, dim=0), torch.cat(ret_v_cache, dim=0)

if __name__ == "__main__":
    args, dataset = parse_args()
    # model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=CACHE_DIR, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=CACHE_DIR, device_map="auto", quantization_config=bnb_config, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=CACHE_DIR, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    kv_classifier = KVClassifier(n_layers=n_layers, n_heads=n_heads, seq_len=seq_len, n_dim=n_dim).to(kv_classifier_device)
    print(f"Loading checkpoint from {args.checkpoint_path}...")
    kv_classifier.load_state_dict(torch.load(args.checkpoint_path, map_location=kv_classifier_device))    
    k_cache_columns, v_cache_columns = prep_kv_cache(dataset, args.data_prep_batch_size)
    print("KV cache prepared, now evaluating...")
    kv_classifier.eval()
    
    print('Deleting LLM to free up memory...')
    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    print(f"CUDA memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"CUDA memory reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
    
    # Evaluate the KV classifier
    fout = args.dataset_path.replace('.jsonl', '_results_{}_{}.jsonl'.format(seq_len, n_layers))
    with open(fout, 'w') as f_out:
        for i in tqdm.tqdm(range(0, k_cache_columns.shape[0], args.eval_batch_size)):
            # print(i, i+args.eval_batch_size)
            batch_k_cache = k_cache_columns[i:i+args.eval_batch_size]
            batch_v_cache = v_cache_columns[i:i+args.eval_batch_size]
            batch_k_cache = batch_k_cache.to(kv_classifier_device).bfloat16()
            batch_v_cache = batch_v_cache.to(kv_classifier_device).bfloat16()
            
            with torch.no_grad():
                predictions = kv_classifier(batch_k_cache, batch_v_cache).bfloat16()
                
            
            for j in range(batch_k_cache.shape[0]):
                f_out.write(json.dumps({
                    'prompt': dataset[i+j]['question'] if 'question' in dataset[i+j] else dataset[i+j]['problem'] if 'problem' in dataset[i+j] else dataset[i+j]['prompt'],
                    'difficulty': predictions[j].item(),
                }) + '\n')