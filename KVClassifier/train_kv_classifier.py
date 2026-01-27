import torch 
from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache, BitsAndBytesConfig
import tqdm
import argparse
import json
import gc
import datasets
from kv_classfier import KVClassifier

bnb_config = BitsAndBytesConfig(
    load_in_8bit_fp8=True  # <-- FP8 quantization
)

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

model_name = "Qwen/Qwen3-32B"
tot_layers = 64      # number of transformer blocks
n_heads = 8         # attention heads per layer
n_dim = 128          # dimension per head

seq_len = 64 # ust last 64 tokens for KV Classifier
n_layers = 4 # use KV cache from the last n layers
selected_layers = -1 # -1 means last n_layers, otherwise use the list


kv_classifier_device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def parse_args():
    parser = argparse.ArgumentParser(description="Run dual thinking trail on a math problem.")
    # parser.add_argument('--dataset_path', type=str, required=False, help='path to dataset', default='data/gsm8k/train_trail_prep_1000.jsonl,data/math/train_trail_prep_1000.jsonl')
    parser.add_argument('--dataset_path', type=str, required=False, help='path to dataset', default='data/gsm8k/train_trail_prep.jsonl,data/math/train_trail_prep.jsonl')
    parser.add_argument('--seq_len', type=int, required=False, help='sequence length for KV cache', default=64)
    parser.add_argument('--n_layers', type=int, required=False, help='number of layers to use for KV cache', default=4)
    parser.add_argument('--selected_layers', type=str, required=False, help='selected layers for KV cache, e.g., -1 for last n_layers, or a list of layer indices', default='-1')
    parser.add_argument('--data_prep_batch_size', type=int, required=False, help='batch size for preprocessing', default=128)
    parser.add_argument('--batch_size', type=int, required=False, help='batch size for training', default=1024)
    parser.add_argument('--learning_rate', type=float, required=False, help='learning rate for training', default=1e-5)
    parser.add_argument('--train_iter', type=int, required=False, help='training iteration', default=500)
    parser.add_argument('--save_period', type=int, required=False, help='save period for training', default=10)
    args = parser.parse_args()
    global seq_len, n_layers, selected_layers
    seq_len = args.seq_len
    n_layers = args.n_layers
    if args.selected_layers == '-1':
        selected_layers = list(range(tot_layers - n_layers, tot_layers))
    else:
        selected_layers = [int(x) for x in args.selected_layers.split(',')]
    print(f"Using sequence length: {seq_len}, number of layers: {n_layers}, selected layers: {selected_layers}")
    datasets_path = args.dataset_path.split(',')
    dataset = datasets.load_dataset('json', data_files=datasets_path, split='train')
    return args, dataset


def get_kv_cache(prompts, model, tokenizer):
    inputs = tokenizer(prompts, return_tensors="pt", truncation=True, padding='max_length', max_length=seq_len).input_ids.cuda()
    
    
    past_key_values = DynamicCache()
    model.generate(inputs, past_key_values=past_key_values, use_cache=True, max_new_tokens=1)
    return past_key_values

def parse_prompts(prompts):
    ret = []
    for prompt in prompts:
        prompt_think = """<|im_start|>user
You are a math problem solver. For the following question, solve it using your thinking ability.
Here is the question:
{}
<|im_end|>
<|im_start|>assistant\n""".format(prompt)
        if 'DeepSeek' in model_name:
            prompt_think = """｜User｜>
You are a math problem solver. For the following question, solve it using your thinking ability.
Here is the question:
{}
｜Assistant｜\n""".format(prompt)
        ret.append(prompt_think)
    return ret

def prep_kv_cache(dataset, batch_size, parse=True):
    print("Preparing KV cache...")
    ret_k_cache = []
    ret_v_cache = []
    for i in tqdm.tqdm(range(0, len(dataset), batch_size)):
        batch = dataset[i:i+batch_size]

        kv_cache = get_kv_cache(parse_prompts(batch['prompt']) if parse else batch['prompt'], model, tokenizer)

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
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        # torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=CACHE_DIR, trust_remote_code=True, truncation_side='left', padding_side='left')
    tokenizer.pad_token = tokenizer.eos_token
    
    kv_classifier = KVClassifier(n_layers=n_layers, n_heads=n_heads, seq_len=seq_len, n_dim=n_dim).to(kv_classifier_device) 
    kv_classifier.train()
    optimizer = torch.optim.Adam(kv_classifier.parameters(), lr=args.learning_rate)
    
    dataset = dataset.select(range(50))  # for testing, use a smaller subset
    k_cache_columns_parsed, v_cache_columns_parsed = prep_kv_cache(dataset, args.data_prep_batch_size)
    labels_parsed = torch.tensor(dataset['difficulty']).bfloat16()
    k_cache_columns_nonparsed, v_cache_columns_nonparsed = prep_kv_cache(dataset, args.data_prep_batch_size, parse=False)
    labels_nonparsed = labels_parsed
    
    # cat parsed and non-parsed kv_cache_columns together
    # kv_cache_columns = torch.cat([kv_cache_columns_parsed, kv_cache_columns_nonparsed], dim=0)
    k_cache_columns = torch.cat([k_cache_columns_parsed, k_cache_columns_nonparsed], dim=0)
    v_cache_columns = torch.cat([v_cache_columns_parsed, v_cache_columns_nonparsed], dim=0)
    labels_columns = torch.cat([labels_parsed, labels_nonparsed], dim=0)
    
    # print(k_cache_columns.shape, v_cache_columns.shape, labels_columns.shape)
    
    print('Deleting LLM to free up memory...')
    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    print(f"CUDA memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"CUDA memory reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
    
    # random select 10% of the dataset for evaluation
    eval_size = int(k_cache_columns.shape[0] * 0.1)
    eval_indices = torch.randperm(k_cache_columns.shape[0])[:eval_size]
    
    eval_k_cache_columns = k_cache_columns[eval_indices]
    eval_v_cache_columns = v_cache_columns[eval_indices]
    eval_labels = labels_columns[eval_indices]

    batch_size = args.batch_size
    for iteration in range(args.train_iter):  # number of training iterations
        for i in tqdm.tqdm(range(0, k_cache_columns.shape[0], batch_size)):
            # kv_cache = kv_cache_columns[i:i+batch_size].to(kv_classifier_device)
            k_cache = k_cache_columns[i:i+batch_size].to(kv_classifier_device).bfloat16()
            v_cache = v_cache_columns[i:i+batch_size].to(kv_classifier_device).bfloat16()
            labels = labels_columns[i:i+batch_size].to(kv_classifier_device).bfloat16()
            
            print(k_cache.shape, v_cache.shape, labels.shape)

            output = kv_classifier(k_cache, v_cache)
            loss = torch.nn.functional.mse_loss(output, labels)
            # print(loss, output.shape, labels.shape)
            kv_classifier.zero_grad()
            loss.backward()
            optimizer.step()
            # print(f"Batch {i//batch_size + 1}, Loss: {loss.item()}")

        eval_k_cache = eval_k_cache_columns.to(kv_classifier_device).bfloat16()
        eval_v_cache = eval_v_cache_columns.to(kv_classifier_device).bfloat16()
        eval_labels = eval_labels.to(kv_classifier_device).bfloat16()
        output = kv_classifier(eval_k_cache, eval_v_cache)
        loss = torch.nn.functional.mse_loss(output, eval_labels)
        # print(output, eval_labels)
        print(f"Iteration {iteration + 1}/{args.train_iter}, Evaluation Loss: {loss.item()}")
        
        # Save the model after each iteration
        # add seq_len, n_layers, selected_layers, batch_size, learning_rate, iteration to the filename
        if (iteration + 1) % args.save_period == 0:
            print(f"Saving model at iteration {iteration + 1}...")
            save_filename = f"checkpoints_re/kv_classifier_iter_{iteration}_seq_len_{seq_len}_n_layers_{n_layers}_selected_layers_{','.join(map(str, selected_layers))}_batch_size_{batch_size}_lr_{args.learning_rate}.pt"
            # create directory if not exists
            import os
            os.makedirs(os.path.dirname(save_filename), exist_ok=True)
            torch.save(kv_classifier.state_dict(), save_filename)
    print("Training completed.")
        
    