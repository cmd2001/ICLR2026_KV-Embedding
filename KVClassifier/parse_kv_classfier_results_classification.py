import argparse
import datasets
import json
import tqdm
from eval.parser import extract_answer

def load_generation_results(generation_result_path):
    ret = {
        'output_fast_think': [],
        'output_slow_think': [],
        'output_fast_think_tokens': [],
        'output_slow_think_tokens': []
    }
    with open(generation_result_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            if 'output_fast_think' in data:
                ret['output_fast_think'].append(data['output_fast_think'])
                ret['output_slow_think'].append(data['output_slow_think'])
                ret['output_fast_think_tokens'].append(data['output_fast_think_tokens'])
                ret['output_slow_think_tokens'].append(data['output_slow_think_tokens'])
            elif 'fast_think_output' in data:
                ret['output_fast_think'].append(data['fast_think_output'])
                ret['output_slow_think'].append(data['slow_think_output'])
                ret['output_fast_think_tokens'].append(data['fast_think_output_tokens'])
                ret['output_slow_think_tokens'].append(data['slow_think_output_tokens'])
            else:
                raise ValueError("Data does not contain expected output columns. Please check the generation result format.")
    return ret
def parse_args():
    parser = argparse.ArgumentParser(description="Parse KV Classifier results.")
    parser.add_argument('--dataset_path', type=str, required=False, help='path to dataset', default='../math500/math500_test.jsonl')
    parser.add_argument('--eval_result_path', type=str, required=False, help='path to KVClassfier Eval Results', default='../math500/math500_test_results.jsonl')
    parser.add_argument('--generation_result_path', type=str, required=False, help='path to Base Model Generation Results', default='../math500/qwen3_32b-q8_0_math500_results_method1.jsonl')
    
    args = parser.parse_args()
    
    dataset = datasets.load_dataset('json', data_files=args.dataset_path, split='train')
    dataset_eval_result = datasets.load_dataset('json', data_files=args.eval_result_path, split='train')
    dataset_generation_result = load_generation_results(args.generation_result_path)
    
    assert len(dataset) == len(dataset_eval_result) == len(dataset_generation_result['output_fast_think']), "Dataset, Eval Result and Generation Result must have the same length."
    return args, dataset, dataset_eval_result, dataset_generation_result


if __name__ == "__main__":
    args, dataset, dataset_eval_result, dataset_generation_result = parse_args()
    
    dataset_type = 'gsm8k' if 'gsm8k' in args.dataset_path else 'math'
    
    is_correct_fast, avg_tokens_fast = 0, 0
    is_correct_slow, avg_tokens_slow = 0, 0
    is_correct_hybrid, avg_tokens_hybrid = 0, 0
    for i in tqdm.tqdm(range(len(dataset))):
        
        require_slow_thinking = dataset_eval_result['difficulty'][i] > 50
        answer_ref = extract_answer(dataset['answer'][i], dataset_type)
        
        if require_slow_thinking:
            think_output = dataset_generation_result['output_slow_think'][i]
            think_output_tokens = dataset_generation_result['output_slow_think_tokens'][i]
        else:
            think_output = dataset_generation_result['output_fast_think'][i]
            think_output_tokens = dataset_generation_result['output_fast_think_tokens'][i]
            
        think_output_answer = extract_answer(think_output, dataset_type)
        
        is_correct_hybrid += think_output_answer == answer_ref
        avg_tokens_hybrid += think_output_tokens
        
        is_correct_fast += extract_answer(dataset_generation_result['output_fast_think'][i], dataset_type) == answer_ref
        avg_tokens_fast += dataset_generation_result['output_fast_think_tokens'][i]
        is_correct_slow += extract_answer(dataset_generation_result['output_slow_think'][i], dataset_type) == answer_ref
        avg_tokens_slow += dataset_generation_result['output_slow_think_tokens'][i]
    avg_tokens_hybrid /= len(dataset)
    avg_tokens_fast /= len(dataset)
    avg_tokens_slow /= len(dataset)
    print(f"Dataset: {args.dataset_path.split('/')[-1]}")
    print(f"Fast Thinking Correct: {is_correct_fast} / {len(dataset)} ({is_correct_fast / len(dataset) :.3f})")
    print(f"Average Tokens Fast: {avg_tokens_fast:.2f}")
    print(f"Slow Thinking Correct: {is_correct_slow} / {len(dataset)} ({is_correct_slow / len(dataset) :.3f})")
    print(f"Average Tokens Slow: {avg_tokens_slow:.2f}")
    print(f"Hybrid Thinking Correct: {is_correct_hybrid} / {len(dataset)} ({is_correct_hybrid / len(dataset) :.3f})")
    print(f"Average Tokens Hybrid: {avg_tokens_hybrid:.2f}")