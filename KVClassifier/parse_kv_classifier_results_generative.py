import argparse
import datasets
import json
import tqdm
from eval.parser import extract_answer

def parse_args():
    parser = argparse.ArgumentParser(description="Parse KV Classifier results.")
    parser.add_argument('--dataset_path', type=str, required=False, help='path to dataset', default='../gsm8k/gsm8k_test.jsonl')
    parser.add_argument('--eval_result_path', type=str, required=False, help='path to KVClassfier Eval Results', default='../gsm8k/gsm8k_test_results_multi_32_8.jsonl')

    args = parser.parse_args()
    
    return args


if __name__ == "__main__":
    args = parse_args()

    dataset_type = 'gsm8k' if 'gsm8k' in args.dataset_path else 'math'
    
    # for each line, read output_second_str if not, read output_first_str
    # test if the output is correct, then add output_second_tokens or output_first_tokens to the total tokens
    # reference column: answer
    # all parsed by extract_answer
    dataset = datasets.load_dataset('json', data_files=args.dataset_path, split='train')
    with open(args.eval_result_path, 'r') as f:
        dataset_eval_result = [json.loads(line) for line in f.readlines()]
    correct = 0
    tokens = 0
    for i in tqdm.tqdm(range(len(dataset_eval_result))):
        output = dataset_eval_result[i]['output_second_str'] if 'output_second_str' in dataset_eval_result[i] else dataset_eval_result[i]['output_first_str']
        # tokens += dataset_eval_result[i]['output_second_tokens'] if 'output_second_tokens' in dataset_eval_result[i] else dataset_eval_result[i]['output_first_tokens']
        # output = dataset_eval_result[i]['output_first_str']
        tokens += dataset_eval_result[i]['output_second_tokens'] if 'output_second_tokens' in dataset_eval_result[i] else dataset_eval_result[i]['output_first_tokens']
        answer = extract_answer(output, dataset_type)
        if answer == extract_answer(dataset[i]['answer'], dataset_type):
            correct += 1
    print(f'Correct: {correct}/{len(dataset_eval_result)}')
    print(f'Accuracy: {correct / len(dataset_eval_result):.4f}')
    print(f'Tokens per example: {tokens / len(dataset_eval_result):.4f}')