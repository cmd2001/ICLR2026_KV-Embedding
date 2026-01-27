import tqdm
import argparse
from eval.parser import extract_answer
import json


def parse_args():
    parser = argparse.ArgumentParser(description="Prep data")
    parser.add_argument('--filename', type=str, required=False, help='path to dataset', default='data/gsm8k/train_trail.jsonl')
    return parser.parse_args().filename, 'gsm8k' if 'gsm8k' in parser.parse_args().filename else 'math'


filename, dataset_name = parse_args()
filename_ref = filename.replace('_trail', '')
fou = filename.replace('.jsonl', '_prep.jsonl')
with open(filename, 'r') as f, open(filename_ref, 'r') as f_ref, open(fou, 'w') as fout:
    for line, line_ref in tqdm.tqdm(zip(f.readlines(), f_ref.readlines())):
        line = json.loads(line)
        line_ref = json.loads(line_ref)
        answer_fast_think = extract_answer(line['output_fast_think'], dataset_name)
        answer_slow_think = extract_answer(line['output_slow_think'], dataset_name)
        answer_ref = extract_answer(line_ref['answer'], dataset_name)
        
        slow_think_correct = answer_slow_think == answer_ref
        fast_think_correct = answer_fast_think == answer_ref
        
        difficulty = -1
        if fast_think_correct:
            if line['output_fast_think_tokens'] < 128:
                difficulty = 0
            else:
                difficulty = 25
        elif slow_think_correct:
            difficulty = 75
        else:
            difficulty = 100
        
        fout.write(json.dumps({
            'prompt': line['prompt'],
            'slow_think_correct': slow_think_correct,
            'fast_think_correct': fast_think_correct,
            'output_fast_think_tokens': line['output_fast_think_tokens'],
            'output_slow_think_tokens': line['output_slow_think_tokens'],
            'difficulty': difficulty,
        }) + '\n')

# generate first 50 slice and first 1000 slice
import os
os.system(f'head -n 50 {fou} > {fou.replace(".jsonl", "_50.jsonl")}')
os.system(f'head -n 1000 {fou} > {fou.replace(".jsonl", "_1000.jsonl")}')