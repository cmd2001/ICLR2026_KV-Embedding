import json
from collections import Counter

def count_difficulty(jsonl_path):
    difficulty_counter = Counter()

    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Skipping line {line_num}: JSON decode error: {e}")
                continue

            if "difficulty" not in obj:
                print(f"Skipping line {line_num}: no 'difficulty' field")
                continue

            difficulty = obj["difficulty"]
            difficulty_counter[difficulty] += 1

    # Print result sorted by difficulty
    for diff in sorted(difficulty_counter.keys()):
        print(f"difficulty={diff}: {difficulty_counter[diff]}")

    print("\nTotal items:", sum(difficulty_counter.values()))


if __name__ == "__main__":
    # Change 'data.jsonl' to your file path
    count_difficulty("data/math/train_trail_prep.jsonl")
