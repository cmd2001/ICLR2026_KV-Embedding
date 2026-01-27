# Beyond Speedup – Utilizing KV Cache for Sampling and Reasoning
**Accepted by ICLR 2026**

This repository contains the code used to evaluate KV-cache–based methods proposed in our ICLR 2026 submission, including MTEB evaluation, KV-CoE inference, and KV-based classification.

## MTEB
We follow the official MTEB evaluation protocol without modification.

Set up the environment according to the official MTEB instructions.

After the environment is ready, run `custom_model.py`, and another baselines with different suffix names.

The script evaluates our KV-cache–based setup by exposing KV representations as embeddings within the MTEB framework.

## KV-CoE
Create the environment using `requirements.txt`.

Run `Scripts/llm_infer.sh` to perform LLM inference based KV-based representations.

Run `Scripts/llm_eval.sh` to evaluate the performance.

## KVClassifier
The KVClassifier pipeline consists of the following steps:

0. Set up the environment using `requirements.txt`.
1. Run `prep_fast_slow_thinking_results.py` to generate baseline fast and slow thinking results.
2. Run `prep_kv_classfier_training_data.py` to construct the training dataset for KVClassifier. The distribution of difficulty labels can be inspected using `count_difficulty.py`.
3. Run `train_kv_classifier.py` to train the KVClassifier.
4. Run `eval_kv_classifier_classification.py`, followed by `parse_kv_classifier_results_classification.py`, to evaluate the KVClassifier in the classification setting.
5. Run `eval_kv_classifier_generative.py`, followed by `parse_kv_classifier_results_generative.py`, to evaluate the KVClassifier in the generative setting.

The core implementation of KVClassifier is provided in `kv_classfier.py`.

## Reference

```bibtex
@inproceedings{
anonymous2026beyond,
title={Beyond Speedup - Utilizing {KV} Cache for Sampling and Reasoning},
author={Xing, Zeyu and Li, Xing and Zhen, Hui-Ling and Yuan, Mingxuan and Pan, Sinno Jialin},
booktitle={The Fourteenth International Conference on Learning Representations},
year={2026},
url={https://openreview.net/forum?id=GUhmiJaAzv}
}
```