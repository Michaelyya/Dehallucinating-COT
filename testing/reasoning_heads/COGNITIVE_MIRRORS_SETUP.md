# CognitiveMirrors Setup Guide

This guide explains how to use the reasoning head discovery framework with the CognitiveMirrors dataset.

## Step 1: Preprocess the Dataset

First, filter the dataset to keep only Logical Reasoning questions:

```bash
cd testing/reasoning_heads
python preprocess_cognitive_mirrors.py
```

This will:
- Load `../CognitiveMirrors/dataset/balanced_cot_train_data.json`
- Filter for questions with `"cognitive_skill": "Logical Reasoning"`
- Save to `cognitive_mirrors_logical_reasoning.json`

## Step 2: Discover Reasoning Heads

Run the discovery process:

```bash
python main.py \
    --discover \
    --model_name "meta-llama/Meta-Llama-3-8B-Instruct" \
    --scoring_method ablation \
    --n_examples 20 \
    --top_k 10 \
    --single_subtask "logical_reasoning" \
    --min_score 0.0 \
    --min_confidence 0.0 \
    --backward_chaining_dir "../CognitiveMirrors"
```

## Step 3: Evaluate Discovered Heads

After discovery, evaluate the heads:

```bash
python main.py \
    --evaluate \
    --model_name "meta-llama/Meta-Llama-3-8B-Instruct" \
    --heads_file "discovered_heads.json" \
    --subtask_filter "logical_reasoning" \
    --top_k_heads 10
```

## Key Differences from Backward-Chaining

1. **Dataset Format**: Uses JSON instead of line-by-line text
2. **Evaluation Metric**: Uses BLEU score instead of exact match
3. **Prompt Format**: Includes main question, context, and subquestion
4. **Subtask**: Focuses on "logical_reasoning" instead of "path_finding"

## BLEU Score Evaluation

The framework uses BLEU score to evaluate free-form text generation:
- Compares generated answers to reference answers
- Uses NLTK or evaluate library
- Reports average BLEU score as the "accuracy" metric

## Head Selection

Heads are selected based on:
- **Score**: Performance drop when head is masked (higher = more important)
- **Confidence**: Based on number of examples evaluated
- **Top K**: Selects top K heads per subtask

Heads that decrease BLEU score the most when masked are considered the most important for logical reasoning.

