# Task Logic Explanation

This document explains how each subtask is constructed and how the head scoring works.

## Overview

The framework discovers reasoning heads by:
1. **Identifying subtasks** from backward-chaining-circuits dataset and code
2. **Scoring heads** using ablation: measure performance drop when a head is masked
3. **Ranking heads** by their importance to each subtask

## Subtask Construction

### 1. Edge Parsing (`edge_parsing`)
**What it does**: Parse edge tokens from input and construct graph representation

**Logic**:
- Input format: `"12>4,14>12,1>2,...|13:10>7>3>5>..."`
- The model needs to parse the comma-separated edges (e.g., "12>4" means node 12 points to node 4)
- Extract edges and build a graph structure
- **Scoring**: Heads that help parse edge tokens are important

**Example**:
```
Input: "12>4,14>12,1>2|13:10>7>3"
Edges: [(12,4), (14,12), (1,2)]
Goal: 13
```

### 2. Goal Identification (`goal_identification`)
**What it does**: Identify the goal node from input

**Logic**:
- The input contains a goal node after the `|` separator
- Format: `"edges|goal:path"`
- The model needs to identify which node is the target
- **Scoring**: Heads that attend to the goal token are important

**Example**:
```
Input: "12>4,14>12|13:10>7>3"
Goal: 13 (the number after | and before :)
```

### 3. Path Finding (`path_finding`)
**What it does**: Find path from root to goal using backward-chaining

**Logic**:
- Given a graph and goal, find the path from root to goal
- This is the core backward-chaining reasoning task
- The model must traverse the graph backward from goal to root
- **Scoring**: Heads that help navigate the graph are critical

**Example**:
```
Graph: 12->4, 14->12, 1->2
Goal: 13
Path: 10>7>3>5>0>1>2>6>14>12>15>9>11>13
```

### 4. Node Traversal (`node_traversal`)
**What it does**: Traverse graph nodes step-by-step in backward direction

**Logic**:
- Similar to path finding but focuses on individual steps
- Each step: current node -> previous node in path
- Requires maintaining state of current position
- **Scoring**: Heads that track position and select next node are important

### 5. Graph Construction (`graph_construction`)
**What it does**: Construct graph representation from edge tokens

**Logic**:
- Convert edge list into a graph data structure
- Build adjacency relationships
- **Scoring**: Heads that help build representations are important

### 6. Backward Chain Step (`backward_chain_step`)
**What it does**: Execute one step of backward-chaining reasoning

**Logic**:
- Single reasoning step: given current node, find parent node
- Iterative process that builds the path
- **Scoring**: Heads that perform individual reasoning steps

### 7. Path Validation (`path_validation`)
**What it does**: Validate that generated path is correct

**Logic**:
- Check if generated path matches expected path
- Verify path is valid in the graph
- **Scoring**: Heads that help verify correctness

### 8. Token Prediction (`token_prediction`)
**What it does**: Predict next token in backward-chaining sequence

**Logic**:
- Standard language modeling task
- Predict next node in the path sequence
- **Scoring**: Heads that help with token prediction

## Head Scoring Logic

### Ablation Method

**How it works**:
1. **Baseline**: Run model on examples without masking any heads
   - Measure accuracy: how many examples are "correct"
   - For backward-chaining: check if path nodes appear in output

2. **Ablated**: Run model with specific head masked (set to zero)
   - Same examples, but one head is disabled
   - Measure accuracy again

3. **Score calculation**:
   ```
   score = (baseline_accuracy - ablated_accuracy) / baseline_accuracy
   ```
   - Higher score = head is more important
   - If masking a head causes big performance drop, it's important

**Why scores might be low/zero**:
- The model (LLaMA-3-8B) wasn't trained on backward-chaining tasks
- It may not perform well on this task at all
- But we can still find heads that are *relatively* more important
- Even if baseline accuracy is low, we can see which heads matter more

### Correctness Checking

**Current approach** (lenient):
- For path-finding: Check if 50%+ of path nodes appear in output
- For goal identification: Check if goal number appears in output
- For edge parsing: Check if edge information appears

**Why lenient**:
- General LLMs aren't trained for this specific task
- We're looking for *relative* importance, not absolute correctness
- The ablation comparison (baseline vs masked) is what matters

## Running Single Subtask

To run and debug a single subtask:

```bash
python testing/reasoning_heads/main.py --discover \
    --model_name "meta-llama/Meta-Llama-3-8B-Instruct" \
    --scoring_method ablation \
    --n_examples 20 \
    --top_k 10 \
    --single_subtask "path_finding" \
    --min_score 0.0 \
    --min_confidence 0.0
```

This will:
1. Process only the "path_finding" subtask
2. Show detailed scoring information
3. Display top scores even if they're low
4. Help you understand what's happening

## Understanding the Results

**If you see 0 heads found**:
- All head scores might be very low (close to 0)
- Try lowering `--min_score` and `--min_confidence` to 0.0
- Look at the "Top 5 scores" output to see actual values
- Even low scores can be meaningful if they're consistently higher than others

**Score interpretation**:
- Score of 0.1 = masking this head causes 10% performance drop
- Score of 0.0 = masking this head has no effect (head is not important)
- Negative score = masking improves performance (head might be harmful)

**Next steps**:
1. Run with `--single_subtask` to focus on one task
2. Check the top scores even if they're low
3. Adjust thresholds based on what you see
4. The framework will still identify the *most* important heads even if absolute scores are low

