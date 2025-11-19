# Full Pipeline: Discover Reasoning Heads and Test on Benchmarks

This guide shows you how to discover reasoning heads from CognitiveMirrors and use them in HotpotQA, MEQA, and MuSiQue benchmarks.

## Quick Start

### Step 1: Discover Reasoning Heads

```bash
cd testing/reasoning_heads
python main.py \
    --discover \
    --model_name "meta-llama/Meta-Llama-3-8B-Instruct" \
    --scoring_method ablation \
    --n_examples 20 \
    --top_k 10
```

This will:
- Load CognitiveMirrors Logical Reasoning examples
- Score each attention head by measuring BLEU score drop when masked
- Select top 10 heads with largest performance drops
- Save to:
  - `discovered_heads.json` (full format)
  - `../retrieval_heads/Meta-Llama-3-8B-Instruct.json` (DeCoReEntropy format)

### Step 2: Test HotpotQA with Discovered Heads

**Option A: Using the simple test script**

```bash
python test_with_discovered_heads.py \
    --benchmark hotpotqa \
    --config ../configs/hotpotqa_model_config.yaml \
    --num_retrieval_heads 10
```

**Option B: Using the original test script (heads automatically loaded)**

The discovered heads are saved to `../retrieval_heads/Meta-Llama-3-8B-Instruct.json`. 

Your config file (`configs/hotpotqa_model_config.yaml`) already has:
```yaml
decoder:
  name: "DeCoReEntropy"
  method: "DeCoReEntropy"
  configs:
    retrieval_heads_dir: "../retrieval_heads/"
    num_retrieval_heads: 10
```

So you can just run:
```bash
cd testing
python test_hotpotqa.py --config configs/hotpotqa_model_config.yaml
```

The DeCoReEntropy model will automatically load the heads from `../retrieval_heads/Meta-Llama-3-8B-Instruct.json`!

### Step 3: Run Full Pipeline (All Benchmarks)

```bash
cd testing/reasoning_heads
python run_full_pipeline.py \
    --model_name "meta-llama/Meta-Llama-3-8B-Instruct" \
    --scoring_method ablation \
    --n_examples 20 \
    --top_k 10 \
    --benchmarks hotpotqa meqa musique
```

This will:
1. Discover reasoning heads
2. Run baseline tests on all benchmarks
3. Run DeCoReEntropy tests with discovered heads
4. Generate a comparison report

## How It Works

### Head Discovery
- Uses **ablation scoring**: Masks each head and measures BLEU score drop
- **Score = baseline_bleu - ablated_bleu**
- Positive scores = head is important (masking hurts performance)
- Selects heads with **largest positive scores** (most important)

### Head Format for DeCoReEntropy
Heads are saved in format:
```json
{"0-5": [0.123], "1-3": [0.456], "2-7": [0.789], ...}
```

Where:
- Key: `"layer-head"` (e.g., `"0-5"` = layer 0, head 5)
- Value: `[score]` (list with single score)

DeCoReEntropy loads this file and uses the top K heads based on score.

### Reference Answer Extraction
For CognitiveMirrors, reference answers are extracted to:
- `"yes"` - if answer indicates yes/true/factual
- `"no"` - if answer indicates no/not factual/not true
- `"unanswerable"` - if answer indicates cannot determine

BLEU score is computed using these simplified references.

## File Locations

After discovery, you'll have:
- `testing/reasoning_heads/discovered_heads.json` - Full format with metadata
- `testing/retrieval_heads/Meta-Llama-3-8B-Instruct.json` - DeCoReEntropy format

The DeCoReEntropy model automatically loads from:
- `{retrieval_heads_dir}/{model_base_name}.json`
- Where `model_base_name` = `"Meta-Llama-3-8B-Instruct"` (from `meta-llama/Meta-Llama-3-8B-Instruct`)

## Testing Individual Benchmarks

### HotpotQA
```bash
cd testing
python test_hotpotqa.py --config configs/hotpotqa_model_config.yaml
```

### MEQA
```bash
cd testing
python test_meqa.py --config configs/meqa_model_config.yaml
```

### MuSiQue
```bash
cd testing
python test_musique.py --config configs/musique_model_config.yaml
```

All of these will automatically use the discovered reasoning heads if they exist in `../retrieval_heads/Meta-Llama-3-8B-Instruct.json`!

## Troubleshooting

### Heads not loading?
1. Check that `../retrieval_heads/Meta-Llama-3-8B-Instruct.json` exists
2. Verify the file format is correct (single-line JSON)
3. Check that `retrieval_heads_dir` in config points to the right directory

### Wrong model name?
The file name must match the model base name. For `meta-llama/Meta-Llama-3-8B-Instruct`, the file should be `Meta-Llama-3-8B-Instruct.json`.

### Want to use different number of heads?
Update `num_retrieval_heads` in your config file, or use the `--num_retrieval_heads` argument in the test script.

