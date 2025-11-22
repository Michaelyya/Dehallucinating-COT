from head_scoring import AblationScorer
import torch

# Load your model and tokenizer
scorer = AblationScorer(model, tokenizer, "cuda")

# Test ablation directly
test_text = "The capital of France is"
test_ids = tokenizer.encode(test_text, return_tensors="pt").cuda()

# Normal generation
with torch.no_grad():
    normal = model.generate(test_ids, max_new_tokens=5)
print("Normal:", tokenizer.decode(normal[0]))

# With ablation
with torch.no_grad():
    ablated = scorer._generate_with_ablation(test_ids, [(0, 0)])
print("Ablated:", tokenizer.decode(ablated[0]))

# They should be different!