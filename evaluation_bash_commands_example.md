# DeCoRe Entropy - LLaMA3 8B Instruct

```bash
python scripts/main.py experiment=truthfulqa/decore_entropy/llama3_8b_instruct decoder.configs.num_retrieval_heads=10
python scripts/main.py experiment=truthfulqa/decore_entropy/llama3_8b_instruct decoder.configs.num_retrieval_heads=20
python scripts/main.py experiment=truthfulqa/decore_entropy/llama3_8b_instruct decoder.configs.num_retrieval_heads=30
python scripts/main.py experiment=truthfulqa/decore_entropy/llama3_8b_instruct decoder.configs.num_retrieval_heads=40
python scripts/main.py experiment=truthfulqa/decore_entropy/llama3_8b_instruct decoder.configs.num_retrieval_heads=50
python scripts/main.py experiment=truthfulqa/decore_entropy/llama3_8b_instruct decoder.configs.num_retrieval_heads=60
python scripts/main.py experiment=truthfulqa/decore_entropy/llama3_8b_instruct decoder.configs.num_retrieval_heads=70
python scripts/main.py experiment=truthfulqa/decore_entropy/llama3_8b_instruct decoder.configs.num_retrieval_heads=80
python scripts/main.py experiment=truthfulqa/decore_entropy/llama3_8b_instruct decoder.configs.num_retrieval_heads=90
python scripts/main.py experiment=truthfulqa/decore_entropy/llama3_8b_instruct decoder.configs.num_retrieval_heads=100

python scripts/main.py experiment=xsum/decore_entropy/llama3_8b_instruct decoder.configs.num_retrieval_heads=10
python scripts/main.py experiment=xsum/decore_entropy/llama3_8b_instruct decoder.configs.num_retrieval_heads=20
python scripts/main.py experiment=xsum/decore_entropy/llama3_8b_instruct decoder.configs.num_retrieval_heads=30
python scripts/main.py experiment=xsum/decore_entropy/llama3_8b_instruct decoder.configs.num_retrieval_heads=40
python scripts/main.py experiment=xsum/decore_entropy/llama3_8b_instruct decoder.configs.num_retrieval_heads=50
python scripts/main.py experiment=xsum/decore_entropy/llama3_8b_instruct decoder.configs.num_retrieval_heads=60
python scripts/main.py experiment=xsum/decore_entropy/llama3_8b_instruct decoder.configs.num_retrieval_heads=70
python scripts/main.py experiment=xsum/decore_entropy/llama3_8b_instruct decoder.configs.num_retrieval_heads=80
python scripts/main.py experiment=xsum/decore_entropy/llama3_8b_instruct decoder.configs.num_retrieval_heads=90
python scripts/main.py experiment=xsum/decore_entropy/llama3_8b_instruct decoder.configs.num_retrieval_heads=100

python scripts/main.py experiment=memo_trap/decore_entropy/llama3_8b_instruct decoder.configs.num_retrieval_heads=10
python scripts/main.py experiment=memo_trap/decore_entropy/llama3_8b_instruct decoder.configs.num_retrieval_heads=20
python scripts/main.py experiment=memo_trap/decore_entropy/llama3_8b_instruct decoder.configs.num_retrieval_heads=30
python scripts/main.py experiment=memo_trap/decore_entropy/llama3_8b_instruct decoder.configs.num_retrieval_heads=40
python scripts/main.py experiment=memo_trap/decore_entropy/llama3_8b_instruct decoder.configs.num_retrieval_heads=50
python scripts/main.py experiment=memo_trap/decore_entropy/llama3_8b_instruct decoder.configs.num_retrieval_heads=60
python scripts/main.py experiment=memo_trap/decore_entropy/llama3_8b_instruct decoder.configs.num_retrieval_heads=70
python scripts/main.py experiment=memo_trap/decore_entropy/llama3_8b_instruct decoder.configs.num_retrieval_heads=80
python scripts/main.py experiment=memo_trap/decore_entropy/llama3_8b_instruct decoder.configs.num_retrieval_heads=90
python scripts/main.py experiment=memo_trap/decore_entropy/llama3_8b_instruct decoder.configs.num_retrieval_heads=100

python scripts/main.py experiment=nq/decore_entropy/llama3_8b_instruct decoder.configs.num_retrieval_heads=10 data.variation=closed_book
python scripts/main.py experiment=nq/decore_entropy/llama3_8b_instruct decoder.configs.num_retrieval_heads=20 data.variation=closed_book
python scripts/main.py experiment=nq/decore_entropy/llama3_8b_instruct decoder.configs.num_retrieval_heads=30 data.variation=closed_book
python scripts/main.py experiment=nq/decore_entropy/llama3_8b_instruct decoder.configs.num_retrieval_heads=40 data.variation=closed_book
python scripts/main.py experiment=nq/decore_entropy/llama3_8b_instruct decoder.configs.num_retrieval_heads=50 data.variation=closed_book
python scripts/main.py experiment=nq/decore_entropy/llama3_8b_instruct decoder.configs.num_retrieval_heads=60 data.variation=closed_book
python scripts/main.py experiment=nq/decore_entropy/llama3_8b_instruct decoder.configs.num_retrieval_heads=70 data.variation=closed_book
python scripts/main.py experiment=nq/decore_entropy/llama3_8b_instruct decoder.configs.num_retrieval_heads=80 data.variation=closed_book
python scripts/main.py experiment=nq/decore_entropy/llama3_8b_instruct decoder.configs.num_retrieval_heads=90 data.variation=closed_book
python scripts/main.py experiment=nq/decore_entropy/llama3_8b_instruct decoder.configs.num_retrieval_heads=100 data.variation=closed_book

python scripts/main.py experiment=nq/decore_entropy/llama3_8b_instruct decoder.configs.num_retrieval_heads=10 data.variation=oracle
python scripts/main.py experiment=nq/decore_entropy/llama3_8b_instruct decoder.configs.num_retrieval_heads=20 data.variation=oracle
python scripts/main.py experiment=nq/decore_entropy/llama3_8b_instruct decoder.configs.num_retrieval_heads=30 data.variation=oracle
python scripts/main.py experiment=nq/decore_entropy/llama3_8b_instruct decoder.configs.num_retrieval_heads=40 data.variation=oracle
python scripts/main.py experiment=nq/decore_entropy/llama3_8b_instruct decoder.configs.num_retrieval_heads=50 data.variation=oracle
python scripts/main.py experiment=nq/decore_entropy/llama3_8b_instruct decoder.configs.num_retrieval_heads=60 data.variation=oracle
python scripts/main.py experiment=nq/decore_entropy/llama3_8b_instruct decoder.configs.num_retrieval_heads=70 data.variation=oracle
python scripts/main.py experiment=nq/decore_entropy/llama3_8b_instruct decoder.configs.num_retrieval_heads=80 data.variation=oracle
python scripts/main.py experiment=nq/decore_entropy/llama3_8b_instruct decoder.configs.num_retrieval_heads=90 data.variation=oracle
python scripts/main.py experiment=nq/decore_entropy/llama3_8b_instruct decoder.configs.num_retrieval_heads=100 data.variation=oracle

python scripts/main.py experiment=nq_swap/decore_entropy/llama3_8b_instruct decoder.configs.num_retrieval_heads=10
python scripts/main.py experiment=nq_swap/decore_entropy/llama3_8b_instruct decoder.configs.num_retrieval_heads=20
python scripts/main.py experiment=nq_swap/decore_entropy/llama3_8b_instruct decoder.configs.num_retrieval_heads=30
python scripts/main.py experiment=nq_swap/decore_entropy/llama3_8b_instruct decoder.configs.num_retrieval_heads=40
python scripts/main.py experiment=nq_swap/decore_entropy/llama3_8b_instruct decoder.configs.num_retrieval_heads=50
python scripts/main.py experiment=nq_swap/decore_entropy/llama3_8b_instruct decoder.configs.num_retrieval_heads=60
python scripts/main.py experiment=nq_swap/decore_entropy/llama3_8b_instruct decoder.configs.num_retrieval_heads=70
python scripts/main.py experiment=nq_swap/decore_entropy/llama3_8b_instruct decoder.configs.num_retrieval_heads=80
python scripts/main.py experiment=nq_swap/decore_entropy/llama3_8b_instruct decoder.configs.num_retrieval_heads=90
python scripts/main.py experiment=nq_swap/decore_entropy/llama3_8b_instruct decoder.configs.num_retrieval_heads=100

python scripts/main.py experiment=musique/decore_entropy/llama3_8b_instruct decoder.configs.num_retrieval_heads=10 data.variation=direct_closed_book
python scripts/main.py experiment=musique/decore_entropy/llama3_8b_instruct decoder.configs.num_retrieval_heads=20 data.variation=direct_closed_book
python scripts/main.py experiment=musique/decore_entropy/llama3_8b_instruct decoder.configs.num_retrieval_heads=30 data.variation=direct_closed_book
python scripts/main.py experiment=musique/decore_entropy/llama3_8b_instruct decoder.configs.num_retrieval_heads=40 data.variation=direct_closed_book
python scripts/main.py experiment=musique/decore_entropy/llama3_8b_instruct decoder.configs.num_retrieval_heads=50 data.variation=direct_closed_book
python scripts/main.py experiment=musique/decore_entropy/llama3_8b_instruct decoder.configs.num_retrieval_heads=60 data.variation=direct_closed_book
python scripts/main.py experiment=musique/decore_entropy/llama3_8b_instruct decoder.configs.num_retrieval_heads=70 data.variation=direct_closed_book
python scripts/main.py experiment=musique/decore_entropy/llama3_8b_instruct decoder.configs.num_retrieval_heads=80 data.variation=direct_closed_book
python scripts/main.py experiment=musique/decore_entropy/llama3_8b_instruct decoder.configs.num_retrieval_heads=90 data.variation=direct_closed_book
python scripts/main.py experiment=musique/decore_entropy/llama3_8b_instruct decoder.configs.num_retrieval_heads=100 data.variation=direct_closed_book

python scripts/main.py experiment=musique/decore_entropy/llama3_8b_instruct decoder.configs.num_retrieval_heads=10 data.variation=direct_open_book
python scripts/main.py experiment=musique/decore_entropy/llama3_8b_instruct decoder.configs.num_retrieval_heads=20 data.variation=direct_open_book
python scripts/main.py experiment=musique/decore_entropy/llama3_8b_instruct decoder.configs.num_retrieval_heads=30 data.variation=direct_open_book
python scripts/main.py experiment=musique/decore_entropy/llama3_8b_instruct decoder.configs.num_retrieval_heads=40 data.variation=direct_open_book
python scripts/main.py experiment=musique/decore_entropy/llama3_8b_instruct decoder.configs.num_retrieval_heads=50 data.variation=direct_open_book
python scripts/main.py experiment=musique/decore_entropy/llama3_8b_instruct decoder.configs.num_retrieval_heads=60 data.variation=direct_open_book
python scripts/main.py experiment=musique/decore_entropy/llama3_8b_instruct decoder.configs.num_retrieval_heads=70 data.variation=direct_open_book
python scripts/main.py experiment=musique/decore_entropy/llama3_8b_instruct decoder.configs.num_retrieval_heads=80 data.variation=direct_open_book
python scripts/main.py experiment=musique/decore_entropy/llama3_8b_instruct decoder.configs.num_retrieval_heads=90 data.variation=direct_open_book
python scripts/main.py experiment=musique/decore_entropy/llama3_8b_instruct decoder.configs.num_retrieval_heads=100 data.variation=direct_open_book

python scripts/main.py experiment=popqa/decore_entropy/llama3_8b_instruct decoder.configs.num_retrieval_heads=10
python scripts/main.py experiment=popqa/decore_entropy/llama3_8b_instruct decoder.configs.num_retrieval_heads=20
python scripts/main.py experiment=popqa/decore_entropy/llama3_8b_instruct decoder.configs.num_retrieval_heads=30
python scripts/main.py experiment=popqa/decore_entropy/llama3_8b_instruct decoder.configs.num_retrieval_heads=40
python scripts/main.py experiment=popqa/decore_entropy/llama3_8b_instruct decoder.configs.num_retrieval_heads=50
python scripts/main.py experiment=popqa/decore_entropy/llama3_8b_instruct decoder.configs.num_retrieval_heads=60
python scripts/main.py experiment=popqa/decore_entropy/llama3_8b_instruct decoder.configs.num_retrieval_heads=70
python scripts/main.py experiment=popqa/decore_entropy/llama3_8b_instruct decoder.configs.num_retrieval_heads=80
python scripts/main.py experiment=popqa/decore_entropy/llama3_8b_instruct decoder.configs.num_retrieval_heads=90
python scripts/main.py experiment=popqa/decore_entropy/llama3_8b_instruct decoder.configs.num_retrieval_heads=100

python scripts/main.py experiment=musique/decore_entropy/llama3_8b_instruct decoder.configs.num_retrieval_heads=10 data.variation=cot_closed_book
python scripts/main.py experiment=musique/decore_entropy/llama3_8b_instruct decoder.configs.num_retrieval_heads=20 data.variation=cot_closed_book
python scripts/main.py experiment=musique/decore_entropy/llama3_8b_instruct decoder.configs.num_retrieval_heads=30 data.variation=cot_closed_book
python scripts/main.py experiment=musique/decore_entropy/llama3_8b_instruct decoder.configs.num_retrieval_heads=40 data.variation=cot_closed_book
python scripts/main.py experiment=musique/decore_entropy/llama3_8b_instruct decoder.configs.num_retrieval_heads=50 data.variation=cot_closed_book
python scripts/main.py experiment=musique/decore_entropy/llama3_8b_instruct decoder.configs.num_retrieval_heads=60 data.variation=cot_closed_book
python scripts/main.py experiment=musique/decore_entropy/llama3_8b_instruct decoder.configs.num_retrieval_heads=70 data.variation=cot_closed_book
python scripts/main.py experiment=musique/decore_entropy/llama3_8b_instruct decoder.configs.num_retrieval_heads=80 data.variation=cot_closed_book
python scripts/main.py experiment=musique/decore_entropy/llama3_8b_instruct decoder.configs.num_retrieval_heads=90 data.variation=cot_closed_book
python scripts/main.py experiment=musique/decore_entropy/llama3_8b_instruct decoder.configs.num_retrieval_heads=100 data.variation=cot_closed_book

python scripts/main.py experiment=triviaqa/decore_entropy/llama3_8b_instruct decoder.configs.num_retrieval_heads=10
python scripts/main.py experiment=triviaqa/decore_entropy/llama3_8b_instruct decoder.configs.num_retrieval_heads=20
python scripts/main.py experiment=triviaqa/decore_entropy/llama3_8b_instruct decoder.configs.num_retrieval_heads=30
python scripts/main.py experiment=triviaqa/decore_entropy/llama3_8b_instruct decoder.configs.num_retrieval_heads=40
python scripts/main.py experiment=triviaqa/decore_entropy/llama3_8b_instruct decoder.configs.num_retrieval_heads=50
python scripts/main.py experiment=triviaqa/decore_entropy/llama3_8b_instruct decoder.configs.num_retrieval_heads=60
python scripts/main.py experiment=triviaqa/decore_entropy/llama3_8b_instruct decoder.configs.num_retrieval_heads=70
python scripts/main.py experiment=triviaqa/decore_entropy/llama3_8b_instruct decoder.configs.num_retrieval_heads=80
python scripts/main.py experiment=triviaqa/decore_entropy/llama3_8b_instruct decoder.configs.num_retrieval_heads=90
python scripts/main.py experiment=triviaqa/decore_entropy/llama3_8b_instruct decoder.configs.num_retrieval_heads=100

python scripts/main.py experiment=musique/decore_entropy/llama3_8b_instruct decoder.configs.num_retrieval_heads=10 data.variation=cot_open_book
python scripts/main.py experiment=musique/decore_entropy/llama3_8b_instruct decoder.configs.num_retrieval_heads=20 data.variation=cot_open_book
python scripts/main.py experiment=musique/decore_entropy/llama3_8b_instruct decoder.configs.num_retrieval_heads=30 data.variation=cot_open_book
python scripts/main.py experiment=musique/decore_entropy/llama3_8b_instruct decoder.configs.num_retrieval_heads=40 data.variation=cot_open_book
python scripts/main.py experiment=musique/decore_entropy/llama3_8b_instruct decoder.configs.num_retrieval_heads=50 data.variation=cot_open_book
python scripts/main.py experiment=musique/decore_entropy/llama3_8b_instruct decoder.configs.num_retrieval_heads=60 data.variation=cot_open_book
python scripts/main.py experiment=musique/decore_entropy/llama3_8b_instruct decoder.configs.num_retrieval_heads=70 data.variation=cot_open_book
python scripts/main.py experiment=musique/decore_entropy/llama3_8b_instruct decoder.configs.num_retrieval_heads=80 data.variation=cot_open_book
python scripts/main.py experiment=musique/decore_entropy/llama3_8b_instruct decoder.configs.num_retrieval_heads=90 data.variation=cot_open_book
python scripts/main.py experiment=musique/decore_entropy/llama3_8b_instruct decoder.configs.num_retrieval_heads=100 data.variation=cot_open_book
```