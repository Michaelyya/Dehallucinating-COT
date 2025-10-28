---
license: mit
dataset_info:
  features:
  - name: question
    dtype: string
  - name: org_context
    dtype: string
  - name: org_answer
    sequence: string
  - name: sub_context
    dtype: string
  - name: sub_answer
    sequence: string
  splits:
  - name: dev
    num_bytes: 10056243
    num_examples: 4746
  download_size: 2754938
  dataset_size: 10056243
configs:
- config_name: default
  data_files:
  - split: dev
    path: data/dev-*
---
