# Reasoning Head Detection 方法说明

## 1. 如何找到 Reasoning Head？

### 核心思路

代码通过分析**生成过程中的注意力模式**来识别 reasoning head。具体流程：

1. **生成阶段跟踪** (`decode_with_reasoning_tracking`, 213-248行)
   - 在模型生成每个token时，获取该层的attention weights
   - 对每个head，检查其注意力主要关注哪些位置

2. **Reasoning Score 计算** (`reasoning_calculate`, 167-201行)
   ```python
   # 对每个layer和head：
   # 1. 获取top-k注意力位置
   values, idx = attention_matrix[layer_idx][0][head_idx][-1].topk(topk)
   
   # 2. 检查注意力是否关注"推理相关"的位置：
   #    - 之前生成的token（chain-of-thought风格，190行）
   #    - query区域（194-196行）
   
   # 3. 如果关注这些位置，累积attention weights作为reasoning score
   if reasoning_attention_score > 0:
       reasoning_score[layer_idx][head_idx][0] += reasoning_attention_score
   ```

3. **成功样本过滤** (319行)
   - 只有当模型生成结果与ground truth的ROUGE分数>50时，才累积该样本的head scores
   - 这样可以确保只分析"真正成功的推理"所激活的head

4. **跨样本累积** (`reasoning_head_accumulate`, 203-211行)
   - 将所有成功样本的reasoning scores累积到`head_counter`中
   - 最终通过平均分数排序，找出最活跃的reasoning heads

### 与 Retrieval Head 的区别

| 方面 | Retrieval Head | Reasoning Head |
|------|---------------|----------------|
| **检测目标** | 关注context中的特定信息（needle） | 关注推理过程（已生成token + query） |
| **匹配条件** | 生成token必须与needle中的token**完全匹配** | 关注attention weights的**累积值** |
| **评分方式** | 计数匹配：`1/(needle_end - needle_start)` | 累积attention weights |
| **适用场景** | 信息检索任务 | 推理/问答任务 |

---

## 2. 借鉴 Retrieval Head 的地方

### 直接借鉴的部分

1. **整体框架结构** ✅
   - `decode_with_reasoning_tracking` 方法（213行）借鉴了 `retrieval_head_detection.py` 的 `decode` 方法（237行）
   - 都使用 `output_attentions=True` 获取attention weights
   - 都使用 `past_key_values` 进行增量解码

2. **ROUGE > 50 阈值** ✅ **（直接借鉴）**
   ```python
   # retrieval_head_detection.py 第300行
   if score > 50:
       self.retrieval_head_accumulate(retrieval_score)
   
   # reasoning_head_detection.py 第319行
   if score > 50:
       self.reasoning_head_accumulate(reasoning_score)
   ```
   这个阈值直接来自retrieval head论文的方法。

3. **Head Score 累积机制** ✅
   - `reasoning_head_accumulate` (203行) 借鉴了 `retrieval_head_accumulate` (232行)
   - 都使用 `head_counter` 字典存储每个head的分数列表
   - 格式：`{layer_idx-head_idx: [score1, score2, ...]}`

4. **保存和统计方式** ✅
   - `save_reasoning_heads` (365行) 借鉴了retrieval head的保存方式
   - 都计算mean、std、count、max等统计量

5. **Query/Needle 位置查找** ✅
   - `find_query_idx` (250行) 借鉴了 `find_needle_idx` (251行)
   - 都使用token overlap的方式在prompt中定位目标区域

### 修改的部分

1. **Score 计算逻辑**（核心差异）
   - **Retrieval**: 检查生成token是否与needle中的token匹配（228行）
   - **Reasoning**: 检查attention是否关注已生成token和query区域（190-196行）

2. **评分方式**
   - **Retrieval**: 简单计数 `1/(needle_end - needle_start)`
   - **Reasoning**: 累积attention weights值（更关注注意力强度）

---

## 3. 为什么 ROUGE > 50 代表有用？

### 来源

这个阈值来自 **retrieval head论文** (`retrieval_head_detection.py` 第300行的注释)：
```python
## if recall > 50, we determine this retrieval succeed and update the retrieval score
if score > 50:
    self.retrieval_head_accumulate(retrieval_score)
```

### 含义

1. **ROUGE Recall 的含义**
   - ROUGE-1 recall = (ground truth中出现在生成结果中的token数) / (ground truth的总token数)
   - 如果recall > 50%，说明生成结果覆盖了ground truth中**超过一半**的信息

2. **为什么用50%作为阈值？**
   - **过滤噪声**: 只分析"真正成功"的推理样本，避免学习到错误的head pattern
   - **确保质量**: 如果模型生成结果与正确答案完全不相关（recall < 50%），那么该样本激活的head可能不是"真正的reasoning head"
   - **提高稳定性**: 只累积高质量样本的head scores，使最终结果更可靠

3. **实际效果**
   - 在retrieval head论文中，这个阈值被证明能有效识别出最有用的retrieval heads
   - 类比到reasoning head：只有模型**真正进行推理**时（生成结果与GT匹配度高），才认为激活的head是reasoning head

### 注意事项

- **阈值可能调整**: 对于不同任务（如数学推理vs文本推理），50%可能不是最优值
- **ROUGE的限制**: ROUGE主要衡量token重叠，对于语义相似但表达不同的答案可能不够敏感
- **建议**: 可以尝试其他阈值（如40%、60%）或使用其他评估指标（如BLEU、BERTScore）进行对比

---



