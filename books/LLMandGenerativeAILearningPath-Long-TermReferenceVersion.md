# LLM 与生成式 AI 学习路线（长期参考版）

> 目标：从深度学习基础出发，系统掌握现代 LLM 的预训练、架构、Scaling Law、后训练、推理、RAG、Agent、多模态与生成式 AI。
>
> 推荐主线：**D2L → Stanford CS336 → Berkeley LLM Agents → 专题论文 → 多模态 / Agent 深入**
>
> 使用原则：**课程建立体系，论文跟进前沿，项目负责内化。不要同时刷太多课程。**

---

## 一、总路线

```text
Deep Learning 基础
    │
    ├── Dive into Deep Learning
    │
    ▼
Transformer / Language Modeling
    │
    ├── Stanford CS336: Language Modeling from Scratch
    │
    ├── Stanford CS224N（选学）
    │
    ▼
Pretraining / Scaling / Systems
    │
    ├── Tokenization
    ├── Transformer
    ├── MoE
    ├── GPU / Triton / Distributed Training
    ├── Scaling Laws
    ├── Data
    └── Inference
    │
    ▼
Post-training
    │
    ├── SFT
    ├── RLHF
    ├── DPO
    ├── RLVR / Reasoning RL
    └── Inference-time Scaling
    │
    ▼
RAG / Tool Use / Reasoning
    │
    ├── RAG
    ├── ReAct
    ├── Toolformer
    └── Evaluation
    │
    ▼
Agent
    │
    ├── Berkeley LLM Agents
    ├── Berkeley Advanced LLM Agents
    └── Agentic AI
    │
    ▼
Multimodal
    │
    ├── CLIP
    ├── LLaVA
    ├── Chameleon
    └── Transfusion
```

---

# 二、主线课程

## 1. Dive into Deep Learning（D2L）

### 定位

负责打牢现代深度学习基础和 PyTorch 实践能力。

### 重点内容

- MLP 与反向传播
- CNN / ResNet
- RNN 基础
- Attention
- Transformer
- 优化与正则化
- GPU 训练
- 基本模型工程

### 链接

- 官方英文版：<https://d2l.ai/>
- 中文版：<https://zh.d2l.ai/>
- GitHub：<https://github.com/d2l-ai/d2l-en>

### 学习建议

不必把整本书逐页看完。目标是达到：

- 能解释反向传播；
- 能独立写训练循环；
- 能解释 Transformer 的核心结构；
- 能使用 PyTorch 完成普通模型训练与调试。

完成核心章节后立即进入 CS336。

---

# 三、核心课程：Stanford CS336

## 2. Stanford CS336 — Language Modeling from Scratch

### 定位

这是整条路线中最重要的一门课程。

它不是教你调用 LLM API，而是带你从底层理解一个现代语言模型如何被：

> 数据准备 → Tokenization → Transformer → 训练 → 分布式计算 → Scaling → 推理 → 后训练 → 评测

完整构建出来。

### 官方资源

- 2026 课程主页：<https://cs336.stanford.edu/>
- 2025 课程主页：<https://cs336.stanford.edu/spring2025/>

2026 页面本身包含：

- Lecture schedule
- Slides
- Readings
- Assignments
- YouTube recordings

### 推荐重点

优先学习：

1. Tokenization
2. Transformer architecture
3. Mixture of Experts
4. GPU / accelerators
5. Triton / kernels
6. Parallelism
7. Scaling Laws
8. Inference
9. Pretraining Data
10. SFT
11. RLHF / Preference Learning
12. RLVR / Reasoning
13. Multimodality

### 作业建议

至少认真完成或复现：

- tokenizer
- Transformer
- optimizer
- training loop

有余力再做：

- FlashAttention
- distributed training
- inference optimization
- post-training

### 达标标准

学完后应能解释：

- 为什么 Transformer 可以扩展到超大规模；
- 参数量、token 数量、算力之间如何权衡；
- Pretraining 和 Post-training 的边界；
- KV Cache 为什么重要；
- FlashAttention 为什么能加速；
- Data Parallel / Tensor Parallel / Pipeline Parallel 的区别；
- SFT、RLHF、DPO、RLVR 的关系；
- 推理阶段为什么也可以换算力换性能。

---

# 四、辅助课程：Stanford CS224N

## 3. Stanford CS224N — Natural Language Processing with Deep Learning

### 定位

CS224N 更适合补：

- NLP 基础；
- Transformer 的语言视角；
- LLM evaluation；
- RAG；
- Agent；
- 多模态语言模型。

不建议在已经学习 CS336 的情况下完整重刷。

### 官方资源

- 2026 课程主页：<https://web.stanford.edu/class/cs224n/>
- 课程公开视频入口通常可从课程主页进入。
- 历年 syllabus / slides：<https://web.stanford.edu/class/archive/cs/cs224n/>

### 推荐选学主题

重点看：

- Transformer
- Pretraining
- LLM
- Evaluation
- Retrieval-Augmented Generation
- Reasoning
- Agents
- Multimodal Models

---

# 五、Agent 主线课程

## 4. UC Berkeley — Large Language Model Agents

### 定位

从“LLM”跨入“Agent”的核心课程。

内容包括：

- Reasoning
- Planning
- Tool Use
- Agent Infrastructure
- RAG
- Code Agents
- Multimodal Agents
- Robotics
- Evaluation
- Safety
- Multi-Agent

### 官方链接

- 课程主页：<https://rdi.berkeley.edu/llm-agents/f24>
- Berkeley RDI 公共课程主页：<https://rdi.berkeley.edu/publicCourses>

### 推荐学习方式

不要只看视频。

每个主题采用：

```text
Lecture
→ Slides
→ Supplemental Reading
→ 自己实现 mini demo
```

例如学完 ReAct 后，至少自己实现：

```text
User
 ↓
LLM reasoning
 ↓
Tool selection
 ↓
Tool execution
 ↓
Observation
 ↓
LLM
 ↓
Final answer
```

而不是直接调用成熟 Agent Framework。

---

# 六、高级 Agent / Reasoning

## 5. UC Berkeley — Advanced Large Language Model Agents

### 定位

重点研究：

- inference-time reasoning；
- post-training for reasoning；
- search；
- planning；
- agentic workflow；
- tool/function calling；
- code generation；
- program verification；
- theorem proving。

### 官方链接

- 课程主页：<https://rdi.berkeley.edu/adv-llm-agents/sp25>

### 推荐优先内容

对工程和工业 AI 更重要的是：

1. Inference-Time Reasoning
2. Post-training for Reasoning
3. Search & Planning
4. Tool Use
5. Function Calling
6. Code Agents
7. Verification

数学定理证明部分可以降低优先级。

---

# 七、更新版本：Berkeley Agentic AI

## 6. UC Berkeley — Agentic AI

如果希望了解 Berkeley 后续对 Agent 课程的升级，可以继续参考：

- 课程主页：<https://rdi.berkeley.edu/agentic-ai/f25>

重点关注：

- Agent architecture
- Reasoning
- Planning
- Agent frameworks
- Infrastructure
- Code generation
- Robotics
- Web agents
- Scientific agents

---

# 八、前沿 Seminar

## 7. Stanford CS25 — Transformers United

### 定位

它不是系统教材，而是前沿研究 seminar。

适合在已经掌握基础体系后，用来持续更新认知。

### 官方链接

- 当前课程：<https://web.stanford.edu/class/cs25/>
- 历史课程 V5：<https://web.stanford.edu/class/cs25/past/cs25-v5/>
- 历史课程 V3：<https://web.stanford.edu/class/cs25/past/cs25-v3/>

### 使用方式

不要从第一讲刷到最后一讲。

正确做法：

```text
学到一个主题
→ 找 CS25 对应专家 lecture
→ 看专家如何理解该领域
→ 再阅读论文
```

---

# 九、非常推荐的工程直觉课程

## 8. Andrej Karpathy — Neural Networks: Zero to Hero

### 定位

建立从神经网络、反向传播到 GPT 的直觉。

尤其适合真正搞懂：

```text
micrograd
→ MLP
→ language model
→ tokenization
→ Transformer
→ GPT
```

### 官方资源

- 课程主页：<https://karpathy.ai/zero-to-hero.html>
- Karpathy YouTube：<https://www.youtube.com/@AndrejKarpathy>

### 推荐

如果已经学习 D2L：

不用全部完成。

重点看：

- micrograd
- makemore
- GPT / Transformer 相关课程

然后进入 CS336。

---

# 十、工程工具链课程

## 9. Hugging Face LLM Course

### 定位

偏工程实践和工具生态，而不是 LLM 底层理论。

适合掌握：

- Transformers
- Tokenizers
- Datasets
- pretrained models
- fine-tuning
- inference
- Hugging Face ecosystem

### 官方链接

- 课程主页：<https://huggingface.co/learn/llm-course/chapter0/1>
- Hugging Face Learn：<https://huggingface.co/learn>

### 学习原则

把它当作：

> 工程工具手册

而不是 LLM 理论主教材。

---

# 十一、核心论文路线

下面不要一次全部读完。

按照学习阶段阅读。

---

## A. Transformer 基础

### 1. Attention Is All You Need

Vaswani et al., 2017

### 为什么读

Transformer 的源头。

重点理解：

- Self-Attention
- Multi-Head Attention
- Position Encoding
- Encoder / Decoder
- Residual Connection

### 链接

- arXiv：<https://arxiv.org/abs/1706.03762>
- PDF：<https://arxiv.org/pdf/1706.03762>

### 阅读等级

**必须精读**

---

# 十二、Scaling Law

## 2. Scaling Laws for Neural Language Models

Kaplan et al., 2020

### 为什么读

理解：

> 模型规模、数据规模、算力与 loss 的关系。

### 链接

- OpenAI 页面：<https://openai.com/index/scaling-laws-for-neural-language-models/>
- arXiv：<https://arxiv.org/abs/2001.08361>
- PDF：<https://arxiv.org/pdf/2001.08361>

### 阅读等级

**必须理解**

---

## 3. Training Compute-Optimal Large Language Models（Chinchilla）

Hoffmann et al., 2022

### 为什么读

它修正了早期“只增大模型”的 scaling 思路。

核心问题：

> 给定固定计算预算，模型参数和训练 token 应该如何分配？

### 链接

- arXiv：<https://arxiv.org/abs/2203.15556>
- PDF：<https://arxiv.org/pdf/2203.15556>

### 阅读等级

**必须精读**

---

# 十三、LLM 能力形成

## 4. Language Models are Few-Shot Learners（GPT-3）

Brown et al., 2020

### 链接

- arXiv：<https://arxiv.org/abs/2005.14165>
- PDF：<https://arxiv.org/pdf/2005.14165>

### 重点

理解：

- In-context Learning
- Few-shot Learning
- Scaling 后能力变化

### 阅读等级

**理解核心思想即可**

---

# 十四、Instruction Tuning

## 5. Finetuned Language Models Are Zero-Shot Learners（FLAN）

Wei et al., 2021

### 链接

- arXiv：<https://arxiv.org/abs/2109.01652>
- PDF：<https://arxiv.org/pdf/2109.01652>

### 重点

理解：

> 为什么 instruction tuning 能显著增强模型遵循自然语言任务描述的能力。

### 阅读等级

**理解核心思想**

---

# 十五、RLHF / Alignment

## 6. Training Language Models to Follow Instructions with Human Feedback

Ouyang et al., 2022

即 InstructGPT。

### 链接

- arXiv：<https://arxiv.org/abs/2203.02155>
- PDF：<https://arxiv.org/pdf/2203.02155>

### 必须理解

完整训练链：

```text
Pretrained LM
    ↓
SFT
    ↓
Human Preference Data
    ↓
Reward Model
    ↓
PPO / RLHF
    ↓
Aligned Model
```

### 阅读等级

**必须精读**

---

# 十六、Preference Optimization

## 7. Direct Preference Optimization（DPO）

Rafailov et al., 2023

### 重点

理解：

- 为什么可以不显式训练 Reward Model；
- 为什么 DPO 相比传统 PPO-RLHF 更简单；
- preference optimization 的数学目标。

### 链接

- arXiv：<https://arxiv.org/abs/2305.18290>
- PDF：<https://arxiv.org/pdf/2305.18290>

### 阅读等级

**必须精读**

---

# 十七、Reasoning

## 8. Chain-of-Thought Prompting Elicits Reasoning in Large Language Models

Wei et al., 2022

### 链接

- arXiv：<https://arxiv.org/abs/2201.11903>
- PDF：<https://arxiv.org/pdf/2201.11903>

### 重点

理解：

- intermediate reasoning steps；
- reasoning capability；
- scale 对推理能力的影响。

### 阅读等级

**必须精读**

---

# 十八、Agent 基础论文

## 9. ReAct: Synergizing Reasoning and Acting in Language Models

Yao et al., 2022 / ICLR 2023

### 重要性

这是理解现代 Agent 架构最值得读的论文之一。

核心思想：

```text
Thought
  ↓
Action
  ↓
Observation
  ↓
Thought
  ↓
Action
```

### 链接

- arXiv：<https://arxiv.org/abs/2210.03629>
- PDF：<https://arxiv.org/pdf/2210.03629>
- 项目主页：<https://react-lm.github.io/>

### 阅读等级

**必须精读**

---

# 十九、RAG

## 10. Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks

Lewis et al., 2020

### 核心思想

模型同时利用：

```text
Parametric Memory
+
Non-parametric Memory
```

即：

```text
LLM Parameters
+
Retriever / Vector Index
```

### 链接

- arXiv：<https://arxiv.org/abs/2005.11401>
- PDF：<https://arxiv.org/pdf/2005.11401>

### 阅读等级

**必须精读**

---

# 二十、Tool Use

## 11. Toolformer: Language Models Can Teach Themselves to Use Tools

Schick et al., 2023

### 链接

- arXiv：<https://arxiv.org/abs/2302.04761>
- PDF：<https://arxiv.org/pdf/2302.04761>

### 重点

理解：

> 模型如何学习什么时候调用工具、调用什么工具、如何利用工具结果继续生成。

### 阅读等级

**推荐精读**

---

# 二十一、训练 / 推理系统

## 12. FlashAttention

Dao et al., 2022

### 链接

- arXiv：<https://arxiv.org/abs/2205.14135>
- PDF：<https://arxiv.org/pdf/2205.14135>
- 官方 GitHub：<https://github.com/Dao-AILab/flash-attention>

### 重点

不要只记“FlashAttention 更快”。

真正理解：

- HBM
- SRAM
- IO complexity
- Tiling
- 为什么减少 memory traffic 能显著加速 Attention

### 阅读等级

**系统方向重点精读**

---

# 二十二、MoE

## 13. Switch Transformers

Fedus et al., 2021

### 链接

- arXiv：<https://arxiv.org/abs/2101.03961>
- PDF：<https://arxiv.org/pdf/2101.03961>

### 重点

理解：

```text
Dense Model
vs.
Sparse Mixture-of-Experts
```

以及：

- Router
- Expert
- Load balancing
- Parameter scaling

### 阅读等级

**理解核心思想**

---

# 二十三、Parameter-Efficient Fine-Tuning

## 14. LoRA: Low-Rank Adaptation of Large Language Models

Hu et al., 2021

### 链接

- arXiv：<https://arxiv.org/abs/2106.09685>
- PDF：<https://arxiv.org/pdf/2106.09685>

### 重点

理解：

- 为什么冻结原模型；
- 为什么低秩矩阵能有效适配；
- rank 的含义；
- LoRA 与 full fine-tuning 的权衡。

### 阅读等级

**必须掌握**

---

# 二十四、多模态基础

## 15. CLIP

Learning Transferable Visual Models From Natural Language Supervision

Radford et al., 2021

### 链接

- arXiv：<https://arxiv.org/abs/2103.00020>
- PDF：<https://arxiv.org/pdf/2103.00020>
- OpenAI GitHub：<https://github.com/openai/CLIP>

### 重点

理解：

```text
Image Encoder
       │
       ▼
Embedding Space
       ▲
       │
Text Encoder
```

以及：

- Contrastive Learning
- Image-text alignment
- Zero-shot classification

### 阅读等级

**做 VLM 必须精读**

---

# 二十五、VLM

## 16. Visual Instruction Tuning（LLaVA）

Liu et al., 2023

### 链接

- arXiv：<https://arxiv.org/abs/2304.08485>
- PDF：<https://arxiv.org/pdf/2304.08485>
- 项目主页：<https://llava-vl.github.io/>
- GitHub：<https://github.com/haotian-liu/LLaVA>

### 重点

理解经典 VLM 架构：

```text
Vision Encoder
    ↓
Projection / Connector
    ↓
LLM
```

以及 Visual Instruction Tuning。

### 阅读等级

**多模态方向必须精读**

---

# 二十六、统一多模态模型

## 17. Chameleon: Mixed-Modal Early-Fusion Foundation Models

Meta, 2024

### 链接

- arXiv：<https://arxiv.org/abs/2405.09818>
- PDF：<https://arxiv.org/pdf/2405.09818>
- GitHub：<https://github.com/facebookresearch/chameleon>

### 重点

理解：

- early fusion；
- image / text unified token sequence；
- mixed-modal generation；
- unified multimodal modeling。

### 阅读等级

**进阶**

---

## 18. Transfusion

Transfusion: Predict the Next Token and Diffuse Images with One Multi-Modal Model

### 链接

- arXiv：<https://arxiv.org/abs/2408.11039>
- PDF：<https://arxiv.org/pdf/2408.11039>
- Meta Research：<https://ai.meta.com/research/publications/transfusion-predict-the-next-token-and-diffuse-images-with-one-multi-modal-model/>

### 核心思想

统一：

```text
Text → Next-token prediction

Image → Diffusion

        ↓

Shared Transformer
```

### 阅读等级

**进阶**

---

# 二十七、推荐阅读优先级

## S 级：必须精读

这些论文建议真正做笔记：

1. Attention Is All You Need
2. Training Compute-Optimal Large Language Models
3. InstructGPT
4. DPO
5. Chain-of-Thought
6. ReAct
7. RAG
8. LoRA
9. CLIP
10. LLaVA

---

## A 级：理解主要思想

1. Scaling Laws for Neural Language Models
2. GPT-3
3. FLAN
4. Toolformer
5. FlashAttention
6. Switch Transformers

---

## B 级：进入相关方向后再深入

1. Chameleon
2. Transfusion
3. 更复杂 Reasoning RL / RLVR 论文
4. 大规模分布式训练论文
5. 推理优化论文
6. Agent Evaluation
7. World Model
8. Robotics Agent

---

# 二十八、学习顺序建议

## 阶段 1：深度学习基础

主资料：

- D2L

目标：

```text
PyTorch
+
Backprop
+
Optimization
+
CNN
+
Transformer
```

---

## 阶段 2：LLM 底层体系

主资料：

- Stanford CS336

配合论文：

1. Attention Is All You Need
2. Scaling Laws
3. Chinchilla
4. FlashAttention
5. Switch Transformer

目标：

真正理解：

```text
Data
↓
Tokenizer
↓
Transformer
↓
Pretraining
↓
Scaling
↓
Distributed Training
↓
Inference
```

---

# 二十九、阶段 3：Post-training

学习：

```text
SFT
↓
RLHF
↓
DPO
↓
Reasoning RL / RLVR
↓
Inference-time Scaling
```

论文：

- InstructGPT
- DPO
- Chain-of-Thought

目标：

理解：

> 为什么 pretrained model 不等于 ChatGPT 类产品。

---

# 三十、阶段 4：RAG / Agent

课程：

- Berkeley LLM Agents

论文：

- RAG
- ReAct
- Toolformer

自己实现至少一个：

```text
LLM
+
Memory
+
RAG
+
Tool Calling
+
State
+
Planning
```

的小型 Agent。

不要一上来完全依赖 LangChain / LangGraph。

建议先手写核心循环，再使用框架。

---

# 三十一、阶段 5：多模态

论文：

1. CLIP
2. LLaVA
3. Chameleon
4. Transfusion

目标：

理解：

```text
Vision Encoder
↓
Visual Tokens
↓
Projector
↓
LLM
↓
Multimodal Reasoning
```

然后进一步研究：

```text
Image
Video
Audio
Sensor
Time-series
Industrial Signals
```

如何统一进入 Agent。

---

# 三十二、10 周建议版本

如果希望快速形成完整框架，可以按以下方式执行。

## Week 1

D2L：

- MLP
- Backprop
- Optimization
- CNN

---

## Week 2

D2L：

- Attention
- Transformer

论文：

- Attention Is All You Need

---

## Week 3

CS336：

- Tokenization
- Transformer implementation
- Training

论文：

- GPT-3

---

## Week 4

CS336：

- GPU
- FlashAttention
- Distributed Training

论文：

- FlashAttention

---

## Week 5

CS336：

- Scaling Laws
- Data

论文：

- Scaling Laws
- Chinchilla

---

## Week 6

CS336：

- SFT
- RLHF
- DPO
- RLVR

论文：

- InstructGPT
- DPO
- Chain-of-Thought

---

## Week 7

Berkeley LLM Agents：

- Reasoning
- Planning
- Tool Use

论文：

- ReAct
- Toolformer

---

## Week 8

RAG：

论文：

- RAG

项目：

```text
Industrial Knowledge RAG
```

包含：

- 文档 ingestion
- chunk
- embedding
- retrieval
- rerank
- generation
- citation
- evaluation

---

## Week 9

Multimodal：

论文：

- CLIP
- LLaVA

项目：

```text
Image
+
Text
+
LLM
+
Tool
```

---

## Week 10

Agent：

综合：

```text
VLM
+
RAG
+
Tool
+
Workflow
+
Memory
+
Evaluation
```

构造一个：

> Industrial Multimodal Agent

---

# 三十三、最终知识结构

最终不要把知识记成一堆论文。

应该形成下面的结构：

```text
                    Foundation Model
                          │
            ┌─────────────┴─────────────┐
            │                           │
          Model                       System
            │                           │
     Transformer                 GPU / Distributed
            │                           │
          MoE                     Memory / IO
            │                           │
       Pretraining                Inference
            │                           │
       Scaling Law                Serving
            │
       Post-training
            │
    ┌───────┼────────┐
    │       │        │
   SFT     DPO      RL/RLVR
    │
Reasoning
    │
    ├── CoT
    ├── Search
    ├── Planning
    │
   Agent
    │
    ├── RAG
    ├── Tool
    ├── Memory
    ├── Workflow
    └── Evaluation
    │
Multimodal
    │
    ├── CLIP
    ├── VLM
    ├── Image
    ├── Video
    └── Sensor
    │
Industrial AI Agent
```

---

# 三十四、资料选择原则

以后看到新的课程或论文，可以用以下问题判断是否值得投入：

### 1. 它补的是哪一层？

```text
Model
Training
Inference
Post-training
Reasoning
Agent
Multimodal
System
```

如果说不出来，大概率暂时不需要学。

### 2. 它是否替代已有资料？

例如一个新的“LLM 入门教程”，如果只是重新讲 Transformer，则没有必要重新学。

### 3. 是基础知识还是时效性知识？

基础：

- Transformer
- Optimization
- Scaling
- Distributed Systems

应该深入。

时效性知识：

- 某个 Agent Framework API
- 某个 Prompt 技巧
- 某个模型产品

应该快速理解，不要重仓记忆。

---

# 三十五、建议长期跟踪的网站

### Stanford CS336

<https://cs336.stanford.edu/>

### Stanford CS224N

<https://web.stanford.edu/class/cs224n/>

### Stanford CS25

<https://web.stanford.edu/class/cs25/>

### Berkeley RDI

<https://rdi.berkeley.edu/publicCourses>

### Hugging Face Learn

<https://huggingface.co/learn>

### arXiv

<https://arxiv.org/>

### Papers with Code

<https://paperswithcode.com/>

---

# 三十六、最终推荐

如果时间有限，只维护四条主线：

```text
1. D2L
   ↓
深度学习基础

2. Stanford CS336
   ↓
LLM 原理 + 训练 + 系统

3. Berkeley LLM Agents
   ↓
Reasoning + Tool + RAG + Agent

4. 核心论文
   ↓
Post-training + Multimodal + 前沿
```

不要试图同时学习十几门课程。

真正重要的是：

> **用一门课建立体系，用论文升级体系，用项目验证体系。**

