
Lecture Notes

Lecture 2
- Pytorch and Resource accounting
- Helpful video about floating point formats: https://www.youtube.com/watch?v=bbkcEiUjehk
- Introduces a lot of good basics about calculating memory and compute requirements for a model. Check notebook for a good review.

Lecture 3
- Talks about some good tricks about LLM hyperparameters
- Weight decay point was interesting, can be tried.
- Stability tricks are good to know, refer to his slides for more details.
- Best resource for practical LLM tricks.

Lecture 4
- MOE
- Talks about the routing function to get top k experts
- Stability issues with MOE
- Load balancing loss is critical to get all experts to be used. Several aspects of this are not yet fully clear including aux z-loss.
- Deepseek series and OLMOe are good papers to review next.
- Implement MOE in vlms.

Lecture 5
- Talks about GPUs and how to use them effectively
- Shared memory and global memory (DRAM)
- Tiling, burst sections, memory coallescing are some key ideas to understand GPU performance for models
- Flash attention walkthrough - tiling, operator fusion and online softmax are key ideas.

Lecture 6
- Talks about benchmarking and profiling to start the lecture.
- Very important to profile, pytorch profiler is one initial tool to use, but use Nvidia profiler for additional insights
- Triton kernels are a good way to fuse operations and improve performance
- Torch compile should be the first thing to try to improve performance

Lecture 7
- Distributed training (part 1)
- Talks about different types of parallelism - Data Parallel, Model Parallel (Tensor Parallel, Pipeline Parallel) and sequence parallel
- Zero Stage 1 and 2 have no communication overhead and are a no brainer to use. Zero stage 3 (FSDP) has overhead, but still worth it.
- Pipeline parallel is hard to implement and bubbles remain an issue.
- Tensor Parallel is easier to implement, but only works within a single node due to high communication overhead.
- Sequence parallel makes pointwise operations like LayerNorm also parallelized as it helps distribute the activations across machines.
- Rule of thumb is fit the model in memory (using Tensor parallel and then pipeline parallel) and then use DP to utilize all the GPUs
- Batch size is an important lever for PP and DP.
- Typically combine these parallelism approaches (3D, 4D parallelism)
- TPU Book and HuggingFace book remain important references for more details.

Lecture 8
- Explain parallelism techniques using code.
We want to saturate our GPUs
- NVLink vs NVSwitch
- World Size vs rank
- Collective operations
    - Broadcast
    - Scatter
- All-reduce = reduce-Scatter + all-gather
- Dist barrier is important
- Data parallelism (cut along the batch dimension)
- Tensor parallelism (cut along the hidden dimension)
    - Split along the hidden dims
- Pipeline parallelism
- JAX handles this automatically once we define the sharding structure.

Lecture 9
- Scaling Laws
- Empiricial in nature, we don't have good bounds for theoretical scaling laws.
- Data scaling: follows power law, which means linear relationship between log-data size and log-model error.
- When scaling data size, you want the model to be big enough so that you are not in irreproducible error regime.
- Scaling model size, learning rate (muP) and batch size (critical batch size) are important.
- Kaplan scaling was off with predicting the coeffs of how loss changes with scaling model size and number of tokens.
- Chinchilla paper does a better job of doing this (especially Approach 2)
- Number of tokens per parameter is increasing as inference costs increase -> goal is to train the best small model.
- Karpathy's scaling law notebook is a good one to repro Chinchilla results: https://github.com/karpathy/nanoGPT/blob/master/scaling_laws.ipynb

Lecture 10
- LLM Inference
- Metrics that matter: Time to first token, Latency (seconds/token), Throughput (tokens/second)
- Open source packages: vLLM, SGLang, Tensor-RT
- We want arithmetic intensity to he high during inference.
- Two stages of Inference
  - Prefill: Fill the kv-cache given a prompt in parallel like training
  - Generation: Generate one token at a time sequenctially.
- Prefill is compute limited, generation is memory limited.
- Attention arithmetic intensity is especially hard to improve as each sequence needs its own Q,K,V vectors. In MLP, each sequence has the same MLP weights.
- Inference is memory limited.
- For reducing memory consumption, we try to reduce the size of KV cache.
    - Grouped Query Attention (use k keys/values for N queries)
    - MLA used by Deepseek et.al
    - Cross Layer Attention (share KVs across layers just as GQA shares them across heads)
    - Local Attention which looks at just the local context, not the whole sequence. For example, sliding window attention.
- SSM and Diffusion Models are alternatives to transformers that are more efficient.
- Quantization
   - Reduces memory so better for inference.
- Model pruning
   - Trim the model to make it smaller and then use distillation to achieve parity with the original model.
- Speculative decoding

Lecture 11
- Scaling Laws continued
- Chinchilla recipe states that model size and training tokens should scale together. Optimal ratio is 20 tokens per parameters, so for a 70B param model, it should be trained on 1.4T tokens.
- Cerebras-GPT
  - Uses chinchilla recipe for scaling
  - muP parametrization - scale learning rate and initialize as usual using fan_in.
  - MiniCPM does the same
- Chinchilla trains a new model from scratch for every experiment which is expensive. This is due to cosine learning rate. In cosine decay, rate of LR depends on total number of steps.
- WSD is an alternative as we can reuse the warmup and stable part for each iteration.
- Trend is to increase data to model ratio from 20.
- Deepseek estimates optimal LR and batch size using scaling law analysis, don't use muP.
- IsoFlop style scaling curve is commonly used.
- muP allows you to use the same base LR as you scale the model, initialization scales by 1/sqrt(m_width) and lr by 1/m_width. m_width is just scaling factor d_m / d_mbase.
- Can go a lot deeper.

Lecture 12
- Evaluating LLMs
- Find out how good the model is.
- Benchmarks are the way to measure this recently (MMLU, GPQA etc)
- Andre Karpathy points to an evaluation crisis, we are not sure how good the LLMs are anymore.
- Perplexity is a common metric to measure model performance. It is exp(-sum(logprob)/N) where N is the number of tokens. Lower is better.
- There are some tasks that are close to perplexity like LAMBADA, HellaSwag.
- Knowledge Benchmarks
  - MMLU: multiple choice questions, more about testing knowledge than language understanding.
  - MMLU-Pro
  - GPQA
  - Humanity's Last Exam
- Instruction Following Benchmarks
  - Chatbot Arena
  - IFEval
  - AlpacaEval
- Agent Benchmarks
  - SWEBench
  - MLEBench
- Reasoning benchmarks
  - ARC-AGI
- Safety benchmarks
  - Jailbreaking
  - Pre deployment testing of models.
- Model capability and propensity matters.
- We evaluate systems not just models these days.

Lecture 13
- Data
- What data should we train on.
- Data is the most important component to get right in training LLMs, which is one reason why companies don't disclose it.
- Data quality scales with human effort.
- From pre-training to post-training, data volume decreases, but data quality improves.
- Pretraining datasets
  - Books corpus
  - Wikipedia
  - gpt-2 webtext
    - Pages that are outgoing links from reddit posts with >= 3 karma (surrogate for quality)
  - CommonCrawl
    - Runs a web crawl every month.
    - Two formats: WARC (raw HTTP response like HTML), WET (converted to text)
    - There are several tools to convert HTML to text and conversion matters for downstream task accuracy.
  - CCNet
    - Keeps documents that look like Wikipedia using a KenLM 5-gram model.
  - C4
    - Produced with T5 paper.
    - Filters Common Crawl using manual heuristics, resulting dataset showed promising results.
  - The Pile
    - Effort to produce more open source datasets comparable to what gpt-3 used.
  - Project Gutenberg: mostly books that have received copyright clearance
  - Books3: taken down due to copyright issues.
  - Stackexchange: QA datasets
  - Github: coding data
    - The Stack is one collection of permissively licensed code.
- Previously rule based quality filtering was more common, model based filtering was avoided as it could introduce biases.
- DCLM-pool -> DCLM-baseline
- Nemotron-cc uses classifier ensemble to not filter as much as dclm, but still produce high quality data.
- copyright
  - key issue in GenAI
  - Applies to original works, not collections, so things like telephone directories are not copyrightable.
  - Applies to expression, not idea like Quicksort
  - Most things on internet like personal websites are actually copyrighted
  - To use a copyrighted work
    - Get a license
    - Appeal to fair use
- Instruction datasets

Lecture 14
- Data continued
- Data filtering
  - KenLM model which is a n-gram language model, runs fast and is available on HuggingFace. Based on Kneser-Ney smoothing.
  - fasttext classifier based on word embeddings. Very simple and fast model, but very effective to filter large datasets.
  - DSIR which involves data selection via importance resampling.
- Filtering applications:
  - language identification like filter only english text. Fasttext language identification is good. Dolma used this to filter out non-english text.
  - quality filtering. Model based filtering is becoming the norm.
  - Toxicity filtering, use a classifier to filter out toxic text.
- Deduplication: exact duplicates vs near duplicates. Important not only for improving efficiency of training, but also to avoid memorization caused by seeing the same data many times.
- Exact match filtering used by C4 using hashing like murmurhash3.
- Bloom filters are a efficienct approximate data structure to test set membership.
  - If return no, then it is definitely not a duplicate.
  - If return yes, then mostly yes, small probabability of a no. So, there could be some false positives.
  - False positive rate is configurable by using more bins or by using more hash functions.
  - Go deeper by implementing a bloom filter in pytorch.
- Jaccard similarity is a good way to measure similarity between two sets: intersection/union.
- MinHash is a hash function so that Pr[h(A) = h(B)] = Jaccard(A,B)
- Locality sensitive hashing builds on minhash to sharpen the probability.
- Use n hash functions, break up into b bands of r hash functions each. n = b * r
- A and B collide if for some band, all hash functions return the same value. This and or construction really sharpens the probabability of being a near duplicate.

Lecture 15
- RLHF / Alignment
- Post training
- Instruction Following
- Collect data of behaviors we do want from LLM like chat, safety.
- First post training step is SFT, typically QA data.
- Fine-tuning a model on facts it doesn't know makes it hallucinate. It is tricky to know what model does know though. We should teach the model how to extract behaviors learnt during pre-training in a desired format. Adding new factual data can hurt the model sometimes.
- Safety tuning is easy conceptually, but tricky part is to balance this with over-refusals. Model will refuse to answer even simple questions.
- SFT data is now pretty big, so we incorporate SFT data during mid-training, this also helps with avoiding catastrophic forgetting.
- After SFT, we do RLHF.
- RLHF maximizes the reward from the model response.
- LMs are policies
- Typically pairwise feedback
- PPO is one approach, although it is quite tricky to make it work in practice
- DPO is a good alternate. It is not clear which of PPO or DPO is better.

Lecture 16
- Reinforcement Learning from Verifiable Rewards (RLVR)
- DPO improves the LM policy by using pos gradient on y(w) and neg gradient on y(l)
- Scales by prediction error and tries to maximize the likelihood of y(w)
- RLVR is what gets us reasoning.
- PPO is trick to implement, and needs a value model during training which adds additional memory overhead.
- DPO inherently operates on pairwise data, so can't be used as is in RLVR settings.
- GRPO is a new algo which removes these limitations.
- Uses rewards within a group to calculate advantages
- Dr.GRPO improves this in a couple of ways: removes division by std, only center rewards around mean.
  - Also removes length normalization. Length norm makes the correct answers shorter and incorrect answers bigger as model is incentivized to blabber to get small penalties for mistakes.
- Few case studies
- Deepseek R1
  - Use outcome rewards functions: accuracy and format rewards
  - SFT init was useful for RL.
  - Usual SFT/RLHF happens after RLVR.
- Kimi K1.5
  - Uses curriculum learning to train from easy to hard samples
  - Considers samples model can't answer correctly for RL: not too difficult, not too easy
- Qwen 3
  - Filtering for difficulty
  - RL with GRPO on only 3995 examples
  - Show test time scaling with more thinking budget.
  - Need to read more about thinking fusion.

Lecture 17
- Dive Deep into the mechanics of RLVR
- RL systems are more complex than pretraining as you need to manage inference workloads, and manage multiple models.
- Rewards, which are usually outcome rewards and verifiable rewards.
- Policy: pi(a/s)
- Naive policy gradient suffers from high variance, so we use baseline to get the advantage function (rewards - baseline)
- GRPO improves on PPO as it removes the critic (aka value function)
- Implementing this is tricky and requires a lot of details to get right.
