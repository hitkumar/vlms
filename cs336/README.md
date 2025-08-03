
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
